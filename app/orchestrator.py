"""Coordinates the five agents and streams progress events to the client.

Pipeline:  Planner → Researchers (parallel) → Fact-Checkers (parallel)
           → Conflict Detector → Report Writer
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from app.agents import (
    ConflictDetectorAgent,
    FactCheckerAgent,
    PlannerAgent,
    ReportWriterAgent,
    ResearcherAgent,
)
from app.agents.researcher import ResearchResult
from app.config import Settings
from app.events import Event, agent_event, error_event
from app.llm import LLM, LLMError
from app.retrieval import Retriever
from app.schemas import Claim, Conflict, Depth, PlannedQuestion, Report, Verdict
from app.sources import SourceRegistry
from app.storage import ReportStore, report_key
from app.text import is_near_duplicate

logger = logging.getLogger(__name__)

BUSY_STATUSES = frozenset({429, 503, 504, None})


@dataclass(frozen=True)
class DepthProfile:
    """How much breadth a run may use. ``None`` means the configured maximum."""

    sub_questions: int | None
    claims_per_question: int | None


# Quick mode trades breadth for speed: roughly 40% of the model calls of a standard run.
DEPTH_PROFILES = {
    Depth.QUICK: DepthProfile(sub_questions=2, claims_per_question=2),
    Depth.STANDARD: DepthProfile(sub_questions=None, claims_per_question=None),
}


class ResearchOrchestrator:
    """Runs the agent pipeline for a question and yields :mod:`app.events`."""

    def __init__(
        self, llm: LLM, retriever: Retriever, store: ReportStore, settings: Settings
    ) -> None:
        self._retriever = retriever
        self._store = store
        self.planner = PlannerAgent(llm, settings.max_sub_questions)
        self.researcher = ResearcherAgent(llm, retriever, settings.max_claims_per_question)
        self.fact_checker = FactCheckerAgent(llm, retriever)
        self.conflict_detector = ConflictDetectorAgent(llm)
        self.report_writer = ReportWriterAgent(llm)

    async def run(self, question: str, depth: Depth = Depth.STANDARD) -> AsyncIterator[Event]:
        """Answer from cache when possible, otherwise run the pipeline.

        Failures never escape as exceptions: they end the stream with a
        user-safe error event (details go to the logs only).
        """
        key = report_key(question, depth.value)
        cached = await self._store.get(key)
        if cached is not None:
            yield {"type": "cached", "detail": "Served from cache: answered recently."}
            yield {"type": "report", "report": cached.model_dump(mode="json")}
            return

        try:
            async for event in self._pipeline(key, question, depth):
                yield event
        except LLMError as exc:
            logger.warning("pipeline_llm_error", extra={"status": exc.status})
            yield error_event(
                "The AI service is busy or out of quota. Please try again in a minute."
                if exc.status in BUSY_STATUSES
                else "The AI service rejected the request."
            )
        except Exception:
            logger.exception("pipeline_failed")
            yield error_event("Something went wrong while researching.")

    async def _pipeline(self, key: str, question: str, depth: Depth) -> AsyncIterator[Event]:
        started = time.perf_counter()
        profile = DEPTH_PROFILES[depth]
        registry = SourceRegistry()

        yield agent_event("planner", "running", "Breaking the topic into sub-questions…")
        plan = await self.planner.run(question, profile.sub_questions)
        yield agent_event("planner", "done", f"Planned {len(plan)} sub-questions.")
        yield {"type": "plan", "sub_questions": [p.model_dump() for p in plan]}

        results: list[ResearchResult] = []
        async for event in self._research(plan, registry, profile, results):
            yield event
        claims = assemble_claims(results)
        if not claims:
            yield error_event("No usable evidence was found for this topic.")
            return
        yield agent_event(
            "researcher",
            "done",
            f"Extracted {len(claims)} claims from {len(registry.all())} sources "
            f"(via {self._retriever.name}).",
        )

        yield agent_event("fact_checker", "running", f"Verifying {len(claims)} claims…")
        checked = await self._fact_check(claims, registry)
        supported = sum(c.verdict == Verdict.SUPPORTED for c in checked)
        yield agent_event(
            "fact_checker", "done", f"{supported} of {len(checked)} claims supported."
        )
        yield {"type": "claims", "claims": [c.model_dump(mode="json") for c in checked]}

        yield agent_event("conflict_detector", "running", "Looking for contradictory evidence…")
        conflicts = await self.conflict_detector.run(checked)
        yield agent_event("conflict_detector", "done", f"Found {len(conflicts)} conflicts.")

        yield agent_event("report_writer", "running", "Writing the evidence-backed report…")
        report = await self._write_report(
            key, question, depth, plan, checked, conflicts, registry, started
        )
        await self._store.put(report)
        yield agent_event("report_writer", "done", "Report ready.")
        logger.info(
            "report_completed",
            extra={"claims": len(checked), "sources": len(report.sources), "s": report.duration_s},
        )
        yield {"type": "report", "report": report.model_dump(mode="json")}

    async def _research(
        self,
        plan: list[PlannedQuestion],
        registry: SourceRegistry,
        profile: DepthProfile,
        results: list[ResearchResult],
    ) -> AsyncIterator[Event]:
        """Research every sub-question concurrently, reporting progress as each finishes.

        Results are appended to ``results``; one failed branch never sinks the others.
        """
        total = len(plan)
        yield agent_event(
            "researcher", "running", f"Researching {total} sub-questions in parallel…"
        )
        tasks = [
            asyncio.create_task(
                self.researcher.run(i, question, registry, profile.claims_per_question)
            )
            for i, question in enumerate(plan)
        ]
        for done, task in enumerate(asyncio.as_completed(tasks), start=1):
            try:
                results.append(await task)
            except Exception as exc:  # noqa: BLE001 - isolate the failure to this branch
                logger.warning("researcher_failed", extra={"error": type(exc).__name__})
            yield agent_event(
                "researcher", "progress", f"Finished {done} of {total} sub-questions."
            )

    async def _fact_check(self, claims: list[Claim], registry: SourceRegistry) -> list[Claim]:
        """Verify each sub-question's claims as one concurrent batch.

        If a batch fails, its claims are kept with their default "unverified" verdict.
        """
        groups: dict[int, list[Claim]] = {}
        for claim in claims:
            groups.setdefault(claim.sub_question, []).append(claim)
        outcomes = await asyncio.gather(
            *(self.fact_checker.run(group, registry) for group in groups.values()),
            return_exceptions=True,
        )
        checked: list[Claim] = []
        for original, outcome in zip(groups.values(), outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning("fact_check_group_failed", extra={"error": type(outcome).__name__})
                checked.extend(original)
            else:
                checked.extend(outcome)
        return sorted(checked, key=lambda c: int(c.id.removeprefix("C")))

    async def _write_report(
        self,
        key: str,
        question: str,
        depth: Depth,
        plan: list[PlannedQuestion],
        claims: list[Claim],
        conflicts: list[Conflict],
        registry: SourceRegistry,
        started: float,
    ) -> Report:
        """Have the Report Writer draft the report, then attach the run's metadata."""
        sources = registry.all()
        draft = await self.report_writer.run(question, claims, conflicts, sources)
        return Report(
            id=key,
            question=question,
            depth=depth,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            search_backend=self._retriever.name,
            title=draft.title,
            summary=draft.summary,
            key_findings=draft.key_findings,
            sections=draft.sections,
            limitations=draft.limitations,
            sub_questions=plan,
            claims=claims,
            conflicts=conflicts,
            sources=sources,
            duration_s=round(time.perf_counter() - started, 1),
        )


def assemble_claims(results: Iterable[ResearchResult]) -> list[Claim]:
    """Number claims C1..Cn in sub-question order, dropping repeats across researchers."""
    claims: list[Claim] = []
    for result in sorted(results, key=lambda r: r.index):
        for statement, source_ids in result.claims:
            if is_near_duplicate(statement, [c.statement for c in claims]):
                continue
            claims.append(
                Claim(
                    id=f"C{len(claims) + 1}",
                    statement=statement,
                    sub_question=result.index,
                    source_ids=source_ids,
                )
            )
    return claims
