"""Coordinates the agents and streams progress events to the client.

Pipeline:  Planner → Researchers (parallel) → Fact-Checkers (parallel)
           → Conflict Detector → Report Writer
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from app.agents import (
    ConflictDetectorAgent,
    FactCheckerAgent,
    PlannerAgent,
    ReportWriterAgent,
    ResearcherAgent,
)
from app.agents.researcher import is_near_duplicate
from app.config import Settings
from app.llm import LLM, LLMError
from app.schemas import Claim, Depth, Report
from app.search import Retriever
from app.sources import SourceRegistry
from app.storage import ReportStore, report_key

logger = logging.getLogger(__name__)

Event = dict[str, Any]

# Quick mode trades breadth for speed: ~40% of the model calls of a standard run.
QUICK_SUB_QUESTIONS = 2
QUICK_CLAIMS_PER_QUESTION = 2


def _agent(agent: str, status: str, detail: str) -> Event:
    return {"type": "agent", "agent": agent, "status": status, "detail": detail}


class ResearchOrchestrator:
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
        """Run the full pipeline, yielding progress events and finally the report."""
        key = report_key(question, depth.value)
        cached = await self._store.get(key)
        if cached is not None:
            yield {"type": "cached", "detail": "Served from cache: answered recently."}
            yield {"type": "report", "report": cached.model_dump(mode="json")}
            return

        started = time.perf_counter()
        try:
            async for event in self._pipeline(key, question, depth, started):
                yield event
        except LLMError as exc:
            logger.warning("pipeline_llm_error", extra={"status": exc.status})
            message = (
                "The AI service is busy or out of quota. Please try again in a minute."
                if exc.status in {429, 503, 504, None}
                else "The AI service rejected the request."
            )
            yield {"type": "error", "message": message}
        except Exception:
            logger.exception("pipeline_failed")
            yield {"type": "error", "message": "Something went wrong while researching."}

    async def _pipeline(
        self, key: str, question: str, depth: Depth, started: float
    ) -> AsyncIterator[Event]:
        registry = SourceRegistry()
        quick = depth == Depth.QUICK
        sub_question_limit = QUICK_SUB_QUESTIONS if quick else None
        claim_limit = QUICK_CLAIMS_PER_QUESTION if quick else None

        # 1. Plan
        yield _agent("planner", "running", "Breaking the topic into sub-questions…")
        plan = await self.planner.run(question, sub_question_limit)
        yield _agent("planner", "done", f"Planned {len(plan)} sub-questions.")
        yield {"type": "plan", "sub_questions": [p.model_dump() for p in plan]}

        # 2. Research all sub-questions concurrently
        yield _agent("researcher", "running", f"Researching {len(plan)} sub-questions in parallel…")
        tasks = [
            asyncio.create_task(self.researcher.run(i, q, registry, claim_limit))
            for i, q in enumerate(plan)
        ]
        results = []
        for done, future in enumerate(asyncio.as_completed(tasks), start=1):
            try:
                results.append(await future)
            except Exception as exc:  # noqa: BLE001 - one failed branch must not sink the rest
                logger.warning("researcher_failed", extra={"error": type(exc).__name__})
            yield _agent("researcher", "progress", f"Finished {done} of {len(plan)} sub-questions.")
        results.sort(key=lambda r: r.index)

        claims: list[Claim] = []
        for result in results:
            for statement, source_ids in result.claims:
                if is_near_duplicate(statement, [c.statement for c in claims]):
                    continue  # the same fact surfaced by two researchers
                claims.append(
                    Claim(
                        id=f"C{len(claims) + 1}",
                        statement=statement,
                        sub_question=result.index,
                        source_ids=source_ids,
                    )
                )
        if not claims:
            yield {"type": "error", "message": "No usable evidence was found for this topic."}
            return
        yield _agent(
            "researcher",
            "done",
            f"Extracted {len(claims)} claims from {len(registry.all())} sources "
            f"(via {self._retriever.name}).",
        )

        # 3. Fact-check each sub-question's claims concurrently
        yield _agent("fact_checker", "running", f"Verifying {len(claims)} claims…")
        groups: dict[int, list[Claim]] = {}
        for claim in claims:
            groups.setdefault(claim.sub_question, []).append(claim)
        checked_groups = await asyncio.gather(
            *(self.fact_checker.run(group, registry) for group in groups.values()),
            return_exceptions=True,
        )
        checked: list[Claim] = []
        for original, outcome in zip(groups.values(), checked_groups, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning("fact_check_group_failed", extra={"error": type(outcome).__name__})
                checked.extend(original)
            else:
                checked.extend(outcome)
        checked.sort(key=lambda c: int(c.id[1:]))
        supported = sum(c.verdict == "supported" for c in checked)
        yield _agent("fact_checker", "done", f"{supported} of {len(checked)} claims supported.")
        yield {"type": "claims", "claims": [c.model_dump(mode="json") for c in checked]}

        # 4. Detect conflicts
        yield _agent("conflict_detector", "running", "Looking for contradictory evidence…")
        conflicts = await self.conflict_detector.run(checked)
        yield _agent("conflict_detector", "done", f"Found {len(conflicts)} conflicts.")

        # 5. Write the report
        yield _agent("report_writer", "running", "Writing the evidence-backed report…")
        sources = registry.all()
        draft = await self.report_writer.run(question, checked, conflicts, sources)
        yield _agent("report_writer", "done", "Report ready.")

        report = Report(
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
            claims=checked,
            conflicts=conflicts,
            sources=sources,
            duration_s=round(time.perf_counter() - started, 1),
        )
        await self._store.put(report)
        logger.info(
            "report_completed",
            extra={"claims": len(checked), "sources": len(sources), "seconds": report.duration_s},
        )
        yield {"type": "report", "report": report.model_dump(mode="json")}
