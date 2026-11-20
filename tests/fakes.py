"""Test doubles: a scripted LLM and a deterministic retriever (no network, no credentials)."""

from __future__ import annotations

import itertools
import zlib
from collections.abc import Callable
from typing import Any

from app.llm import GroundedResult
from app.schemas import (
    ClaimJudgement,
    ConflictOutput,
    DetectedConflict,
    Evidence,
    ExtractedClaim,
    ExtractionOutput,
    FactCheckOutput,
    PlannedQuestion,
    PlanOutput,
    ReportDraft,
    ReportParagraph,
    ReportSection,
    Verdict,
)
from app.sources import SourceRegistry

Responder = Callable[[str], Any]


class FakeLLM:
    """Returns a scripted output per schema and records every prompt it receives.

    A response may be a model instance, a callable taking the prompt, or an
    exception to raise.
    """

    def __init__(self, responses: dict[type, Any] | None = None) -> None:
        self.responses: dict[type, Any] = responses or {}
        self.calls: list[tuple[str, str, str]] = []  # (schema name, tier, prompt)
        self.grounded: GroundedResult | Exception | None = None

    async def generate_json(
        self, prompt: str, schema: type, *, system: str, tier: str = "fast"
    ) -> Any:
        self.calls.append((schema.__name__, tier, prompt))
        response = self.responses.get(schema)
        if isinstance(response, Exception):
            raise response
        if callable(response) and not isinstance(response, type):
            return response(prompt)
        if response is None:
            raise AssertionError(f"No fake response for {schema.__name__}")
        return response

    async def grounded_search(self, prompt: str) -> GroundedResult:
        self.calls.append(("grounded_search", "search", prompt))
        if isinstance(self.grounded, Exception):
            raise self.grounded
        return self.grounded or GroundedResult(text="", sources=[])


class FakeRetriever:
    """Each query yields two deterministic sources: a .gov page and a blog."""

    name = "fake"

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(
        self, query: str, registry: SourceRegistry, keywords: str | None = None
    ) -> Evidence:
        self.queries.append(query)
        slug = zlib.crc32(query.encode())
        gov = registry.add(f"https://agency.gov/{slug}", "Agency report", "Official statistics.")
        blog = registry.add(f"https://someblog.blogspot.com/{slug}", "A blog", "Opinion piece.")
        return Evidence(
            text=f"[{gov.id}] Official data on {query}. [{blog.id}] A blog disagrees.",
            sources=[gov, blog],
            backend=self.name,
        )

    async def search_many(self, queries: list[str], registry: SourceRegistry) -> Evidence:
        merged = [await self.search(q, registry) for q in queries]
        return Evidence(
            text="\n".join(e.text for e in merged),
            sources=[s for e in merged for s in e.sources],
            backend=self.name,
        )


class ScriptedExtraction:
    """Extraction responder: distinct facts per call, plus a hallucinated ID and a repeat."""

    FACTS = (
        (
            "Official statistics show a ten percent increase in output.",
            "A personal blog insists nothing changed.",
        ),
        (
            "Government surveys report fewer commuting hours overall.",
            "Critics allege managers observed lower engagement.",
        ),
        (
            "University researchers measured stable quarterly results.",
            "Forum posters describe widespread burnout.",
        ),
    )

    def __init__(self) -> None:
        self._facts = itertools.cycle(self.FACTS)

    def __call__(self, prompt: str) -> ExtractionOutput:
        ids = [token.strip("[]") for token in prompt.split() if token.startswith("[S")]
        first = ids[0] if ids else "S1"
        supported, disputed = next(self._facts)
        return ExtractionOutput(
            claims=[
                ExtractedClaim(statement=supported, source_ids=[first]),
                ExtractedClaim(statement=disputed, source_ids=["S999"]),  # hallucinated ID
                ExtractedClaim(statement=supported.rstrip("."), source_ids=[first]),  # repeat
            ]
        )


def scripted_fact_check(prompt: str) -> FactCheckOutput:
    """Even-numbered claims are supported, odd ones disputed, citing pool source IDs."""
    claim_ids = [line.split(":")[0] for line in prompt.splitlines() if line.startswith("C")]
    pool = [line[1:].split("]")[0] for line in prompt.splitlines() if line.startswith("[S")]
    first, second = [*pool, "S1", "S2"][:2]
    return FactCheckOutput(
        judgements=[
            ClaimJudgement(
                claim_id=claim_id,
                verdict=Verdict.SUPPORTED,
                confidence=0.9,
                rationale="Official data.",
                supporting_source_ids=[first],
                contradicting_source_ids=[],
            )
            if i % 2 == 0
            else ClaimJudgement(
                claim_id=claim_id,
                verdict=Verdict.DISPUTED,
                confidence=0.6,
                rationale="Sources disagree.",
                supporting_source_ids=[second],
                contradicting_source_ids=[first, "S404"],  # S404 does not exist
            )
            for i, claim_id in enumerate(claim_ids)
        ]
    )


def default_responses() -> dict[type, Any]:
    """A complete, realistic script for one pipeline run (with deliberate bad IDs)."""
    return {
        PlanOutput: PlanOutput(
            sub_questions=[
                PlannedQuestion(
                    question="What does official data say?", search_query="official data"
                ),
                PlannedQuestion(question="What do critics say?", search_query="criticism"),
                PlannedQuestion(question="what does official data say?", search_query="dupe"),
            ]
        ),
        ExtractionOutput: ScriptedExtraction(),
        FactCheckOutput: scripted_fact_check,
        ConflictOutput: ConflictOutput(
            conflicts=[
                DetectedConflict(
                    topic="Size of increase",
                    claim_ids=["C1", "C2", "C99"],
                    explanation="Official data and the blog use different baselines.",
                    reason="methodology",
                ),
                DetectedConflict(topic="Bogus", claim_ids=["C42"], explanation="x", reason="other"),
            ]
        ),
        ReportDraft: ReportDraft(
            title="Test report",
            summary="Evidence mostly supports an increase.",
            key_findings=["Official data shows an increase.", "  "],
            sections=[
                ReportSection(
                    heading="Findings",
                    paragraphs=[
                        ReportParagraph(
                            text="Official data shows growth.", citations=["S1", "S77"]
                        ),
                        ReportParagraph(text="   ", citations=["S1"]),
                    ],
                ),
                ReportSection(heading="Empty", paragraphs=[]),
            ],
            limitations=["Small sample of sources."],
        ),
    }
