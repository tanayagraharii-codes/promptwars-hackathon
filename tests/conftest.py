"""Shared fakes: a scripted LLM and an in-memory retriever (no network, no credentials)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.config import Settings
from app.llm import GroundedResult, LLMError
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
)
from app.sources import SourceRegistry

Responder = Callable[[str], Any]


class FakeLLM:
    """Returns scripted outputs per schema; records every prompt it receives."""

    def __init__(self, responses: dict[type, Responder | Any] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, str, str]] = []  # (schema name, tier, prompt)
        self.grounded: GroundedResult | Exception | None = None

    async def generate_json(self, prompt: str, schema: type, *, system: str, tier: str = "fast"):
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

    async def search(self, query: str, registry: SourceRegistry, keywords: str | None = None):
        self.queries.append(query)
        slug = str(abs(hash(query)) % 10_000)
        gov = registry.add(f"https://agency.gov/{slug}", "Agency report", "Official statistics.")
        blog = registry.add(f"https://someblog.blogspot.com/{slug}", "A blog", "Opinion piece.")
        return Evidence(
            text=f"[{gov.id}] Official data on {query}. [{blog.id}] A blog disagrees.",
            sources=[gov, blog],
            backend=self.name,
        )

    async def search_many(self, queries: list[str], registry: SourceRegistry):
        merged = [await self.search(q, registry) for q in queries]
        return Evidence(
            text="\n".join(e.text for e in merged),
            sources=[s for e in merged for s in e.sources],
            backend=self.name,
        )


_FACTS = [
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
]


def _extraction(prompt: str) -> ExtractionOutput:
    ids = [token.strip("[]") for token in prompt.split() if token.startswith("[S")]
    first = ids[0] if ids else "S1"
    supported, disputed = _FACTS[_extraction.calls % len(_FACTS)]
    _extraction.calls += 1
    return ExtractionOutput(
        claims=[
            ExtractedClaim(statement=supported, source_ids=[first]),
            ExtractedClaim(statement=disputed, source_ids=["S999"]),
            ExtractedClaim(statement=supported.rstrip("."), source_ids=[first]),  # paraphrase
        ]
    )


_extraction.calls = 0


def _fact_check(prompt: str) -> FactCheckOutput:
    claim_ids = [line.split(":")[0] for line in prompt.splitlines() if line.startswith("C")]
    pool = [line[1:].split("]")[0] for line in prompt.splitlines() if line.startswith("[S")]
    first, second = ([*pool, "S1", "S2"])[:2]
    judgements = []
    for i, cid in enumerate(claim_ids):
        if i % 2 == 0:
            judgements.append(
                ClaimJudgement(
                    claim_id=cid,
                    verdict="supported",
                    confidence=0.9,
                    rationale="Official data.",
                    supporting_source_ids=[first],
                    contradicting_source_ids=[],
                )
            )
        else:
            judgements.append(
                ClaimJudgement(
                    claim_id=cid,
                    verdict="disputed",
                    confidence=0.6,
                    rationale="Sources disagree.",
                    supporting_source_ids=[second],
                    contradicting_source_ids=[first, "S404"],
                )
            )
    return FactCheckOutput(judgements=judgements)


def default_responses() -> dict[type, Any]:
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
        ExtractionOutput: _extraction,
        FactCheckOutput: _fact_check,
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


@pytest.fixture
def settings() -> Settings:
    return Settings(
        max_sub_questions=3,
        max_claims_per_question=3,
        rate_limit_per_minute=3,
        max_concurrent_jobs=2,
        firestore_enabled=False,
        allowed_origins=(),
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(default_responses())


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever()


__all__ = ["FakeLLM", "FakeRetriever", "LLMError", "default_responses"]
