"""Researcher agent: gathers evidence for one sub-question and extracts claims."""

from __future__ import annotations

from dataclasses import dataclass

from app.llm import LLM
from app.retrieval import Retriever
from app.schemas import Evidence, ExtractionOutput, PlannedQuestion
from app.security import wrap_untrusted
from app.sources import SourceRegistry, format_sources_for_prompt
from app.text import is_near_duplicate

from .base import clamp_limit, known_ids

SYSTEM = (
    "You are a Researcher in a multi-agent fact-checking team. From the evidence provided, "
    "extract the most important specific, checkable factual claims (numbers, dates, study "
    "results, official positions). Each claim must be a single self-contained sentence and "
    "must cite the source IDs (like S3) that state it. Never invent sources or facts that are "
    "not in the evidence. If sources disagree, extract each side as a separate claim. "
    "Claims must be about the topic, never about the sources themselves (e.g. not 'the "
    "sources do not mention X'). Do not repeat the same claim in different words."
)
MAX_STATEMENT_CHARS = 500

ExtractedClaims = list[tuple[str, list[str]]]  # (statement, cited source IDs)


@dataclass(frozen=True)
class ResearchResult:
    """Evidence gathered for one sub-question and the claims extracted from it."""

    index: int
    question: PlannedQuestion
    evidence: Evidence
    claims: ExtractedClaims


class ResearcherAgent:
    """Searches for evidence on one sub-question and extracts checkable claims."""

    name = "researcher"

    def __init__(self, llm: LLM, retriever: Retriever, max_claims: int) -> None:
        self._llm = llm
        self._retriever = retriever
        self._max_claims = max_claims

    async def run(
        self,
        index: int,
        question: PlannedQuestion,
        registry: SourceRegistry,
        max_claims: int | None = None,
    ) -> ResearchResult:
        """Research ``question``; skips the model call entirely when no evidence is found."""
        limit = clamp_limit(max_claims, self._max_claims)
        evidence = await self._retriever.search(
            question.question, registry, keywords=question.search_query
        )
        if not evidence.text.strip():
            return ResearchResult(index, question, evidence, [])

        prompt = (
            f"Sub-question: {question.question}\n\n"
            f"Sources:\n{format_sources_for_prompt(evidence.sources)}\n\n"
            f"{wrap_untrusted('research', evidence.text)}\n\n"
            f"Extract up to {limit} claims most relevant to the sub-question. Partial or "
            "indirect evidence counts: extract what the sources do say about the topic. Return "
            "an empty list only if the evidence is entirely unrelated to the topic."
        )
        output = await self._llm.generate_json(prompt, ExtractionOutput, system=SYSTEM)

        valid_ids = {s.id for s in evidence.sources}
        claims: ExtractedClaims = []
        for claim in output.claims[:limit]:
            statement = claim.statement.strip()
            if statement and not is_near_duplicate(statement, [c for c, _ in claims]):
                claims.append(
                    (statement[:MAX_STATEMENT_CHARS], known_ids(claim.source_ids, valid_ids))
                )
        return ResearchResult(index, question, evidence, claims)
