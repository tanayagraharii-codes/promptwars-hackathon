"""Researcher agent: gathers evidence for one sub-question and extracts claims."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.llm import LLM
from app.schemas import Evidence, ExtractionOutput, PlannedQuestion
from app.search import Retriever
from app.security import wrap_untrusted
from app.sources import SourceRegistry, format_sources_for_prompt

SYSTEM = (
    "You are a Researcher in a multi-agent fact-checking team. From the evidence provided, "
    "extract the most important specific, checkable factual claims (numbers, dates, study "
    "results, official positions). Each claim must be a single self-contained sentence and "
    "must cite the source IDs (like S3) that state it. Never invent sources or facts that are "
    "not in the evidence. If sources disagree, extract each side as a separate claim. "
    "Claims must be about the topic, never about the sources themselves (e.g. not 'the "
    "sources do not mention X'). Do not repeat the same claim in different words."
)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def is_near_duplicate(statement: str, existing: list[str], threshold: float = 0.7) -> bool:
    """Jaccard similarity on content words: catches paraphrased repeats cheaply."""
    words = _words(statement)
    for other in existing:
        other_words = _words(other)
        union = words | other_words
        if union and len(words & other_words) / len(union) >= threshold:
            return True
    return False


@dataclass
class ResearchResult:
    index: int
    question: PlannedQuestion
    evidence: Evidence
    claims: list[tuple[str, list[str]]]  # (statement, source ids)


class ResearcherAgent:
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
        limit = (
            self._max_claims if max_claims is None else max(1, min(max_claims, self._max_claims))
        )
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
        claims: list[tuple[str, list[str]]] = []
        for claim in output.claims[:limit]:
            statement = claim.statement.strip()
            if statement and not is_near_duplicate(statement, [c for c, _ in claims]):
                ids = [i for i in dict.fromkeys(claim.source_ids) if i in valid_ids]
                claims.append((statement[:500], ids))
        return ResearchResult(index, question, evidence, claims)
