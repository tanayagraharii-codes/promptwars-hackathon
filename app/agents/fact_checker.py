"""Fact-checker agent: verifies claims against independent evidence."""

from __future__ import annotations

from app.llm import LLM
from app.retrieval import Retriever
from app.schemas import Claim, ClaimJudgement, FactCheckOutput, Verdict
from app.security import wrap_untrusted
from app.sources import SourceRegistry, format_sources_for_prompt

from .base import known_ids

SYSTEM = (
    "You are the Fact-Checker in a multi-agent research team. For each claim, weigh the "
    "original sources and the independent verification evidence. Give a verdict:\n"
    "- supported: independent credible sources agree;\n"
    "- disputed: credible sources disagree or the claim needs important qualification;\n"
    "- refuted: credible evidence contradicts the claim;\n"
    "- unverified: not enough evidence either way.\n"
    "Weigh sources by their credibility score; one low-credibility source is never enough "
    "for 'supported'. Only cite source IDs that appear in the source list. Keep each "
    "rationale to one or two sentences."
)

UNVERIFIED_CAP = 0.35  # a claim without usable evidence can never look confident
TOP_SOURCES = 3  # evidence strength is the mean credibility of the best three sources
SINGLE_SOURCE_CORROBORATION = 0.6  # one source → 0.6, each extra source +0.2, max 1.0
CORROBORATION_STEP = 0.2
MODEL_WEIGHT = 0.5
MAX_RATIONALE_CHARS = 600
NOT_ASSESSED = "Not assessed."


def score_confidence(
    verdict: Verdict,
    model_confidence: float,
    supporting: list[float],
    contradicting: list[float],
) -> float:
    """Blend the model's confidence with the credibility of the cited evidence.

    The model can be over-confident, so the final score is anchored to how
    trustworthy the cited sources are and how many of them agree.
    """
    model_confidence = _clamp(model_confidence)
    if verdict == Verdict.SUPPORTED:
        relevant = supporting
    elif verdict == Verdict.DISPUTED:
        relevant = supporting + contradicting
    else:
        relevant = contradicting
    if verdict == Verdict.UNVERIFIED or not relevant:
        return round(min(model_confidence, UNVERIFIED_CAP), 2)

    top = sorted(relevant, reverse=True)[:TOP_SOURCES]
    evidence_strength = sum(top) / len(top)
    corroboration = min(1.0, SINGLE_SOURCE_CORROBORATION + CORROBORATION_STEP * (len(relevant) - 1))
    score = MODEL_WEIGHT * model_confidence + (1 - MODEL_WEIGHT) * evidence_strength * corroboration
    return round(_clamp(score), 2)


class FactCheckerAgent:
    """Runs an independent verification search and judges every claim in a batch."""

    name = "fact_checker"

    def __init__(self, llm: LLM, retriever: Retriever) -> None:
        self._llm = llm
        self._retriever = retriever

    async def run(self, claims: list[Claim], registry: SourceRegistry) -> list[Claim]:
        """Return ``claims`` with verdicts, confidence scores and validated evidence IDs."""
        if not claims:
            return []
        verification = await self._retriever.search_many([c.statement for c in claims], registry)
        pool_ids = {i for c in claims for i in c.source_ids} | {s.id for s in verification.sources}
        pool = [s for s in registry.all() if s.id in pool_ids]
        claim_lines = "\n".join(
            f"{c.id}: {c.statement} (originally cited: {', '.join(c.source_ids) or 'none'})"
            for c in claims
        )
        prompt = (
            f"Claims to verify:\n{claim_lines}\n\n"
            f"Sources:\n{format_sources_for_prompt(pool)}\n\n"
            f"{wrap_untrusted('verification', verification.text)}\n\n"
            "Return exactly one judgement per claim ID."
        )
        output = await self._llm.generate_json(prompt, FactCheckOutput, system=SYSTEM, tier="smart")
        judgements = {j.claim_id: j for j in output.judgements}
        return [
            apply_judgement(claim, judgements.get(claim.id), pool_ids, registry) for claim in claims
        ]


def apply_judgement(
    claim: Claim,
    judgement: ClaimJudgement | None,
    pool_ids: set[str],
    registry: SourceRegistry,
) -> Claim:
    """Validate a model judgement and fold it into the claim.

    Only source IDs from the evidence pool survive, and a verdict left without
    any evidence is downgraded to "unverified".
    """
    if judgement is None:
        return claim.model_copy(update={"rationale": NOT_ASSESSED})
    supporting = known_ids(judgement.supporting_source_ids, pool_ids)
    contradicting = known_ids(judgement.contradicting_source_ids, pool_ids)
    verdict = judgement.verdict
    if verdict != Verdict.UNVERIFIED and not (supporting or contradicting):
        verdict = Verdict.UNVERIFIED
    confidence = score_confidence(
        verdict,
        judgement.confidence,
        [registry.credibility(i) for i in supporting],
        [registry.credibility(i) for i in contradicting],
    )
    return claim.model_copy(
        update={
            "verdict": verdict,
            "confidence": confidence,
            "rationale": judgement.rationale.strip()[:MAX_RATIONALE_CHARS],
            "supporting_source_ids": supporting,
            "contradicting_source_ids": contradicting,
        }
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
