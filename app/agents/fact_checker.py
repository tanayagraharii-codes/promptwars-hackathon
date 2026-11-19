"""Fact-checker agent: verifies claims against independent evidence."""

from __future__ import annotations

from app.llm import LLM
from app.schemas import Claim, FactCheckOutput, Verdict
from app.search import Retriever
from app.security import wrap_untrusted
from app.sources import SourceRegistry, format_sources_for_prompt

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

UNVERIFIED_CAP = 0.35


def score_confidence(
    verdict: Verdict,
    model_confidence: float,
    supporting: list[float],
    contradicting: list[float],
) -> float:
    """Blend the model's confidence with the credibility of the cited evidence.

    The model can be over-confident, so the final score is anchored to how
    trustworthy and how many the cited sources are.
    """
    model_confidence = max(0.0, min(1.0, model_confidence))
    relevant = supporting if verdict == Verdict.SUPPORTED else contradicting
    if verdict == Verdict.DISPUTED:
        relevant = supporting + contradicting
    if verdict == Verdict.UNVERIFIED or not relevant:
        return round(min(model_confidence, UNVERIFIED_CAP), 2)
    top = sorted(relevant, reverse=True)[:3]
    evidence_strength = sum(top) / len(top)
    corroboration = min(1.0, 0.6 + 0.2 * (len(relevant) - 1))  # 1 src → 0.6, 3+ → 1.0
    score = 0.5 * model_confidence + 0.5 * evidence_strength * corroboration
    return round(max(0.0, min(1.0, score)), 2)


class FactCheckerAgent:
    name = "fact_checker"

    def __init__(self, llm: LLM, retriever: Retriever) -> None:
        self._llm = llm
        self._retriever = retriever

    async def run(self, claims: list[Claim], registry: SourceRegistry) -> list[Claim]:
        if not claims:
            return []
        verification = await self._retriever.search_many([c.statement for c in claims], registry)

        cited = {i for c in claims for i in c.source_ids}
        pool_ids = cited | {s.id for s in verification.sources}
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

        checked: list[Claim] = []
        for claim in claims:
            judgement = judgements.get(claim.id)
            if judgement is None:
                checked.append(claim.model_copy(update={"rationale": "Not assessed."}))
                continue
            supporting = [
                i for i in dict.fromkeys(judgement.supporting_source_ids) if i in pool_ids
            ]
            contradicting = [
                i for i in dict.fromkeys(judgement.contradicting_source_ids) if i in pool_ids
            ]
            verdict = judgement.verdict
            if verdict != Verdict.UNVERIFIED and not (supporting or contradicting):
                verdict = Verdict.UNVERIFIED  # a verdict with no evidence cannot stand
            confidence = score_confidence(
                verdict,
                judgement.confidence,
                [_cred(registry, i) for i in supporting],
                [_cred(registry, i) for i in contradicting],
            )
            checked.append(
                claim.model_copy(
                    update={
                        "verdict": verdict,
                        "confidence": confidence,
                        "rationale": judgement.rationale.strip()[:600],
                        "supporting_source_ids": supporting,
                        "contradicting_source_ids": contradicting,
                    }
                )
            )
        return checked


def _cred(registry: SourceRegistry, source_id: str) -> float:
    source = registry.get(source_id)
    return source.credibility if source else 0.0
