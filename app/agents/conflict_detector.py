"""Conflict-detector agent: finds claims that contradict each other and explains why."""

from __future__ import annotations

from app.llm import LLM
from app.schemas import Claim, Conflict, ConflictOutput, Verdict

SYSTEM = (
    "You are the Conflict Detector in a multi-agent fact-checking team. Identify groups of "
    "claims that contradict or are in tension with each other, and claims marked disputed. "
    "For each conflict explain, neutrally, the most likely reason the sources disagree "
    "(different methodology, timeframe, definitions, source bias, outdated data, or other). "
    "Only reference claim IDs from the list. Return an empty list if there are no conflicts."
)


class ConflictDetectorAgent:
    name = "conflict_detector"

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def run(self, claims: list[Claim]) -> list[Conflict]:
        if len(claims) < 2 and not any(c.verdict == Verdict.DISPUTED for c in claims):
            return []
        lines = "\n".join(
            f"{c.id} [{c.verdict.value}, confidence {c.confidence:.2f}]: {c.statement}"
            + (f" — fact-check note: {c.rationale}" if c.rationale else "")
            for c in claims
        )
        prompt = f"Claims:\n{lines}"
        output = await self._llm.generate_json(prompt, ConflictOutput, system=SYSTEM, tier="smart")

        known = {c.id for c in claims}
        conflicts: list[Conflict] = []
        for item in output.conflicts:
            ids = [i for i in dict.fromkeys(item.claim_ids) if i in known]
            if not ids or not item.explanation.strip():
                continue
            conflicts.append(
                Conflict(
                    id=f"K{len(conflicts) + 1}",
                    topic=item.topic.strip()[:200],
                    claim_ids=ids,
                    explanation=item.explanation.strip()[:800],
                    reason=item.reason,
                )
            )
        return conflicts
