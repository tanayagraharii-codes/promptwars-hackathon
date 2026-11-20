"""Conflict-detector agent: finds claims that contradict each other and explains why."""

from __future__ import annotations

from app.llm import LLM
from app.schemas import Claim, Conflict, ConflictOutput, Verdict

from .base import known_ids

SYSTEM = (
    "You are the Conflict Detector in a multi-agent fact-checking team. Identify groups of "
    "claims that contradict or are in tension with each other, and claims marked disputed. "
    "For each conflict explain, neutrally, the most likely reason the sources disagree "
    "(different methodology, timeframe, definitions, source bias, outdated data, or other). "
    "Only reference claim IDs from the list. Return an empty list if there are no conflicts."
)
MAX_TOPIC_CHARS = 200
MAX_EXPLANATION_CHARS = 800


class ConflictDetectorAgent:
    """Groups contradictory claims and explains the likely cause of each disagreement."""

    name = "conflict_detector"

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def run(self, claims: list[Claim]) -> list[Conflict]:
        """Return validated conflicts (IDs K1..Kn); skips the model when none are possible."""
        if len(claims) < 2 and not any(c.verdict == Verdict.DISPUTED for c in claims):
            return []
        lines = "\n".join(
            f"{c.id} [{c.verdict.value}, confidence {c.confidence:.2f}]: {c.statement}"
            + (f" — fact-check note: {c.rationale}" if c.rationale else "")
            for c in claims
        )
        output = await self._llm.generate_json(
            f"Claims:\n{lines}", ConflictOutput, system=SYSTEM, tier="smart"
        )

        valid_ids = {c.id for c in claims}
        conflicts: list[Conflict] = []
        for item in output.conflicts:
            ids = known_ids(item.claim_ids, valid_ids)
            if not ids or not item.explanation.strip():
                continue
            conflicts.append(
                Conflict(
                    id=f"K{len(conflicts) + 1}",
                    topic=item.topic.strip()[:MAX_TOPIC_CHARS],
                    claim_ids=ids,
                    explanation=item.explanation.strip()[:MAX_EXPLANATION_CHARS],
                    reason=item.reason,
                )
            )
        return conflicts
