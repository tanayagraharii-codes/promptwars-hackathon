"""Planner agent: decomposes a complex topic into researchable sub-questions."""

from __future__ import annotations

from app.llm import LLM
from app.schemas import PlannedQuestion, PlanOutput

SYSTEM = (
    "You are the Planner in a multi-agent fact-checking team. Break the user's topic into "
    "distinct, non-overlapping sub-questions that together cover the evidence needed for a "
    "balanced answer. Include at least one sub-question that looks for counter-evidence, "
    "criticism or competing viewpoints. Each sub-question must be answerable with facts."
)


class PlannerAgent:
    name = "planner"

    def __init__(self, llm: LLM, max_sub_questions: int) -> None:
        self._llm = llm
        self._max = max_sub_questions

    async def run(self, question: str, limit: int | None = None) -> list[PlannedQuestion]:
        """Plan up to ``limit`` sub-questions (never more than the configured maximum)."""
        limit = self._max if limit is None else max(1, min(limit, self._max))
        prompt = (
            f"Topic: {question}\n\n"
            f"Return between {min(2, limit)} and {limit} sub-questions, each with a short 2-4 word "
            "keyword search query naming the core concepts in full words, never abbreviations "
            "(e.g. 'telecommuting productivity', 'electric car emissions')."
        )
        plan = await self._llm.generate_json(prompt, PlanOutput, system=SYSTEM, tier="fast")
        return self._normalise(plan.sub_questions, question, limit)

    def _normalise(
        self, items: list[PlannedQuestion], question: str, limit: int
    ) -> list[PlannedQuestion]:
        seen: set[str] = set()
        result: list[PlannedQuestion] = []
        for item in items:
            text = item.question.strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            query = item.search_query.strip() or text
            result.append(PlannedQuestion(question=text[:300], search_query=query[:120]))
            if len(result) == limit:
                break
        return result or [PlannedQuestion(question=question, search_query=question[:120])]
