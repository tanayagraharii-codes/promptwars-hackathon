"""Planner agent: decomposes a complex topic into researchable sub-questions."""

from __future__ import annotations

from app.llm import LLM
from app.schemas import PlannedQuestion, PlanOutput

from .base import clamp_limit

SYSTEM = (
    "You are the Planner in a multi-agent fact-checking team. Break the user's topic into "
    "distinct, non-overlapping sub-questions that together cover the evidence needed for a "
    "balanced answer. Include at least one sub-question that looks for counter-evidence, "
    "criticism or competing viewpoints. Each sub-question must be answerable with facts."
)
MAX_QUESTION_CHARS = 300
MAX_QUERY_CHARS = 120


class PlannerAgent:
    """Turns one broad question into focused sub-questions plus keyword queries."""

    name = "planner"

    def __init__(self, llm: LLM, max_sub_questions: int) -> None:
        self._llm = llm
        self._max = max_sub_questions

    async def run(self, question: str, limit: int | None = None) -> list[PlannedQuestion]:
        """Plan up to ``limit`` sub-questions (never more than the configured maximum)."""
        limit = clamp_limit(limit, self._max)
        prompt = (
            f"Topic: {question}\n\n"
            f"Return between {min(2, limit)} and {limit} sub-questions, each with a short 2-4 word "
            "keyword search query naming the core concepts in full words, never abbreviations "
            "(e.g. 'telecommuting productivity', 'electric car emissions')."
        )
        plan = await self._llm.generate_json(prompt, PlanOutput, system=SYSTEM, tier="fast")
        return normalise_plan(plan.sub_questions, question, limit)


def normalise_plan(
    items: list[PlannedQuestion], question: str, limit: int
) -> list[PlannedQuestion]:
    """De-duplicate and trim the plan; fall back to the original question if it is empty."""
    seen: set[str] = set()
    plan: list[PlannedQuestion] = []
    for item in items:
        text = item.question.strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        query = item.search_query.strip() or text
        plan.append(
            PlannedQuestion(
                question=text[:MAX_QUESTION_CHARS], search_query=query[:MAX_QUERY_CHARS]
            )
        )
        if len(plan) == limit:
            break
    return plan or [PlannedQuestion(question=question, search_query=question[:MAX_QUERY_CHARS])]
