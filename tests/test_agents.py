"""Tests for the five agents, focusing on validation of untrusted model output."""

from __future__ import annotations

import pytest

from app.agents import (
    ConflictDetectorAgent,
    FactCheckerAgent,
    PlannerAgent,
    ReportWriterAgent,
    ResearcherAgent,
    score_confidence,
)
from app.agents.base import clamp_limit, clean_lines, known_ids
from app.schemas import (
    Claim,
    ClaimJudgement,
    Evidence,
    FactCheckOutput,
    PlannedQuestion,
    PlanOutput,
    Verdict,
)
from app.sources import SourceRegistry
from tests.fakes import FakeLLM, FakeRetriever


class EmptyRetriever:
    name = "empty"

    async def search(
        self, query: str, registry: SourceRegistry, keywords: str | None = None
    ) -> Evidence:
        return Evidence(text="  ", sources=[], backend=self.name)

    async def search_many(self, queries: list[str], registry: SourceRegistry) -> Evidence:
        return Evidence(text="", sources=[], backend=self.name)


def make_claim(claim_id: str, verdict: Verdict = Verdict.UNVERIFIED, rationale: str = "") -> Claim:
    return Claim(
        id=claim_id,
        statement=f"Statement {claim_id}",
        sub_question=0,
        source_ids=[],
        verdict=verdict,
        rationale=rationale,
    )


def judgement(
    claim_id: str, verdict: Verdict, supporting: list[str], contradicting: list[str]
) -> ClaimJudgement:
    return ClaimJudgement(
        claim_id=claim_id,
        verdict=verdict,
        confidence=0.9,
        rationale="Reason.",
        supporting_source_ids=supporting,
        contradicting_source_ids=contradicting,
    )


# ----------------------------------------------------------------- shared helpers


def test_known_ids_filters_and_deduplicates_in_order():
    assert known_ids(["S2", "S9", "S1", "S2"], {"S1", "S2"}) == ["S2", "S1"]


@pytest.mark.parametrize(("requested", "expected"), [(None, 4), (2, 2), (0, 1), (9, 4)])
def test_clamp_limit(requested: int | None, expected: int):
    assert clamp_limit(requested, 4) == expected


def test_clean_lines():
    assert clean_lines([" a ", "", "  ", "b", "c"], limit=2) == ["a", "b"]


# ----------------------------------------------------------------- planner


async def test_planner_deduplicates_and_caps(fake_llm: FakeLLM):
    plan = await PlannerAgent(fake_llm, max_sub_questions=3).run("Topic?")
    assert [p.question for p in plan] == ["What does official data say?", "What do critics say?"]


async def test_planner_respects_a_smaller_per_request_limit(fake_llm: FakeLLM):
    plan = await PlannerAgent(fake_llm, max_sub_questions=3).run("Topic?", limit=1)
    assert len(plan) == 1
    assert "between 1 and 1 sub-questions" in fake_llm.calls[0][2]


async def test_planner_falls_back_to_the_original_question():
    llm = FakeLLM({PlanOutput: PlanOutput(sub_questions=[])})
    plan = await PlannerAgent(llm, max_sub_questions=3).run("Is coffee healthy for adults?")
    assert plan == [
        PlannedQuestion(
            question="Is coffee healthy for adults?", search_query="Is coffee healthy for adults?"
        )
    ]


# ----------------------------------------------------------------- researcher


async def test_researcher_drops_unknown_ids_and_repeated_claims(
    fake_llm: FakeLLM, fake_retriever: FakeRetriever
):
    question = PlannedQuestion(question="What does official data say?", search_query="data")
    result = await ResearcherAgent(fake_llm, fake_retriever, max_claims=5).run(
        0, question, SourceRegistry()
    )
    assert len(result.claims) == 2  # the third claim repeats the first
    assert result.claims[0][1] == ["S1"]
    assert result.claims[1][1] == []  # S999 was hallucinated and removed
    assert "<evidence" in fake_llm.calls[-1][2]  # web text is fenced as untrusted


async def test_researcher_skips_the_model_when_there_is_no_evidence(fake_llm: FakeLLM):
    question = PlannedQuestion(question="Q", search_query="q")
    result = await ResearcherAgent(fake_llm, EmptyRetriever(), 3).run(0, question, SourceRegistry())
    assert result.claims == []
    assert fake_llm.calls == []


# ----------------------------------------------------------------- confidence scoring


def test_confidence_rewards_credible_corroboration():
    weak = score_confidence(Verdict.SUPPORTED, 0.9, [0.3], [])
    strong = score_confidence(Verdict.SUPPORTED, 0.9, [0.9, 0.9, 0.9], [])
    assert 0 <= weak < strong <= 1


def test_confidence_is_capped_without_evidence():
    assert score_confidence(Verdict.UNVERIFIED, 0.95, [0.9], []) <= 0.35
    assert score_confidence(Verdict.SUPPORTED, 0.95, [], []) <= 0.35
    assert score_confidence(Verdict.REFUTED, 0.8, [], [0.9]) > 0.35
    assert score_confidence(Verdict.SUPPORTED, 5.0, [1.0, 1.0, 1.0], []) == 1.0


@pytest.mark.parametrize("verdict", list(Verdict))
def test_confidence_stays_in_bounds(verdict: Verdict):
    assert 0.0 <= score_confidence(verdict, 0.5, [0.5], [0.5]) <= 1.0


# ----------------------------------------------------------------- fact-checker


async def test_fact_checker_validates_ids_and_downgrades_evidence_free_verdicts(
    fake_llm: FakeLLM, fake_retriever: FakeRetriever
):
    registry = SourceRegistry()
    registry.add("https://agency.gov/base", "Agency")
    claims = [
        Claim(id="C1", statement="Official data shows growth.", sub_question=0, source_ids=["S1"]),
        Claim(id="C2", statement="A blog says no growth.", sub_question=0, source_ids=[]),
        Claim(id="C3", statement="Unjudged claim", sub_question=0, source_ids=[]),
    ]
    fake_llm.responses[FactCheckOutput] = FactCheckOutput(
        judgements=[
            judgement("C1", Verdict.SUPPORTED, supporting=["S1", "S1", "S500"], contradicting=[]),
            judgement("C2", Verdict.REFUTED, supporting=[], contradicting=["S500"]),  # missing
        ]
    )
    checked = await FactCheckerAgent(fake_llm, fake_retriever).run(claims, registry)

    assert checked[0].verdict == Verdict.SUPPORTED
    assert checked[0].supporting_source_ids == ["S1"]
    assert checked[1].verdict == Verdict.UNVERIFIED
    assert checked[1].confidence <= 0.35
    assert checked[2].rationale == "Not assessed."
    assert fake_llm.calls[-1][1] == "smart"


async def test_fact_checker_does_nothing_for_no_claims(
    fake_llm: FakeLLM, fake_retriever: FakeRetriever
):
    assert await FactCheckerAgent(fake_llm, fake_retriever).run([], SourceRegistry()) == []
    assert fake_llm.calls == []


# ----------------------------------------------------------------- conflict detector


async def test_conflict_detector_keeps_only_known_claims(fake_llm: FakeLLM):
    claims = [
        make_claim("C1", Verdict.SUPPORTED),
        make_claim("C2", Verdict.DISPUTED, rationale="Sources disagree."),
    ]
    conflicts = await ConflictDetectorAgent(fake_llm).run(claims)
    assert [(k.id, k.claim_ids) for k in conflicts] == [("K1", ["C1", "C2"])]


async def test_conflict_detector_skips_a_single_undisputed_claim(fake_llm: FakeLLM):
    assert await ConflictDetectorAgent(fake_llm).run([make_claim("C1")]) == []
    assert fake_llm.calls == []


# ----------------------------------------------------------------- report writer


async def test_report_writer_strips_invalid_citations_and_empty_content(fake_llm: FakeLLM):
    registry = SourceRegistry()
    registry.add("https://who.int/x", "WHO")
    draft = await ReportWriterAgent(fake_llm).run("Q?", [], [], registry.all())
    assert [len(s.paragraphs) for s in draft.sections] == [1]
    assert draft.sections[0].paragraphs[0].citations == ["S1"]
    assert draft.key_findings == ["Official data shows an increase."]
