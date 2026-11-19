import pytest

from app.agents import (
    ConflictDetectorAgent,
    FactCheckerAgent,
    PlannerAgent,
    ReportWriterAgent,
    ResearcherAgent,
    score_confidence,
)
from app.schemas import Claim, PlannedQuestion, PlanOutput, Verdict
from app.sources import SourceRegistry
from tests.conftest import FakeLLM


async def test_planner_deduplicates_and_caps(fake_llm):
    plan = await PlannerAgent(fake_llm, max_sub_questions=3).run("Topic?")
    assert [p.question for p in plan] == ["What does official data say?", "What do critics say?"]


async def test_planner_falls_back_to_original_question():
    llm = FakeLLM({PlanOutput: PlanOutput(sub_questions=[])})
    plan = await PlannerAgent(llm, max_sub_questions=3).run("Is coffee healthy for adults?")
    assert plan == [
        PlannedQuestion(
            question="Is coffee healthy for adults?", search_query="Is coffee healthy for adults?"
        )
    ]


async def test_researcher_drops_unknown_source_ids(fake_llm, fake_retriever):
    registry = SourceRegistry()
    q = PlannedQuestion(question="What does official data say?", search_query="official data")
    result = await ResearcherAgent(fake_llm, fake_retriever, max_claims=5).run(0, q, registry)
    assert len(result.claims) == 2  # third claim was a near-duplicate
    assert result.claims[0][1] == ["S1"]
    assert result.claims[1][1] == []  # S999 was hallucinated and removed
    # Evidence was fenced as untrusted content in the prompt
    assert "<evidence" in fake_llm.calls[-1][2]


async def test_researcher_skips_llm_when_no_evidence(fake_llm):
    class EmptyRetriever:
        name = "empty"

        async def search(self, query, registry, keywords=None):
            from app.schemas import Evidence

            return Evidence(text="  ", sources=[], backend="empty")

    q = PlannedQuestion(question="Q", search_query="q")
    result = await ResearcherAgent(fake_llm, EmptyRetriever(), 3).run(0, q, SourceRegistry())
    assert result.claims == []
    assert fake_llm.calls == []


def test_score_confidence_rewards_credible_corroboration():
    weak = score_confidence(Verdict.SUPPORTED, 0.9, [0.3], [])
    strong = score_confidence(Verdict.SUPPORTED, 0.9, [0.9, 0.9, 0.9], [])
    assert strong > weak
    assert 0 <= weak <= 1 and 0 <= strong <= 1


def test_score_confidence_caps_unverified_and_evidence_free():
    assert score_confidence(Verdict.UNVERIFIED, 0.95, [0.9], []) <= 0.35
    assert score_confidence(Verdict.SUPPORTED, 0.95, [], []) <= 0.35
    assert score_confidence(Verdict.REFUTED, 0.8, [], [0.9]) > 0.35
    assert score_confidence(Verdict.SUPPORTED, 5.0, [1.0, 1.0, 1.0], []) == 1.0


async def test_fact_checker_validates_ids_and_downgrades_unsupported(fake_llm, fake_retriever):
    registry = SourceRegistry()
    registry.add("https://agency.gov/base", "Agency")
    claims = [
        Claim(
            id="C1",
            statement="Official data shows a 10% increase.",
            sub_question=0,
            source_ids=["S1"],
        ),
        Claim(id="C2", statement="A blog says no increase.", sub_question=0, source_ids=[]),
        Claim(id="C3", statement="Unjudged claim", sub_question=0, source_ids=[]),
    ]

    def judge(prompt):
        from app.schemas import ClaimJudgement, FactCheckOutput

        return FactCheckOutput(
            judgements=[
                ClaimJudgement(
                    claim_id="C1",
                    verdict="supported",
                    confidence=0.9,
                    rationale="ok",
                    supporting_source_ids=["S1", "S1", "S500"],
                    contradicting_source_ids=[],
                ),
                ClaimJudgement(
                    claim_id="C2",
                    verdict="refuted",
                    confidence=0.9,
                    rationale="no ids",
                    supporting_source_ids=[],
                    contradicting_source_ids=["S500"],
                ),
            ]
        )

    from app.schemas import FactCheckOutput

    fake_llm.responses[FactCheckOutput] = judge
    checked = await FactCheckerAgent(fake_llm, fake_retriever).run(claims, registry)

    assert checked[0].verdict == Verdict.SUPPORTED
    assert checked[0].supporting_source_ids == ["S1"]
    assert checked[1].verdict == Verdict.UNVERIFIED  # cited only a non-existent source
    assert checked[1].confidence <= 0.35
    assert checked[2].rationale == "Not assessed."
    assert fake_llm.calls[-1][1] == "smart"


async def test_fact_checker_noop_on_empty(fake_llm, fake_retriever):
    assert await FactCheckerAgent(fake_llm, fake_retriever).run([], SourceRegistry()) == []


async def test_conflict_detector_filters_unknown_claims(fake_llm):
    claims = [
        Claim(id="C1", statement="a", sub_question=0, source_ids=[], verdict=Verdict.SUPPORTED),
        Claim(
            id="C2",
            statement="b",
            sub_question=1,
            source_ids=[],
            verdict=Verdict.DISPUTED,
            rationale="disagree",
        ),
    ]
    conflicts = await ConflictDetectorAgent(fake_llm).run(claims)
    assert len(conflicts) == 1
    assert conflicts[0].id == "K1"
    assert conflicts[0].claim_ids == ["C1", "C2"]


async def test_conflict_detector_skips_single_undisputed_claim(fake_llm):
    claims = [Claim(id="C1", statement="a", sub_question=0, source_ids=[])]
    assert await ConflictDetectorAgent(fake_llm).run(claims) == []
    assert fake_llm.calls == []


async def test_report_writer_strips_invalid_citations(fake_llm):
    registry = SourceRegistry()
    registry.add("https://who.int/x", "WHO")
    draft = await ReportWriterAgent(fake_llm).run("Q?", [], [], registry.all())
    assert len(draft.sections) == 1
    assert draft.sections[0].paragraphs[0].citations == ["S1"]
    assert len(draft.sections[0].paragraphs) == 1
    assert draft.key_findings == ["Official data shows an increase."]


@pytest.mark.parametrize("verdict", list(Verdict))
def test_score_confidence_bounds(verdict):
    assert 0.0 <= score_confidence(verdict, 0.5, [0.5], [0.5]) <= 1.0


def test_near_duplicate_detection():
    from app.agents.researcher import is_near_duplicate

    existing = ["Opponents argue remote technology cannot fully replace face-to-face contact."]
    assert is_near_duplicate(
        "Opponents argue remote technology cannot fully replace face to face contact", existing
    )
    assert not is_near_duplicate("Remote workers saved 40 minutes of commuting daily.", existing)
    assert not is_near_duplicate("Anything", [])
