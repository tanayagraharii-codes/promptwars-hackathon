"""End-to-end tests of the agent pipeline with a fake LLM and retriever."""

from __future__ import annotations

from app.agents.researcher import ResearchResult
from app.config import Settings
from app.events import Event
from app.llm import LLMError
from app.orchestrator import ResearchOrchestrator, assemble_claims
from app.schemas import (
    Depth,
    Evidence,
    ExtractionOutput,
    FactCheckOutput,
    PlannedQuestion,
    PlanOutput,
    Report,
    Verdict,
)
from app.storage import MemoryReportStore, ReportStore
from tests.fakes import FakeLLM, FakeRetriever

QUESTION = "Did X increase?"


def make_orchestrator(
    llm: FakeLLM, retriever: FakeRetriever, settings: Settings, store: ReportStore | None = None
) -> ResearchOrchestrator:
    return ResearchOrchestrator(llm, retriever, store or MemoryReportStore(10, 60), settings)


async def collect(
    orchestrator: ResearchOrchestrator, question: str = QUESTION, depth: Depth = Depth.STANDARD
) -> list[Event]:
    return [event async for event in orchestrator.run(question, depth)]


def final_report(events: list[Event]) -> Report:
    last = events[-1]
    assert last["type"] == "report"
    return Report.model_validate(last["report"])


async def test_full_pipeline_produces_a_validated_report(fake_llm, fake_retriever, settings):
    events = await collect(make_orchestrator(fake_llm, fake_retriever, settings))

    finished = [e["agent"] for e in events if e["type"] == "agent" and e["status"] == "done"]
    assert finished == [
        "planner",
        "researcher",
        "fact_checker",
        "conflict_detector",
        "report_writer",
    ]
    report = final_report(events)
    assert len(report.sub_questions) == 2
    assert [c.id for c in report.claims] == ["C1", "C2", "C3", "C4"]
    assert {c.verdict for c in report.claims} == {Verdict.SUPPORTED, Verdict.DISPUTED}
    known = {s.id for s in report.sources}
    for checked in report.claims:
        assert set(checked.supporting_source_ids + checked.contradicting_source_ids) <= known
    assert report.conflicts[0].claim_ids == ["C1", "C2"]
    assert all(set(p.citations) <= known for s in report.sections for p in s.paragraphs)
    assert report.search_backend == "fake"


async def test_every_sub_question_is_researched(fake_llm, fake_retriever, settings):
    await collect(make_orchestrator(fake_llm, fake_retriever, settings))
    assert set(fake_retriever.queries[:2]) == {
        "What does official data say?",
        "What do critics say?",
    }


async def test_repeat_question_is_served_from_cache(fake_llm, fake_retriever, settings):
    orchestrator = make_orchestrator(fake_llm, fake_retriever, settings)
    await collect(orchestrator)
    calls_after_first_run = len(fake_llm.calls)

    events = await collect(orchestrator, "  did x INCREASE ")
    assert [e["type"] for e in events] == ["cached", "report"]
    assert len(fake_llm.calls) == calls_after_first_run


async def test_quick_depth_limits_breadth_and_is_cached_separately(
    fake_llm, fake_retriever, settings
):
    orchestrator = make_orchestrator(fake_llm, fake_retriever, settings)

    quick = final_report(await collect(orchestrator, depth=Depth.QUICK))
    assert quick.depth == Depth.QUICK
    assert len(quick.sub_questions) <= 2
    per_question: dict[int, int] = {}
    for checked in quick.claims:
        per_question[checked.sub_question] = per_question.get(checked.sub_question, 0) + 1
    assert max(per_question.values()) <= 2
    assert "between 2 and 2 sub-questions" in fake_llm.calls[0][2]

    standard_events = await collect(orchestrator, depth=Depth.STANDARD)
    assert standard_events[0]["type"] != "cached"
    assert final_report(standard_events).id != quick.id


async def test_llm_outage_ends_with_a_friendly_error(fake_llm, fake_retriever, settings):
    fake_llm.responses[PlanOutput] = LLMError("down", status=429)
    events = await collect(make_orchestrator(fake_llm, fake_retriever, settings))
    assert events[-1] == {
        "type": "error",
        "message": "The AI service is busy or out of quota. Please try again in a minute.",
    }


async def test_rejected_request_has_its_own_message(fake_llm, fake_retriever, settings):
    fake_llm.responses[PlanOutput] = LLMError("bad request", status=400)
    events = await collect(make_orchestrator(fake_llm, fake_retriever, settings))
    assert events[-1] == {"type": "error", "message": "The AI service rejected the request."}


async def test_unexpected_errors_are_not_leaked(fake_llm, fake_retriever, settings):
    fake_llm.responses[PlanOutput] = RuntimeError("secret internal detail")
    events = await collect(make_orchestrator(fake_llm, fake_retriever, settings))
    assert events[-1] == {"type": "error", "message": "Something went wrong while researching."}


async def test_no_claims_ends_with_an_error(fake_llm, fake_retriever, settings):
    fake_llm.responses[ExtractionOutput] = ExtractionOutput(claims=[])
    events = await collect(make_orchestrator(fake_llm, fake_retriever, settings))
    assert events[-1] == {
        "type": "error",
        "message": "No usable evidence was found for this topic.",
    }


async def test_failed_researcher_does_not_sink_the_run(fake_llm, fake_retriever, settings):
    extract = fake_llm.responses[ExtractionOutput]
    calls = 0

    def flaky(prompt: str) -> ExtractionOutput:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("one branch fails")
        result: ExtractionOutput = extract(prompt)
        return result

    fake_llm.responses[ExtractionOutput] = flaky
    report = final_report(await collect(make_orchestrator(fake_llm, fake_retriever, settings)))
    assert report.claims  # the other researcher's claims still made it through


async def test_failed_fact_check_keeps_claims_unverified(fake_llm, fake_retriever, settings):
    fake_llm.responses[FactCheckOutput] = LLMError("busy", status=503)
    report = final_report(await collect(make_orchestrator(fake_llm, fake_retriever, settings)))
    assert all(c.verdict == Verdict.UNVERIFIED for c in report.claims)


def test_assemble_claims_numbers_in_order_and_drops_cross_researcher_repeats():
    evidence = Evidence(text="", sources=[], backend="fake")
    question = PlannedQuestion(question="q", search_query="q")
    results = [
        ResearchResult(1, question, evidence, [("Prices rose ten percent in 2024.", ["S2"])]),
        ResearchResult(0, question, evidence, [("Wages fell in 2023 across sectors.", ["S1"])]),
        ResearchResult(2, question, evidence, [("Prices rose ten percent in 2024", ["S3"])]),
    ]
    claims = assemble_claims(results)
    assert [(c.id, c.sub_question) for c in claims] == [("C1", 0), ("C2", 1)]
