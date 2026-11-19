from app.llm import LLMError
from app.orchestrator import ResearchOrchestrator
from app.schemas import PlanOutput, Report, Verdict
from app.storage import MemoryReportStore


async def _collect(orchestrator, question, *args):
    return [event async for event in orchestrator.run(question, *args)]


def _orchestrator(fake_llm, fake_retriever, settings, store=None):
    return ResearchOrchestrator(
        fake_llm, fake_retriever, store or MemoryReportStore(10, 60), settings
    )


async def test_full_pipeline_produces_validated_report(fake_llm, fake_retriever, settings):
    events = await _collect(_orchestrator(fake_llm, fake_retriever, settings), "Did X increase?")

    agents_done = [e["agent"] for e in events if e["type"] == "agent" and e["status"] == "done"]
    assert agents_done == [
        "planner",
        "researcher",
        "fact_checker",
        "conflict_detector",
        "report_writer",
    ]
    report = Report.model_validate(events[-1]["report"])
    assert events[-1]["type"] == "report"
    assert len(report.sub_questions) == 2
    assert [c.id for c in report.claims] == ["C1", "C2", "C3", "C4"]
    assert {c.verdict for c in report.claims} == {Verdict.SUPPORTED, Verdict.DISPUTED}
    known = {s.id for s in report.sources}
    for claim in report.claims:
        assert set(claim.supporting_source_ids + claim.contradicting_source_ids) <= known
    assert report.conflicts[0].claim_ids == ["C1", "C2"]
    assert all(set(p.citations) <= known for s in report.sections for p in s.paragraphs)
    assert report.search_backend == "fake"


async def test_researchers_run_for_every_sub_question(fake_llm, fake_retriever, settings):
    await _collect(_orchestrator(fake_llm, fake_retriever, settings), "Did X increase?")
    assert fake_retriever.queries[:2] == [
        "What does official data say?",
        "What do critics say?",
    ] or set(fake_retriever.queries[:2]) == {"What does official data say?", "What do critics say?"}


async def test_second_request_is_served_from_cache(fake_llm, fake_retriever, settings):
    store = MemoryReportStore(10, 60)
    orchestrator = _orchestrator(fake_llm, fake_retriever, settings, store)
    await _collect(orchestrator, "Did X increase?")
    calls_after_first = len(fake_llm.calls)

    events = await _collect(orchestrator, "  did x INCREASE ")
    assert [e["type"] for e in events] == ["cached", "report"]
    assert len(fake_llm.calls) == calls_after_first


async def test_llm_outage_yields_friendly_error(fake_llm, fake_retriever, settings):
    fake_llm.responses[PlanOutput] = LLMError("down", status=429)
    events = await _collect(_orchestrator(fake_llm, fake_retriever, settings), "Did X increase?")
    assert events[-1]["type"] == "error"
    assert "try again" in events[-1]["message"]


async def test_unexpected_error_is_not_leaked(fake_llm, fake_retriever, settings):
    fake_llm.responses[PlanOutput] = RuntimeError("secret internal detail")
    events = await _collect(_orchestrator(fake_llm, fake_retriever, settings), "Did X increase?")
    assert events[-1] == {"type": "error", "message": "Something went wrong while researching."}


async def test_no_claims_ends_with_error(fake_llm, fake_retriever, settings):
    from app.schemas import ExtractionOutput

    fake_llm.responses[ExtractionOutput] = ExtractionOutput(claims=[])
    events = await _collect(_orchestrator(fake_llm, fake_retriever, settings), "Did X increase?")
    assert events[-1]["type"] == "error"
    assert "No usable evidence" in events[-1]["message"]


async def test_failed_fact_check_keeps_claims_unverified(fake_llm, fake_retriever, settings):
    from app.schemas import FactCheckOutput

    fake_llm.responses[FactCheckOutput] = LLMError("busy", status=503)
    events = await _collect(_orchestrator(fake_llm, fake_retriever, settings), "Did X increase?")
    report = Report.model_validate(events[-1]["report"])
    assert all(c.verdict == Verdict.UNVERIFIED for c in report.claims)


async def test_quick_depth_limits_breadth_and_is_cached_separately(
    fake_llm, fake_retriever, settings
):
    from app.schemas import Depth

    store = MemoryReportStore(10, 60)
    orchestrator = _orchestrator(fake_llm, fake_retriever, settings, store)

    quick = Report.model_validate(
        (await _collect(orchestrator, "Did X increase?", Depth.QUICK))[-1]["report"]
    )
    assert quick.depth == Depth.QUICK
    assert len(quick.sub_questions) <= 2
    per_question: dict[int, int] = {}
    for claim in quick.claims:
        per_question[claim.sub_question] = per_question.get(claim.sub_question, 0) + 1
    assert max(per_question.values()) <= 2
    assert "between 2 and 2 sub-questions" in fake_llm.calls[0][2]

    # Same question at standard depth must not reuse the quick report.
    events = await _collect(orchestrator, "Did X increase?", Depth.STANDARD)
    assert events[0]["type"] != "cached"
    assert Report.model_validate(events[-1]["report"]).id != quick.id
