"""Tests for the Gemini wrapper: parsing, retries, model fallback and client selection."""

from __future__ import annotations

import asyncio
from asyncio import sleep as real_sleep  # bound before the no_sleep fixture patches it
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors

from app import llm as llm_module
from app.config import Settings
from app.llm import GeminiLLM, LLMError, build_genai_client
from app.schemas import PlanOutput

pytestmark = pytest.mark.usefixtures("no_sleep")

EMPTY_PLAN = SimpleNamespace(text='{"sub_questions": []}')


class FakeModels:
    """Plays back a script of responses/exceptions and records which model was called."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.models_called: list[str] = []

    async def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.models_called.append(model)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def make_llm(script: list[Any]) -> tuple[GeminiLLM, FakeModels]:
    models = FakeModels(script)
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    settings = Settings(
        fast_models=("m1", "m2"), smart_models=("m1",), search_models=("s1",), llm_timeout_s=5
    )
    return GeminiLLM(settings, client=client), models  # type: ignore[arg-type]


def api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "x", "status": "X"}})


async def test_generate_json_parses_schema():
    ok = SimpleNamespace(text='{"sub_questions": [{"question": "q", "search_query": "s"}]}')
    llm, _ = make_llm([ok])
    plan = await llm.generate_json("p", PlanOutput, system="sys")
    assert plan.sub_questions[0].question == "q"


async def test_quota_error_falls_back_and_cools_the_model_down():
    llm, models = make_llm([api_error(429), EMPTY_PLAN, EMPTY_PLAN])
    await llm.generate_json("p", PlanOutput, system="sys")
    await llm.generate_json("p", PlanOutput, system="sys")
    assert models.models_called == ["m1", "m2", "m2"]  # m1 skipped while cooling down


async def test_transient_errors_are_retried_on_the_same_model():
    llm, models = make_llm([api_error(503), EMPTY_PLAN])
    await llm.generate_json("p", PlanOutput, system="sys")
    assert models.models_called == ["m1", "m1"]


async def test_timeout_fails_over_to_the_next_model():
    llm, models = make_llm([TimeoutError(), EMPTY_PLAN])
    await llm.generate_json("p", PlanOutput, system="sys")
    assert models.models_called == ["m1", "m2"]


async def test_non_retryable_error_raises_with_status():
    llm, _ = make_llm([api_error(400)])
    with pytest.raises(LLMError) as info:
        await llm.generate_json("p", PlanOutput, system="sys")
    assert info.value.status == 400


async def test_invalid_json_gets_one_repair_attempt_then_fails():
    bad = SimpleNamespace(text="not json")
    llm, models = make_llm([bad, bad])
    with pytest.raises(LLMError, match="PlanOutput"):
        await llm.generate_json("p", PlanOutput, system="sys")
    assert len(models.models_called) == 2


async def test_all_models_exhausted_reports_last_status():
    llm, _ = make_llm([api_error(429), api_error(429)])
    with pytest.raises(LLMError) as info:
        await llm.generate_json("p", PlanOutput, system="sys")
    assert info.value.status == 429


async def test_grounded_search_extracts_sources_and_supports():
    web = SimpleNamespace(uri="https://r/1", title="who.int")
    support = SimpleNamespace(segment=SimpleNamespace(end_index=5), grounding_chunk_indices=[0])
    metadata = SimpleNamespace(
        grounding_chunks=[SimpleNamespace(web=web), SimpleNamespace(web=None)],
        grounding_supports=[support, SimpleNamespace(segment=None, grounding_chunk_indices=[1])],
    )
    response = SimpleNamespace(
        text="Hello", candidates=[SimpleNamespace(grounding_metadata=metadata)]
    )
    llm, models = make_llm([response])
    result = await llm.grounded_search("q")
    assert result.sources == [("https://r/1", "who.int"), ("", "")]
    assert result.supports == [(5, [0])]
    assert models.models_called == ["s1"]


async def test_concurrency_is_bounded_by_settings():
    in_flight = 0
    peak = 0

    class SlowModels:
        async def generate_content(self, **_: Any) -> Any:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await real_sleep(0)  # let other calls run while this one is "in flight"
            in_flight -= 1
            return EMPTY_PLAN

    settings = Settings(fast_models=("m1",), llm_concurrency=2)
    client = SimpleNamespace(aio=SimpleNamespace(models=SlowModels()))
    llm = GeminiLLM(settings, client=client)  # type: ignore[arg-type]
    await asyncio.gather(*(llm.generate_json("p", PlanOutput, system="s") for _ in range(6)))
    assert peak <= 2


def test_build_genai_client_selects_vertex_or_api_key(monkeypatch):
    created: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    monkeypatch.setattr(llm_module.genai, "Client", FakeClient)
    build_genai_client(
        Settings(use_vertex_ai=True, gcp_project="demo-project", gcp_location="europe-west1")
    )
    build_genai_client(Settings(use_vertex_ai=False))
    assert created == [
        {"vertexai": True, "project": "demo-project", "location": "europe-west1"},
        {},
    ]
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        build_genai_client(Settings(use_vertex_ai=True, gcp_project=""))
