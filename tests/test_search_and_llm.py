"""Tests for retrieval backends, citation annotation and the Gemini wrapper (all offline)."""

from types import SimpleNamespace

import httpx
import pytest

from app.config import Settings
from app.llm import GeminiLLM, GroundedResult, LLMError
from app.schemas import PlanOutput
from app.search import (
    AutoRetriever,
    GoogleSearchRetriever,
    WikipediaRetriever,
    annotate_citations,
    build_retriever,
    widening_queries,
)
from app.sources import SourceRegistry
from tests.conftest import FakeLLM

# ----------------------------------------------------------------- citations


def test_annotate_citations_inserts_markers_at_byte_offsets():
    text = "Café sales rose. Tea fell."
    first_end = len("Café sales rose.".encode())
    out = annotate_citations(
        text, [(first_end, [0, 1]), (len(text.encode()), [1, 5])], {0: "S1", 1: "S2"}
    )
    assert out == "Café sales rose.[S1][S2] Tea fell.[S2]"


def test_annotate_citations_ignores_out_of_range():
    assert annotate_citations("abc", [(99, [0])], {0: "S1"}) == "abc"


# ----------------------------------------------------------------- google retriever


async def test_google_retriever_maps_grounding_chunks():
    llm = FakeLLM()
    llm.grounded = GroundedResult(
        text="EV emissions are lower.",
        sources=[
            ("https://redirect/1", "iea.org"),
            ("javascript:alert(1)", "evil"),
            ("https://redirect/2", "reddit.com"),
        ],
        supports=[(len("EV emissions are lower."), [0, 1, 2])],
    )
    registry = SourceRegistry()
    evidence = await GoogleSearchRetriever(llm).search("EVs?", registry)
    assert [s.domain for s in evidence.sources] == ["iea.org", "reddit.com"]
    assert evidence.text.endswith("[S1][S2]")
    assert evidence.backend == "google_search"


async def test_google_retriever_search_many_numbers_queries():
    llm = FakeLLM()
    await GoogleSearchRetriever(llm).search_many(["claim a", "claim b"], SourceRegistry())
    assert "1. claim a" in llm.calls[-1][2] and "2. claim b" in llm.calls[-1][2]


# ----------------------------------------------------------------- wikipedia retriever


_NO_ARTICLE = {"query": {"pages": []}}


def _is_article_request(request: httpx.Request) -> bool:
    return "titles" in request.url.params


def _wiki_transport(payload, status=200, article=_NO_ARTICLE):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "en.wikipedia.org"
        if _is_article_request(request) and status == 200:
            return httpx.Response(200, json=article)
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


async def test_wikipedia_retriever_parses_pages():
    payload = {
        "query": {
            "pages": [
                {
                    "index": 2,
                    "title": "B",
                    "fullurl": "https://en.wikipedia.org/wiki/B",
                    "extract": "Bee <b>text</b>",
                },
                {
                    "index": 1,
                    "title": "A",
                    "fullurl": "https://en.wikipedia.org/wiki/A",
                    "extract": "Aye text",
                },
                {"index": 3, "title": "C", "fullurl": "javascript:x", "extract": "bad"},
            ]
        }
    }
    client = httpx.AsyncClient(transport=_wiki_transport(payload))
    registry = SourceRegistry()
    evidence = await WikipediaRetriever(client).search("q", registry, keywords="kw")
    assert [s.title for s in evidence.sources] == ["A", "B"]
    assert "<b>" not in evidence.text
    assert evidence.text.startswith("[S1] A: Aye text")


async def test_wikipedia_retriever_handles_http_errors():
    client = httpx.AsyncClient(transport=_wiki_transport({}, status=500))
    evidence = await WikipediaRetriever(client).search_many(["a", "b"], SourceRegistry())
    assert evidence.sources == [] and evidence.text == ""


# ----------------------------------------------------------------- auto retriever


class _Stub:
    def __init__(self, name, error=None):
        self.name = name
        self.error = error
        self.calls = 0

    async def search(self, query, registry, keywords=None):
        self.calls += 1
        if self.error:
            raise self.error
        from app.schemas import Evidence

        return Evidence(text=self.name, sources=[], backend=self.name)

    async def search_many(self, queries, registry):
        return await self.search(queries[0], registry)


async def test_auto_retriever_trips_circuit_on_quota():
    primary, fallback = _Stub("google_search", LLMError("quota", status=429)), _Stub("wikipedia")
    auto = AutoRetriever(primary, fallback, cooldown_s=600)
    assert (await auto.search("q", SourceRegistry())).backend == "wikipedia"
    assert (await auto.search_many(["q"], SourceRegistry())).backend == "wikipedia"
    assert primary.calls == 1  # circuit open: primary not retried
    assert auto.name == "wikipedia"


async def test_auto_retriever_keeps_primary_on_transient_error():
    primary, fallback = _Stub("google_search", LLMError("boom", status=500)), _Stub("wikipedia")
    auto = AutoRetriever(primary, fallback)
    await auto.search("q", SourceRegistry())
    await auto.search("q", SourceRegistry())
    assert primary.calls == 2
    assert auto.name == "google_search"


def test_build_retriever_variants():
    llm = FakeLLM()
    assert isinstance(build_retriever("google", llm), GoogleSearchRetriever)
    assert isinstance(build_retriever("wikipedia", llm), WikipediaRetriever)
    assert isinstance(build_retriever("auto", llm), AutoRetriever)


# ----------------------------------------------------------------- Gemini wrapper


class _FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.models_called = []

    async def generate_content(self, *, model, contents, config):
        self.models_called.append(model)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _gemini(script, **overrides):
    models = _FakeModels(script)
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    settings = Settings(
        fast_models=("m1", "m2"),
        smart_models=("m1",),
        search_models=("s1",),
        llm_timeout_s=5,
        **overrides,
    )
    return GeminiLLM(settings, client=client), models


def _api_error(code):
    from google.genai import errors

    return errors.APIError(code, {"error": {"code": code, "message": "x", "status": "X"}})


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr("app.llm.asyncio.sleep", instant)


async def test_gemini_generate_json_parses_schema():
    ok = SimpleNamespace(text='{"sub_questions": [{"question": "q", "search_query": "s"}]}')
    llm, _ = _gemini([ok])
    plan = await llm.generate_json("p", PlanOutput, system="sys")
    assert plan.sub_questions[0].question == "q"


async def test_gemini_falls_back_on_quota_and_cools_model_down():
    ok = SimpleNamespace(text='{"sub_questions": []}')
    llm, models = _gemini([_api_error(429), ok, ok])
    await llm.generate_json("p", PlanOutput, system="sys")
    await llm.generate_json("p", PlanOutput, system="sys")
    assert models.models_called == ["m1", "m2", "m2"]  # m1 skipped while cooling down


async def test_gemini_retries_transient_errors():
    ok = SimpleNamespace(text='{"sub_questions": []}')
    llm, models = _gemini([_api_error(503), ok])
    await llm.generate_json("p", PlanOutput, system="sys")
    assert models.models_called == ["m1", "m1"]


async def test_gemini_non_retryable_error_raises():
    llm, _ = _gemini([_api_error(400)])
    with pytest.raises(LLMError) as info:
        await llm.generate_json("p", PlanOutput, system="sys")
    assert info.value.status == 400


async def test_gemini_invalid_json_is_repaired_then_fails():
    bad = SimpleNamespace(text="not json")
    llm, _ = _gemini([bad, bad])
    with pytest.raises(LLMError):
        await llm.generate_json("p", PlanOutput, system="sys")


async def test_gemini_all_models_exhausted():
    llm, _ = _gemini([_api_error(429), _api_error(429)])
    with pytest.raises(LLMError) as info:
        await llm.generate_json("p", PlanOutput, system="sys")
    assert info.value.status == 429


async def test_gemini_grounded_search_extracts_sources():
    web = SimpleNamespace(uri="https://r/1", title="who.int")
    support = SimpleNamespace(segment=SimpleNamespace(end_index=5), grounding_chunk_indices=[0])
    metadata = SimpleNamespace(
        grounding_chunks=[SimpleNamespace(web=web)], grounding_supports=[support]
    )
    response = SimpleNamespace(
        text="Hello", candidates=[SimpleNamespace(grounding_metadata=metadata)]
    )
    llm, models = _gemini([response])
    result = await llm.grounded_search("q")
    assert result.sources == [("https://r/1", "who.int")]
    assert result.supports == [(5, [0])]
    assert models.models_called == ["s1"]


def test_widening_queries_drop_stopwords_and_shrink():
    queries = widening_queries("What is the effect of remote work on productivity?")
    assert queries == ["effect remote work productivity", "effect remote work", "effect remote"]
    assert widening_queries("the of and") == ["the of and"]


async def test_wikipedia_retries_on_rate_limit(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr("app.search.asyncio.sleep", instant)
    statuses = [429, 200]
    payload = {
        "query": {
            "pages": [
                {
                    "index": 1,
                    "title": "A",
                    "fullurl": "https://en.wikipedia.org/wiki/A",
                    "extract": "x",
                },
            ]
        }
    }

    def handler(request):
        if _is_article_request(request):
            return httpx.Response(200, json=_NO_ARTICLE)
        return httpx.Response(statuses.pop(0), json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    retriever = WikipediaRetriever(client, results_per_query=1)
    evidence = await retriever.search("a", SourceRegistry())
    assert [s.title for s in evidence.sources] == ["A"]


def test_build_genai_client_vertex_and_api_key(monkeypatch):
    from app import llm as llm_module

    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr("google.genai.Client", FakeClient)
    llm_module.build_genai_client(
        Settings(use_vertex_ai=True, gcp_project="demo-project", gcp_location="europe-west1")
    )
    llm_module.build_genai_client(Settings(use_vertex_ai=False))
    assert created == [
        {"vertexai": True, "project": "demo-project", "location": "europe-west1"},
        {},
    ]
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        llm_module.build_genai_client(Settings(use_vertex_ai=True, gcp_project=""))


def test_firestore_defaults_on_for_cloud_run(monkeypatch):
    monkeypatch.delenv("FIRESTORE_ENABLED", raising=False)
    monkeypatch.setenv("K_SERVICE", "evidentia")
    assert Settings().firestore_enabled is True
    monkeypatch.delenv("K_SERVICE")
    assert Settings().firestore_enabled is False
    monkeypatch.setenv("FIRESTORE_ENABLED", "false")
    monkeypatch.setenv("K_SERVICE", "evidentia")
    assert Settings().firestore_enabled is False


async def test_wikipedia_caches_results_but_not_failures(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr("app.search.asyncio.sleep", instant)
    payload = {
        "query": {
            "pages": [
                {
                    "index": 1,
                    "title": "A",
                    "fullurl": "https://en.wikipedia.org/wiki/A",
                    "extract": "x",
                }
            ]
        }
    }
    statuses = [500, 200]
    requests = []

    def handler(request):
        if _is_article_request(request):
            return httpx.Response(200, json=_NO_ARTICLE)
        requests.append(request)
        return httpx.Response(statuses.pop(0) if statuses else 200, json=payload)

    retriever = WikipediaRetriever(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), results_per_query=1
    )
    assert (await retriever.search("alpha", SourceRegistry())).sources == []  # 500: not cached
    assert len((await retriever.search("alpha", SourceRegistry())).sources) == 1
    await retriever.search("ALPHA", SourceRegistry())  # case-insensitive cache hit
    assert len(requests) == 2


def test_retry_after_header_parsing():
    from app.search import _retry_after

    assert _retry_after(httpx.Response(429, headers={"retry-after": "3"}), default=1) == 3.0
    assert _retry_after(httpx.Response(429, headers={"retry-after": "999"}), default=1) == 10.0
    assert _retry_after(httpx.Response(429, headers={"retry-after": "soon"}), default=1.5) == 1.5
    assert _retry_after(httpx.Response(429), default=2.0) == 2.0


def test_wikipedia_user_agent_includes_contact():
    retriever = WikipediaRetriever(contact="https://github.com/team/evidentia")
    assert "https://github.com/team/evidentia" in retriever._client.headers["user-agent"]


async def test_wikipedia_mines_full_article_for_relevant_passages():
    search = {
        "query": {
            "pages": [
                {
                    "index": 1,
                    "title": "Electric car",
                    "fullurl": "https://en.wikipedia.org/wiki/Electric_car",
                    "extract": "An electric car is a car propelled by electric motors.",
                }
            ]
        }
    }
    article_text = "\n".join(
        [
            "An electric car is a passenger automobile propelled by one or more electric motors.",
            "== History ==",
            "The first practical electric cars were produced in the 1880s by several inventors.",
            "== Environmental aspects ==",
            "Lifecycle greenhouse gas emissions of electric cars are lower than petrol cars "
            "in most countries, even accounting for battery manufacturing emissions.",
        ]
    )
    article = {"query": {"pages": [{"title": "Electric car", "extract": article_text}]}}
    client = httpx.AsyncClient(transport=_wiki_transport(search, article=article))
    evidence = await WikipediaRetriever(client).search(
        "Do electric cars have lower lifecycle emissions than petrol cars?",
        SourceRegistry(),
        keywords="electric car emissions",
    )
    assert "Lifecycle greenhouse gas emissions" in evidence.text
    assert "1880s" not in evidence.text  # irrelevant history paragraph skipped
    assert "== " not in evidence.text


def test_select_passages_keeps_lead_and_respects_budget():
    from app.search import select_passages

    article = "Lead paragraph about coffee and its history in the world.\n" + "\n".join(
        f"Coffee consumption study {i} found caffeine health effects on heart outcomes."
        for i in range(100)
    )
    out = select_passages(article, "coffee caffeine health heart", max_chars=500)
    assert out.startswith("Lead paragraph")
    assert len(out) <= 500
    assert select_passages("short", "x") == "short"
