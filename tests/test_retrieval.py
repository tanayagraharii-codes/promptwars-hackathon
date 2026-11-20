"""Tests for the evidence backends: Google Search grounding, Wikipedia and the fallback."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.llm import GroundedResult, LLMError
from app.retrieval import (
    AutoRetriever,
    GoogleSearchRetriever,
    WikipediaRetriever,
    annotate_citations,
    build_retriever,
    retry_after,
)
from app.schemas import Evidence
from app.sources import SourceRegistry
from tests.fakes import FakeLLM

NO_ARTICLE: dict[str, Any] = {"query": {"pages": []}}


def search_payload(*pages: dict[str, Any]) -> dict[str, Any]:
    return {"query": {"pages": list(pages)}}


def wiki_page(index: int, title: str, extract: str, url: str | None = None) -> dict[str, Any]:
    return {
        "index": index,
        "title": title,
        "fullurl": url or f"https://en.wikipedia.org/wiki/{title}",
        "extract": extract,
    }


def is_article_request(request: httpx.Request) -> bool:
    return "titles" in request.url.params


def wiki_client(
    search: dict[str, Any], status: int = 200, article: dict[str, Any] = NO_ARTICLE
) -> httpx.AsyncClient:
    """A client whose search and full-article requests are answered from fixtures."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "en.wikipedia.org"
        if is_article_request(request) and status == 200:
            return httpx.Response(200, json=article)
        return httpx.Response(status, json=search)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ----------------------------------------------------------------- citations


def test_annotate_citations_inserts_markers_at_byte_offsets():
    text = "Café sales rose. Tea fell."
    first_end = len("Café sales rose.".encode())
    out = annotate_citations(
        text, [(first_end, [0, 1]), (len(text.encode()), [1, 5])], {0: "S1", 1: "S2"}
    )
    assert out == "Café sales rose.[S1][S2] Tea fell.[S2]"


def test_annotate_citations_ignores_out_of_range_offsets():
    assert annotate_citations("abc", [(99, [0])], {0: "S1"}) == "abc"


# ----------------------------------------------------------------- Google Search grounding


async def test_google_retriever_maps_grounding_chunks_and_drops_unsafe_urls():
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
    evidence = await GoogleSearchRetriever(llm).search("EVs?", SourceRegistry())
    assert [s.domain for s in evidence.sources] == ["iea.org", "reddit.com"]
    assert evidence.text.endswith("[S1][S2]")
    assert evidence.backend == "google_search"


async def test_google_retriever_verifies_all_claims_in_one_call():
    llm = FakeLLM()
    await GoogleSearchRetriever(llm).search_many(["claim a", "claim b"], SourceRegistry())
    assert len(llm.calls) == 1
    assert "1. claim a" in llm.calls[0][2] and "2. claim b" in llm.calls[0][2]


# ----------------------------------------------------------------- Wikipedia


async def test_wikipedia_orders_pages_strips_html_and_drops_unsafe_urls():
    search = search_payload(
        wiki_page(2, "B", "Bee <b>text</b>"),
        wiki_page(1, "A", "Aye text"),
        wiki_page(3, "C", "bad", url="javascript:x"),
    )
    evidence = await WikipediaRetriever(wiki_client(search)).search(
        "q", SourceRegistry(), keywords="kw"
    )
    assert [s.title for s in evidence.sources] == ["A", "B"]
    assert "<b>" not in evidence.text
    assert evidence.text.startswith("[S1] A: Aye text")


async def test_wikipedia_returns_empty_evidence_on_http_errors():
    retriever = WikipediaRetriever(wiki_client({}, status=500))
    evidence = await retriever.search_many(["a", "b"], SourceRegistry())
    assert evidence.sources == [] and evidence.text == ""


async def test_wikipedia_mines_full_article_for_relevant_passages():
    search = search_payload(wiki_page(1, "Electric car", "An electric car uses motors."))
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
    evidence = await WikipediaRetriever(wiki_client(search, article=article)).search(
        "Do electric cars have lower lifecycle emissions than petrol cars?",
        SourceRegistry(),
        keywords="electric car emissions",
    )
    assert "Lifecycle greenhouse gas emissions" in evidence.text
    assert "1880s" not in evidence.text  # irrelevant history paragraph skipped
    assert "== " not in evidence.text


@pytest.mark.usefixtures("no_sleep")
async def test_wikipedia_retries_when_rate_limited():
    statuses = [429, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        if is_article_request(request):
            return httpx.Response(200, json=NO_ARTICLE)
        return httpx.Response(statuses.pop(0), json=search_payload(wiki_page(1, "A", "x")))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    evidence = await WikipediaRetriever(client, results_per_query=1).search("a", SourceRegistry())
    assert [s.title for s in evidence.sources] == ["A"]


@pytest.mark.usefixtures("no_sleep")
async def test_wikipedia_caches_results_but_not_failures():
    statuses = [500, 200]
    search_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if is_article_request(request):
            return httpx.Response(200, json=NO_ARTICLE)
        search_requests.append(request)
        status = statuses.pop(0) if statuses else 200
        return httpx.Response(status, json=search_payload(wiki_page(1, "A", "x")))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    retriever = WikipediaRetriever(client, results_per_query=1)
    assert (await retriever.search("alpha", SourceRegistry())).sources == []  # 500: not cached
    assert len((await retriever.search("alpha", SourceRegistry())).sources) == 1
    await retriever.search("ALPHA", SourceRegistry())  # case-insensitive cache hit
    assert len(search_requests) == 2


@pytest.mark.parametrize(
    ("header", "default", "expected"),
    [("3", 1.0, 3.0), ("999", 1.0, 10.0), ("soon", 1.5, 1.5), (None, 2.0, 2.0)],
)
def test_retry_after_parsing(header: str | None, default: float, expected: float):
    headers = {"retry-after": header} if header is not None else {}
    assert retry_after(httpx.Response(429, headers=headers), default=default) == expected


def test_wikipedia_user_agent_includes_contact():
    retriever = WikipediaRetriever(contact="https://github.com/team/evidentia")
    assert "https://github.com/team/evidentia" in retriever._client.headers["user-agent"]


# ----------------------------------------------------------------- automatic fallback


class StubRetriever:
    """Counts calls; raises ``error`` when set, otherwise returns empty evidence."""

    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name
        self.error = error
        self.calls = 0

    async def search(
        self, query: str, registry: SourceRegistry, keywords: str | None = None
    ) -> Evidence:
        self.calls += 1
        if self.error:
            raise self.error
        return Evidence(text=self.name, sources=[], backend=self.name)

    async def search_many(self, queries: list[str], registry: SourceRegistry) -> Evidence:
        return await self.search(queries[0], registry)


async def test_auto_retriever_trips_circuit_on_quota_errors():
    primary = StubRetriever("google_search", LLMError("quota", status=429))
    auto = AutoRetriever(primary, StubRetriever("wikipedia"), cooldown_s=600)  # type: ignore[arg-type]
    assert (await auto.search("q", SourceRegistry())).backend == "wikipedia"
    assert (await auto.search_many(["q"], SourceRegistry())).backend == "wikipedia"
    assert primary.calls == 1  # circuit open: the primary is not called again
    assert auto.name == "wikipedia"


async def test_auto_retriever_keeps_primary_after_transient_errors():
    primary = StubRetriever("google_search", LLMError("boom", status=500))
    auto = AutoRetriever(primary, StubRetriever("wikipedia"))  # type: ignore[arg-type]
    await auto.search("q", SourceRegistry())
    await auto.search("q", SourceRegistry())
    assert primary.calls == 2
    assert auto.name == "google_search"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("google", GoogleSearchRetriever),
        ("wikipedia", WikipediaRetriever),
        ("auto", AutoRetriever),
        ("unknown", AutoRetriever),
    ],
)
def test_build_retriever(backend: str, expected: type):
    assert isinstance(build_retriever(backend, FakeLLM()), expected)


async def test_auto_retriever_falls_back_for_batches_after_transient_errors():
    primary = StubRetriever("google_search", LLMError("boom", status=500))
    auto = AutoRetriever(primary, StubRetriever("wikipedia"))  # type: ignore[arg-type]
    assert (await auto.search_many(["claim"], SourceRegistry())).backend == "wikipedia"
    assert auto.name == "google_search"  # transient errors do not open the circuit
