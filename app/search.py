"""Evidence retrieval backends.

* ``GoogleSearchRetriever``: Gemini with Google Search grounding (primary).
* ``WikipediaRetriever``: MediaWiki API, used when grounding is unavailable
  (e.g. free-tier quota), so the system degrades instead of failing.
* ``AutoRetriever``: tries Google first and trips a circuit breaker on
  quota/permission errors to avoid wasting calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Protocol

import httpx

from app.llm import LLM, LLMError
from app.schemas import Evidence
from app.security import is_safe_http_url
from app.sources import SourceRegistry

logger = logging.getLogger(__name__)

_HTML_TAGS = re.compile(r"<[^>]+>")


class Retriever(Protocol):
    name: str

    async def search(
        self, query: str, registry: SourceRegistry, keywords: str | None = None
    ) -> Evidence: ...

    async def search_many(self, queries: list[str], registry: SourceRegistry) -> Evidence: ...


class GoogleSearchRetriever:
    name = "google_search"

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def search(
        self, query: str, registry: SourceRegistry, keywords: str | None = None
    ) -> Evidence:
        prompt = (
            "Research the following question using Google Search. Report concrete facts, "
            "figures, dates and study findings, and note where sources disagree.\n\n"
            f"Question: {query}"
        )
        return await self._run(prompt, registry)

    async def search_many(self, queries: list[str], registry: SourceRegistry) -> Evidence:
        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))
        prompt = (
            "Independently verify each statement below using Google Search. For each one, "
            "report evidence that confirms it AND evidence that contradicts or qualifies it. "
            "Prefer primary sources, official statistics and peer-reviewed research.\n\n"
            f"{numbered}"
        )
        return await self._run(prompt, registry)

    async def _run(self, prompt: str, registry: SourceRegistry) -> Evidence:
        result = await self._llm.grounded_search(prompt)
        chunk_ids: dict[int, str] = {}
        for index, (uri, title) in enumerate(result.sources):
            if is_safe_http_url(uri):
                chunk_ids[index] = registry.add(uri, title).id
        text = annotate_citations(result.text, result.supports, chunk_ids)
        sources = [s for s in (registry.get(i) for i in dict.fromkeys(chunk_ids.values())) if s]
        return Evidence(text=text, sources=sources, backend=self.name)


class WikipediaRetriever:
    name = "wikipedia"
    API = "https://en.wikipedia.org/w/api.php"

    MIN_INTERVAL_S = 0.15  # polite pacing: Wikimedia throttles bursts
    CACHE_SIZE = 256
    FULL_TEXT_PAGES = 2  # top hits whose full article is mined for relevant passages
    PASSAGE_CHARS = 2500

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        results_per_query: int = 3,
        contact: str | None = None,
    ) -> None:
        # Wikimedia's API policy asks clients to identify themselves with a contact URL.
        contact = contact or os.getenv("WIKIPEDIA_CONTACT", "fact-checking research demo")
        self._client = client or httpx.AsyncClient(
            timeout=10.0, headers={"User-Agent": f"EvidentiaResearchBot/1.0 ({contact})"}
        )
        self._k = results_per_query
        self._lock = asyncio.Lock()
        self._last_request = 0.0
        self._cache: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self._articles: OrderedDict[str, str] = OrderedDict()

    async def search(
        self, query: str, registry: SourceRegistry, keywords: str | None = None
    ) -> Evidence:
        pages = await self._query_widening(keywords or query)
        # Lead sections are often too general (e.g. no lifecycle-emissions figures), so for
        # the top hits pull the paragraphs of the full article most relevant to the question.
        focus = f"{query} {keywords or ''}"
        for page in pages[: self.FULL_TEXT_PAGES]:
            article = await self._article(page["title"])
            if article:
                page["extract"] = select_passages(article, focus, self.PASSAGE_CHARS)
        sources = []
        text_parts = []
        for page in pages:
            source = registry.add(page["url"], page["title"], page["extract"])
            sources.append(source)
            text_parts.append(f"[{source.id}] {page['title']}: {page['extract']}")
        return Evidence(text="\n\n".join(text_parts), sources=_unique(sources), backend=self.name)

    async def search_many(self, queries: list[str], registry: SourceRegistry) -> Evidence:
        results = await asyncio.gather(*(self.search(q, registry) for q in queries))
        sources = _unique([s for r in results for s in r.sources])
        text = "\n\n".join(r.text for r in results if r.text)
        return Evidence(text=text, sources=sources, backend=self.name)

    async def _query_widening(self, query: str) -> list[dict[str, str]]:
        """Wikipedia search ANDs every term, so retry with fewer terms until we have results."""
        pages: dict[str, dict[str, str]] = {}
        for candidate in widening_queries(query):
            for page in await self._query(candidate):
                pages.setdefault(page["url"], page)
            if len(pages) >= self._k:
                break
        return list(pages.values())[: self._k]

    async def _query(self, query: str) -> list[dict[str, str]]:
        """One request: search + lead-section text for every hit.

        ``exintro`` matters: without it the API returns text for only one page.
        Results are cached (LRU) so repeated and overlapping queries cost nothing.
        """
        key = query.lower()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        data = await self._get(
            {
                "generator": "search",
                "gsrsearch": query[:300],
                "gsrlimit": str(self._k),
                "prop": "extracts|info",
                "exintro": "1",
                "explaintext": "1",
                "exlimit": "max",
                "inprop": "url",
            }
        )
        pages = sorted(data.get("query", {}).get("pages", []), key=lambda p: p.get("index", 0))
        results = []
        for page in pages:
            url = page.get("fullurl", "")
            extract = _HTML_TAGS.sub("", page.get("extract", "")).strip()[:3000]
            if is_safe_http_url(url) and extract:
                results.append({"title": page.get("title", ""), "url": url, "extract": extract})
        if data:  # never cache failures
            self._cache[key] = results
            while len(self._cache) > self.CACHE_SIZE:
                self._cache.popitem(last=False)
        return results

    async def _article(self, title: str) -> str:
        """Full plain text of one article (cached; one page per request is an API limit)."""
        if title in self._articles:
            self._articles.move_to_end(title)
            return self._articles[title]
        data = await self._get(
            {"titles": title, "prop": "extracts", "explaintext": "1", "exsectionformat": "raw"}
        )
        pages = data.get("query", {}).get("pages", [])
        text = _HTML_TAGS.sub("", pages[0].get("extract", ""))[:80_000] if pages else ""
        if data:
            self._articles[title] = text
            while len(self._articles) > 64:
                self._articles.popitem(last=False)
        return text

    async def _get(self, params: dict[str, str]) -> dict:
        """Serialised, paced GET with Retry-After-aware backoff on 429/503."""
        query = {"action": "query", "format": "json", "formatversion": "2", **params}
        async with self._lock:
            for attempt in range(4):
                wait = self._last_request + self.MIN_INTERVAL_S - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_request = time.monotonic()
                try:
                    response = await self._client.get(self.API, params=query)
                    if response.status_code in {429, 503} and attempt < 3:
                        await asyncio.sleep(_retry_after(response, default=2.0**attempt))
                        continue
                    response.raise_for_status()
                    return response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    logger.warning("wikipedia_request_failed", extra={"error": type(exc).__name__})
                    return {}
        return {}


def _retry_after(response: httpx.Response, default: float) -> float:
    """Seconds to wait, from a numeric Retry-After header (capped), else ``default``."""
    try:
        return min(10.0, max(0.0, float(response.headers.get("retry-after", default))))
    except ValueError:
        return default


class AutoRetriever:
    """Google Search grounding with automatic fallback to Wikipedia.

    Quota/permission errors open a circuit breaker for ``cooldown_s`` so we
    stop spending calls on a backend that is known to be unavailable.
    """

    def __init__(
        self,
        primary: GoogleSearchRetriever,
        fallback: WikipediaRetriever,
        cooldown_s: float = 600.0,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._cooldown_s = cooldown_s
        self._disabled_until = 0.0

    @property
    def primary_available(self) -> bool:
        return time.monotonic() >= self._disabled_until

    @property
    def name(self) -> str:
        return self._primary.name if self.primary_available else self._fallback.name

    async def search(
        self, query: str, registry: SourceRegistry, keywords: str | None = None
    ) -> Evidence:
        if self.primary_available:
            try:
                return await self._primary.search(query, registry)
            except LLMError as exc:
                self._on_failure(exc)
        return await self._fallback.search(query, registry, keywords)

    async def search_many(self, queries: list[str], registry: SourceRegistry) -> Evidence:
        if self.primary_available:
            try:
                return await self._primary.search_many(queries, registry)
            except LLMError as exc:
                self._on_failure(exc)
        return await self._fallback.search_many(queries, registry)

    def _on_failure(self, exc: LLMError) -> None:
        if exc.status in {401, 403, 429}:
            self._disabled_until = time.monotonic() + self._cooldown_s
            logger.warning("google_search_circuit_open", extra={"status": exc.status})
        else:
            logger.warning("google_search_failed", extra={"status": exc.status})


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "more",
        "most",
        "of",
        "on",
        "or",
        "than",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "versus",
        "vs",
        "about",
        "compared",
    }
)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def _stems(text: str) -> set[str]:
    """Crude stemming: content words truncated to 5 chars ("emissions" ~ "emission")."""
    return {
        w.lower()[:5] for w in _WORD.findall(text) if w.lower() not in _STOPWORDS and len(w) > 2
    }


def select_passages(article: str, focus: str, max_chars: int = 2500) -> str:
    """Pick the article's lead plus the paragraphs that best match ``focus``, in order.

    Cheap lexical relevance (shared word stems) keeps prompts small and on-topic
    without an extra model call.
    """
    paragraphs = [
        p.strip()
        for p in article.split("\n")
        if p.strip() and not p.strip().startswith("==") and len(p.strip()) > 40
    ]
    if not paragraphs:
        return article[:max_chars].strip()
    terms = _stems(focus)
    lead = paragraphs[0][:600]
    scored = [(len(terms & _stems(p)), i) for i, p in enumerate(paragraphs[1:], start=1) if terms]
    # Topic words appear all over an article, so keep only paragraphs that score at
    # least half as well as the best match (and always share >= 2 terms).
    threshold = max(2, max((score for score, _ in scored), default=0) / 2)
    chosen: list[int] = []
    budget = max_chars - len(lead)
    for score, index in sorted(scored, key=lambda item: (-item[0], item[1])):
        if score < threshold or budget <= 0:
            break
        chosen.append(index)
        budget -= len(paragraphs[index][:1200]) + 2
    body = [paragraphs[i][:1200] for i in sorted(chosen)]
    return "\n\n".join([lead, *body])[:max_chars]


def widening_queries(text: str, max_terms: int = 6) -> list[str]:
    """Turn a question or claim into keyword queries, from most to least specific."""
    terms = [w for w in _WORD.findall(text) if w.lower() not in _STOPWORDS][:max_terms]
    if not terms:
        return [text[:120]]
    queries = [" ".join(terms[:n]) for n in (len(terms), 3, 2) if n <= len(terms)]
    return list(dict.fromkeys(queries))


def annotate_citations(
    text: str, supports: list[tuple[int, list[int]]], chunk_ids: dict[int, str]
) -> str:
    """Insert ``[S#]`` markers after each grounded segment.

    Gemini reports segment offsets in UTF-8 bytes, so we splice on the encoded
    text, working backwards so earlier offsets stay valid.
    """
    raw = text.encode("utf-8")
    markers: dict[int, list[str]] = {}
    for end, indices in supports:
        ids = [chunk_ids[i] for i in indices if i in chunk_ids]
        if ids and 0 <= end <= len(raw):
            bucket = markers.setdefault(end, [])
            bucket.extend(i for i in ids if i not in bucket)
    for end in sorted(markers, reverse=True):
        tag = "".join(f"[{i}]" for i in markers[end]).encode("utf-8")
        raw = raw[:end] + tag + raw[end:]
    return raw.decode("utf-8", errors="ignore")


def build_retriever(backend: str, llm: LLM) -> Retriever:
    if backend == "google":
        return GoogleSearchRetriever(llm)
    if backend == "wikipedia":
        return WikipediaRetriever()
    return AutoRetriever(GoogleSearchRetriever(llm), WikipediaRetriever())


def _unique(sources: list) -> list:
    seen: set[str] = set()
    out = []
    for s in sources:
        if s.id not in seen:
            seen.add(s.id)
            out.append(s)
    return out
