"""Evidence from the Wikipedia (MediaWiki) API: the fallback when grounding is unavailable.

The client is a polite API citizen: requests are serialised and paced,
``Retry-After`` is honoured, results are cached, and the User-Agent carries a
contact URL as Wikimedia's API policy asks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import OrderedDict
from typing import Any, TypedDict

import httpx

from app.schemas import Evidence
from app.security import is_safe_http_url
from app.sources import SourceRegistry
from app.text import select_passages, strip_html, widening_queries

from .base import unique_sources

logger = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
MIN_INTERVAL_S = 0.15  # Wikimedia throttles bursts of parallel requests
MAX_ATTEMPTS = 4
MAX_RETRY_AFTER_S = 10.0
THROTTLED_STATUSES = frozenset({429, 503})
SEARCH_CACHE_SIZE = 256
ARTICLE_CACHE_SIZE = 64
MAX_ARTICLE_CHARS = 80_000
MAX_EXTRACT_CHARS = 3000
FULL_TEXT_PAGES = 2  # top hits whose full article is mined for relevant passages
PASSAGE_CHARS = 2500

JSON = dict[str, Any]


class Page(TypedDict):
    title: str
    url: str
    extract: str


class WikipediaRetriever:
    """Searches Wikipedia and returns the passages most relevant to each question."""

    name = "wikipedia"

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        results_per_query: int = 3,
        contact: str | None = None,
    ) -> None:
        contact = contact or os.getenv("WIKIPEDIA_CONTACT", "fact-checking research demo")
        self._client = client or httpx.AsyncClient(
            timeout=10.0, headers={"User-Agent": f"EvidentiaResearchBot/1.0 ({contact})"}
        )
        self._k = results_per_query
        self._lock = asyncio.Lock()
        self._last_request = 0.0
        self._search_cache: OrderedDict[str, list[Page]] = OrderedDict()
        self._article_cache: OrderedDict[str, str] = OrderedDict()

    async def search(
        self, query: str, registry: SourceRegistry, keywords: str | None = None
    ) -> Evidence:
        """Search with ``keywords`` (or the query), then mine the top articles for passages."""
        pages = await self._search_widening(keywords or query)
        # Lead sections are often too general, so for the top hits use the paragraphs
        # of the full article that best match the question.
        focus = f"{query} {keywords or ''}"
        for page in pages[:FULL_TEXT_PAGES]:
            article = await self._article(page["title"])
            if article:
                page["extract"] = select_passages(article, focus, PASSAGE_CHARS)

        sources = [registry.add(p["url"], p["title"], p["extract"]) for p in pages]
        text = "\n\n".join(
            f"[{source.id}] {page['title']}: {page['extract']}"
            for source, page in zip(sources, pages, strict=True)
        )
        return Evidence(text=text, sources=unique_sources(sources), backend=self.name)

    async def search_many(self, queries: list[str], registry: SourceRegistry) -> Evidence:
        """Search each claim separately (requests are paced, results cached)."""
        results = await asyncio.gather(*(self.search(q, registry) for q in queries))
        return Evidence(
            text="\n\n".join(r.text for r in results if r.text),
            sources=unique_sources(s for r in results for s in r.sources),
            backend=self.name,
        )

    async def _search_widening(self, query: str) -> list[Page]:
        """Retry with fewer terms until there are enough hits (search ANDs every term)."""
        pages: dict[str, Page] = {}
        for candidate in widening_queries(query):
            for page in await self._search(candidate):
                pages.setdefault(page["url"], page)
            if len(pages) >= self._k:
                break
        return list(pages.values())[: self._k]

    async def _search(self, query: str) -> list[Page]:
        """One request: search plus the lead section of every hit (LRU-cached).

        ``exintro`` matters: without it the API returns text for only one page.
        """
        key = query.lower()
        if key in self._search_cache:
            self._search_cache.move_to_end(key)
            return self._search_cache[key]
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
        raw_pages = sorted(_pages(data), key=lambda p: int(p.get("index", 0)))
        results: list[Page] = []
        for page in raw_pages:
            url = str(page.get("fullurl", ""))
            extract = strip_html(str(page.get("extract", ""))).strip()[:MAX_EXTRACT_CHARS]
            if is_safe_http_url(url) and extract:
                results.append(Page(title=str(page.get("title", "")), url=url, extract=extract))
        if data:  # never cache failures
            _remember(self._search_cache, key, results, SEARCH_CACHE_SIZE)
        return results

    async def _article(self, title: str) -> str:
        """Full plain text of one article (LRU-cached; the API returns one per request)."""
        if title in self._article_cache:
            self._article_cache.move_to_end(title)
            return self._article_cache[title]
        data = await self._get(
            {"titles": title, "prop": "extracts", "explaintext": "1", "exsectionformat": "raw"}
        )
        pages = _pages(data)
        text = strip_html(str(pages[0].get("extract", "")))[:MAX_ARTICLE_CHARS] if pages else ""
        if data:
            _remember(self._article_cache, title, text, ARTICLE_CACHE_SIZE)
        return text

    async def _get(self, params: dict[str, str]) -> JSON:
        """Serialised, paced GET with ``Retry-After``-aware backoff. Returns {} on failure."""
        query = {"action": "query", "format": "json", "formatversion": "2", **params}
        async with self._lock:
            for attempt in range(MAX_ATTEMPTS):
                await self._pace()
                try:
                    response = await self._client.get(API_URL, params=query)
                    if response.status_code in THROTTLED_STATUSES and attempt < MAX_ATTEMPTS - 1:
                        await asyncio.sleep(retry_after(response, default=2.0**attempt))
                        continue
                    response.raise_for_status()
                    data: JSON = response.json()
                    return data
                except (httpx.HTTPError, ValueError) as exc:
                    logger.warning("wikipedia_request_failed", extra={"error": type(exc).__name__})
                    return {}
        return {}

    async def _pace(self) -> None:
        wait = self._last_request + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = time.monotonic()


def retry_after(response: httpx.Response, default: float) -> float:
    """Seconds to wait, from a numeric ``Retry-After`` header (capped), else ``default``."""
    try:
        seconds = float(response.headers.get("retry-after", default))
    except ValueError:
        return default
    return min(MAX_RETRY_AFTER_S, max(0.0, seconds))


def _pages(data: JSON) -> list[JSON]:
    pages: list[JSON] = data.get("query", {}).get("pages", [])
    return pages


def _remember[V](cache: OrderedDict[str, V], key: str, value: V, max_size: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)
