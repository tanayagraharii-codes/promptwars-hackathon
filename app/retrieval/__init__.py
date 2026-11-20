"""Evidence retrieval backends.

* :class:`GoogleSearchRetriever`: Gemini with Grounding with Google Search (primary).
* :class:`WikipediaRetriever`: the MediaWiki API, used when grounding is unavailable
  (for example free-tier quota), so the system degrades instead of failing.
* :class:`AutoRetriever`: Google first, with a circuit breaker and Wikipedia fallback.
"""

from __future__ import annotations

from typing import Literal

from app.llm import LLM

from .auto import AutoRetriever
from .base import Retriever, unique_sources
from .google_search import GoogleSearchRetriever, annotate_citations
from .wikipedia import WikipediaRetriever, retry_after

Backend = Literal["auto", "google", "wikipedia"]


def build_retriever(backend: str, llm: LLM) -> Retriever:
    """Create the retriever for a ``SEARCH_BACKEND`` setting (unknown values mean auto)."""
    if backend == "google":
        return GoogleSearchRetriever(llm)
    if backend == "wikipedia":
        return WikipediaRetriever()
    return AutoRetriever(GoogleSearchRetriever(llm), WikipediaRetriever())


__all__ = [
    "AutoRetriever",
    "Backend",
    "GoogleSearchRetriever",
    "Retriever",
    "WikipediaRetriever",
    "annotate_citations",
    "build_retriever",
    "retry_after",
    "unique_sources",
]
