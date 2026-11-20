"""Google Search grounding with an automatic, circuit-broken fallback to Wikipedia."""

from __future__ import annotations

import logging
import time

from app.llm import LLMError
from app.schemas import Evidence
from app.sources import SourceRegistry

from .google_search import GoogleSearchRetriever
from .wikipedia import WikipediaRetriever

logger = logging.getLogger(__name__)

CIRCUIT_BREAKING_STATUSES = frozenset({401, 403, 429})
DEFAULT_COOLDOWN_S = 600.0


class AutoRetriever:
    """Prefers Google Search; falls back to Wikipedia when it fails.

    Quota and permission errors open a circuit breaker for ``cooldown_s`` so no
    further calls are spent on a backend that is known to be unavailable.
    """

    def __init__(
        self,
        primary: GoogleSearchRetriever,
        fallback: WikipediaRetriever,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
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
        """The backend currently answering requests."""
        return self._primary.name if self.primary_available else self._fallback.name

    async def search(
        self, query: str, registry: SourceRegistry, keywords: str | None = None
    ) -> Evidence:
        if self.primary_available:
            try:
                return await self._primary.search(query, registry)
            except LLMError as exc:
                self._record_failure(exc)
        return await self._fallback.search(query, registry, keywords)

    async def search_many(self, queries: list[str], registry: SourceRegistry) -> Evidence:
        if self.primary_available:
            try:
                return await self._primary.search_many(queries, registry)
            except LLMError as exc:
                self._record_failure(exc)
        return await self._fallback.search_many(queries, registry)

    def _record_failure(self, exc: LLMError) -> None:
        if exc.status in CIRCUIT_BREAKING_STATUSES:
            self._disabled_until = time.monotonic() + self._cooldown_s
            logger.warning("google_search_circuit_open", extra={"status": exc.status})
        else:
            logger.warning("google_search_failed", extra={"status": exc.status})
