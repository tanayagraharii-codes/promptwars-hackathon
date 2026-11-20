"""Report persistence and caching.

Reports are keyed by a hash of the normalised question and depth, so repeated
questions are served instantly and cost no Gemini quota. An in-memory TTL/LRU
cache is always used; Cloud Firestore adds durable, shareable reports.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Protocol

from google.cloud import firestore

from app.config import Settings
from app.schemas import Report

logger = logging.getLogger(__name__)

REPORT_ID_LENGTH = 20
FIRESTORE_TIMEOUT_S = 10.0


def report_key(question: str, depth: str = "standard") -> str:
    """Stable report ID for a question + depth (case- and whitespace-insensitive)."""
    normalised = " ".join(question.lower().split()).rstrip("?.! ") + f"|{depth}"
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:REPORT_ID_LENGTH]


class ReportStore(Protocol):
    """Where finished reports are kept so they can be re-served and shared."""

    async def get(self, key: str) -> Report | None: ...

    async def put(self, report: Report) -> None: ...


class MemoryReportStore:
    """Bounded LRU cache with a per-entry time-to-live."""

    def __init__(self, max_items: int, ttl_s: float) -> None:
        self._items: OrderedDict[str, tuple[float, Report]] = OrderedDict()
        self._max = max_items
        self._ttl = ttl_s

    async def get(self, key: str) -> Report | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        stored_at, report = entry
        if time.monotonic() - stored_at > self._ttl:
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return report

    async def put(self, report: Report) -> None:
        self._items[report.id] = (time.monotonic(), report)
        self._items.move_to_end(report.id)
        while len(self._items) > self._max:
            self._items.popitem(last=False)


class FirestoreReportStore:
    """Durable store in Cloud Firestore, fronted by the in-memory cache.

    Persistence is best-effort: a Firestore outage is logged and the request
    is still answered from the cache, never failed.
    """

    def __init__(
        self,
        collection: str,
        cache: MemoryReportStore,
        client: firestore.AsyncClient | None = None,
    ) -> None:
        self._collection = (client or firestore.AsyncClient()).collection(collection)
        self._cache = cache

    async def get(self, key: str) -> Report | None:
        cached = await self._cache.get(key)
        if cached is not None:
            return cached
        try:
            snapshot = await self._collection.document(key).get()
        except Exception:
            logger.exception("firestore_read_failed")
            return None
        if not snapshot.exists:
            return None
        report = Report.model_validate(snapshot.to_dict())
        await self._cache.put(report)
        return report

    async def put(self, report: Report) -> None:
        await self._cache.put(report)
        try:
            await asyncio.wait_for(
                self._collection.document(report.id).set(report.model_dump(mode="json")),
                timeout=FIRESTORE_TIMEOUT_S,
            )
        except Exception:
            logger.exception("firestore_write_failed")


def build_store(settings: Settings) -> ReportStore:
    """Firestore when enabled (falling back to memory if it cannot start), else memory."""
    cache = MemoryReportStore(settings.cache_max_items, settings.cache_ttl_s)
    if settings.firestore_enabled:
        try:
            return FirestoreReportStore(settings.firestore_collection, cache)
        except Exception:
            logger.exception("firestore_unavailable_using_memory")
    return cache
