"""Report persistence and caching.

Reports are keyed by a hash of the normalised question, so repeated questions
are served instantly (and cost no Gemini quota). An in-memory TTL/LRU cache is
always used; Cloud Firestore can be enabled for durable, shareable reports.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Protocol

from app.config import Settings
from app.schemas import Report

logger = logging.getLogger(__name__)


def report_key(question: str, depth: str = "standard") -> str:
    """Stable ID for a question + depth: case/whitespace-insensitive SHA-256 prefix."""
    normalised = " ".join(question.lower().split()).rstrip("?.! ") + f"|{depth}"
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:20]


class ReportStore(Protocol):
    async def get(self, key: str) -> Report | None: ...

    async def put(self, report: Report) -> None: ...


class MemoryReportStore:
    """Bounded LRU cache with per-entry TTL."""

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
    """Durable store in Cloud Firestore, fronted by the in-memory cache."""

    def __init__(self, collection: str, cache: MemoryReportStore, client: object | None = None):
        if client is None:
            from google.cloud import firestore  # optional dependency

            client = firestore.AsyncClient()
        self._collection = client.collection(collection)
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
                timeout=10,
            )
        except Exception:
            logger.exception("firestore_write_failed")


def build_store(settings: Settings) -> ReportStore:
    cache = MemoryReportStore(settings.cache_max_items, settings.cache_ttl_s)
    if settings.firestore_enabled:
        try:
            return FirestoreReportStore(settings.firestore_collection, cache)
        except Exception:
            logger.exception("firestore_unavailable_using_memory")
    return cache
