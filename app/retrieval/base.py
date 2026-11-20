"""The retrieval interface shared by every evidence backend."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from app.schemas import Evidence, Source
from app.sources import SourceRegistry


class Retriever(Protocol):
    """Finds evidence for questions and claims, registering every source it cites."""

    @property
    def name(self) -> str: ...

    async def search(
        self, query: str, registry: SourceRegistry, keywords: str | None = None
    ) -> Evidence:
        """Gather evidence answering ``query`` (``keywords`` may help keyword engines)."""
        ...

    async def search_many(self, queries: list[str], registry: SourceRegistry) -> Evidence:
        """Gather independent evidence for several claims at once."""
        ...


def unique_sources(sources: Iterable[Source]) -> list[Source]:
    """De-duplicate sources by ID, keeping first-seen order."""
    return list({source.id: source for source in sources}.values())
