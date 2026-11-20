"""Evidence from Gemini with Grounding with Google Search (the primary backend)."""

from __future__ import annotations

from app.llm import LLM
from app.schemas import Evidence
from app.security import is_safe_http_url
from app.sources import SourceRegistry

RESEARCH_PROMPT = (
    "Research the following question using Google Search. Report concrete facts, figures, "
    "dates and study findings, and note where sources disagree.\n\nQuestion: {query}"
)
VERIFY_PROMPT = (
    "Independently verify each statement below using Google Search. For each one, report "
    "evidence that confirms it AND evidence that contradicts or qualifies it. Prefer primary "
    "sources, official statistics and peer-reviewed research.\n\n{statements}"
)


class GoogleSearchRetriever:
    """Turns grounded Gemini answers into evidence with numbered ``[S#]`` citations."""

    name = "google_search"

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def search(
        self, query: str, registry: SourceRegistry, keywords: str | None = None
    ) -> Evidence:
        """Research one question; ``keywords`` is unused because Google handles prose."""
        return await self._run(RESEARCH_PROMPT.format(query=query), registry)

    async def search_many(self, queries: list[str], registry: SourceRegistry) -> Evidence:
        """Verify several statements in a single grounded call (one call, not N)."""
        statements = "\n".join(f"{i}. {q}" for i, q in enumerate(queries, start=1))
        return await self._run(VERIFY_PROMPT.format(statements=statements), registry)

    async def _run(self, prompt: str, registry: SourceRegistry) -> Evidence:
        result = await self._llm.grounded_search(prompt)
        chunk_ids = {
            index: registry.add(uri, title).id
            for index, (uri, title) in enumerate(result.sources)
            if is_safe_http_url(uri)
        }
        text = annotate_citations(result.text, result.supports, chunk_ids)
        cited = (registry.get(source_id) for source_id in dict.fromkeys(chunk_ids.values()))
        return Evidence(text=text, sources=[s for s in cited if s], backend=self.name)


def annotate_citations(
    text: str, supports: list[tuple[int, list[int]]], chunk_ids: dict[int, str]
) -> str:
    """Insert ``[S#]`` markers after each grounded segment.

    Gemini reports segment offsets in UTF-8 bytes, so markers are spliced into
    the encoded text, working backwards so earlier offsets stay valid.
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
