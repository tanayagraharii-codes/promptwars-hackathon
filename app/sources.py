"""Source credibility scoring and a de-duplicating source registry."""

from __future__ import annotations

import threading
from urllib.parse import urlparse

from app.schemas import CredibilityTier, Source

# Curated, deliberately conservative lists. Scores are priors, not verdicts:
# the fact-checker still has to find agreement across independent sources.
_HIGH_TRUST_DOMAINS: dict[str, float] = {
    "who.int": 0.95,
    "un.org": 0.9,
    "worldbank.org": 0.9,
    "imf.org": 0.9,
    "oecd.org": 0.9,
    "ipcc.ch": 0.95,
    "nature.com": 0.92,
    "science.org": 0.92,
    "thelancet.com": 0.92,
    "nejm.org": 0.92,
    "bmj.com": 0.9,
    "jamanetwork.com": 0.9,
    "cochranelibrary.com": 0.95,
    "pubmed.ncbi.nlm.nih.gov": 0.92,
    "ncbi.nlm.nih.gov": 0.9,
    "cell.com": 0.9,
    "pnas.org": 0.9,
    "sciencedirect.com": 0.85,
    "springer.com": 0.85,
    "wiley.com": 0.85,
    "plos.org": 0.85,
    "arxiv.org": 0.72,
    "reuters.com": 0.82,
    "apnews.com": 0.82,
    "bbc.com": 0.8,
    "bbc.co.uk": 0.8,
    "britannica.com": 0.8,
    "economist.com": 0.78,
    "ft.com": 0.78,
    "nytimes.com": 0.76,
    "theguardian.com": 0.75,
    "wikipedia.org": 0.7,
}

_LOW_TRUST_DOMAINS: dict[str, float] = {
    "reddit.com": 0.25,
    "quora.com": 0.2,
    "twitter.com": 0.2,
    "x.com": 0.2,
    "facebook.com": 0.2,
    "tiktok.com": 0.15,
    "instagram.com": 0.2,
    "youtube.com": 0.3,
    "medium.com": 0.35,
    "substack.com": 0.35,
    "blogspot.com": 0.3,
    "wordpress.com": 0.3,
    "pinterest.com": 0.15,
}

_HIGH_TRUST_SUFFIXES: dict[str, float] = {
    ".gov": 0.9,
    ".mil": 0.85,
    ".int": 0.88,
    ".edu": 0.85,
    ".gov.uk": 0.9,
    ".gov.in": 0.88,
    ".nic.in": 0.85,
    ".gc.ca": 0.88,
    ".gov.au": 0.88,
    ".europa.eu": 0.88,
    ".ac.uk": 0.85,
    ".edu.au": 0.85,
    ".ac.in": 0.82,
}

DEFAULT_SCORE = 0.5


def extract_domain(url: str, title: str = "") -> str:
    """Return the bare host for a URL.

    Google Search grounding returns redirect URLs whose real domain is given
    as the chunk title (e.g. ``"nasa.gov"``), so prefer that when it looks
    like a domain.
    """
    candidate = title.strip().lower()
    if candidate and " " not in candidate and "." in candidate:
        host = candidate
    else:
        host = (urlparse(url).hostname or "").lower()
    return host.removeprefix("www.")


def _matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def score_domain(domain: str) -> float:
    """Prior credibility of a domain in [0, 1]."""
    domain = domain.lower().removeprefix("www.")
    if not domain:
        return 0.3
    for known, score in _HIGH_TRUST_DOMAINS.items():
        if _matches(domain, known):
            return score
    for known, score in _LOW_TRUST_DOMAINS.items():
        if _matches(domain, known):
            return score
    best = None
    for suffix, score in _HIGH_TRUST_SUFFIXES.items():
        if domain.endswith(suffix) and (best is None or score > best):
            best = score
    if best is not None:
        return best
    if domain.endswith(".org"):
        return 0.6
    return DEFAULT_SCORE


def tier_for(score: float) -> CredibilityTier:
    if score >= 0.75:
        return CredibilityTier.HIGH
    if score >= 0.45:
        return CredibilityTier.MEDIUM
    return CredibilityTier.LOW


MAX_TITLE_CHARS = 200
MAX_SNIPPET_CHARS = 500
PROMPT_SNIPPET_CHARS = 300


class SourceRegistry:
    """Assigns stable IDs (S1, S2, ...) to sources and de-duplicates them by URL.

    One registry is shared by every agent in a run, so a page found by two
    researchers keeps a single ID and citations stay consistent.
    """

    def __init__(self) -> None:
        self._by_url: dict[str, Source] = {}
        self._by_id: dict[str, Source] = {}
        self._lock = threading.Lock()

    def add(self, url: str, title: str, snippet: str = "") -> Source:
        """Register a source (or return the existing one for the same URL)."""
        key = url.split("#")[0].rstrip("/")
        with self._lock:
            existing = self._by_url.get(key)
            if existing is not None:
                if snippet and not existing.snippet:
                    existing.snippet = snippet[:MAX_SNIPPET_CHARS]
                return existing
            domain = extract_domain(url, title)
            score = score_domain(domain)
            source = Source(
                id=f"S{len(self._by_url) + 1}",
                url=url,
                title=(title or domain or url)[:MAX_TITLE_CHARS],
                domain=domain,
                credibility=score,
                tier=tier_for(score),
                snippet=snippet[:MAX_SNIPPET_CHARS],
            )
            self._by_url[key] = source
            self._by_id[source.id] = source
            return source

    def get(self, source_id: str) -> Source | None:
        return self._by_id.get(source_id)

    def credibility(self, source_id: str) -> float:
        """Credibility of a registered source; unknown IDs count as zero."""
        source = self._by_id.get(source_id)
        return source.credibility if source else 0.0

    def ids(self) -> set[str]:
        return set(self._by_id)

    def all(self) -> list[Source]:
        """Every registered source, in ID order."""
        return list(self._by_id.values())


def format_sources_for_prompt(sources: list[Source]) -> str:
    """Compact numbered listing used inside agent prompts."""
    lines = []
    for s in sources:
        snippet = f" | {s.snippet[:PROMPT_SNIPPET_CHARS]}" if s.snippet else ""
        lines.append(f"[{s.id}] {s.title} ({s.domain}, credibility {s.credibility:.2f}){snippet}")
    return "\n".join(lines)
