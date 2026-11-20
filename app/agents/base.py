"""Small helpers shared by the agents.

Model output is never trusted blindly: every ID an agent receives from Gemini
is filtered against the IDs that actually exist before it is used.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable


def known_ids(ids: Iterable[str], valid: Collection[str]) -> list[str]:
    """Keep only IDs present in ``valid``, de-duplicated, in their original order."""
    return [i for i in dict.fromkeys(ids) if i in valid]


def clamp_limit(requested: int | None, maximum: int) -> int:
    """A per-request limit within ``[1, maximum]``; ``None`` means the maximum."""
    return maximum if requested is None else max(1, min(requested, maximum))


def clean_lines(items: Iterable[str], limit: int) -> list[str]:
    """Strip whitespace, drop empty entries, and keep at most ``limit`` items."""
    return [item.strip() for item in items if item.strip()][:limit]
