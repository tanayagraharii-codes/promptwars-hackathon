"""Tests for the deterministic text utilities."""

from __future__ import annotations

import pytest

from app.text import (
    content_words,
    is_near_duplicate,
    select_passages,
    stems,
    strip_html,
    widening_queries,
)


def test_widening_queries_drop_stopwords_and_shrink():
    queries = widening_queries("What is the effect of remote work on productivity?")
    assert queries == ["effect remote work productivity", "effect remote work", "effect remote"]


def test_widening_queries_fall_back_to_the_raw_text():
    assert widening_queries("the of and") == ["the of and"]


def test_content_words_and_stems():
    assert content_words("How do the emissions compare?") == ["emissions", "compare"]
    assert stems("emission emissions") == {"emiss"}


def test_strip_html():
    assert strip_html("a <b>bold</b> move") == "a bold move"


def test_select_passages_keeps_lead_and_respects_budget():
    article = "Lead paragraph about coffee and its history in the world.\n" + "\n".join(
        f"Coffee consumption study {i} found caffeine health effects on heart outcomes."
        for i in range(100)
    )
    passages = select_passages(article, "coffee caffeine health heart", max_chars=500)
    assert passages.startswith("Lead paragraph")
    assert len(passages) <= 500


def test_select_passages_returns_short_text_unchanged():
    assert select_passages("short", "x") == "short"


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("Opponents argue remote technology cannot fully replace face to face contact", True),
        ("Remote workers saved 40 minutes of commuting daily.", False),
    ],
)
def test_near_duplicate_detection(statement: str, expected: bool):
    existing = ["Opponents argue remote technology cannot fully replace face-to-face contact."]
    assert is_near_duplicate(statement, existing) is expected


def test_near_duplicate_with_nothing_to_compare():
    assert not is_near_duplicate("Anything at all", [])
