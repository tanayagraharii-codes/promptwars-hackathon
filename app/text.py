"""Lightweight, dependency-free text utilities shared by retrieval and agents.

Everything here is deterministic and cheap, so it runs without model calls:
keyword extraction for search, lexical passage selection, and near-duplicate
detection for extracted claims.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    "a an and are as at be been by can could did do does for from has have how in into is it "
    "its more most of on or than that the their there these this those to was were what when "
    "which who why will with would versus vs about compared".split()
)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_HTML_TAGS = re.compile(r"<[^>]+>")

STEM_LENGTH = 5  # "emissions" and "emission" share the stem "emiss"
MIN_PARAGRAPH_CHARS = 40
LEAD_CHARS = 600
PARAGRAPH_CHARS = 1200
NEAR_DUPLICATE_THRESHOLD = 0.7


def strip_html(text: str) -> str:
    """Remove HTML tags (the MediaWiki API occasionally leaves some in extracts)."""
    return _HTML_TAGS.sub("", text)


def content_words(text: str) -> list[str]:
    """Words in ``text`` that carry meaning (stopwords removed), in order."""
    return [w for w in _WORD.findall(text) if w.lower() not in STOPWORDS]


def stems(text: str) -> set[str]:
    """Crude stems of the content words, used for cheap lexical relevance scoring."""
    return {w.lower()[:STEM_LENGTH] for w in content_words(text) if len(w) > 2}


def widening_queries(text: str, max_terms: int = 6) -> list[str]:
    """Turn a question or claim into keyword queries, from most to least specific.

    Encyclopedia search engines AND every term, so a long query often returns
    nothing; callers try each query in turn until they have enough results.
    """
    terms = content_words(text)[:max_terms]
    if not terms:
        return [text[:120]]
    queries = [" ".join(terms[:n]) for n in (len(terms), 3, 2) if n <= len(terms)]
    return list(dict.fromkeys(queries))


def select_passages(article: str, focus: str, max_chars: int = 2500) -> str:
    """Return the article's lead plus the paragraphs that best match ``focus``, in order.

    Topic words appear all over an article, so only paragraphs scoring at least
    half as well as the best match (and sharing two or more stems) are kept.
    """
    paragraphs = [
        line.strip()
        for line in article.split("\n")
        if len(line.strip()) > MIN_PARAGRAPH_CHARS and not line.strip().startswith("==")
    ]
    if not paragraphs:
        return article[:max_chars].strip()

    terms = stems(focus)
    lead = paragraphs[0][:LEAD_CHARS]
    scored = [(len(terms & stems(p)), i) for i, p in enumerate(paragraphs[1:], start=1)]
    threshold = max(2, max((score for score, _ in scored), default=0) / 2)

    chosen: list[int] = []
    budget = max_chars - len(lead)
    for score, index in sorted(scored, key=lambda item: (-item[0], item[1])):
        if score < threshold or budget <= 0:
            break
        chosen.append(index)
        budget -= len(paragraphs[index][:PARAGRAPH_CHARS]) + 2

    body = [paragraphs[i][:PARAGRAPH_CHARS] for i in sorted(chosen)]
    return "\n\n".join([lead, *body])[:max_chars]


def is_near_duplicate(
    statement: str, existing: list[str], threshold: float = NEAR_DUPLICATE_THRESHOLD
) -> bool:
    """Jaccard similarity of word sets: catches reworded repeats of the same claim.

    Deliberately lexical rather than semantic: embeddings rate "X is higher"
    and "X is lower" as near-identical, and merging contradictions would hide
    exactly the conflicts a fact-checker must surface.
    """
    words = _word_set(statement)
    for other in existing:
        other_words = _word_set(other)
        union = words | other_words
        if union and len(words & other_words) / len(union) >= threshold:
            return True
    return False


def _word_set(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}
