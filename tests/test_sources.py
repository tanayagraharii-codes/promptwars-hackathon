import pytest

from app.schemas import CredibilityTier
from app.sources import (
    SourceRegistry,
    extract_domain,
    format_sources_for_prompt,
    score_domain,
    tier_for,
)


@pytest.mark.parametrize(
    ("domain", "expected_tier"),
    [
        ("who.int", CredibilityTier.HIGH),
        ("pubmed.ncbi.nlm.nih.gov", CredibilityTier.HIGH),
        ("www.cdc.gov", CredibilityTier.HIGH),
        ("ox.ac.uk", CredibilityTier.HIGH),
        ("stanford.edu", CredibilityTier.HIGH),
        ("reuters.com", CredibilityTier.HIGH),
        ("en.wikipedia.org", CredibilityTier.MEDIUM),
        ("example.com", CredibilityTier.MEDIUM),
        ("redcross.org", CredibilityTier.MEDIUM),
        ("reddit.com", CredibilityTier.LOW),
        ("someone.blogspot.com", CredibilityTier.LOW),
        ("", CredibilityTier.LOW),
    ],
)
def test_domain_tiers(domain, expected_tier):
    assert tier_for(score_domain(domain)) == expected_tier


def test_lookalike_domain_is_not_trusted():
    # "evilwho.int.example.com" must not inherit who.int's score.
    assert score_domain("notwho.int.example.com") == score_domain("example.com")
    assert score_domain("fakenature.com") < score_domain("nature.com")


def test_extract_domain_prefers_title_for_grounding_redirects():
    redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc"
    assert extract_domain(redirect, "nasa.gov") == "nasa.gov"
    assert extract_domain("https://www.bbc.co.uk/news/1", "BBC News article") == "bbc.co.uk"


def test_registry_deduplicates_and_numbers_sources():
    registry = SourceRegistry()
    a = registry.add("https://who.int/a#section", "WHO", "snippet")
    b = registry.add("https://who.int/a/", "WHO duplicate")
    c = registry.add("https://example.com/x", "Example")
    assert a.id == b.id == "S1"
    assert c.id == "S2"
    assert registry.ids() == {"S1", "S2"}
    assert registry.get("S2").domain == "example.com"
    assert registry.get("S9") is None
    assert [s.id for s in registry.all()] == ["S1", "S2"]


def test_registry_backfills_missing_snippet():
    registry = SourceRegistry()
    registry.add("https://example.com", "Example")
    source = registry.add("https://example.com", "Example", "later snippet")
    assert source.snippet == "later snippet"


def test_format_sources_for_prompt_includes_credibility():
    registry = SourceRegistry()
    registry.add("https://cdc.gov/x", "CDC", "Vaccines are tested.")
    text = format_sources_for_prompt(registry.all())
    assert "[S1] CDC (cdc.gov, credibility 0.90)" in text
    assert "Vaccines are tested." in text
