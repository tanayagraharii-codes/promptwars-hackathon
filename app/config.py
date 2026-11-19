"""Application settings, read once from environment variables.

Secrets (the Gemini API key) are never stored here; the google-genai SDK reads
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` itself, or uses the Cloud Run service
account when ``GOOGLE_GENAI_USE_VERTEXAI=true``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env_list(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 10_000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the service."""

    # Model chains: the first model is preferred; later ones are fallbacks used
    # when a model is overloaded or out of quota.
    fast_models: tuple[str, ...] = field(
        default_factory=lambda: _env_list(
            "GEMINI_FAST_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite"
        )
    )
    smart_models: tuple[str, ...] = field(
        default_factory=lambda: _env_list(
            "GEMINI_SMART_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"
        )
    )
    search_models: tuple[str, ...] = field(
        default_factory=lambda: _env_list(
            "GEMINI_SEARCH_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite"
        )
    )
    # "auto" = Google Search grounding, falling back to Wikipedia if unavailable.
    search_backend: str = field(default_factory=lambda: os.getenv("SEARCH_BACKEND", "auto"))

    max_sub_questions: int = field(
        default_factory=lambda: _env_int("MAX_SUB_QUESTIONS", 4, maximum=6)
    )
    max_claims_per_question: int = field(
        default_factory=lambda: _env_int("MAX_CLAIMS_PER_QUESTION", 3, maximum=6)
    )
    llm_concurrency: int = field(default_factory=lambda: _env_int("LLM_CONCURRENCY", 4, maximum=16))
    llm_timeout_s: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT_S", 40, maximum=300))
    max_concurrent_jobs: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT_JOBS", 4, maximum=64)
    )

    rate_limit_per_minute: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 5, maximum=1000)
    )
    trust_proxy_headers: bool = field(default_factory=lambda: _env_bool("TRUST_PROXY", True))

    cache_ttl_s: int = field(default_factory=lambda: _env_int("CACHE_TTL_S", 6 * 3600))
    cache_max_items: int = field(default_factory=lambda: _env_int("CACHE_MAX_ITEMS", 200))
    # On Cloud Run (K_SERVICE is set) reports persist to Firestore by default.
    firestore_enabled: bool = field(
        default_factory=lambda: _env_bool("FIRESTORE_ENABLED", running_on_cloud_run())
    )
    firestore_collection: str = field(
        default_factory=lambda: os.getenv("FIRESTORE_COLLECTION", "evidentia_reports")
    )

    allowed_origins: tuple[str, ...] = field(
        default_factory=lambda: _env_list("ALLOWED_ORIGINS", "")
    )

    # Vertex AI: authenticate with the Cloud Run service account instead of an API key.
    use_vertex_ai: bool = field(
        default_factory=lambda: _env_bool("GOOGLE_GENAI_USE_VERTEXAI", False)
    )
    gcp_project: str = field(default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    gcp_location: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    )


def running_on_cloud_run() -> bool:
    """Cloud Run sets K_SERVICE for every revision."""
    return bool(os.getenv("K_SERVICE"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
