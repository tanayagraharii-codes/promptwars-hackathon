"""Async wrapper around the Google Gen AI SDK (Gemini).

Responsibilities:
* structured JSON generation validated against Pydantic schemas;
* Google Search grounded generation that returns the cited web sources;
* bounded concurrency, timeouts, retries with backoff, and model fallback with
  a per-model cooldown so an overloaded model is skipped rather than retried.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Tier = Literal["fast", "smart", "search"]

RETRYABLE_STATUSES = frozenset({500, 502, 503, 504})
SKIP_MODEL_STATUSES = frozenset({404, 429})  # model retired / out of quota: try the next one
TIMEOUT_STATUS = 504
MODEL_COOLDOWN_S = 120.0
ATTEMPTS_PER_MODEL = 2
JSON_REPAIR_ATTEMPTS = 2
MAX_BACKOFF_S = 4.0


class LLMError(RuntimeError):
    """Raised when no configured model could fulfil a request."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class GroundedResult:
    """Text produced with Google Search grounding, plus the web sources behind it."""

    text: str
    sources: list[tuple[str, str]]  # (uri, title)
    # (segment end byte offset, indices into ``sources``) from grounding supports
    supports: list[tuple[int, list[int]]] = field(default_factory=list)


class LLM(Protocol):
    """Interface the agents depend on; tests supply a fake implementation."""

    async def generate_json(
        self, prompt: str, schema: type[T], *, system: str, tier: Tier = "fast"
    ) -> T: ...

    async def grounded_search(self, prompt: str) -> GroundedResult: ...


def build_genai_client(settings: Settings) -> genai.Client:
    """Create a client for Vertex AI (service account) or the Gemini Developer API.

    Vertex AI is preferred in production: no API key exists to leak, access is
    governed by IAM, and usage is billed and audited in the Google Cloud project.
    """
    if settings.use_vertex_ai:
        if not settings.gcp_project:
            raise ValueError("GOOGLE_CLOUD_PROJECT is required when using Vertex AI.")
        return genai.Client(
            vertexai=True, project=settings.gcp_project, location=settings.gcp_location
        )
    return genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY


def parse_grounding(response: types.GenerateContentResponse) -> GroundedResult:
    """Extract cited web sources and their text spans from grounding metadata."""
    candidates = response.candidates or []
    metadata = candidates[0].grounding_metadata if candidates else None
    sources = [
        (chunk.web.uri or "", chunk.web.title or "") if chunk.web else ("", "")
        for chunk in (metadata.grounding_chunks if metadata else None) or []
    ]
    supports = [
        (support.segment.end_index, list(support.grounding_chunk_indices or []))
        for support in (metadata.grounding_supports if metadata else None) or []
        if support.segment is not None and support.segment.end_index is not None
    ]
    return GroundedResult(text=response.text or "", sources=sources, supports=supports)


class GeminiLLM:
    """Production :class:`LLM` backed by ``google-genai``."""

    def __init__(self, settings: Settings, client: genai.Client | None = None) -> None:
        self._settings = settings
        self._client = client or build_genai_client(settings)
        self._semaphore = asyncio.Semaphore(settings.llm_concurrency)
        self._chains: dict[Tier, tuple[str, ...]] = {
            "fast": settings.fast_models,
            "smart": settings.smart_models,
            "search": settings.search_models,
        }
        self._cooldown_until: dict[tuple[Tier, str], float] = {}

    async def generate_json(
        self, prompt: str, schema: type[T], *, system: str, tier: Tier = "fast"
    ) -> T:
        """Generate JSON that validates against ``schema`` (with one repair attempt)."""
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        )
        last_error: ValidationError | None = None
        for _ in range(JSON_REPAIR_ATTEMPTS):
            response = await self._generate(tier, prompt, config)
            try:
                return schema.model_validate_json(response.text or "")
            except ValidationError as exc:
                last_error = exc
                logger.warning("schema_validation_failed", extra={"schema": schema.__name__})
        raise LLMError(f"Model output did not match {schema.__name__}: {last_error}")

    async def grounded_search(self, prompt: str) -> GroundedResult:
        """Answer ``prompt`` with Google Search grounding and return cited sources."""
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.1,
        )
        return parse_grounding(await self._generate("search", prompt, config))

    async def _generate(
        self, tier: Tier, prompt: str, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        """Try each healthy model in the tier's chain until one succeeds."""
        last_status: int | None = None
        async with self._semaphore:
            for model in self._healthy_models(tier):
                response, last_status = await self._try_model(model, prompt, config)
                if response is not None:
                    return response
                # Skip this model for a while so other calls go straight to a fallback.
                self._cooldown_until[(tier, model)] = time.monotonic() + MODEL_COOLDOWN_S
        raise LLMError("All Gemini models are unavailable right now.", status=last_status)

    def _healthy_models(self, tier: Tier) -> tuple[str, ...]:
        """Models not cooling down; if every model is cooling down, try them all."""
        now = time.monotonic()
        chain = self._chains[tier]
        healthy = tuple(m for m in chain if self._cooldown_until.get((tier, m), 0.0) <= now)
        return healthy or chain

    async def _try_model(
        self, model: str, prompt: str, config: types.GenerateContentConfig
    ) -> tuple[types.GenerateContentResponse | None, int | None]:
        """Call one model with retries. Returns (response, None) or (None, last status)."""
        status: int | None = None
        for attempt in range(ATTEMPTS_PER_MODEL):
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=model, contents=prompt, config=config
                    ),
                    timeout=self._settings.llm_timeout_s,
                )
                return response, None
            except errors.APIError as exc:
                status = exc.code
                logger.warning(
                    "gemini_call_failed",
                    extra={"model": model, "status": exc.code, "attempt": attempt},
                )
                if exc.code in SKIP_MODEL_STATUSES:
                    return None, status
                if exc.code not in RETRYABLE_STATUSES:
                    raise LLMError("Gemini request was rejected.", status=exc.code) from exc
            except TimeoutError:
                logger.warning("gemini_timeout", extra={"model": model})
                return None, TIMEOUT_STATUS  # a slow model is likely saturated: fail over
            await asyncio.sleep(_backoff(attempt))
        return None, status


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter (jitter only spreads load, so it need not be secure)."""
    return min(MAX_BACKOFF_S, 0.5 * 2.0**attempt) + random.random() * 0.3  # noqa: S311
