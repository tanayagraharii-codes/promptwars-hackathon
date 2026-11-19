"""Thin async wrapper around the Google Gen AI SDK (Gemini).

Responsibilities:
* structured JSON generation validated against Pydantic schemas;
* Google Search grounded generation that returns the cited web sources;
* bounded concurrency, timeouts, retries with backoff and model fallback.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRYABLE = {500, 502, 503, 504}
_SKIP_MODEL = {404, 429}  # model retired / out of quota: try the next model
MODEL_COOLDOWN_S = 120.0


class LLMError(RuntimeError):
    """Raised when no configured model could fulfil a request."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class GroundedResult:
    text: str
    sources: list[tuple[str, str]]  # (uri, title)
    # (segment end byte offset, indices into ``sources``) from grounding supports
    supports: list[tuple[int, list[int]]] = field(default_factory=list)


class LLM(Protocol):
    """Interface the agents depend on; tests supply a fake implementation."""

    async def generate_json(
        self, prompt: str, schema: type[T], *, system: str, tier: str = "fast"
    ) -> T: ...

    async def grounded_search(self, prompt: str) -> GroundedResult: ...


def build_genai_client(settings: Settings) -> object:
    """Create a Gen AI client for Vertex AI (service account) or the Gemini Developer API.

    Vertex AI is preferred in production: no API key exists to leak, access is
    governed by IAM, and usage is billed and audited in the Google Cloud project.
    """
    from google import genai  # imported lazily so tests never need credentials

    if settings.use_vertex_ai:
        if not settings.gcp_project:
            raise ValueError("GOOGLE_CLOUD_PROJECT is required when using Vertex AI.")
        return genai.Client(
            vertexai=True, project=settings.gcp_project, location=settings.gcp_location
        )
    return genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY


class GeminiLLM:
    """Production implementation backed by ``google-genai``."""

    def __init__(self, settings: Settings, client: object | None = None) -> None:
        self._settings = settings
        self._client = client or build_genai_client(settings)
        self._semaphore = asyncio.Semaphore(settings.llm_concurrency)
        self._models = {
            "fast": settings.fast_models,
            "smart": settings.smart_models,
            "search": settings.search_models,
        }
        self._cooldown: dict[tuple[str, str], float] = {}

    async def generate_json(
        self, prompt: str, schema: type[T], *, system: str, tier: str = "fast"
    ) -> T:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        )
        last_error: Exception | None = None
        for _ in range(2):  # one repair attempt if the JSON fails validation
            response = await self._call(tier, prompt, config)
            try:
                return schema.model_validate_json(response.text or "")
            except ValidationError as exc:
                last_error = exc
                logger.warning("schema_validation_failed", extra={"schema": schema.__name__})
        raise LLMError(f"Model output did not match {schema.__name__}: {last_error}")

    async def grounded_search(self, prompt: str) -> GroundedResult:
        from google.genai import types

        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.1,
        )
        response = await self._call("search", prompt, config)
        sources: list[tuple[str, str]] = []
        candidates = getattr(response, "candidates", None) or []
        metadata = getattr(candidates[0], "grounding_metadata", None) if candidates else None
        for chunk in getattr(metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            sources.append((web.uri or "", web.title or "") if web else ("", ""))
        supports = [
            (s.segment.end_index, list(s.grounding_chunk_indices or []))
            for s in getattr(metadata, "grounding_supports", None) or []
            if s.segment is not None and s.segment.end_index is not None
        ]
        return GroundedResult(text=response.text or "", sources=sources, supports=supports)

    async def _call(self, tier: str, prompt: str, config: object) -> object:
        """Try each model in the tier's chain, skipping models that recently failed."""
        from google.genai import errors

        chain = self._models[tier]
        now = time.monotonic()
        healthy = [m for m in chain if self._cooldown.get((tier, m), 0.0) <= now]
        last_status: int | None = None
        async with self._semaphore:
            for model in healthy or list(chain):
                for attempt in range(2):
                    try:
                        return await asyncio.wait_for(
                            self._client.aio.models.generate_content(
                                model=model, contents=prompt, config=config
                            ),
                            timeout=self._settings.llm_timeout_s,
                        )
                    except errors.APIError as exc:
                        last_status = exc.code
                        logger.warning(
                            "gemini_call_failed",
                            extra={"model": model, "status": exc.code, "attempt": attempt},
                        )
                        if exc.code in _SKIP_MODEL:
                            break
                        if exc.code not in _RETRYABLE:
                            raise LLMError("Gemini request was rejected.", status=exc.code) from exc
                    except TimeoutError:
                        last_status = 504
                        logger.warning("gemini_timeout", extra={"model": model})
                        break  # a slow model is likely saturated: fail over immediately
                    await asyncio.sleep(min(4.0, 0.5 * 2**attempt) + random.random() * 0.3)  # noqa: S311 - jitter, not crypto
                # This model failed: skip it for a while so other calls go straight to a fallback.
                self._cooldown[(tier, model)] = time.monotonic() + MODEL_COOLDOWN_S
        raise LLMError("All Gemini models are unavailable right now.", status=last_status)
