"""Application services and the FastAPI dependencies that expose them to routes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from app.config import Settings
from app.llm import GeminiLLM
from app.orchestrator import ResearchOrchestrator
from app.retrieval import build_retriever
from app.security import RateLimiter, client_identifier
from app.storage import ReportStore


@dataclass
class Services:
    """Long-lived objects shared by all requests, stored on ``app.state``."""

    settings: Settings
    store: ReportStore
    limiter: RateLimiter
    jobs: asyncio.Semaphore
    _orchestrator: ResearchOrchestrator | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        settings: Settings,
        store: ReportStore,
        orchestrator: ResearchOrchestrator | None = None,
    ) -> Services:
        return cls(
            settings=settings,
            store=store,
            limiter=RateLimiter(settings.rate_limit_per_minute),
            jobs=asyncio.Semaphore(settings.max_concurrent_jobs),
            _orchestrator=orchestrator,
        )

    @property
    def orchestrator(self) -> ResearchOrchestrator:
        """Built on first use, so health checks work before credentials are needed."""
        if self._orchestrator is None:
            llm = GeminiLLM(self.settings)
            retriever = build_retriever(self.settings.search_backend, llm)
            self._orchestrator = ResearchOrchestrator(llm, retriever, self.store, self.settings)
        return self._orchestrator


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def enforce_rate_limit(request: Request, services: ServicesDep) -> None:
    """Per-client sliding-window limit (HTTP 429 when exceeded)."""
    client = client_identifier(
        request.headers.get("x-forwarded-for"),
        request.client.host if request.client else None,
        services.settings.trust_proxy_headers,
    )
    if not services.limiter.allow(client):
        raise HTTPException(429, "Too many requests. Please wait a minute and try again.")


def ensure_capacity(services: ServicesDep) -> None:
    """Global cap on concurrent research jobs (HTTP 503 when all slots are busy)."""
    if services.jobs.locked():
        raise HTTPException(503, "The service is busy. Please try again shortly.")
