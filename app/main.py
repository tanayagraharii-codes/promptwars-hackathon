"""Application entry point: wires settings, services, middleware and routes together."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import (
    STATIC_DIR,
    BodySizeLimitMiddleware,
    SecurityHeadersMiddleware,
    Services,
    TraceContextMiddleware,
    router,
    validation_error_handler,
)
from app.config import Settings, get_settings
from app.logging_config import configure_logging
from app.orchestrator import ResearchOrchestrator
from app.storage import ReportStore, build_store

GZIP_MIN_BYTES = 1024


def create_app(
    settings: Settings | None = None,
    orchestrator: ResearchOrchestrator | None = None,
    store: ReportStore | None = None,
) -> FastAPI:
    """Build the FastAPI app. Tests inject a fake orchestrator and store."""
    settings = settings or get_settings()
    app = FastAPI(
        title="Evidentia",
        version=__version__,
        description="Multi-agent research and fact-checking powered by Gemini.",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.services = Services.create(
        settings, store or build_store(settings), orchestrator=orchestrator
    )

    # Starlette runs the last-added middleware first (outermost).
    app.add_middleware(GZipMiddleware, minimum_size=GZIP_MIN_BYTES)
    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(TraceContextMiddleware)

    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


configure_logging(project=get_settings().gcp_project)
app = create_app()
