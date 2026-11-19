"""FastAPI entry point: API routes, streaming, security middleware and static UI."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import Settings, get_settings
from app.logging_config import configure_logging
from app.orchestrator import ResearchOrchestrator
from app.schemas import Report, ResearchRequest
from app.security import RateLimiter, client_identifier
from app.storage import ReportStore, build_store

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MAX_BODY_BYTES = 4096
REPORT_ID = re.compile(r"^[0-9a-f]{20}$")

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; "
        "form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), interest-cohort=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Cross-Origin-Opener-Policy": "same-origin",
}


def _build_default_orchestrator(settings: Settings, store: ReportStore) -> ResearchOrchestrator:
    from app.llm import GeminiLLM
    from app.search import build_retriever

    llm = GeminiLLM(settings)
    return ResearchOrchestrator(llm, build_retriever(settings.search_backend, llm), store, settings)


def create_app(
    settings: Settings | None = None,
    orchestrator: ResearchOrchestrator | None = None,
    store: ReportStore | None = None,
) -> FastAPI:
    """Application factory; tests inject fakes for the orchestrator and store."""
    settings = settings or get_settings()
    store = store or build_store(settings)
    state: dict[str, ResearchOrchestrator | None] = {"orchestrator": orchestrator}
    limiter = RateLimiter(settings.rate_limit_per_minute)
    jobs = asyncio.Semaphore(settings.max_concurrent_jobs)

    def get_orchestrator() -> ResearchOrchestrator:
        if state["orchestrator"] is None:  # created lazily so /healthz works without creds
            state["orchestrator"] = _build_default_orchestrator(settings, store)
        return state["orchestrator"]

    app = FastAPI(
        title="Evidentia",
        version=__version__,
        description="Multi-agent research and fact-checking powered by Gemini.",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            return JSONResponse({"detail": "Request body too large."}, status_code=413)
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        message = str(first.get("msg", "Invalid request.")).removeprefix("Value error, ")
        return JSONResponse({"detail": message}, status_code=422)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/api/research", summary="Research and fact-check a topic (Server-Sent Events)")
    async def research(body: ResearchRequest, request: Request) -> StreamingResponse:
        client = client_identifier(
            request.headers.get("x-forwarded-for"),
            request.client.host if request.client else None,
            settings.trust_proxy_headers,
        )
        if not limiter.allow(client):
            raise HTTPException(429, "Too many requests. Please wait a minute and try again.")
        if jobs.locked():
            raise HTTPException(503, "The service is busy. Please try again shortly.")

        orchestrator = get_orchestrator()

        async def stream() -> AsyncIterator[str]:
            async with jobs:
                async for event in orchestrator.run(body.question, body.depth):
                    if await request.is_disconnected():
                        logger.info("client_disconnected")
                        return
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/reports/{report_id}", response_model=Report, summary="Fetch a saved report")
    async def get_report(report_id: str) -> Report:
        if not REPORT_ID.fullmatch(report_id):
            raise HTTPException(404, "Report not found.")
        report = await store.get(report_id)
        if report is None:
            raise HTTPException(404, "Report not found.")
        return report

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


configure_logging()
app = create_app()
