"""HTTP routes: the streaming research endpoint, saved reports, health and the UI."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app import __version__
from app.events import Event
from app.orchestrator import ResearchOrchestrator
from app.schemas import Report, ResearchRequest

from .dependencies import ServicesDep, enforce_rate_limit, ensure_capacity

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
REPORT_ID = re.compile(r"^[0-9a-f]{20}$")
SSE_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}

router = APIRouter()


@router.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.post(
    "/api/research",
    summary="Research and fact-check a topic (Server-Sent Events)",
    dependencies=[Depends(enforce_rate_limit), Depends(ensure_capacity)],
)
async def research(
    body: ResearchRequest, request: Request, services: ServicesDep
) -> StreamingResponse:
    """Stream agent progress events, ending with the report or an error event."""

    async def stream(orchestrator: ResearchOrchestrator) -> AsyncIterator[str]:
        async with services.jobs:
            async for event in orchestrator.run(body.question, body.depth):
                if await request.is_disconnected():
                    logger.info("client_disconnected")
                    return
                yield format_sse(event)

    return StreamingResponse(
        stream(services.orchestrator), media_type="text/event-stream", headers=SSE_HEADERS
    )


@router.get("/api/reports/{report_id}", response_model=Report, summary="Fetch a saved report")
async def get_report(report_id: str, services: ServicesDep) -> Report:
    """Return a previously generated report (IDs are validated before any lookup)."""
    report = await services.store.get(report_id) if REPORT_ID.fullmatch(report_id) else None
    if report is None:
        raise HTTPException(404, "Report not found.")
    return report


@router.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def format_sse(event: Event) -> str:
    """Serialise one event as a Server-Sent Events ``data:`` frame."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def validation_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Return the first validation problem as a short, user-friendly message."""
    errors = exc.errors() if isinstance(exc, RequestValidationError) else []
    message = str(errors[0].get("msg", "Invalid request.")) if errors else "Invalid request."
    return JSONResponse({"detail": message.removeprefix("Value error, ")}, status_code=422)
