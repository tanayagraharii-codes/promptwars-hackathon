"""HTTP-level tests: streaming, validation, limits, headers, saved reports and tracing."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import SECURITY_HEADERS, Services
from app.config import Settings
from app.logging_config import CloudLoggingFormatter, TraceContext, current_trace
from app.main import create_app
from app.orchestrator import ResearchOrchestrator
from app.storage import MemoryReportStore
from tests.fakes import FakeLLM, FakeRetriever

VALID_QUESTION = {"question": "Did X increase last year?"}
TRACE_ID = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def client(fake_llm: FakeLLM, fake_retriever: FakeRetriever, settings: Settings) -> TestClient:
    store = MemoryReportStore(10, 60)
    orchestrator = ResearchOrchestrator(fake_llm, fake_retriever, store, settings)
    return TestClient(create_app(settings=settings, orchestrator=orchestrator, store=store))


def services_of(client: TestClient) -> Services:
    services: Services = client.app.state.services  # type: ignore[attr-defined]
    return services


def sse_events(response: Any) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_healthz(client: TestClient):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_index_is_served_with_security_headers(client: TestClient, method: str):
    response = client.request(method, "/")
    assert response.status_code == 200
    for header in SECURITY_HEADERS:
        assert header in response.headers
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "cache-control" not in response.headers  # only API responses are no-store


@pytest.mark.parametrize(
    "path", ["/static/js/app.js", "/static/js/lib.js", "/static/js/render.js", "/static/styles.css"]
)
def test_static_assets_are_served(client: TestClient, path: str):
    assert client.get(path).status_code == 200


def test_research_streams_events_and_saves_the_report(client: TestClient):
    response = client.post("/api/research", json=VALID_QUESTION)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    events = sse_events(response)
    assert events[0]["type"] == "agent"
    assert events[-1]["type"] == "report"

    saved = client.get(f"/api/reports/{events[-1]['report']['id']}")
    assert saved.status_code == 200
    assert saved.json()["title"] == "Test report"


def test_research_accepts_quick_depth(client: TestClient):
    response = client.post("/api/research", json={**VALID_QUESTION, "depth": "quick"})
    assert sse_events(response)[-1]["report"]["depth"] == "quick"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "short"},
        {"question": "x" * 501},
        {},
        {"question": 123},
        {**VALID_QUESTION, "depth": "extreme"},
    ],
)
def test_invalid_input_is_rejected_with_a_message(client: TestClient, payload: dict[str, Any]):
    response = client.post("/api/research", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]
    assert "Value error" not in response.json()["detail"]


@pytest.mark.parametrize("length", ["99999", "not-a-number"])
def test_oversized_or_malformed_bodies_are_rejected(client: TestClient, length: str):
    response = client.post(
        "/api/research",
        content=json.dumps(VALID_QUESTION),
        headers={"Content-Type": "application/json", "Content-Length": length},
    )
    assert response.status_code == 413


def test_rate_limit(client: TestClient):
    for _ in range(3):
        assert client.post("/api/research", json=VALID_QUESTION).status_code == 200
    blocked = client.post("/api/research", json=VALID_QUESTION)
    assert blocked.status_code == 429
    assert "Too many requests" in blocked.json()["detail"]


def test_busy_service_returns_503(client: TestClient):
    services_of(client).jobs = asyncio.Semaphore(0)  # every job slot taken
    assert client.post("/api/research", json=VALID_QUESTION).status_code == 503


@pytest.mark.parametrize("report_id", ["0123456789abcdef0123", "../../etc/passwd", "ZZZ"])
def test_unknown_or_malformed_report_ids_return_404(client: TestClient, report_id: str):
    assert client.get(f"/api/reports/{report_id}").status_code == 404


def test_api_docs_are_available(client: TestClient):
    assert client.get("/api/docs").status_code == 200


def test_orchestrator_is_built_lazily(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    built: list[Settings] = []

    class FakeGemini(FakeLLM):
        def __init__(self, config: Settings) -> None:
            super().__init__()
            built.append(config)

    monkeypatch.setattr("app.api.dependencies.GeminiLLM", FakeGemini)
    services = Services.create(settings, MemoryReportStore(1, 1))
    assert built == []
    assert services.orchestrator is services.orchestrator
    assert built == [settings]


# ----------------------------------------------------------------- Cloud Trace correlation


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (f"{TRACE_ID}/123;o=1", TraceContext(TRACE_ID, "123", sampled=True)),
        (TRACE_ID.upper(), TraceContext(TRACE_ID, None, sampled=False)),
        ("not-a-trace/1;o=1", None),
        (None, None),
    ],
)
def test_trace_header_parsing(header: str | None, expected: TraceContext | None):
    assert TraceContext.parse(header) == expected


def test_log_lines_carry_the_request_trace():
    formatter = CloudLoggingFormatter(project="demo-project")
    record = logging.makeLogRecord({"msg": "hello", "levelname": "INFO", "claims": 3})
    token = current_trace.set(TraceContext(TRACE_ID, "42", sampled=True))
    try:
        payload = json.loads(formatter.format(record))
    finally:
        current_trace.reset(token)
    assert payload["logging.googleapis.com/trace"] == f"projects/demo-project/traces/{TRACE_ID}"
    assert payload["logging.googleapis.com/spanId"] == "42"
    assert payload["claims"] == 3
    assert "logging.googleapis.com/trace" not in json.loads(formatter.format(record))


def test_trace_context_is_set_only_during_a_request(client: TestClient):
    seen: list[TraceContext | None] = []

    @client.app.get("/trace-probe")  # type: ignore[attr-defined]
    async def probe() -> dict[str, str]:
        seen.append(current_trace.get())
        return {}

    client.get("/trace-probe", headers={"X-Cloud-Trace-Context": f"{TRACE_ID}/7;o=1"})
    assert seen == [TraceContext(TRACE_ID, "7", sampled=True)]
    assert current_trace.get() is None


def test_middleware_passes_through_non_http_scopes(client: TestClient):
    with client:  # entering the client runs the ASGI lifespan protocol through every middleware
        assert client.get("/healthz").status_code == 200
