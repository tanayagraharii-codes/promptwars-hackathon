import json

import pytest
from fastapi.testclient import TestClient

from app.main import SECURITY_HEADERS, create_app
from app.orchestrator import ResearchOrchestrator
from app.storage import MemoryReportStore


@pytest.fixture
def client(fake_llm, fake_retriever, settings):
    store = MemoryReportStore(10, 60)
    orchestrator = ResearchOrchestrator(fake_llm, fake_retriever, store, settings)
    return TestClient(create_app(settings=settings, orchestrator=orchestrator, store=store))


def _events(response):
    return [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_served_with_security_headers(client):
    response = client.get("/")
    assert response.status_code == 200
    assert '<html lang="en">' in response.text
    for header in SECURITY_HEADERS:
        assert header in response.headers
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_static_assets(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_research_streams_events_and_report(client):
    response = client.post("/api/research", json={"question": "Did X increase last year?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    events = _events(response)
    assert events[0]["type"] == "agent"
    assert events[-1]["type"] == "report"

    report_id = events[-1]["report"]["id"]
    fetched = client.get(f"/api/reports/{report_id}")
    assert fetched.status_code == 200
    assert fetched.json()["title"] == "Test report"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "short"},
        {"question": "x" * 501},
        {},
        {"question": 123},
        {"question": "A perfectly valid question?", "depth": "extreme"},
    ],
)
def test_research_rejects_invalid_input(client, payload):
    response = client.post("/api/research", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_body_size_limit(client):
    response = client.post(
        "/api/research",
        content=json.dumps({"question": "y" * 10_000}),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_rate_limit(client):
    for _ in range(3):
        assert (
            client.post("/api/research", json={"question": "Did X increase last year?"}).status_code
            == 200
        )
    blocked = client.post("/api/research", json={"question": "Did X increase last year?"})
    assert blocked.status_code == 429
    assert "Too many requests" in blocked.json()["detail"]


@pytest.mark.parametrize("report_id", ["0123456789abcdef0123", "../../etc/passwd", "ZZZ"])
def test_unknown_or_malformed_report_ids_404(client, report_id):
    assert client.get(f"/api/reports/{report_id}").status_code == 404


def test_api_docs_available(client):
    assert client.get("/api/docs").status_code == 200


def test_research_accepts_quick_depth(client):
    response = client.post(
        "/api/research", json={"question": "Did X increase last year?", "depth": "quick"}
    )
    assert response.status_code == 200
    assert _events(response)[-1]["report"]["depth"] == "quick"


def test_frontend_library_served(client):
    response = client.get("/static/lib.js")
    assert response.status_code == 200
    assert "createSseParser" in response.text
