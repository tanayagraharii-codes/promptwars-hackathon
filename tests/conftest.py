"""Shared pytest fixtures. Test doubles live in :mod:`tests.fakes`."""

from __future__ import annotations

import pytest

from app.config import Settings
from tests.fakes import FakeLLM, FakeRetriever, default_responses


@pytest.fixture
def settings() -> Settings:
    return Settings(
        max_sub_questions=3,
        max_claims_per_question=3,
        rate_limit_per_minute=3,
        max_concurrent_jobs=2,
        firestore_enabled=False,
        allowed_origins=(),
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(default_responses())


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever()


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``asyncio.sleep`` instant (retry backoff, request pacing)."""

    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr("asyncio.sleep", instant)
