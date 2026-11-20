from types import SimpleNamespace

from app.config import Settings
from app.schemas import Report
from app.storage import FirestoreReportStore, MemoryReportStore, build_store, report_key


def _report(key: str) -> Report:
    return Report(
        id=key,
        question="q",
        created_at="2026-01-01T00:00:00+00:00",
        search_backend="fake",
        title="t",
        summary="s",
        key_findings=[],
        sections=[],
        limitations=[],
        sub_questions=[],
        claims=[],
        conflicts=[],
        sources=[],
        duration_s=1.0,
    )


def test_report_key_is_normalised():
    assert report_key("Is coffee healthy?") == report_key("  is   COFFEE healthy ")
    assert report_key("a question") != report_key("another question")
    assert len(report_key("x")) == 20


async def test_memory_store_ttl_and_lru(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("app.storage.time.monotonic", lambda: clock[0])
    store = MemoryReportStore(max_items=2, ttl_s=10)
    await store.put(_report("a"))
    await store.put(_report("b"))
    assert await store.get("a") is not None  # "a" now most recent
    await store.put(_report("c"))  # evicts "b"
    assert await store.get("b") is None
    clock[0] = 11
    assert await store.get("a") is None  # expired


class _Doc:
    def __init__(self, db, key):
        self.db, self.key = db, key

    async def get(self):
        data = self.db.get(self.key)
        return SimpleNamespace(exists=data is not None, to_dict=lambda: data)

    async def set(self, data):
        self.db[self.key] = data


class _FakeFirestore:
    def __init__(self):
        self.db = {}

    def collection(self, _name):
        return SimpleNamespace(document=lambda key: _Doc(self.db, key))


async def test_firestore_store_round_trip():
    fake = _FakeFirestore()
    store = FirestoreReportStore("reports", MemoryReportStore(5, 60), client=fake)
    await store.put(_report("abc"))
    assert "abc" in fake.db

    fresh = FirestoreReportStore("reports", MemoryReportStore(5, 60), client=fake)
    loaded = await fresh.get("abc")
    assert loaded is not None and loaded.id == "abc"
    assert await fresh.get("missing") is None


async def test_firestore_errors_do_not_break_requests():
    class Broken:
        def collection(self, _):
            async def boom(*_a, **_k):
                raise RuntimeError("down")

            return SimpleNamespace(document=lambda _k: SimpleNamespace(get=boom, set=boom))

    store = FirestoreReportStore("r", MemoryReportStore(5, 60), client=Broken())
    await store.put(_report("x"))  # logged, not raised
    assert await store.get("x") is not None  # still served from cache
    assert await store.get("y") is None


def test_build_store_defaults_to_memory():
    assert isinstance(build_store(Settings(firestore_enabled=False)), MemoryReportStore)


def test_build_store_falls_back_to_memory_when_firestore_cannot_start(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("no Google Cloud credentials")

    monkeypatch.setattr("app.storage.firestore.AsyncClient", unavailable)
    assert isinstance(build_store(Settings(firestore_enabled=True)), MemoryReportStore)
