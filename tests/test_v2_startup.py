"""V2 must not wipe a populated vector DB on every startup."""
import asyncio
import os
import threading
from unittest.mock import MagicMock

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from app import main_demo  # noqa: E402


def test_startup_skips_ingest_when_db_already_populated(monkeypatch):
    fake_collection = MagicMock()
    fake_collection.count.return_value = 834
    monkeypatch.setattr(main_demo, "collection", fake_collection)
    called = threading.Event()
    monkeypatch.setattr(main_demo, "run_ingest", called.set)

    asyncio.run(main_demo.startup_ingest())

    assert not called.wait(timeout=1), "ingest ran on a populated DB (would wipe it)"
    assert main_demo.ingest_status["done"] is True
    assert main_demo.ingest_status["count"] == 834


def test_startup_runs_ingest_when_db_is_empty(monkeypatch):
    fake_collection = MagicMock()
    fake_collection.count.return_value = 0
    monkeypatch.setattr(main_demo, "collection", fake_collection)
    called = threading.Event()
    monkeypatch.setattr(main_demo, "run_ingest", called.set)

    asyncio.run(main_demo.startup_ingest())

    assert called.wait(timeout=5), "ingest did not run on an empty DB"
