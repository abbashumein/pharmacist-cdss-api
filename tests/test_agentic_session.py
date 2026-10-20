"""Regression tests for the V3 agent: state must not leak between turns.

Bug this guards against: MemorySaver keeps state per session_id. After
turn 1, tool_called stayed True, so turn 2 in the same session skipped
routing + retrieval and answered with turn 1's FDA evidence.

Gemini and the FDA tool are faked, so these tests need no API quota.
They still import app.main_agentic, which loads the local embedding
model and ChromaDB, so run them where those are available.
"""
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from fastapi.testclient import TestClient  # noqa: E402

from app import main_agentic  # noqa: E402

API_KEY = "test-cdss-key"


def _fake_fda(query, extracted_drug=None):
    """Return evidence that names the drug asked about in THIS query."""
    drug = extracted_drug or ("metformin" if "metformin" in query.lower() else "warfarin")
    return {
        "has_usable_evidence": True,
        "retrieval_distance": 0.4,
        "evidence_text": f"EVIDENCE-FOR-{drug.upper()}",
        "chunks": [f"EVIDENCE-FOR-{drug.upper()}"],
    }


def _fake_gemini_factory(routing_calls, generation_calls):
    def fake(model, contents, config=None, max_retries=2):
        if config is not None:
            # Routing call: always ask for the FDA tool, with a drug name.
            routing_calls.append(contents)
            drug = "metformin" if "metformin" in contents.lower() else "warfarin"
            fn = SimpleNamespace(name="fda_lookup", args={"drug_name": drug})
            part = SimpleNamespace(function_call=fn)
            return SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))]
            )
        # Generation call: echo the evidence block so the test can see it.
        generation_calls.append(contents)
        evidence = contents.split("FDA EVIDENCE:")[1].split("USER QUERY:")[0].strip()
        return SimpleNamespace(text=f"Answer using: {evidence}\nRISK_LEVEL: LOW")

    return fake


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CDSS_API_KEY", API_KEY)
    return TestClient(main_agentic.app)


def _chat(client, session_id, message):
    res = client.post(
        "/chat",
        json={"session_id": session_id, "message": message},
        headers={"X-API-KEY": API_KEY},
    )
    assert res.status_code == 200, res.text
    return res.json()


def test_second_turn_same_session_gets_fresh_evidence(client, monkeypatch):
    routing_calls, generation_calls = [], []
    monkeypatch.setattr(main_agentic, "check_fda_database", _fake_fda)
    monkeypatch.setattr(
        main_agentic,
        "call_gemini_with_retry",
        _fake_gemini_factory(routing_calls, generation_calls),
    )

    sid = "two-turn-session"
    first = _chat(client, sid, "Can warfarin and aspirin be taken together?")
    second = _chat(client, sid, "What are the side effects of metformin?")

    # Turn 1 is grounded in warfarin evidence.
    assert "EVIDENCE-FOR-WARFARIN" in first["response"]

    # Turn 2 must be routed again AND use new evidence, not turn 1's.
    assert len(routing_calls) == 2, "turn 2 skipped routing (stale tool_called)"
    assert "EVIDENCE-FOR-METFORMIN" in second["response"]
    assert "EVIDENCE-FOR-WARFARIN" not in second["response"]
    assert second["tool_called"] is True
    assert second["fda_evidence_used"] is True


def test_out_of_scope_turn_after_clinical_turn_does_not_reuse_tool(client, monkeypatch):
    """A non-drug question after a drug question must not run the FDA tool."""
    tool_runs = []

    def counting_fda(query, extracted_drug=None):
        tool_runs.append(query)
        return _fake_fda(query, extracted_drug)

    def fake_gemini(model, contents, config=None, max_retries=2):
        if config is not None:
            if "weather" in contents.lower():
                return SimpleNamespace(
                    candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))]
                )
            return _fake_gemini_factory([], [])(model, contents, config)
        return _fake_gemini_factory([], [])(model, contents, config) if "FDA EVIDENCE:" in contents \
            else SimpleNamespace(text="Not a drug question.\nRISK_LEVEL: LOW")

    monkeypatch.setattr(main_agentic, "check_fda_database", counting_fda)
    monkeypatch.setattr(main_agentic, "call_gemini_with_retry", fake_gemini)

    sid = "scope-session"
    _chat(client, sid, "Can warfarin and aspirin be taken together?")
    second = _chat(client, sid, "What is the weather like today?")

    assert len(tool_runs) == 1, "FDA tool ran for a non-drug question"
    assert second["tool_called"] is False
    assert second["fda_evidence_used"] is False


def test_build_turn_state_resets_every_per_turn_field():
    state = main_agentic.build_turn_state("s1", "hello")
    assert state["tool_called"] is False
    assert state["fda_evidence"] == ""
    assert state["has_usable_evidence"] is False
    assert state["retrieval_distance"] is None
    assert state["extracted_drug"] is None
    assert state["final_response"] == ""
    assert state["risk_level"] == "LOW"
    assert state["messages"] == [{"role": "user", "content": "hello"}]


def test_chat_endpoints_are_sync_so_they_run_in_a_threadpool():
    """/chat does blocking work (Gemini, CrossEncoder, ChromaDB).

    Declared as `async def`, it would run on the event loop and freeze every
    other request while it waits. As plain `def`, FastAPI runs it in a worker
    thread. This test guards against someone changing it back to `async def`.
    """
    import inspect

    assert not inspect.iscoroutinefunction(main_agentic.agent_chat)