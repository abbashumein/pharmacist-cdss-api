"""One drug must not take every evidence slot (breaks interaction answers)."""
import os

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from app import main_agentic  # noqa: E402


def _item(score, drug, i):
    return (score, f"{drug.upper()}-DOC-{i}", 0.5, drug)


def test_select_evidence_gives_each_drug_a_share():
    usable = [_item(9 - i * 0.1, "warfarin", i) for i in range(5)]      # high scores
    usable += [_item(1 - i * 0.1, "aspirin", i) for i in range(5)]      # low scores

    picked = main_agentic.select_evidence(usable)

    drugs = [p[3] for p in picked]
    assert drugs.count("aspirin") >= 3, "aspirin was crowded out of the evidence"
    assert drugs.count("warfarin") >= 3
    assert len(picked) <= main_agentic.MAX_EVIDENCE_CHUNKS


def test_select_evidence_single_drug_keeps_best_chunks_in_order():
    usable = [_item(s, "warfarin", i) for i, s in enumerate([1, 5, 3, 4, 2])]
    picked = main_agentic.select_evidence(usable)
    assert [p[0] for p in picked] == [5, 4, 3, 2, 1]


class _FakeCollection:
    """Returns 5 chunks that name the drug in the query."""
    def query(self, query_texts, n_results, include):
        import re
        drug = re.search(r"Drug: (\w+)", query_texts[0]).group(1).upper()
        return {"documents": [[f"{drug}-DOC-{i}" for i in range(5)]],
                "distances": [[0.5] * 5]}


class _FakeReranker:
    """Always prefers warfarin text, like the real reranker did in Q1."""
    def predict(self, pairs):
        return [5.0 if "WARFARIN" in doc else 1.0 for _, doc in pairs]


def test_two_drug_query_returns_evidence_for_both(monkeypatch):
    monkeypatch.setattr(main_agentic, "collection", _FakeCollection())
    monkeypatch.setattr(main_agentic, "reranker", _FakeReranker())

    result = main_agentic.check_fda_database("Can warfarin interact with aspirin?")

    assert result["has_usable_evidence"] is True
    text = result["evidence_text"]
    assert "WARFARIN-DOC" in text
    assert "ASPIRIN-DOC" in text, "second drug missing: model will refuse the interaction"
