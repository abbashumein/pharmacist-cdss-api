"""Chunking regression tests (no network, no models for the pure parts)."""
import json
import os
from types import SimpleNamespace

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from app.chunking import MAX_CHUNK_CHARS, build_chunks, get_drug_name, split_text


def _warfarin(interactions_text):
    return {
        "openfda": {"generic_name": ["WARFARIN SODIUM"], "brand_name": ["Coumadin"]},
        "drug_interactions": [interactions_text],
        "boxed_warning": ["WARNING: BLEEDING RISK. Warfarin can cause major or fatal bleeding."],
        "warnings_and_cautions": ["Tissue necrosis and calciphylaxis have been reported."],
        "dosage_and_administration": ["Individualize dosing. Monitor INR regularly."],
    }


def test_split_text_windows_are_small_and_lose_nothing():
    sentences = [f"Sentence number {i} about a drug effect." for i in range(150)]
    parts = split_text(" ".join(sentences))
    assert len(parts) > 1
    assert all(len(p) <= MAX_CHUNK_CHARS for p in parts)
    joined = " ".join(parts)
    assert all(s in joined for s in sentences), "text was lost between chunks"


def test_fact_late_in_a_long_section_is_not_cut_off():
    """The proven bug: warfarin's aspirin text sat after the old 1,200-char cut."""
    filler = "General interaction text about CYP2C9 and INR monitoring. " * 120   # ~7k chars
    record = _warfarin(filler + "Drugs that increase bleeding risk include aspirin and NSAIDs.")
    chunks = build_chunks(record, 0)
    hits = [c for c in chunks if "aspirin" in c["text"].lower()]
    assert hits, "aspirin text was lost"
    assert hits[0]["text"].startswith("Drug: WARFARIN SODIUM (Coumadin) | Section: Interactions |")


def test_every_chunk_carries_drug_and_section_header_and_metadata():
    chunks = build_chunks(_warfarin("Interacts with aspirin."), 3)
    sections = {c["metadata"]["section"] for c in chunks}
    assert {"interactions", "boxed warning", "warnings", "dosage"} <= sections
    for c in chunks:
        assert c["text"].startswith("Drug: WARFARIN SODIUM")
        assert f"Section: {c['metadata']['section'].title()}" in c["text"].replace("Boxed Warning", "Boxed Warning")
        assert c["metadata"]["drug"] == "warfarin sodium"


def test_ids_are_unique_and_empty_sections_skipped():
    chunks = build_chunks(_warfarin("x " * 3000), 7)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert not any(c["metadata"]["section"] == "side effects" for c in chunks)  # field absent


def test_label_without_a_name_or_clinical_text_gives_no_chunks():
    assert build_chunks({"drug_interactions": ["text"]}, 0) == []
    assert get_drug_name({})["generic"] == ""
    assert build_chunks({"openfda": {"generic_name": ["X"]}}, 0) == []


def test_run_ingest_stores_chunks_with_metadata(monkeypatch):
    """End to end through V2's run_ingest with a fake FDA API and a fake DB."""
    from app import main_demo

    record = _warfarin("Warfarin interacts with aspirin, which raises bleeding risk.")

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"results": [record]}).encode()

    class FakeCollection:
        def __init__(self): self.added = []
        def add(self, documents, ids, metadatas): self.added.append((documents, ids, metadatas))
        def count(self): return sum(len(d) for d, _, _ in self.added)

    fake_collection = FakeCollection()
    fake_client = SimpleNamespace(delete_collection=lambda name: None,
                                  create_collection=lambda **kw: fake_collection)
    monkeypatch.setattr(main_demo, "chroma_client", fake_client)
    monkeypatch.setattr(main_demo.urllib.request, "urlopen", lambda url, timeout=30: FakeResponse())

    main_demo.run_ingest()

    assert main_demo.ingest_status["done"] is True, main_demo.ingest_status
    docs = [d for batch, _, _ in fake_collection.added for d in batch]
    metas = [m for _, _, ms in fake_collection.added for m in ms]
    assert any("aspirin" in d.lower() for d in docs)
    assert all({"drug", "section", "part"} <= set(m) for m in metas)
    assert len(docs) == len(set(docs)), "duplicate chunks were stored"


def test_search_text_names_the_section_the_user_asks_about(monkeypatch):
    from app import main_agentic
    seen = []

    class Capture:
        def query(self, query_texts, n_results, include, where=None):
            seen.append(query_texts[0])
            return {"documents": [[]], "distances": [[]]}

    monkeypatch.setattr(main_agentic, "collection", Capture())
    main_agentic.check_fda_database("What does FDA say about warfarin dosing?")
    main_agentic.check_fda_database("What warnings are associated with warfarin?")
    # (each check searches twice here: section-filtered, then all sections, because the fake is empty)
    assert any("Drug: warfarin | Section: Dosage" in q for q in seen)
    assert any("Warnings" in q for q in seen)
