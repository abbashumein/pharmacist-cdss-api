"""Retrieval guards found by the 50-question eval (no network, fake DB and reranker).

Found on the real 11,898-chunk database: vague questions ("is this painkiller safe with
alcohol?") and made-up drugs ("zorbaflex") matched unrelated chunks at distance 0.63-0.89,
which is under the 0.95 cutoff, so junk counted as usable evidence.
"""
import os
from types import SimpleNamespace

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from app import main_agentic  # noqa: E402


class FakeDB:
    def __init__(self, docs):
        self.docs = docs          # list of (chunk text, distance)
        self.calls = []

    def query(self, query_texts, n_results, include, where=None):
        self.calls.append({"q": query_texts[0], "where": where})
        pool = self.docs
        if where:
            allowed = set(where["section"]["$in"])
            pool = [(d, x) for d, x in pool if d.split("Section: ")[1].split(" |")[0].lower() in allowed]
        pool = pool[:n_results]
        return {"documents": [[d for d, _ in pool]], "distances": [[x for _, x in pool]]}


def install(monkeypatch, docs):
    db = FakeDB(docs)
    monkeypatch.setattr(main_agentic, "collection", db)
    monkeypatch.setattr(main_agentic, "reranker", SimpleNamespace(predict=lambda pairs: [1.0] * len(pairs)))
    return db


JUNK = [("Drug: ZOLEDRONIC ACID | Section: Side Effects | nausea", 0.81),
        ("Drug: ALCOHOL (Hand Sanitizer) | Section: Warnings | for external use", 0.83),
        ("Drug: LORAZEPAM | Section: Dosage | 2 mg", 0.68)]


def test_made_up_drug_gets_no_usable_evidence(monkeypatch):
    install(monkeypatch, JUNK)
    res = main_agentic.check_fda_database("What are the side effects of zorbaflex?")
    assert res["has_usable_evidence"] is False and res["chunks"] == []


def test_vague_question_with_no_drug_name_gets_no_usable_evidence(monkeypatch):
    install(monkeypatch, JUNK)
    for q in ["Is this painkiller safe with alcohol?", "What is the dose?",
              "Is the blood thinner safe with my heart pill?", "Which antibiotic is best for me?"]:
        assert main_agentic.check_fda_database(q)["has_usable_evidence"] is False, q


def test_drug_outside_the_core_list_still_works_if_it_is_in_the_database(monkeypatch):
    install(monkeypatch, JUNK + [("Drug: CARVEDILOL | Section: Side Effects | dizziness fatigue", 0.4)])
    res = main_agentic.check_fda_database("What are the side effects of carvedilol?")
    assert res["has_usable_evidence"] and "CARVEDILOL" in res["chunks"][0]


def test_named_drug_never_gets_another_drugs_chunks(monkeypatch):
    install(monkeypatch, [("Drug: METFORMIN | Section: Pregnancy | text", 0.3),
                          ("Drug: AMLODIPINE BESYLATE | Section: Side Effects | edema", 0.5)])
    res = main_agentic.check_fda_database("Is amlodipine safe in pregnancy?")
    assert res["chunks"] and all("AMLODIPINE" in c for c in res["chunks"])


def test_question_about_a_section_searches_only_that_section_first(monkeypatch):
    docs = [(f"Drug: WARFARIN | Section: Interactions | part {i}", 0.4) for i in range(4)] + \
           [("Drug: WARFARIN | Section: Dosage | inr", 0.3)]
    db = install(monkeypatch, docs)
    res = main_agentic.check_fda_database("Can warfarin and aspirin be taken together?")
    assert db.calls[0]["where"] == {"section": {"$in": ["interactions"]}}
    assert all("Section: Interactions" in c for c in res["chunks"])


def test_falls_back_to_all_sections_when_the_section_is_missing(monkeypatch):
    db = install(monkeypatch, [("Drug: AMLODIPINE | Section: Side Effects | edema", 0.5),
                               ("Drug: AMLODIPINE | Section: Dosage | 5 mg", 0.6),
                               ("Drug: AMLODIPINE | Section: Indications | hypertension", 0.6)])
    res = main_agentic.check_fda_database("Is amlodipine safe in pregnancy?")
    assert len(db.calls) == 2 and db.calls[1]["where"] is None   # filtered search found <3, so it widened
    assert res["has_usable_evidence"]


def test_intent_sections_cover_the_questions_that_failed():
    f = main_agentic._intent_sections
    assert "Pregnancy" in f("is amlodipine safe in pregnancy?")
    assert "Stop Use" in f("when should i stop using aspirin?")
    assert "Indications" in f("what is ibuprofen used for?")
    assert "Boxed Warning" in f("what is the boxed warning for warfarin?")


def test_duplicate_label_text_does_not_fill_every_slot(monkeypatch):
    same = "5 WARNINGS Abrupt cessation may exacerbate myocardial ischemia. Heart failure may occur."
    docs = [(f"Drug: METOPROLOL TARTRATE ({n}) | Section: Warnings | {same}", 0.5 + i / 100)
            for i, n in enumerate(["A", "B", "C", "D"])]
    docs.append(("Drug: METOPROLOL | Section: Warnings | Bronchospastic disease: avoid beta-blockers.", 0.6))
    install(monkeypatch, docs)
    res = main_agentic.check_fda_database("What are the warnings for metoprolol?")
    assert sum("Abrupt cessation" in c for c in res["chunks"]) == 1
    assert any("Bronchospastic" in c for c in res["chunks"])


def test_interaction_search_also_mentions_the_other_drug(monkeypatch):
    db = install(monkeypatch, [("Drug: AMIODARONE | Section: Interactions | warfarin potentiation", 0.4),
                               ("Drug: WARFARIN | Section: Interactions | amiodarone", 0.4)] * 2)
    main_agentic.check_fda_database("I take warfarin and amiodarone together. What should I watch for?")
    queries = [c["q"] for c in db.calls]
    assert any(q.startswith("Drug: warfarin") and "amiodarone" in q for q in queries)
    assert any(q.startswith("Drug: amiodarone") and "warfarin" in q for q in queries)
