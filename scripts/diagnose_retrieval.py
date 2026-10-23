"""Check retrieval for every eval query WITHOUT calling Gemini (uses no quota).

Why: when the agent refuses ("I don't have sufficient FDA data"), there are two
possible causes. (1) retrieval found nothing usable (a data/threshold problem),
or (2) retrieval found evidence but the model still refused (a prompt problem).
This script shows which one it is, per query.

Run from the project root:   python scripts/diagnose_retrieval.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEMINI_API_KEY", "not-used-no-gemini-calls-here")

from app import main_agentic  # noqa: E402  (loads MiniLM, CrossEncoder, ChromaDB)

print(f"Collection 'langchain' chunks: {main_agentic.collection.count()}\n")

with open("evaluation/test_cases.json") as f:
    cases = json.load(f)

print(f"{'Q':>3} {'usable':>6} {'best_dist':>9} {'chunks':>6}  drugs in the evidence chunks                | query")
print("-" * 120)
for c in cases:
    if not c.get("should_call_tool"):
        continue
    r = main_agentic.check_fda_database(c["query"])
    names = [ch.split("|")[0].replace("Drug:", "").strip()[:14] for ch in r["chunks"]]
    top = ", ".join(names) if names else "-"
    dist = f"{r['retrieval_distance']:.3f}" if r["retrieval_distance"] is not None else "None"
    print(f"{c['id']:>3} {str(r['has_usable_evidence']):>6} {dist:>9} {len(r['chunks']):>6}  {top:<44} | {c['query'][:45]}")

print("\nusable=False  -> retrieval problem (data, threshold 0.95, or query).")
print("usable=True but the eval still showed a refusal -> prompt/model problem.")
