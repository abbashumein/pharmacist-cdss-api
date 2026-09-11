import requests
import os
import time
import json

# SECURITY FIX: never hardcode keys. Set this in your shell:
#   export CDSS_API_KEY="your-real-key"
API_KEY = os.environ.get("CDSS_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set CDSS_API_KEY environment variable before running.")

# Point this at your LOCAL instance since Azure free tier has expired.
URL = os.environ.get("CDSS_URL", "http://localhost:8000/chat")

# ── EVAL SET ───────────────────────────────────────────────────────────
# Two blocks:
#  1. ORIGINAL cases — direct drug-name mentions (easy mode, kept as a
#     baseline so you can see the score didn't regress).
#  2. NEW hard cases — paraphrased / indirect drug references, no exact
#     keyword match. These are the realistic queries a real user would
#     actually type, and the ones your keyword-based routing/retrieval
#     is most likely to miss.

test_cases = [
    # ── ORIGINAL — direct_lookup ─────────────────────────────────────
    {"q": "What are side effects of aspirin?", "expected": "aspirin", "category": "direct_lookup"},
    {"q": "What are side effects of warfarin?", "expected": "warfarin", "category": "direct_lookup"},
    {"q": "What are side effects of metformin?", "expected": "metformin", "category": "direct_lookup"},

    # ── ORIGINAL — interaction ───────────────────────────────────────
    {"q": "Can a patient take warfarin and aspirin together?", "expected": "warfarin", "category": "interaction"},
    {"q": "Can patient take metformin with lisinopril?", "expected": "metformin", "category": "interaction"},

    # ── ORIGINAL — contraindication ──────────────────────────────────
    {"q": "Warfarin contraindications list", "expected": "warfarin", "category": "contraindication"},
    {"q": "Metformin contraindications", "expected": "metformin", "category": "contraindication"},

    # ── ORIGINAL — out_of_scope ───────────────────────────────────────
    {"q": "What's the weather like today?", "expected": "REFUSE", "category": "out_of_scope"},
    {"q": "Can you diagnose my chest pain?", "expected": "REFUSE", "category": "out_of_scope"},

    # ── NEW — paraphrased (drug named, but not in "textbook" phrasing) ─
    {"q": "Can I take this with alcohol if I'm on warfarin?", "expected": "warfarin", "category": "paraphrased"},
    {"q": "Is it safe to get pregnant while taking metformin?", "expected": "metformin", "category": "paraphrased"},
    {"q": "My doctor put me on aspirin, what should I watch out for?", "expected": "aspirin", "category": "paraphrased"},
    {"q": "Does lisinopril mess with your kidneys?", "expected": "lisinopril", "category": "paraphrased"},

    # ── NEW — indirect_reference (drug described, not named) ──────────
    {"q": "My grandma takes a blood thinner, can she also have aspirin?", "expected": "warfarin", "category": "indirect_reference"},
    {"q": "I'm on a cholesterol pill, any interactions with grapefruit?", "expected": "atorvastatin", "category": "indirect_reference"},
    {"q": "My blood pressure medicine, is it okay with ibuprofen?", "expected": "lisinopril", "category": "indirect_reference"},

    # ── NEW — no_drug_name (system should still ask/route sensibly) ───
    {"q": "Is it okay to combine my heart medication with a painkiller?", "expected": "", "category": "no_drug_name"},
    {"q": "Can I mix my two prescriptions safely?", "expected": "", "category": "no_drug_name"},
]

results = []
category_stats = {}

import sys

# Run a slice of test_cases to stay within daily Gemini quota:
#   python eval_v2_hard.py 0 9    -> first 10 queries (indices 0-9)
#   python eval_v2_hard.py 9 18   -> next 9 queries (indices 9-17)
#   python eval_v2_hard.py        -> all queries (default, may hit quota)
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end = int(sys.argv[2]) + 1 if len(sys.argv) > 2 else len(test_cases)
test_cases = test_cases[start:end]
print(f"Running queries {start} to {end - 1} ({len(test_cases)} total)\n")

for i, case in enumerate(test_cases):
    time.sleep(13)  # 13 seconds = 4-5 requests per minute, safe
    t0 = time.time()
    try:
        res = requests.post(
            URL,
            json={"session_id": f"eval-hard-{i}", "message": case["q"]},
            headers={"X-API-KEY": API_KEY},
            timeout=30,
        )
        data = res.json()
    except Exception as e:
        print(f"Q{i+1}: \u274c ERROR — {e}")
        continue
    latency = time.time() - t0

    # Score against the REAL answer only, not the whole response JSON.
    # Checking the full dump was a false-positive bug: evidence_sources
    # always contains the drug name (from retrieval), so a query would
    # score a "hit" even when generation itself failed (e.g. Gemini
    # quota exhausted, fallback "temporarily unable to respond" message).
    clinical_guidance_raw = data.get("clinical_guidance", "")
    generation_failed = "temporarily unable to respond" in clinical_guidance_raw.lower()
    try:
        parsed = json.loads(clinical_guidance_raw)
        answer_text = f"{parsed.get('answer', '')} {parsed.get('warnings', '')}".lower()
    except (json.JSONDecodeError, TypeError):
        answer_text = clinical_guidance_raw.lower()
    gateway = data.get("audit_log", {}).get("api_gateway_status", "")

    if case["expected"] == "REFUSE":
        hit = gateway != "200_OK_RAG_CONTEXT"
    elif case["expected"] == "":
        # No fixed expected drug — just log it, review manually whether
        # the system asked a clarifying question or gave a sensible reply.
        hit = None
    else:
        hit = (
            not generation_failed
            and case["expected"].lower() in answer_text
            and gateway == "200_OK_RAG_CONTEXT"
        )

    cat = case["category"]
    category_stats.setdefault(cat, {"hits": 0, "total": 0, "manual": 0})
    category_stats[cat]["total"] += 1
    if hit is True:
        category_stats[cat]["hits"] += 1
    elif hit is None:
        category_stats[cat]["manual"] += 1

    status = "\u2705 HIT" if hit else ("\u26aa MANUAL REVIEW" if hit is None else "\u274c MISS")
    print(f"Q{i+1} [{cat}]: {status} ({latency:.1f}s) — {case['q'][:60]}")

    results.append({
        "query": case["q"], "category": cat, "expected": case["expected"],
        "hit": hit, "latency_s": round(latency, 2), "raw_response": data,
    })

print("\n" + "=" * 55)
print("RESULTS BY CATEGORY")
print("=" * 55)
for cat, stats in category_stats.items():
    scored = stats["total"] - stats["manual"]
    pct = (stats["hits"] / scored * 100) if scored else 0
    extra = f" (+{stats['manual']} manual review)" if stats["manual"] else ""
    print(f"{cat:20s}: {stats['hits']}/{scored} = {pct:.0f}%{extra}")

with open("eval_v2_hard_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved detailed results to eval_v2_hard_results.json")