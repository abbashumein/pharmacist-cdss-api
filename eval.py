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
# Run your container locally first:
#   docker run -p 8000:8000 --env-file .env your-cdss-image
URL = os.environ.get("CDSS_URL", "http://localhost:8000/chat")

# ── EVAL SET: expand this with real drugs from your 19 FDA labels ────────
# Check what's actually in your corpus first:
#   python -c "from your_module import get_collection; \
#              print(set(m['drug_name'] for m in get_collection().get()['metadatas']))"
# Replace/extend below with your ACTUAL ingested drugs.

test_cases = [
    # Category: direct_lookup
    {"q": "What are side effects of aspirin?", "expected": "aspirin", "category": "direct_lookup"},
    {"q": "What are side effects of ibuprofen?", "expected": "ibuprofen", "category": "direct_lookup"},
    {"q": "What are side effects of warfarin?", "expected": "warfarin", "category": "direct_lookup"},
    {"q": "What are side effects of amiodarone?", "expected": "amiodarone", "category": "direct_lookup"},
    {"q": "Aspirin dosage for elderly patient", "expected": "aspirin", "category": "direct_lookup"},

    # Category: interaction (needs correct multi-drug retrieval)
    {"q": "Can a patient take warfarin and aspirin together?", "expected": "warfarin", "category": "interaction"},
    {"q": "Is ibuprofen safe with warfarin?", "expected": "warfarin", "category": "interaction"},
    {"q": "What is the interaction between warfarin and ibuprofen?", "expected": "ibuprofen", "category": "interaction"},
    {"q": "Can patient on amiodarone also take aspirin?", "expected": "amiodarone", "category": "interaction"},

    # Category: contraindication / safety-critical
    {"q": "Warfarin contraindications list", "expected": "warfarin", "category": "contraindication"},
    {"q": "Is amiodarone safe long term?", "expected": "amiodarone", "category": "contraindication"},
    {"q": "Patient taking amiodarone has vision changes, is this expected?", "expected": "amiodarone", "category": "contraindication"},
    {"q": "Patient on aspirin complains of stomach pain, is this a known side effect?", "expected": "aspirin", "category": "contraindication"},

    # Category: ambiguous (tests whether it asks for clarification vs guesses)
    {"q": "Patient reports vision changes, what could be causing this?", "expected": "amiodarone", "category": "ambiguous"},
    {"q": "Is this medication safe to combine with painkillers?", "expected": "", "category": "ambiguous"},

    # Category: out_of_scope (tests triage node's refusal — NOT a retrieval test)
    {"q": "What's the weather like today?", "expected": "REFUSE", "category": "out_of_scope"},
    {"q": "Can you diagnose my chest pain?", "expected": "REFUSE", "category": "out_of_scope"},
    {"q": "What stock should I invest in?", "expected": "REFUSE", "category": "out_of_scope"},

    # Category: new drugs (metformin, lisinopril, atorvastatin)

    {"q": "What are side effects of metformin?", "expected": "metformin", "category": "direct_lookup"},
    {"q": "Metformin contraindications", "expected": "metformin", "category": "contraindication"},
    {"q": "Can patient take metformin with lisinopril?", "expected": "metformin", "category": "interaction"},
    {"q": "What are side effects of lisinopril?", "expected": "lisinopril", "category": "direct_lookup"},
    {"q": "Atorvastatin drug interactions", "expected": "atorvastatin", "category": "interaction"},
    {"q": "Is atorvastatin safe long term?", "expected": "atorvastatin", "category": "contraindication"},
]

results = []
category_stats = {}

for i, case in enumerate(test_cases):
    time.sleep(13)  # 13 seconds = 4-5 requests per minute, safe
    t0 = time.time()
    try:
        res = requests.post(
            URL,
            json={"session_id": f"eval-{i}", "message": case["q"]},
            headers={"X-API-KEY": API_KEY},
            timeout=30,
        )
        data = res.json()
    except Exception as e:
        print(f"Q{i+1}: ❌ ERROR — {e}")
        continue
    latency = time.time() - t0

    response_text = json.dumps(data).lower()
    gateway = data.get("audit_log", {}).get("api_gateway_status", "")

    if case["expected"] == "REFUSE":
        # For out-of-scope: correct behavior is declining, NOT returning RAG content
        hit = gateway != "200_OK_RAG_CONTEXT"
    elif case["expected"] == "":
        # Ambiguous case — just log, don't score pass/fail automatically
        hit = None
    else:
        # THE ACTUAL FIX: check the expected term appears in the response,
        # not just that the API returned 200
        hit = case["expected"].lower() in response_text and gateway == "200_OK_RAG_CONTEXT"

    cat = case["category"]
    category_stats.setdefault(cat, {"hits": 0, "total": 0})
    category_stats[cat]["total"] += 1
    if hit:
        category_stats[cat]["hits"] += 1

    status = "✅ HIT" if hit else ("⚪ MANUAL REVIEW" if hit is None else "❌ MISS")
    print(f"Q{i+1} [{cat}]: {status} ({latency:.1f}s) — {case['q'][:55]}")

    results.append({
        "query": case["q"], "category": cat, "expected": case["expected"],
        "hit": hit, "latency_s": round(latency, 2),
    })

print("\n" + "=" * 50)
print("RESULTS BY CATEGORY")
print("=" * 50)
for cat, stats in category_stats.items():
    pct = (stats["hits"] / stats["total"] * 100) if stats["total"] else 0
    print(f"{cat:20s}: {stats['hits']}/{stats['total']} = {pct:.0f}%")

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved detailed results to eval_results.json")