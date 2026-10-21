import json
import os
import sys
import time
import requests

# Set CDSS_API_KEY in your shell (same value the server uses). Never hardcode it.
API_KEY = os.environ.get("CDSS_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set the CDSS_API_KEY environment variable before running.")
BASE_URL = os.environ.get("CDSS_URL", "http://127.0.0.1:8001/chat")

REFUSAL_TEXT = "don't have sufficient fda data"
FALLBACK_TEXT = "temporarily unable"

with open("evaluation/test_cases.json") as f:
    test_cases = json.load(f)

# Run a slice to stay within daily Gemini quota:
#   python evaluate_agent.py 0 9    -> first 10 queries (indices 0-9)
#   python evaluate_agent.py 10 19  -> last 10 queries (indices 10-19)
#   python evaluate_agent.py        -> all queries (default, may hit quota)
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end = int(sys.argv[2]) + 1 if len(sys.argv) > 2 else len(test_cases)
test_cases = test_cases[start:end]
print(f"Running queries {start} to {end - 1} ({len(test_cases)} total)\n")

results = []
category_stats = {}

print("=" * 60)
print("PHARMACIST CDSS V3 — AGENT EVALUATION")
print("=" * 60)

for case in test_cases:
    time.sleep(20)  # Respect Gemini rate limit

    try:
        res = requests.post(
            BASE_URL,
            json={"session_id": f"eval-{case['id']}", "message": case["query"]},
            headers={"X-API-KEY": API_KEY},
            timeout=180  # server may retry on 429 for up to ~2 min
        )
        data = res.json()
    except Exception as e:
        print(f"Q{case['id']}: ERROR — {e}")
        continue

    tool_called = data.get("tool_called", False)
    fda_used = data.get("fda_evidence_used", False)
    response = data.get("response", "").lower()
    latency = data.get("latency_ms", 0)

    # Tool routing accuracy
    routing_correct = tool_called == case["should_call_tool"]

    is_refusal = REFUSAL_TEXT in response
    is_fallback = FALLBACK_TEXT in response

    # Evidence grounding (STRICTER than before).
    # Old rule: the expected drug name appears in the answer. A made-up
    # answer can pass that. New rule for drug questions - ALL must be true:
    #   - the FDA tool ran and found usable evidence
    #   - the answer names the expected drug
    #   - the answer is not a refusal and not a quota/error fallback
    # Refusals are counted separately: a refusal is safe, but it is not an
    # answer, so it must not be counted as "grounded".
    # Still NOT checked: that every claim in the answer appears in the
    # retrieved chunk (claim-level check). Do not describe this as that.
    if case.get("expected_drug"):
        grounded = (
            tool_called
            and fda_used
            and case["expected_drug"].lower() in response
            and not is_refusal
            and not is_fallback
        )
    elif case["type"] == "out_of_scope":
        grounded = not fda_used and not tool_called  # out-of-scope: tool must not run
    else:
        # No expected drug (e.g. an ambiguous query): only require a real reply.
        grounded = not is_fallback

    # Refusal check for out-of-scope
    refused_correctly = True
    if case["type"] == "out_of_scope":
        refused_correctly = not tool_called and not fda_used

    cat = case["type"]
    category_stats.setdefault(cat, {
        "routing_correct": 0,
        "grounded": 0,
        "refusals": 0,
        "total": 0
    })
    if is_refusal:
        category_stats[cat]["refusals"] += 1
    category_stats[cat]["total"] += 1
    if routing_correct:
        category_stats[cat]["routing_correct"] += 1
    if grounded:
        category_stats[cat]["grounded"] += 1

    status = "✅" if routing_correct and grounded else "❌"
    print(f"Q{case['id']} [{cat}]: {status} tool={tool_called} grounded={grounded} ({latency}ms) — {case['query'][:50]}")

    results.append({
        "id": case["id"],
        "query": case["query"],
        "type": cat,
        "tool_called": tool_called,
        "routing_correct": routing_correct,
        "grounded": grounded,
        "refusal": is_refusal,
        "fallback": is_fallback,
        "latency_ms": latency,
        "response": data.get("response", "")
    })

print("\n" + "=" * 60)
print("RESULTS BY CATEGORY")
print("=" * 60)

total_routing = 0
total_grounded = 0
total = 0

for cat, stats in category_stats.items():
    r = stats["routing_correct"]
    g = stats["grounded"]
    t = stats["total"]
    total_routing += r
    total_grounded += g
    total += t
    print(f"{cat:20s}: routing {r}/{t} = {r/t*100:.0f}% | grounded {g}/{t} = {g/t*100:.0f}% | refusals {stats['refusals']}")

print(f"\n{'OVERALL':20s}: routing {total_routing}/{total} = {total_routing/total*100:.0f}% | grounded {total_grounded}/{total} = {total_grounded/total*100:.0f}%")

with open("evaluation/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to evaluation/results.json")