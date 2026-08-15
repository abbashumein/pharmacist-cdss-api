import json
import time
import requests

API_KEY = "local-test-key"
BASE_URL = "http://127.0.0.1:8001/chat"

with open("evaluation/test_cases.json") as f:
    test_cases = json.load(f)

results = []
category_stats = {}

print("=" * 60)
print("PHARMACIST CDSS V3 — AGENT EVALUATION")
print("=" * 60)

for case in test_cases:
    time.sleep(15)  # Respect Gemini rate limit

    try:
        res = requests.post(
            BASE_URL,
            json={"session_id": f"eval-{case['id']}", "message": case["query"]},
            headers={"X-API-KEY": API_KEY},
            timeout=30
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

    # Evidence grounding — did response mention expected drug?
    if case["expected_drug"]:
        grounded = case["expected_drug"].lower() in response
    else:
        grounded = not fda_used  # out-of-scope should not use FDA

    # Refusal check for out-of-scope
    refused_correctly = True
    if case["type"] == "out_of_scope":
        refused_correctly = not tool_called and not fda_used

    cat = case["type"]
    category_stats.setdefault(cat, {
        "routing_correct": 0,
        "grounded": 0,
        "total": 0
    })
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
        "latency_ms": latency
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
    print(f"{cat:20s}: routing {r}/{t} = {r/t*100:.0f}% | grounded {g}/{t} = {g/t*100:.0f}%")

print(f"\n{'OVERALL':20s}: routing {total_routing}/{total} = {total_routing/total*100:.0f}% | grounded {total_grounded}/{total} = {total_grounded/total*100:.0f}%")

with open("evaluation/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to evaluation/results.json")