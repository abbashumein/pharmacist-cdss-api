# test_verification.py
# Run this AFTER starting the V3 server:
#   uvicorn app.main_agentic:app --port 8001

import requests
import time
import json
import uuid

API_KEY = "local-test-key"
HEADERS = {"Content-Type": "application/json", "x-api-key": API_KEY}
V3_URL = "http://127.0.0.1:8001/chat"

test_cases = [
    {"label": "Warfarin + Ibuprofen (V3)", "message": "Can warfarin interact with ibuprofen?"},
    {"label": "Metformin + Lisinopril (V3)", "message": "Can metformin interact with lisinopril?"},
    {"label": "Non-clinical / poem guardrail (V3)", "message": "Write a poem about the ocean"},
]

def run_test(case):
    print(f"\n{'='*60}\n{case['label']}\nQuery: {case['message']}\n{'='*60}")
    payload = {
        "session_id": str(uuid.uuid4()),
        "message": case["message"]
    }
    try:
        resp = requests.post(V3_URL, headers=HEADERS, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        print(json.dumps(data, indent=2)[:2000])

        print(f"\n--- Extracted ---")
        print(f"tool_called: {data.get('tool_called')}")
        print(f"fda_evidence_used: {data.get('fda_evidence_used')}")
        print(f"risk_level: {data.get('risk_level')}")
        print(f"retrieval_distance: {data.get('retrieval_distance')}")
        print(f"latency_ms: {data.get('latency_ms')}")
        print(f"response: {data.get('response')}")

    except Exception as e:
        print(f"ERROR: {e}")

for i, case in enumerate(test_cases):
    run_test(case)
    if i < len(test_cases) - 1:
        print("\nSleeping 20s to respect Gemini free-tier quota...")
        time.sleep(20)

print("\n\nDone. Manually check:")
print("1. Warfarin+ibuprofen response mentions bleeding risk / interaction — not 'insufficient data'")
print("2. Metformin+lisinopril response is coherent and grounded")
print("3. Poem response is an actual poem, NOT the FDA refusal message, and tool_called should be False")
