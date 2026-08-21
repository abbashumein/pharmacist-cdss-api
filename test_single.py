# test_single.py
# Retest just ONE query with a longer timeout.
# Run AFTER V3 server is up: uvicorn app.main_agentic:app --port 8001

import requests
import json
import uuid

API_KEY = "local-test-key"
HEADERS = {"Content-Type": "application/json", "x-api-key": API_KEY}
V3_URL = "http://127.0.0.1:8001/chat"

QUERY = "Can metformin interact with lisinopril?"

payload = {
    "session_id": str(uuid.uuid4()),
    "message": QUERY
}

print(f"Query: {QUERY}\n")
try:
    resp = requests.post(V3_URL, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    print(json.dumps(data, indent=2))

    print(f"\n--- Extracted ---")
    print(f"tool_called: {data.get('tool_called')}")
    print(f"fda_evidence_used: {data.get('fda_evidence_used')}")
    print(f"risk_level: {data.get('risk_level')}")
    print(f"retrieval_distance: {data.get('retrieval_distance')}")
    print(f"latency_ms: {data.get('latency_ms')}")
    print(f"\nresponse: {data.get('response')}")

except Exception as e:
    print(f"ERROR: {e}")
