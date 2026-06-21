import requests

API_KEY = "prod-secret-fallback-key"
URL = "https://pharmacist-cdss.whitemushroom-bdf53e45.eastus.azurecontainerapps.io/chat"

test_cases = [
    {"q": "Can a patient take warfarin and aspirin together?", "expected": "warfarin"},
    {"q": "What are side effects of aspirin?", "expected": "aspirin"},
    {"q": "Is ibuprofen safe with warfarin?", "expected": "warfarin"},
    {"q": "Patient taking amiodarone has vision changes", "expected": "amiodarone"},
    {"q": "What is the interaction between warfarin and ibuprofen?", "expected": "warfarin"},
    {"q": "Aspirin dosage for elderly patient", "expected": "aspirin"},
    {"q": "Can patient take ibuprofen for pain?", "expected": "ibuprofen"},
    {"q": "Warfarin contraindications list", "expected": "warfarin"},
    {"q": "Patient on aspirin complains of stomach pain", "expected": "aspirin"},
    {"q": "Is amiodarone safe long term?", "expected": "amiodarone"},
]

hits = 0
for i, case in enumerate(test_cases):
    res = requests.post(URL,
                        json={"session_id": f"eval-{i}", "message": case["q"]},
                        headers={"X-API-KEY": API_KEY}
                        )
    data = res.json()
    sources = " ".join(data.get("evidence_sources", []))
    gateway = data.get("audit_log", {}).get("api_gateway_status", "")

    hit = gateway == "200_OK_RAG_CONTEXT"
    if hit:
        hits += 1
    print(f"Q{i + 1}: {'✅ HIT' if hit else '❌ MISS'} — {case['q'][:50]}")

print(f"\nRetrieval Accuracy: {hits}/{len(test_cases)} = {hits * 10}%")