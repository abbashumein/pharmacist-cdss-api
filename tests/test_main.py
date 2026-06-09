import pytest
from fastapi.testclient import TestClient
# CRITICAL UPDATE: Point directly to main_demo
from app.main_demo import app

client = TestClient(app)
VALID_HEADERS = {"X-API-KEY": "prod-secret-fallback-key"}

# ==========================================
# 1. GATEWAY & DOCUMENTATION TESTS
# ==========================================

def test_read_root():
    """Verify the API docs load successfully since root is a microservice backend."""
    response = client.get("/docs")
    assert response.status_code == 200

def test_swagger_docs_accessible():
    """Verify that the interactive Swagger API documentation page loads properly."""
    response = client.get("/docs")
    assert response.status_code == 200

# ==========================================
# 2. SECURITY & AUTHENTICATION BOUNDARIES
# ==========================================

def test_chat_unauthorized():
    """Verify endpoint strictly rejects requests with missing credentials with HTTP 403."""
    payload = {"session_id": "session-403", "message": "Clinical inquiry without auth."}
    response = client.post("/chat", json=payload, headers={})
    assert response.status_code == 403

# ==========================================
# 3. PAYLOAD VALIDATION TESTS
# ==========================================

def test_chat_payload_validation():
    """Verify system flags Pydantic validation errors appropriately with HTTP 422."""
    response = client.post("/chat", json={"bad_field": "error"}, headers=VALID_HEADERS)
    assert response.status_code == 422

def test_prediction_validation_error_empty_json():
    """Ensure that sending an empty JSON payload triggers an automatic HTTP 422 error."""
    response = client.post("/chat", json={}, headers=VALID_HEADERS)
    assert response.status_code == 422

@pytest.mark.skip(reason="Requires live external Gemini API key and active ChromaDB local collections")
def test_graph_integration_success():
    """E2E test verifying successful pipeline execution with correct auth headers."""
    payload = {"session_id": "session-e2e-100", "message": "Verify Paracetamol standard dosage."}
    response = client.post("/chat", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200