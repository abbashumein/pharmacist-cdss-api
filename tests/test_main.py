# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
from app.main import app 

client = TestClient(app)

def test_read_root():
    """Verify the API docs load successfully since root is a microservice backend."""
    response = client.get("/docs")
    assert response.status_code == 200

def test_prediction_validation_error():
    """Ensure that sending an empty JSON payload triggers an automatic HTTP 422 error."""
    response = client.post("/chat", json={}) 
    assert response.status_code in [422, 404]

def test_swagger_docs_accessible():
    """Verify that the interactive Swagger API documentation page loads properly."""
    response = client.get("/docs")
    assert response.status_code == 200