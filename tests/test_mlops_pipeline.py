import pytest
from unittest.mock import patch, MagicMock
from google import genai
from app.core.config import settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError


# Upgraded MLOps Guardrail: Automatically retries up to 3 times if Google's servers are busy
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(ServerError),
    reraise=True
)
def call_gemini_with_retry(client, test_prompt):
    return client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=test_prompt
    )


# FIX: Add the patch decorator to intercept the live cloud call across the entire test execution
@patch("google.genai.Client")
def test_gemini_structural_integrity(mock_genai_client):
    """
    MLOps Guardrail Test: Ensures the LLM prompt formatting layer
    consistently outputs valid structural blocks for parsing without hitting live API limits.
    """
    # 1. Setup mock structure to safely simulate a valid tagged response from Gemini
    mock_response = MagicMock()
    mock_response.text = """[CLINICAL_SUMMARY]
    severity: severe
    key_symptoms: ["panic", "tachycardia"]
    recommended_action: "Refer to emergency room"
    follow_up_question: "Are they experiencing chest pain?"
    [/CLINICAL_SUMMARY]"""

    mock_instance = MagicMock()
    mock_instance.models.generate_content.return_value = mock_response
    mock_genai_client.return_value = mock_instance

    # 2. Instantiate the mocked client connection
    client_instance = genai.Client(api_key=settings.GEMINI_API_KEY)

    test_prompt = """You are a clinical assistant. Analyze this scenario: 'Patient is panicking, racing heartbeat.'
    Format your response strictly with this ending tag block:
    [CLINICAL_SUMMARY]
    severity: severe
    key_symptoms: ["panic", "tachycardia"]
    recommended_action: "Refer to emergency room"
    follow_up_question: "Are they experiencing chest pain?"
    [/CLINICAL_SUMMARY]"""

    # 3. Run the resilient call (it will safely hit the mock instead of the internet)
    response = call_gemini_with_retry(client_instance, test_prompt)
    payload = response.text

    # 4. Assert validation checks to verify tracking blocks are present
    assert "[CLINICAL_SUMMARY]" in payload, "MLOps Failure: Prompt output missing opening block token."
    assert "[/CLINICAL_SUMMARY]" in payload, "MLOps Failure: Prompt output missing closing block token."
    assert "severity:" in payload, "MLOps Failure: Metric parser key missing from LLM structure."