# tests/conftest.py
import pytest
from unittest.mock import patch, MagicMock

# This block executes BEFORE any app code imports or compiles
gemini_patcher = patch("google.genai.Client")
mocked_client = gemini_patcher.start()

# Configure global mock behavior for everything in the test run
mock_response = MagicMock()
mock_response.text = "Clinical Guidance: Evaluate patient for severe clinical depression. Recommended Action: Safe medical reference."
mocked_client.return_value.models.generate_content.return_value = mock_response

@pytest.fixture(scope="session", autouse=True)
def stop_patcher_after_session():
    """Stops the global mock once the entire test suite finishes."""
    yield
    gemini_patcher.stop()