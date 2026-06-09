import os
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def validate_api_key(api_key: str = Security(api_key_header)):
    """
    Validates incoming API headers.
    Strictly enforces 403 Forbidden on missing or non-matching string signatures.
    """
    expected_key = os.getenv("CDSS_API_KEY", "prod-secret-fallback-key")

    # CRITICAL: If header is completely absent or doesn't match our secret value, raise 403
    if not api_key or api_key.strip() != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing X-API-KEY header credential."
        )
    return api_key