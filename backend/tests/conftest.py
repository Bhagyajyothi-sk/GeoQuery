"""
GeoQueryAI — pytest configuration and shared fixtures.

Ensures mock mode is always enabled during tests so no real
AI/ML models or network calls are required to run the test suite.
"""

import os
import pytest
from fastapi.testclient import TestClient


# Force mock mode on for all tests — no real Gemini/FAISS/Nominatim calls.
os.environ.setdefault("MOCK_AI", "true")
os.environ.setdefault("MOCK_SEARCH", "true")
os.environ.setdefault("MOCK_GEO", "true")


@pytest.fixture(scope="session")
def client():
    """
    FastAPI TestClient for the full application.
    Created once per test session for speed.
    """
    from app.main import app
    with TestClient(app) as c:
        yield c
