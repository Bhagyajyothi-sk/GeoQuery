"""
GeoQueryAI — pytest configuration and shared fixtures.

Ensures mock mode is always enabled during tests so no real
AI/ML models or network calls are required to run the test suite.
"""

import os
import sys
from pathlib import Path

import pytest

# Force mock mode on for all tests — no real Gemini/FAISS/Nominatim/STAC calls.
# Set before any `app.*` import so app.core.config picks these up.
os.environ.setdefault("MOCK_AI", "true")
os.environ.setdefault("MOCK_SEARCH", "true")
os.environ.setdefault("MOCK_GEO", "true")

# Make the `app` package importable no matter which directory pytest is invoked
# from. This file lives at <project_root>/backend/tests/conftest.py, so the
# backend directory (the package root) is one level up.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(scope="session")
def client():
    """
    FastAPI TestClient for the full application.
    Created once per test session for speed.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
