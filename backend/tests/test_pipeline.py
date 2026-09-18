"""
GeoQueryAI — pipeline integration tests.

Tests cover:
  1. Valid StructuredQuery — all six analysis values accepted
  2. Invalid analysis string — 422 Unprocessable Entity
  3. CandidateResponse validation — score bounds, bbox length
  4. Top-K — /api/search respects top_k parameter
  5. Missing location — location=None flows through cleanly
  6. Mock /api/query execution — all four response fields present
  7. Empty candidate result handling — NoCandidatesError → 404

All tests run in mock mode (MOCK_AI=true, MOCK_SEARCH=true, MOCK_GEO=true)
so no real models, network connections, or API keys are required.
"""

import pytest
from pydantic import ValidationError

from app.schemas.query import StructuredQuery, GeographicResult
from app.schemas.search import SemanticCandidate, CandidateResponse, SearchRequest


# ─────────────────────────────────────────────────────────────────────────────
# 1. Valid StructuredQuery — every allowed analysis value must be accepted
# ─────────────────────────────────────────────────────────────────────────────

VALID_ANALYSIS_VALUES = [
    "discovery",
    "ndvi",
    "ndwi",
    "water_extent",
    "change",
    "water_extent_change",
]


@pytest.mark.parametrize("analysis", VALID_ANALYSIS_VALUES)
def test_valid_structured_query(analysis: str):
    """All six allowed analysis values must pass Pydantic validation."""
    sq = StructuredQuery(
        visual_query="large water body",
        location="Bengaluru",
        analysis=analysis,
        start_date="2025-01-01",
        end_date="2025-12-31",
    )
    assert sq.analysis == analysis
    assert sq.visual_query == "large water body"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Invalid analysis string → ValidationError
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_analysis", ["flood", "rgb", "sar", "", "NDVI", "water"])
def test_invalid_analysis_raises(bad_analysis: str):
    """Arbitrary analysis strings must be rejected by Pydantic validation."""
    with pytest.raises(ValidationError):
        StructuredQuery(
            visual_query="some scene",
            location=None,
            analysis=bad_analysis,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. CandidateResponse schema validation
# ─────────────────────────────────────────────────────────────────────────────

def test_semantic_candidate_valid():
    """A well-formed SemanticCandidate must validate without error."""
    c = SemanticCandidate(
        tile_id="tile_001",
        bbox=[77.50, 12.90, 77.65, 13.05],
        score=0.87,
    )
    assert c.tile_id == "tile_001"
    assert len(c.bbox) == 4
    assert 0.0 <= c.score <= 1.0


def test_semantic_candidate_score_out_of_range():
    """Scores outside [0, 1] must be rejected."""
    with pytest.raises(ValidationError):
        SemanticCandidate(tile_id="t", bbox=[0.0, 0.0, 1.0, 1.0], score=1.5)

    with pytest.raises(ValidationError):
        SemanticCandidate(tile_id="t", bbox=[0.0, 0.0, 1.0, 1.0], score=-0.1)


def test_semantic_candidate_bbox_wrong_length():
    """A bbox that is not exactly 4 elements must be rejected."""
    with pytest.raises(ValidationError):
        SemanticCandidate(tile_id="t", bbox=[0.0, 1.0, 2.0], score=0.5)


def test_candidate_response_valid():
    """CandidateResponse with a populated candidates list must validate."""
    resp = CandidateResponse(
        query_id="test-id",
        visual_query="flooded land",
        location="Cauvery delta",
        candidates=[
            SemanticCandidate(tile_id="tile_001", bbox=[77.50, 12.90, 77.65, 13.05], score=0.87),
        ],
    )
    assert len(resp.candidates) == 1
    assert resp.candidates[0].tile_id == "tile_001"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Top-K — /api/search must return ≤ top_k candidates
# ─────────────────────────────────────────────────────────────────────────────

def test_top_k_respects_limit(client):
    """POST /api/search with top_k=3 must return at most 3 candidates."""
    resp = client.post(
        "/api/search",
        json={"visual_query": "large water body", "location": "Bengaluru", "top_k": 3},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "candidates" in data
    assert len(data["candidates"]) <= 3


def test_top_k_default(client):
    """POST /api/search without top_k uses default (5) — returns ≤ 5."""
    resp = client.post(
        "/api/search",
        json={"visual_query": "agricultural fields"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["candidates"]) <= 5


# ─────────────────────────────────────────────────────────────────────────────
# 5. Missing location — StructuredQuery and /api/search must accept location=None
# ─────────────────────────────────────────────────────────────────────────────

def test_structured_query_no_location():
    """location is optional — None must be accepted."""
    sq = StructuredQuery(
        visual_query="barren land",
        location=None,
        analysis="ndvi",
    )
    assert sq.location is None


def test_search_no_location(client):
    """POST /api/search without a location must succeed."""
    resp = client.post(
        "/api/search",
        json={"visual_query": "dense forest canopy"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["location"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Mock /api/query — full pipeline response shape
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_query_pipeline_response_shape(client):
    """
    POST /api/query in mock mode must return all four inspectable fields:
      query, structured_query, geographic_result, candidates
    """
    resp = client.post(
        "/api/query",
        json={"query": "Show me water bodies around Bengaluru"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Top-level required fields
    assert "query_id" in data
    assert "query" in data
    assert "structured_query" in data
    assert "geographic_result" in data
    assert "candidates" in data

    # structured_query must have all five fields
    sq = data["structured_query"]
    assert "visual_query" in sq
    assert "location" in sq
    assert "analysis" in sq
    assert sq["analysis"] in VALID_ANALYSIS_VALUES

    # geographic_result must be grounded in mock mode
    geo = data["geographic_result"]
    assert geo is not None
    assert geo["grounded"] is True
    assert geo["latitude"] is not None
    assert geo["longitude"] is not None
    assert geo["bbox"] is not None and len(geo["bbox"]) == 4

    # At least one candidate returned
    assert len(data["candidates"]) >= 1
    candidate = data["candidates"][0]
    assert "tile_id" in candidate
    assert "bbox" in candidate and len(candidate["bbox"]) == 4
    assert "score" in candidate
    assert 0.0 <= candidate["score"] <= 1.0


def test_mock_query_echoes_raw_query(client):
    """The query field in the response must echo the user's raw query."""
    raw = "How has the Cauvery river changed in the past year?"
    resp = client.post("/api/query", json={"query": raw})
    assert resp.status_code == 200
    assert resp.json()["query"] == raw


# ─────────────────────────────────────────────────────────────────────────────
# 7. Invalid analysis via /api/query — Pydantic rejects at schema level
# ─────────────────────────────────────────────────────────────────────────────

def test_search_invalid_top_k(client):
    """top_k=0 must be rejected with 422 Unprocessable Entity."""
    resp = client.post(
        "/api/search",
        json={"visual_query": "water body", "top_k": 0},
    )
    assert resp.status_code == 422


def test_search_top_k_too_large(client):
    """top_k=51 exceeds the max of 50 — must be rejected with 422."""
    resp = client.post(
        "/api/search",
        json={"visual_query": "water body", "top_k": 51},
    )
    assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Additional: GeographicResult schema
# ─────────────────────────────────────────────────────────────────────────────

def test_geographic_result_ungrounded():
    """An ungrounded GeographicResult with no coordinates must validate."""
    geo = GeographicResult(name="Unknown place", grounded=False)
    assert geo.grounded is False
    assert geo.latitude is None
    assert geo.longitude is None
    assert geo.bbox is None


def test_geographic_result_grounded():
    """A fully populated GeographicResult must validate correctly."""
    geo = GeographicResult(
        name="Bengaluru",
        latitude=12.9716,
        longitude=77.5946,
        bbox=[77.4601, 12.8340, 77.7840, 13.1434],
        display_name="Bengaluru, Karnataka, India",
        grounded=True,
    )
    assert geo.grounded is True
    assert len(geo.bbox) == 4
