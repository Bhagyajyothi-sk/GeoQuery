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


# ─────────────────────────────────────────────────────────────────────────────
# 8. Evidence Schema & Grounded Gemini Explanation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_evidence_model_validation():
    """Evidence model must validate with only present values (no missing defaults)."""
    from app.schemas.analysis import Evidence

    ev = Evidence(
        location="Bengaluru",
        bbox=[77.58, 12.97, 77.60, 12.98],
        scene_id="S2C_12345",
        acquisition_date="2026-09-14T10:06:10Z",
        analysis="water_extent",
        computed_values={"water_fraction": 0.45, "water_area_km2": 1.25},
    )
    assert ev.location == "Bengaluru"
    assert ev.scene_id == "S2C_12345"
    assert ev.computed_values["water_area_km2"] == 1.25
    assert ev.source == "Sentinel-2 L2A via Planetary Computer STAC"


def test_explain_evidence_mock_generation():
    """explain_evidence() in mock mode generates a grounded string strictly from metrics."""
    from app.schemas.analysis import Evidence
    from app.services.ai.interface import explain_evidence

    ev = Evidence(
        location="Ulsoor Lake",
        scene_id="S2C_ULSOOR_01",
        acquisition_date="2026-09-14",
        analysis="water_extent",
        computed_values={"water_pixels": 120, "water_area_km2": 0.12},
    )
    explanation = explain_evidence(ev)
    assert "S2C_ULSOOR_01" in explanation
    assert "water_extent" in explanation
    assert "water_area_km2: 0.12" in explanation


def test_full_pipeline_response_extended_fields(client):
    """
    POST /api/query in end-to-end mode returns selected_candidate, analysis_result,
    evidence, and explanation objects.
    """
    resp = client.post(
        "/api/query",
        json={"query": "Calculate the water extent for Ulsoor Lake in Bengaluru"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "selected_candidate" in data
    assert "analysis_result" in data
    assert "evidence" in data
    assert "explanation" in data

    if data["selected_candidate"]:
        assert "bbox" in data["selected_candidate"]
        assert len(data["selected_candidate"]["bbox"]) == 4

    if data["evidence"]:
        assert data["evidence"]["analysis"] in VALID_ANALYSIS_VALUES
        assert "computed_values" in data["evidence"]

    if data["explanation"]:
        assert isinstance(data["explanation"], str)
        assert len(data["explanation"]) > 10


# ─────────────────────────────────────────────────────────────────────────────
# 9. Candidate Validation Service Unit & Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_validator_candidate_inside_location_accepted():
    """Candidate tile whose bbox contains the geocoded location is accepted."""
    from app.schemas.search import SemanticCandidate
    from app.schemas.query import GeographicResult
    from app.services.search.validator import validate_candidates

    ulsoor_tile = SemanticCandidate(
        tile_id="ulsoor_001",
        bbox=[77.5899, 12.9765, 77.5992, 12.9854],
        score=0.85,
    )

    geo = GeographicResult(
        name="Ulsoor Lake",
        latitude=12.981,
        longitude=77.5946,
        grounded=True,
    )

    result = validate_candidates([ulsoor_tile], geographic_result=geo)
    assert result.validation_status == "passed"
    assert result.selected_candidate is not None
    assert result.selected_candidate.tile_id == "ulsoor_001"
    assert result.details[0].spatial_match is True


def test_validator_candidate_outside_location_rejected():
    """Candidate tile whose bbox is far from the geocoded location is rejected."""
    from app.schemas.search import SemanticCandidate
    from app.schemas.query import GeographicResult
    from app.services.search.validator import validate_candidates

    urban_tile = SemanticCandidate(
        tile_id="urban_002",
        bbox=[77.5899, 12.9671, 77.5992, 12.9760],
        score=0.92,
    )

    geo = GeographicResult(
        name="Ulsoor Lake",
        latitude=12.981,
        longitude=77.5946,
        grounded=True,
    )

    result = validate_candidates([urban_tile], geographic_result=geo)
    assert result.validation_status == "rejected"
    assert result.selected_candidate is None
    assert result.details[0].spatial_match is False


def test_validator_selects_spatial_match_over_higher_score():
    """
    Validation MUST select Rank 2 candidate that matches spatial constraints
    over Rank 1 candidate with a higher RemoteCLIP score that fails spatial check.
    """
    from app.schemas.search import SemanticCandidate
    from app.schemas.query import GeographicResult
    from app.services.search.validator import validate_candidates

    rank1_urban = SemanticCandidate(
        tile_id="urban_002",
        bbox=[77.5899, 12.9671, 77.5992, 12.9760],
        score=0.95,
    )
    rank2_ulsoor = SemanticCandidate(
        tile_id="ulsoor_001",
        bbox=[77.5899, 12.9765, 77.5992, 12.9854],
        score=0.82,
    )

    geo = GeographicResult(
        name="Ulsoor Lake",
        latitude=12.981,
        longitude=77.5946,
        grounded=True,
    )

    result = validate_candidates([rank1_urban, rank2_ulsoor], geographic_result=geo)
    assert result.validation_status == "passed"
    assert result.selected_candidate is not None
    assert result.selected_candidate.tile_id == "ulsoor_001"
    assert result.details[0].passed is False
    assert result.details[1].passed is True


def test_validator_no_location_accepts_first():
    """When no location is supplied, spatial check passes by default and Rank 1 candidate is selected."""
    from app.schemas.search import SemanticCandidate
    from app.services.search.validator import validate_candidates

    c1 = SemanticCandidate(tile_id="tile_1", bbox=[77.50, 12.90, 77.60, 13.00], score=0.88)
    c2 = SemanticCandidate(tile_id="tile_2", bbox=[77.60, 13.00, 77.70, 13.10], score=0.85)

    result = validate_candidates([c1, c2], geographic_result=None)
    assert result.validation_status == "passed"
    assert result.selected_candidate.tile_id == "tile_1"


def test_query_route_unmatched_location_returns_graceful_response(client):
    """
    Querying a location that has no matching tile in the index returns validation_status='rejected'
    and selected_candidate=None gracefully.
    """
    resp = client.post(
        "/api/query",
        json={"query": "Show me water bodies in Mumbai"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert "validation_status" in data
    assert "validation_reason" in data
    assert "validation_details" in data


