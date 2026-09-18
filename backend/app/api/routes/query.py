"""
GeoQueryAI — POST /api/query

Main orchestration endpoint. Accepts a natural-language query and runs
the GeoQueryAI pipeline up to (and including) semantic retrieval.

Pipeline (this phase):

  raw_query
    ↓  [QUERY_RECEIVED]
  parse_query()                       → StructuredQuery
    ↓  [QUERY_PARSED]
  geocode()          (if location)    → GeographicResult
    ↓  [LOCATION_GROUNDED]
  semantic_search()                   → CandidateResponse
    ↓  [CANDIDATES_RETRIEVED]
  return QueryPipelineResponse

STOP HERE — geospatial analysis (STAC + COG) is NOT run in this phase.
Each intermediate result is returned as a separate inspectable field
so AI/ML components can be debugged in isolation.

Next phase: wire the top candidate bbox into analyze() after candidates
            are validated against STAC availability.
"""

import uuid
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.query import QueryRequest, QueryPipelineResponse
from app.services.ai.interface import parse_query
from app.services.search.interface import semantic_search
from app.services.geo.geocoding_interface import geocode
from app.core.errors import (
    AIServiceError,
    SearchServiceError,
    NoCandidatesError,
)

logger = logging.getLogger("geoquery.query")
router = APIRouter()


@router.post(
    "/query",
    response_model=QueryPipelineResponse,
    summary="Natural-language GeoQueryAI pipeline (up to semantic retrieval)",
    description=(
        "Accepts a free-text natural-language query and runs the GeoQueryAI pipeline: "
        "AI query parsing → geographic grounding → semantic satellite-image retrieval. "
        "Returns all intermediate stage outputs for inspection. "
        "**Geospatial analysis (STAC/COG) is not run in this phase.**"
    ),
    responses={
        200: {"description": "Pipeline completed successfully — candidates returned."},
        404: {"description": "No candidate tiles found for the given query."},
        503: {"description": "An upstream service (AI or search) is unavailable."},
    },
)
async def run_query(body: QueryRequest) -> QueryPipelineResponse:
    """
    GeoQueryAI orchestration endpoint — Phase 2 (up to semantic retrieval).

    Stages:
      1. parse_query()    — AI service (Gemini / mock)
      2. geocode()        — geographic grounding (Nominatim / mock)
      3. semantic_search() — RemoteCLIP + FAISS (mock)

    HTTP error codes:
      503 — AI or search service failure
      404 — No candidate tiles found
    """
    query_id = str(uuid.uuid4())

    # ── Stage 0: Receive query ────────────────────────────────────────────────
    logger.info("QUERY_RECEIVED query_id=%s raw_query=%r", query_id, body.query)

    # ── Stage 1: AI Query Parsing ─────────────────────────────────────────────
    structured = parse_query(body.query)
    logger.info(
        "QUERY_PARSED query_id=%s visual_query=%r location=%r analysis=%s start=%s end=%s",
        query_id,
        structured.visual_query,
        structured.location,
        structured.analysis,
        structured.start_date,
        structured.end_date,
    )

    # ── Stage 2: Geographic Grounding ─────────────────────────────────────────
    # Only ground if the AI parser returned a location.
    # Coordinates from GeographicResult are NEVER used as satellite tile coords —
    # they serve as a hint for semantic search filtering only.
    geographic_result = None
    if structured.location:
        geographic_result = geocode(structured.location)
        logger.info(
            "LOCATION_GROUNDED query_id=%s name=%r lat=%s lon=%s grounded=%s",
            query_id,
            geographic_result.name,
            f"{geographic_result.latitude:.4f}" if geographic_result.latitude is not None else "None",
            f"{geographic_result.longitude:.4f}" if geographic_result.longitude is not None else "None",
            geographic_result.grounded,
        )
    else:
        logger.info("LOCATION_GROUNDED query_id=%s no location provided — skipping geocoding", query_id)

    # ── Stage 3: Semantic Search ───────────────────────────────────────────────
    # visual_query → RemoteCLIP text embedding → FAISS Top-K retrieval
    # location is passed as a hint only — geographic filtering is optional.
    candidate_response = semantic_search(
        visual_query=structured.visual_query,
        location=structured.location,
    )
    # Propagate the same query_id across the entire pipeline.
    candidate_response.query_id = query_id

    logger.info(
        "CANDIDATES_RETRIEVED query_id=%s count=%d top_score=%s",
        query_id,
        len(candidate_response.candidates),
        f"{candidate_response.candidates[0].score:.3f}" if candidate_response.candidates else "N/A",
    )

    if not candidate_response.candidates:
        raise NoCandidatesError()

    # ── Response: expose all intermediate results for inspection ──────────────
    # This multi-field response lets AI/ML components be debugged independently.
    # In the next phase, the top candidate bbox will be validated and passed
    # to analyze() for STAC + COG geospatial computation.
    return QueryPipelineResponse(
        query_id=query_id,
        query=body.query,
        structured_query=structured,
        geographic_result=geographic_result,
        candidates=candidate_response.candidates,
    )
