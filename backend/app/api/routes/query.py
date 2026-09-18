"""
GeoQueryAI — POST /api/query

Main end-to-end orchestration endpoint. Accepts a free-text natural language query
and executes the complete GeoQueryAI pipeline:

Pipeline stages:
  1. parse_query()         → Gemini StructuredQuery (visual_query, location, analysis, dates)
  2. geocode()             → Geographic Grounding (resolves location place name to lat/lon/bbox)
  3. semantic_search()     → RemoteCLIP + FAISS Top-K Candidate Tiles
  4. validate_candidates() → Spatial & Data Integrity Filter (validates location match, selects tile)
  5. analyze()             → STAC Scene Search & COG Raster Analysis (NDVI, NDWI, water extent)
  6. Evidence Layer        → Structured evidence of computed physical metrics
  7. explain_evidence()    → Grounded Gemini Explanation strictly based on Evidence

Architecture Note on Roles:
  - RemoteCLIP          : Semantic visual concept matching (candidate generation)
  - Candidate Validator : Spatial consistency filter (geospatial grounding check)
  - STAC / COG Service  : Scene availability verification and pixel computation
  - Grounded Gemini     : Fact-checked explanation strictly from computed evidence
"""

import uuid
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.schemas.query import QueryRequest, QueryPipelineResponse
from app.schemas.analysis import AnalysisResult, Evidence
from app.services.ai.interface import parse_query, explain_evidence
from app.services.search.interface import semantic_search
from app.services.search.validator import validate_candidates
from app.services.geo.geocoding_interface import geocode
from app.services.geo.interface import analyze
from app.core.errors import (
    AIServiceError,
    SearchServiceError,
    NoCandidatesError,
    GeoAnalysisError,
    NoSatelliteDataError,
)

logger = logging.getLogger("geoquery.query")
router = APIRouter()


@router.post(
    "/query",
    response_model=QueryPipelineResponse,
    summary="Natural-language GeoQueryAI pipeline (end-to-end)",
    description=(
        "Accepts a free-text natural-language query and runs the complete GeoQueryAI pipeline: "
        "Gemini query parsing → geographic grounding → RemoteCLIP + FAISS semantic retrieval → "
        "candidate spatial validation → STAC/COG raster analysis → Evidence JSON → grounded Gemini explanation."
    ),
    responses={
        200: {"description": "Pipeline completed successfully — full response returned."},
        404: {"description": "No candidate tiles found for the given query."},
        503: {"description": "An upstream service (AI or search) is unavailable."},
    },
)
async def run_query(body: QueryRequest) -> QueryPipelineResponse:
    """
    GeoQueryAI end-to-end orchestration endpoint.

    Pipeline stages:
      1. parse_query()         — Gemini / mock query parser
      2. geocode()             — geographic grounding (Nominatim / mock)
      3. semantic_search()     — RemoteCLIP + FAISS candidate retrieval
      4. validate_candidates() — Candidate spatial consistency validation & selection
      5. analyze()             — STAC scene search & COG raster analysis
      6. Evidence layer        — collects only computed & verified metrics
      7. explain_evidence()    — Grounded Gemini explanation strictly from Evidence
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
    candidate_response = semantic_search(
        visual_query=structured.visual_query,
        location=structured.location,
    )
    candidate_response.query_id = query_id

    logger.info(
        "CANDIDATES_RETRIEVED query_id=%s count=%d top_score=%s",
        query_id,
        len(candidate_response.candidates),
        f"{candidate_response.candidates[0].score:.3f}" if candidate_response.candidates else "N/A",
    )

    if not candidate_response.candidates:
        raise NoCandidatesError()

    # ── Stage 4: Candidate Validation & Selection ─────────────────────────────
    validation = validate_candidates(
        candidates=candidate_response.candidates,
        geographic_result=geographic_result,
    )

    selected_candidate = validation.selected_candidate
    logger.info(
        "CANDIDATES_VALIDATED query_id=%s status=%s selected_tile=%s reason=%r",
        query_id,
        validation.validation_status,
        selected_candidate.tile_id if selected_candidate else "None",
        validation.validation_reason,
    )

    # ── Stage 5 & 6: Geospatial Pipeline & Evidence Construction ─────────────
    analysis_result: AnalysisResult | None = None
    evidence: Evidence | None = None
    explanation: str | None = None

    if selected_candidate is not None:
        try:
            analysis_result = analyze(
                bbox=selected_candidate.bbox,
                analysis=structured.analysis,
                start_date=structured.start_date,
                end_date=structured.end_date,
            )

            scene_meta = analysis_result.scene
            evidence = Evidence(
                location=structured.location or (geographic_result.name if geographic_result else None),
                bbox=analysis_result.bbox,
                scene_id=scene_meta.scene_id if scene_meta else None,
                acquisition_date=scene_meta.datetime if scene_meta else None,
                analysis=analysis_result.analysis,
                computed_values=analysis_result.metrics,
                change_metrics=analysis_result.metrics.get("change_metrics", {}),
                source="Sentinel-2 L2A via Planetary Computer STAC",
            )

            explanation = explain_evidence(evidence)

        except (NoSatelliteDataError, GeoAnalysisError) as exc:
            logger.warning("Geospatial analysis skipped/error: %s", exc)
            analysis_result = AnalysisResult(
                analysis=structured.analysis,
                bbox=selected_candidate.bbox,
                status="no_data" if isinstance(exc, NoSatelliteDataError) else "error",
                message=str(exc),
            )
            evidence = Evidence(
                location=structured.location or (geographic_result.name if geographic_result else None),
                bbox=selected_candidate.bbox,
                analysis=structured.analysis,
                computed_values={},
                change_metrics={},
                source="Sentinel-2 L2A via Planetary Computer STAC",
            )
            explanation = explain_evidence(evidence)
    else:
        # No candidate passed spatial validation
        loc_name = structured.location or "requested region"
        explanation = (
            f"No satellite tile imagery covering '{loc_name}' was found in the indexed dataset. "
            f"Candidate validation status: {validation.validation_status}."
        )

    # ── Response: complete end-to-end inspectable response ────────────────────
    return QueryPipelineResponse(
        query_id=query_id,
        query=body.query,
        structured_query=structured,
        geographic_result=geographic_result,
        candidates=candidate_response.candidates,
        selected_candidate=selected_candidate,
        validation_status=validation.validation_status,
        validation_reason=validation.validation_reason,
        validation_details=[d.model_dump() for d in validation.details],
        analysis_result=analysis_result,
        evidence=evidence,
        explanation=explanation,
    )


