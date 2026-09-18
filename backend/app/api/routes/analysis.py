"""
GeoQueryAI — POST /api/analyze

Geospatial analysis endpoint.
Directly exposes the STAC + COG analysis pipeline for standalone use
(e.g. testing analysis logic with a known bbox independently of retrieval).
"""

import logging

from fastapi import APIRouter

from app.schemas.analysis import AnalysisRequest, AnalysisResult
from app.services.geo.interface import analyze

logger = logging.getLogger("geoquery.analysis")
router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="Geospatial analysis on a Sentinel-2 scene (STAC + COG)",
    description=(
        "Searches Planetary Computer for a suitable Sentinel-2 scene covering the given bbox "
        "and runs the requested analysis (NDVI, NDWI, water_extent, change, etc.). "
        "The bbox must come from a validated SemanticCandidate — never from raw LLM output."
    ),
    responses={
        200: {"description": "Analysis completed successfully."},
        404: {"description": "No Sentinel-2 scene found for the given area and dates."},
        422: {"description": "Invalid request (bad bbox, unrecognised analysis type, etc.)."},
        503: {"description": "Geospatial analysis service unavailable."},
    },
)
async def run_analysis(body: AnalysisRequest) -> AnalysisResult:
    """
    Geospatial analysis endpoint.

    Calls the geo analysis interface directly — no AI parsing, no semantic search.
    Use this to test the STAC + COG pipeline with a known bbox.

    HTTP error codes:
      422 — invalid bbox or analysis type (Pydantic validation)
      404 — no satellite data found
      503 — pipeline failure
    """
    logger.info(
        "analyze bbox=%s analysis=%s dates=%s–%s",
        body.bbox, body.analysis, body.start_date, body.end_date,
    )

    result = analyze(
        bbox=body.bbox,
        analysis=body.analysis,
        start_date=body.start_date,
        end_date=body.end_date,
    )

    logger.info("analyze status=%s", result.status)
    return result
