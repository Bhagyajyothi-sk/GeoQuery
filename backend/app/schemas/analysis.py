"""
GeoQueryAI — geospatial analysis schemas.

Defines the contract between:
  - The analysis request    (AnalysisRequest)
  - The analysis result     (AnalysisResult)  ← STAC + COG pipeline populates this
"""

from typing import Any, Literal
from pydantic import BaseModel, Field


# ── Inbound Request ────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Request body for POST /api/analyze."""

    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[min_lon, min_lat, max_lon, max_lat] in EPSG:4326. Must come from a validated SemanticCandidate, never directly from LLM output.",
        examples=[[77.50, 12.90, 77.65, 13.05]],
    )
    analysis: Literal[
        "discovery",
        "ndvi",
        "ndwi",
        "water_extent",
        "change",
        "water_extent_change",
    ] = Field(
        ...,
        description="Geospatial analysis type to perform.",
    )
    start_date: str | None = Field(
        default=None,
        description="ISO 8601 date (YYYY-MM-DD). Start of the scene search window.",
        examples=["2025-09-01"],
    )
    end_date: str | None = Field(
        default=None,
        description="ISO 8601 date (YYYY-MM-DD). End of the scene search window.",
        examples=["2026-09-01"],
    )


# ── Outbound Result ────────────────────────────────────────────────────────────

class SceneMetadata(BaseModel):
    """Metadata for the Sentinel-2 scene used in the analysis."""

    scene_id: str
    datetime: str | None
    cloud_cover: float | None
    valid_pixel_fraction: float | None
    bbox: list[float] | None


class AnalysisResult(BaseModel):
    """
    Output from the geospatial analysis pipeline.

    PLUG-IN POINT:
        services/geo/interface.py :: analyze(bbox, analysis, start_date, end_date) -> AnalysisResult
    """

    analysis: str = Field(..., description="Analysis type that was performed.")
    bbox: list[float] = Field(..., description="AOI used for this analysis.")
    scene: SceneMetadata | None = Field(default=None, description="Sentinel-2 scene metadata.")
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis-specific numeric results (e.g. mean NDVI, water extent km²).",
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Supporting evidence for the Grounded Gemini explanation (future phase).",
    )
    status: Literal["ok", "no_data", "error"] = Field(
        default="ok",
        description="Pipeline execution status.",
    )
    message: str | None = Field(default=None, description="Human-readable status message.")
