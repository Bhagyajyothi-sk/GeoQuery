"""
GeoQueryAI — query schemas.

Defines the contract between:
  - The incoming user request       (QueryRequest)
  - The AI parser output            (StructuredQuery)      ← Gemini populates this
  - The geographic grounding output (GeographicResult)     ← geocoder populates this
  - The pipeline API response       (QueryPipelineResponse)
"""

from typing import Literal
from pydantic import BaseModel, Field


# ── AI Parser Contract ─────────────────────────────────────────────────────────

class StructuredQuery(BaseModel):
    """
    Output of the Gemini natural-language query parser.

    PLUG-IN POINT:
        services/ai/interface.py :: parse_query(text) -> StructuredQuery

    Rules:
      - visual_query  → passed to RemoteCLIP text encoder (no coordinates)
      - location      → passed to geocoder as a place-name string (NOT used as coordinates)
      - analysis      → controls which geospatial computation runs downstream
      - start_date /
        end_date      → STAC temporal filter (YYYY-MM-DD)
    """

    visual_query: str = Field(
        ...,
        description="A visual description suitable for image-embedding (RemoteCLIP input).",
        examples=["large water body", "flooded agricultural land"],
    )
    location: str | None = Field(
        default=None,
        description=(
            "Free-text place name to be geocoded into geographic coordinates. "
            "Never used directly as satellite coordinates."
        ),
        examples=["Bengaluru", "Cauvery delta"],
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
        description="Geospatial analysis type to perform on the retrieved scene.",
    )
    start_date: str | None = Field(
        default=None,
        description="ISO 8601 date string (YYYY-MM-DD). Start of the temporal window.",
        examples=["2025-09-01"],
    )
    end_date: str | None = Field(
        default=None,
        description="ISO 8601 date string (YYYY-MM-DD). End of the temporal window.",
        examples=["2026-09-01"],
    )


# ── Geographic Grounding Contract ─────────────────────────────────────────────

class GeographicResult(BaseModel):
    """
    Output of the geographic grounding stage.

    Produced by geocode() in services/geo/geocoding_interface.py.
    Kept as a separate object — never merged into StructuredQuery or
    used directly as satellite tile coordinates.

    bbox follows [min_lon, min_lat, max_lon, max_lat] (EPSG:4326).
    """

    name: str = Field(..., description="The place name that was geocoded.", examples=["Bengaluru"])
    latitude: float | None = Field(default=None, description="Centroid latitude.", examples=[12.9716])
    longitude: float | None = Field(default=None, description="Centroid longitude.", examples=[77.5946])
    bbox: list[float] | None = Field(
        default=None,
        description="[min_lon, min_lat, max_lon, max_lat] bounding box for the location.",
        examples=[[77.4601, 12.8340, 77.7840, 13.1434]],
    )
    display_name: str | None = Field(
        default=None,
        description="Full resolved display name from the geocoder.",
    )
    grounded: bool = Field(
        default=True,
        description="False if the geocoder could not resolve the location.",
    )


# ── Inbound Request ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Raw user natural-language query sent to POST /api/query."""

    query: str = Field(
        ...,
        min_length=3,
        description="Free-text natural-language query from the end user.",
        examples=["Find the water body around Bengaluru and tell me how its water extent changed since last year."],
    )


from app.schemas.analysis import AnalysisResult, Evidence
from app.schemas.search import SemanticCandidate


# ── Outbound Pipeline Response ─────────────────────────────────────────────────

class QueryPipelineResponse(BaseModel):
    """
    Response from POST /api/query.

    Exposes every intermediate pipeline stage as a separate inspectable object.
    This makes it easy to debug AI/ML output at each stage independently.

    Pipeline stages reflected here:
      query → structured_query → geographic_result → candidates → selected_candidate → analysis_result → evidence → explanation
    """

    query_id: str = Field(..., description="Unique identifier for this request.")
    query: str = Field(..., description="The raw user query, echoed back.")
    structured_query: StructuredQuery = Field(..., description="AI parser (Gemini) output.")
    geographic_result: GeographicResult | None = Field(
        default=None,
        description="Geographic grounding output. None if location was not provided.",
    )
    candidates: list[SemanticCandidate] = Field(
        ...,
        description="Top-K semantically matched satellite tile candidates from RemoteCLIP + FAISS.",
    )
    selected_candidate: SemanticCandidate | None = Field(
        default=None,
        description="The primary candidate tile selected for downstream geospatial analysis after candidate validation.",
    )
    validation_status: str | None = Field(
        default=None,
        description="Candidate validation status ('passed', 'rejected', 'no_candidates').",
    )
    validation_reason: str | None = Field(
        default=None,
        description="Human-readable rationale for candidate validation and selection.",
    )
    validation_details: list[dict] | None = Field(
        default=None,
        description="Inspectable list of per-candidate validation logs.",
    )
    analysis_result: AnalysisResult | None = Field(
        default=None,
        description="Geospatial analysis output computed from STAC scene COG bands.",
    )
    evidence: Evidence | None = Field(
        default=None,
        description="Structured evidence layer containing computed metrics passed to Gemini explanation.",
    )
    explanation: str | None = Field(
        default=None,
        description="Grounded natural-language explanation generated by Gemini strictly from Evidence.",
    )


