"""
GeoQueryAI — semantic search schemas.

Defines the contract between:
  - The search request          (SearchRequest)
  - A single candidate tile     (SemanticCandidate)  ← RemoteCLIP + FAISS produce these
  - The full search response    (CandidateResponse)
"""

from pydantic import BaseModel, Field


# ── RemoteCLIP / FAISS Contract ────────────────────────────────────────────────

class SemanticCandidate(BaseModel):
    """
    A single tile candidate returned by the RemoteCLIP + FAISS retrieval layer.

    bbox follows GeoJSON convention: [min_lon, min_lat, max_lon, max_lat] (EPSG:4326).

    IMPORTANT: These bbox values are the ONLY coordinates trusted downstream.
    Coordinates from the LLM (StructuredQuery.location) are NEVER used directly
    for geospatial analysis — they are only used for geocoding context.
    """

    tile_id: str = Field(..., description="Unique identifier for the tile.", examples=["tile_001"])
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[min_lon, min_lat, max_lon, max_lat] in EPSG:4326.",
        examples=[[77.50, 12.90, 77.65, 13.05]],
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score from RemoteCLIP embedding comparison.",
        examples=[0.87],
    )
    image_url: str | None = Field(
        default=None,
        description="Path or URL to the tile image.",
        examples=["/tiles/tile_001.png"],
    )


class CandidateResponse(BaseModel):
    """
    Full response from the semantic search layer.

    PLUG-IN POINT:
        services/search/interface.py :: semantic_search(...) -> CandidateResponse
    """

    query_id: str = Field(..., description="Unique identifier linking back to the original request.")
    visual_query: str = Field(..., description="The image-embedding query string used for retrieval.")
    location: str | None = Field(default=None, description="Geocoded location hint (for context only).")
    candidates: list[SemanticCandidate] = Field(..., description="Ranked list of candidate tiles.")


# ── Inbound Request ────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Request body for POST /api/search."""

    visual_query: str = Field(
        ...,
        min_length=2,
        description="Visual description for RemoteCLIP image-embedding retrieval.",
        examples=["large water body"],
    )
    location: str | None = Field(
        default=None,
        description="Optional location hint for geographic filtering.",
        examples=["Bengaluru"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of candidate tiles to return.",
    )
