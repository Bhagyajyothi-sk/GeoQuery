"""
GeoQueryAI — Geographic Grounding Interface.

════════════════════════════════════════════════════════════════════
  PLUG-IN POINT: Geographic grounding (location text → coordinates)
════════════════════════════════════════════════════════════════════

`geocode()` is the authoritative adapter between the pipeline and
any geocoding provider. Route handlers and orchestration code call
only this function — never the underlying service directly.

Contract:
    Input  : location (str)  — a free-text place name from StructuredQuery
    Output : GeographicResult — validated Pydantic model

Pipeline position:
  User query
    ↓ parse_query()
  StructuredQuery.location  (text only, never coordinates)
    ↓ geocode()                               ← THIS FILE
  GeographicResult          (lat, lon, bbox)
    ↓ semantic_search()    (location used as hint only)
  CandidateResponse.candidates[*].bbox  ← the ONLY coords trusted downstream

────────────────────────────────────────────────────────────────────
Mock mode:
  Controlled by MOCK_GEO env var (default: true).
  Set MOCK_GEO=false to enable live Nominatim geocoding.

  When mock_geo=True  → returns deterministic Bengaluru coordinates.
  When mock_geo=False → calls geocoding_service.geocode_location()
                        which queries the Nominatim OSM API.

Geographic provider swap guide:
  The real block below delegates to geocoding_service.geocode_location().
  To swap providers (Google Maps, Mapbox, etc.), update geocoding_service.py.
  This file (geocoding_interface.py) does not need to change.
────────────────────────────────────────────────────────────────────
"""

import logging
from typing import Optional

from app.schemas.query import GeographicResult
from app.core.config import settings

logger = logging.getLogger("geoquery.geo")

# ── Mock coordinates (Bengaluru, India) ───────────────────────────────────────
_MOCK_GEO_RESULT = GeographicResult(
    name="Bengaluru",
    latitude=12.9716,
    longitude=77.5946,
    bbox=[77.4601, 12.8340, 77.7840, 13.1434],
    display_name="Bengaluru, Karnataka, India",
    grounded=True,
)


def geocode(location: str) -> GeographicResult:
    """
    Convert a free-text place name into geographic coordinates.

    Returns a GeographicResult with centroid lat/lon and bounding box.
    If the location cannot be resolved, returns a GeographicResult with
    grounded=False and no coordinates.

    ┌──────────────────────────────────────────────────────────────────┐
    │  MOCK MODE  (MOCK_GEO=true, the default)                         │
    │  Returns deterministic Bengaluru coords for development.         │
    │  Set MOCK_GEO=false to enable live Nominatim geocoding.          │
    └──────────────────────────────────────────────────────────────────┘

    IMPORTANT — Coordinate trust rule:
      GeographicResult coordinates are used ONLY as a location hint for
      semantic search filtering. They are NEVER used directly as satellite
      tile coordinates. The only trusted coordinates downstream are
      SemanticCandidate.bbox values returned by the FAISS index.

    Args:
        location: A free-text place name. Comes from StructuredQuery.location.

    Returns:
        GeographicResult: Geocoded location with lat, lon, and bbox.
                         If geocoding fails, grounded=False with no coordinates.
    """
    if settings.mock_geo:
        # ── MOCK: return deterministic Bengaluru result ────────────────────────
        result = _MOCK_GEO_RESULT.model_copy(update={"name": location})
        logger.info(
            "LOCATION_GROUNDED [MOCK] name=%r lat=%.4f lon=%.4f grounded=%s",
            result.name,
            result.latitude,
            result.longitude,
            result.grounded,
        )
        return result
        # ── END MOCK ───────────────────────────────────────────────────────────

    # ── REAL: delegate to the Nominatim geocoding service ─────────────────────
    from app.services.geo.geocoding_service import geocode_location

    raw = geocode_location(location)

    if raw is None:
        # Geocoding failed or location not found — return an ungrounded result.
        result = GeographicResult(
            name=location,
            grounded=False,
        )
        logger.warning("LOCATION_GROUNDED name=%r grounded=False (geocoding failed)", location)
        return result

    # Convert the raw dict from geocoding_service into a clean Pydantic model.
    # geocoding_service bbox format: {min_lat, max_lat, min_lon, max_lon}
    # GeographicResult bbox format: [min_lon, min_lat, max_lon, max_lat]  (GeoJSON)
    raw_bbox = raw.get("bbox")
    bbox: Optional[list[float]] = None
    if raw_bbox and isinstance(raw_bbox, dict):
        bbox = [
            raw_bbox["min_lon"],
            raw_bbox["min_lat"],
            raw_bbox["max_lon"],
            raw_bbox["max_lat"],
        ]

    result = GeographicResult(
        name=raw.get("location", location),
        latitude=raw.get("latitude"),
        longitude=raw.get("longitude"),
        bbox=bbox,
        display_name=raw.get("display_name"),
        grounded=True,
    )

    logger.info(
        "LOCATION_GROUNDED name=%r lat=%s lon=%s grounded=%s",
        result.name,
        f"{result.latitude:.4f}" if result.latitude is not None else "None",
        f"{result.longitude:.4f}" if result.longitude is not None else "None",
        result.grounded,
    )
    return result
