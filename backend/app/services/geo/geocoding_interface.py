"""
GeoQueryAI — Geographic Grounding Interface.

════════════════════════════════════════════════════════════════════
  PLUG-IN POINT: place name → coordinates
════════════════════════════════════════════════════════════════════

Contract:
    Input  : free-text place name (str)
    Output : GeographicResult (validated Pydantic model)

This is the ONLY geocoding entry point used by the pipeline. It wraps
`geocoding_service.geocode_location()` and adapts its raw dict output
into the `GeographicResult` schema.

IMPORTANT — Geographic design rule:
  The coordinates produced here are used ONLY to validate candidate tiles
  (see services/search/validator.py). They are never fed directly into the
  raster pipeline as an AOI. The AOI always comes from a validated
  SemanticCandidate.bbox.

────────────────────────────────────────────────────────────────────
Mock mode:
  Controlled by the MOCK_GEO env var (default: true).

  When mock_geo=True  → resolves against a small built-in gazetteer.
                        No network access, fully deterministic.
  When mock_geo=False → calls live OpenStreetMap Nominatim.
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.schemas.query import GeographicResult
from app.core.config import settings

logger = logging.getLogger("geoquery.geocoding")


# ── Built-in gazetteer used in mock mode ──────────────────────────────────────
# bbox convention: [min_lon, min_lat, max_lon, max_lat] (EPSG:4326)
_MOCK_GAZETTEER: Dict[str, Dict[str, Any]] = {
    "bengaluru": {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "bbox": [77.4601, 12.8340, 77.7840, 13.1434],
        "display_name": "Bengaluru, Bangalore Urban, Karnataka, India",
    },
    "ulsoor lake": {
        "latitude": 12.9810,
        "longitude": 77.5946,
        "bbox": [77.5899, 12.9765, 77.5992, 12.9854],
        "display_name": "Ulsoor Lake, Bengaluru, Karnataka, India",
    },
    "cauvery delta": {
        "latitude": 10.7905,
        "longitude": 79.1378,
        "bbox": [78.8000, 10.4000, 79.8500, 11.2000],
        "display_name": "Cauvery Delta, Thanjavur, Tamil Nadu, India",
    },
    "mumbai": {
        "latitude": 19.0760,
        "longitude": 72.8777,
        "bbox": [72.7758, 18.8900, 72.9866, 19.2760],
        "display_name": "Mumbai, Maharashtra, India",
    },
    "chennai": {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "bbox": [80.1200, 12.8300, 80.3300, 13.2400],
        "display_name": "Chennai, Tamil Nadu, India",
    },
    "delhi": {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "bbox": [76.8380, 28.4041, 77.3486, 28.8835],
        "display_name": "Delhi, India",
    },
    "hyderabad": {
        "latitude": 17.3850,
        "longitude": 78.4867,
        "bbox": [78.2400, 17.2400, 78.6800, 17.6000],
        "display_name": "Hyderabad, Telangana, India",
    },
}

# Aliases → canonical gazetteer keys.
_MOCK_ALIASES: Dict[str, str] = {
    "bangalore": "bengaluru",
    "bengaluru, india": "bengaluru",
    "ulsoor": "ulsoor lake",
    "halasuru lake": "ulsoor lake",
    "bombay": "mumbai",
    "madras": "chennai",
    "new delhi": "delhi",
}


def _lookup_mock(location: str) -> Optional[Dict[str, Any]]:
    """Resolve a place name against the built-in gazetteer (case-insensitive)."""
    key = location.strip().lower()
    key = _MOCK_ALIASES.get(key, key)

    if key in _MOCK_GAZETTEER:
        return _MOCK_GAZETTEER[key]

    # Substring match, longest key first so "ulsoor lake" beats "bengaluru"
    # in a string such as "Ulsoor Lake, Bengaluru".
    for gaz_key in sorted(_MOCK_GAZETTEER, key=len, reverse=True):
        if gaz_key in key:
            return _MOCK_GAZETTEER[gaz_key]

    return None


def geocode(location: str) -> GeographicResult:
    """
    Resolve a free-text place name into coordinates and a bounding box.

    This function never raises on a failed lookup. An unresolvable place name
    returns a GeographicResult with grounded=False and no coordinates, which
    the candidate validator treats as "no spatial constraint" rather than as
    an error.

    Args:
        location: Free-text place name, e.g. "Ulsoor Lake, Bengaluru".

    Returns:
        GeographicResult: Grounded coordinates, or grounded=False if unresolved.
    """
    if not location or not location.strip():
        return GeographicResult(name=location or "", grounded=False)

    # ── MOCK MODE ─────────────────────────────────────────────────────────────
    if settings.mock_geo:
        hit = _lookup_mock(location)
        if hit is None:
            logger.info("LOCATION_GROUNDED [MOCK] name=%r grounded=False (not in gazetteer)", location)
            return GeographicResult(name=location, grounded=False)

        result = GeographicResult(
            name=location,
            latitude=hit["latitude"],
            longitude=hit["longitude"],
            bbox=list(hit["bbox"]),
            display_name=hit["display_name"],
            grounded=True,
        )
        logger.info(
            "LOCATION_GROUNDED [MOCK] name=%r lat=%.4f lon=%.4f",
            result.name, result.latitude, result.longitude,
        )
        return result
    # ── END MOCK ──────────────────────────────────────────────────────────────

    # ── REAL MODE — OpenStreetMap Nominatim ───────────────────────────────────
    from app.services.geo.geocoding_service import geocode_location

    raw = geocode_location(location)
    if raw is None:
        logger.warning("LOCATION_GROUNDED name=%r grounded=False (geocoder returned no match)", location)
        return GeographicResult(name=location, grounded=False)

    # geocoding_service returns bbox as a dict; GeographicResult wants a list
    # ordered [min_lon, min_lat, max_lon, max_lat].
    bbox_list: Optional[list[float]] = None
    raw_bbox = raw.get("bbox")
    if isinstance(raw_bbox, dict):
        bbox_list = [
            float(raw_bbox["min_lon"]),
            float(raw_bbox["min_lat"]),
            float(raw_bbox["max_lon"]),
            float(raw_bbox["max_lat"]),
        ]
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        bbox_list = [float(v) for v in raw_bbox]

    result = GeographicResult(
        name=location,
        latitude=raw.get("latitude"),
        longitude=raw.get("longitude"),
        bbox=bbox_list,
        display_name=raw.get("display_name"),
        grounded=raw.get("latitude") is not None,
    )
    logger.info(
        "LOCATION_GROUNDED name=%r lat=%s lon=%s",
        result.name, result.latitude, result.longitude,
    )
    return result
