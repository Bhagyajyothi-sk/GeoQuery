"""
GeoQueryAI — Geocoding Service.

Converts a free-text place name into latitude, longitude, and bounding box
using the OpenStreetMap Nominatim API with in-memory caching.

Used by the geo interface to ground the location hint from StructuredQuery
before handing off to the STAC/COG pipeline.
"""

from typing import Optional, Dict, Any
import requests

from app.core.config import settings

_GEOCODE_CACHE: Dict[str, Dict[str, Any]] = {}


def geocode_location(location: str) -> Optional[Dict[str, Any]]:
    """
    Convert a place name into latitude, longitude, and bounding box
    using the OpenStreetMap Nominatim geocoding service with in-memory caching.

    Returns None if the location cannot be resolved or if the request fails.
    """
    if not location or not location.strip():
        return None

    clean_loc = location.strip().lower()
    if clean_loc in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[clean_loc]

    params = {
        "q": location,
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": settings.nominatim_user_agent}

    try:
        response = requests.get(
            settings.nominatim_url,
            params=params,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        results = response.json()

        if not results:
            return None

        result = results[0]
        lat = float(result["lat"])
        lon = float(result["lon"])

        # Nominatim bbox: [min_lat, max_lat, min_lon, max_lon]
        raw_bbox = result.get("boundingbox")
        bbox = None
        if raw_bbox and len(raw_bbox) == 4:
            bbox = {
                "min_lat": float(raw_bbox[0]),
                "max_lat": float(raw_bbox[1]),
                "min_lon": float(raw_bbox[2]),
                "max_lon": float(raw_bbox[3]),
            }

        resolved = {
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "display_name": result.get("display_name", location),
            "bbox": bbox,
        }
        _GEOCODE_CACHE[clean_loc] = resolved
        return resolved

    except Exception:
        # Graceful failure — caller decides how to handle a None result
        return None
