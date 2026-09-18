from typing import Optional, Dict, Any

import logging
import requests

logger = logging.getLogger("geoquery.geocoding_service")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {
    "User-Agent": "GeoQuery-AI-GeocodingService/1.0"
}

_GEOCODE_CACHE: Dict[str, Dict[str, Any]] = {}


def geocode_location(location: str) -> Optional[Dict[str, Any]]:
    """
    Convert a place name into latitude, longitude, and bounding box
    using the OpenStreetMap Nominatim geocoding service with in-memory caching.
    """
    if not location or not location.strip():
        return None

    clean_loc = location.strip().lower()
    if clean_loc in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[clean_loc]

    params = {
        "q": location,
        "format": "json",
        "limit": 1
    }

    try:
        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        results = response.json()

        if not results:
            return None

        result = results[0]
        lat = float(result["lat"])
        lon = float(result["lon"])

        # bbox from nominatim: [min_lat, max_lat, min_lon, max_lon]
        raw_bbox = result.get("boundingbox")
        bbox = None
        if raw_bbox and len(raw_bbox) == 4:
            bbox = {
                "min_lat": float(raw_bbox[0]),
                "max_lat": float(raw_bbox[1]),
                "min_lon": float(raw_bbox[2]),
                "max_lon": float(raw_bbox[3])
            }

        resolved = {
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "display_name": result.get("display_name", location),
            "bbox": bbox
        }
        _GEOCODE_CACHE[clean_loc] = resolved
        return resolved

    except Exception as exc:
        # Graceful failure handling — the caller treats None as "ungrounded".
        logger.warning("Nominatim geocoding failed for %r: %s", location, exc)
        return None
