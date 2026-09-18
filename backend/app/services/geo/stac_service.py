"""
GeoQueryAI — STAC Service.

Searches Microsoft's Planetary Computer STAC API for Sentinel-2 L2A scenes
covering a given AOI, with cloud-cover filtering and SCL-based pixel validation.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from pystac_client import Client
import planetary_computer

from app.core.config import settings
from app.services.geo.raster_service import read_geometry_from_cog, normalize_geometry

# Sentinel-2 SCL classes considered invalid (cloud, shadow, defective pixels)
INVALID_SCL_CLASSES = {
    0,   # No data
    1,   # Saturated / defective
    3,   # Cloud shadow
    8,   # Cloud medium probability
    9,   # Cloud high probability
    10,  # Thin cirrus
    11,  # Snow / ice
}


def search_sentinel_scenes(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    start_date: str = "",
    end_date: str = "",
    max_cloud_cover: float = None,
    buffer: float = 0.05,
    aoi: Optional[Union[Dict[str, Any], List[float]]] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search Sentinel-2 satellite imagery metadata using the Planetary Computer STAC API.

    Supports search by:
      - Coordinates (latitude/longitude) with a square buffer
      - Bounding box [min_lon, min_lat, max_lon, max_lat]
      - GeoJSON Polygon / MultiPolygon

    Returns scene metadata (IDs, datetimes, cloud cover, asset keys) without
    reading any pixel data.
    """
    if max_cloud_cover is None:
        max_cloud_cover = settings.stac_max_cloud_cover

    catalog = Client.open(settings.stac_url)
    search_kwargs: Dict[str, Any] = {
        "collections": [settings.stac_collection],
        "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
    }
    if start_date or end_date:
        search_kwargs["datetime"] = f"{start_date or '..'}/{end_date or '..'}"

    if aoi is not None:
        normalized = normalize_geometry(aoi)
        if normalized.get("type") in ("Polygon", "MultiPolygon"):
            search_kwargs["intersects"] = normalized
        else:
            coords = (
                aoi
                if isinstance(aoi, list)
                else [aoi["min_lon"], aoi["min_lat"], aoi["max_lon"], aoi["max_lat"]]
            )
            search_kwargs["bbox"] = coords
    elif latitude is not None and longitude is not None:
        search_kwargs["bbox"] = [
            longitude - buffer,
            latitude - buffer,
            longitude + buffer,
            latitude + buffer,
        ]
    else:
        raise ValueError(
            "Either 'aoi' or ('latitude', 'longitude') must be provided for scene search."
        )

    search = catalog.search(**search_kwargs)
    items = list(search.items())
    signed_items = [planetary_computer.sign(item) for item in items]

    results: List[Dict[str, Any]] = []
    for item in signed_items[:limit]:
        results.append({
            "id": item.id,
            "datetime": item.datetime.isoformat() if item.datetime else None,
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "bbox": item.bbox,
            "assets": list(item.assets.keys()),
        })

    return {
        "total_scenes_found": len(items),
        "scenes": results,
    }


def get_best_sentinel_scene(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    start_date: str = "",
    end_date: str = "",
    max_cloud_cover: float = 100.0,
    cloud_mask: bool = True,
    min_valid_fraction: float = 0.20,
    buffer: float = 0.02,
    aoi: Optional[Union[Dict[str, Any], List[float]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Find and return the best cloud-screened Sentinel-2 scene covering the given AOI.

    Validates coverage using the SCL (Scene Classification Layer) band — only pixels
    inside the target geometry are assessed. Falls back to the least-cloudy scene if
    no scene passes the valid-pixel threshold.

    Returns a dict with signed asset URLs (B02, B03, B04, B08, SCL) or None if
    no scenes are found at all.
    """
    target_aoi = aoi
    if target_aoi is None:
        if latitude is None or longitude is None:
            raise ValueError("Either 'aoi' or ('latitude', 'longitude') must be provided.")
        target_aoi = [
            longitude - buffer,
            latitude - buffer,
            longitude + buffer,
            latitude + buffer,
        ]

    normalized = normalize_geometry(target_aoi)
    catalog = Client.open(settings.stac_url)
    search_kwargs: Dict[str, Any] = {
        "collections": [settings.stac_collection],
        "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
    }
    if start_date or end_date:
        search_kwargs["datetime"] = f"{start_date or '..'}/{end_date or '..'}"

    if normalized.get("type") in ("Polygon", "MultiPolygon"):
        search_kwargs["intersects"] = normalized
    else:
        coords = (
            target_aoi
            if isinstance(target_aoi, list)
            else [target_aoi["min_lon"], target_aoi["min_lat"], target_aoi["max_lon"], target_aoi["max_lat"]]
        )
        search_kwargs["bbox"] = coords

    search = catalog.search(**search_kwargs)
    items = list(search.items())
    if not items:
        return None

    # Sort by overall scene cloud cover ascending
    items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100.0))

    for item in items[:10]:
        signed_item = planetary_computer.sign(item)
        if "SCL" not in signed_item.assets:
            continue

        # Inspect SCL band for valid pixel fraction within the exact AOI
        scl = read_geometry_from_cog(signed_item.assets["SCL"].href, target_aoi)
        scl_data = scl["data"]
        inside_mask = scl.get("inside_mask", np.ones(scl_data.shape, dtype=bool))

        polygon_pixels = scl_data[inside_mask]
        if polygon_pixels.size == 0:
            continue

        valid_mask = ~np.isin(polygon_pixels, list(INVALID_SCL_CLASSES))
        valid_fraction = float(np.mean(valid_mask))

        if valid_fraction >= min_valid_fraction:
            return _build_scene_result(signed_item, valid_fraction)

    # Fallback: return least-cloudy scene regardless of valid pixel fraction
    if items:
        fallback = planetary_computer.sign(items[0])
        return _build_scene_result(fallback, valid_pixel_fraction=0.0)

    return None


def _build_scene_result(signed_item: Any, valid_pixel_fraction: float) -> Dict[str, Any]:
    """Build the standard scene result dict from a signed STAC item."""
    assets = signed_item.assets

    def _href(key: str) -> Optional[str]:
        return assets[key].href if key in assets else None

    return {
        "id": signed_item.id,
        "datetime": signed_item.datetime.isoformat() if signed_item.datetime else None,
        "cloud_cover": signed_item.properties.get("eo:cloud_cover"),
        "valid_pixel_fraction": round(valid_pixel_fraction, 4),
        "B02_url": _href("B02"),
        "B03_url": _href("B03"),
        "B04_url": _href("B04"),
        "B08_url": _href("B08"),
        "SCL_url": _href("SCL"),
        "bbox": signed_item.bbox,
    }
