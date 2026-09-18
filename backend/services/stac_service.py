from typing import List, Dict, Any, Optional, Union
from pystac_client import Client
import planetary_computer
import numpy as np
from services.raster_service import read_geometry_from_cog, normalize_geometry

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Sentinel-2 SCL classes considered invalid (cloud, shadow, defective)
INVALID_SCL_CLASSES = {
    0,   # No data
    1,   # Saturated / defective
    3,   # Cloud shadow
    8,   # Cloud medium probability
    9,   # Cloud high probability
    10,  # Thin cirrus
    11   # Snow / ice
}


def search_sentinel_scenes(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    start_date: str = "",
    end_date: str = "",
    max_cloud_cover: float = 20.0,
    buffer: float = 0.05,
    aoi: Optional[Union[Dict[str, Any], List[float]]] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search Sentinel-2 satellite imagery metadata using Microsoft's Planetary Computer STAC API.
    Supports search by coordinates (with buffer), bounding box, or GeoJSON Polygon.
    """
    catalog = Client.open(STAC_URL)
    search_kwargs = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start_date}/{end_date}",
        "query": {
            "eo:cloud_cover": {
                "lt": max_cloud_cover
            }
        }
    }

    if aoi is not None:
        normalized = normalize_geometry(aoi)
        if normalized.get("type") in ("Polygon", "MultiPolygon"):
            search_kwargs["intersects"] = normalized
        else:
            # Bounding box
            coords = aoi if isinstance(aoi, list) else [aoi["min_lon"], aoi["min_lat"], aoi["max_lon"], aoi["max_lat"]]
            search_kwargs["bbox"] = coords
    elif latitude is not None and longitude is not None:
        bbox = [
            longitude - buffer,
            latitude - buffer,
            longitude + buffer,
            latitude + buffer
        ]
        search_kwargs["bbox"] = bbox
    else:
        raise ValueError("Either 'aoi' or ('latitude', 'longitude') must be provided for scene search.")

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
            "assets": list(item.assets.keys())
        })

    return {
        "total_scenes_found": len(items),
        "scenes": results
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
    aoi: Optional[Union[Dict[str, Any], List[float]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Find and return the best cloud-screened Sentinel-2 scene covering the given AOI.
    Verifies valid pixel coverage specifically in the target geometry using SCL layer inspection.
    """
    # Prepare target AOI geometry
    target_aoi = aoi
    if target_aoi is None:
        if latitude is None or longitude is None:
            raise ValueError("Either 'aoi' or ('latitude', 'longitude') must be provided.")
        target_aoi = [
            longitude - buffer,
            latitude - buffer,
            longitude + buffer,
            latitude + buffer
        ]

    normalized = normalize_geometry(target_aoi)
    catalog = Client.open(STAC_URL)
    search_kwargs = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start_date}/{end_date}",
        "query": {
            "eo:cloud_cover": {
                "lt": max_cloud_cover
            }
        }
    }

    if normalized.get("type") in ("Polygon", "MultiPolygon"):
        search_kwargs["intersects"] = normalized
    else:
        coords = target_aoi if isinstance(target_aoi, list) else [target_aoi["min_lon"], target_aoi["min_lat"], target_aoi["max_lon"], target_aoi["max_lat"]]
        search_kwargs["bbox"] = coords

    search = catalog.search(**search_kwargs)
    items = list(search.items())
    if not items:
        return None

    # Sort candidates by overall scene cloud cover ascending
    items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100.0))

    for item in items[:10]:
        signed_item = planetary_computer.sign(item)
        if "SCL" not in signed_item.assets:
            continue

        # Inspect SCL band for this specific AOI
        scl = read_geometry_from_cog(
            signed_item.assets["SCL"].href,
            target_aoi
        )

        scl_data = scl["data"]
        inside_mask = scl.get("inside_mask", np.ones(scl_data.shape, dtype=bool))
        
        # Consider only pixels inside polygon for valid fraction
        polygon_pixels = scl_data[inside_mask]
        if polygon_pixels.size == 0:
            continue

        valid_mask = ~np.isin(polygon_pixels, list(INVALID_SCL_CLASSES))
        valid_fraction = float(np.mean(valid_mask))

        # Accept this scene if sufficient valid pixels exist in the target AOI
        if valid_fraction >= min_valid_fraction:
            return {
                "id": signed_item.id,
                "datetime": signed_item.datetime.isoformat() if signed_item.datetime else None,
                "cloud_cover": signed_item.properties.get("eo:cloud_cover"),
                "valid_pixel_fraction": round(valid_fraction, 4),
                "B02_url": signed_item.assets.get("B02", signed_item.assets.get("visual")).href if "B02" in signed_item.assets else None,
                "B03_url": signed_item.assets.get("B03").href if "B03" in signed_item.assets else None,
                "B04_url": signed_item.assets.get("B04").href if "B04" in signed_item.assets else None,
                "B08_url": signed_item.assets.get("B08").href if "B08" in signed_item.assets else None,
                "SCL_url": signed_item.assets["SCL"].href,
                "bbox": signed_item.bbox
            }

    # Fallback to least cloudy scene if none pass threshold
    if items:
        fallback = planetary_computer.sign(items[0])
        return {
            "id": fallback.id,
            "datetime": fallback.datetime.isoformat() if fallback.datetime else None,
            "cloud_cover": fallback.properties.get("eo:cloud_cover"),
            "valid_pixel_fraction": 0.0,
            "B02_url": fallback.assets.get("B02").href if "B02" in fallback.assets else None,
            "B03_url": fallback.assets.get("B03").href if "B03" in fallback.assets else None,
            "B04_url": fallback.assets.get("B04").href if "B04" in fallback.assets else None,
            "B08_url": fallback.assets.get("B08").href if "B08" in fallback.assets else None,
            "SCL_url": fallback.assets.get("SCL").href if "SCL" in fallback.assets else None,
            "bbox": fallback.bbox
        }

    return None
