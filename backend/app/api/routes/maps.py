"""
GeoQueryAI — POST /api/maps/*

Rendered-map endpoints. These take a latitude/longitude point rather than a
tile bbox, run the raster pipeline, and additionally write a PNG colour map
that is served back as a static URL under /maps/.

This router preserves the endpoints that previously lived in the standalone
`app/services/main.py` script (a second, unreachable FastAPI app whose imports
pointed at a package that does not exist). They are mounted on the main app
here so there is exactly one application object in the project.

These endpoints read live Sentinel-2 data from Microsoft Planetary Computer
and therefore require network access and MOCK_GEO=false.
"""

import logging
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.errors import GeoAnalysisError, NoSatelliteDataError

logger = logging.getLogger("geoquery.maps")
router = APIRouter()

# Static output directory; mounted at /maps by app.main.
MAPS_DIR = Path(__file__).resolve().parents[2] / "static" / "maps"


# ── Request models ────────────────────────────────────────────────────────────

class PointAnalysisRequest(BaseModel):
    """A point-based analysis request with a square AOI built around it."""

    latitude: float = Field(..., ge=-90.0, le=90.0, examples=[12.9810])
    longitude: float = Field(..., ge=-180.0, le=180.0, examples=[77.5946])
    start_date: str = Field(..., description="ISO date (YYYY-MM-DD)", examples=["2026-01-01"])
    end_date: str = Field(..., description="ISO date (YYYY-MM-DD)", examples=["2026-09-01"])
    max_cloud_cover: float = Field(default=100.0, ge=0.0, le=100.0)
    buffer: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Half-width of the AOI square, in degrees.",
    )


class ChangeDetectionRequest(BaseModel):
    """Two-period NDVI change detection around a point."""

    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)

    first_start_date: str
    first_end_date: str

    second_start_date: str
    second_end_date: str

    max_cloud_cover: float = Field(default=100.0, ge=0.0, le=100.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_real_mode() -> None:
    """These endpoints have no mock path — fail loudly rather than hang."""
    if settings.mock_geo:
        raise GeoAnalysisError(
            "Map rendering endpoints require live satellite data. "
            "Set MOCK_GEO=false (and ensure network access to Planetary Computer) to use them."
        )


def _aoi_around(longitude: float, latitude: float, buffer: float) -> list[float]:
    """Build a square AOI bbox [min_lon, min_lat, max_lon, max_lat] around a point."""
    return [
        longitude - buffer,
        latitude - buffer,
        longitude + buffer,
        latitude + buffer,
    ]


def _render_map(data, filename: str, title: str, label: str, colormap: str) -> str:
    """Write a colour map PNG and return the URL path it is served from."""
    # Imported lazily so a missing matplotlib cannot break application startup.
    from app.services.geo.visualization_service import save_index_map

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    save_index_map(
        data=data,
        output_path=MAPS_DIR / filename,
        title=title,
        colorbar_label=label,
        colormap=colormap,
    )
    return f"/maps/{filename}"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/maps/ndvi",
    summary="NDVI statistics plus a rendered NDVI map image",
    responses={
        200: {"description": "NDVI computed and map rendered."},
        404: {"description": "No suitable Sentinel-2 scene found."},
        503: {"description": "Pipeline unavailable, or MOCK_GEO is enabled."},
    },
)
async def ndvi_map(body: PointAnalysisRequest) -> dict:
    """Compute NDVI around a point and render it as a PNG colour map."""
    _require_real_mode()

    from app.services.geo.ndvi_service import calculate_ndvi
    from app.services.geo.stac_service import get_best_sentinel_scene

    aoi = _aoi_around(body.longitude, body.latitude, body.buffer)

    scene = get_best_sentinel_scene(
        aoi=aoi,
        start_date=body.start_date,
        end_date=body.end_date,
        max_cloud_cover=body.max_cloud_cover,
    )
    if scene is None:
        raise NoSatelliteDataError("No suitable Sentinel-2 scene found for the given area and dates.")

    for band in ("B04_url", "B08_url", "SCL_url"):
        if not scene.get(band):
            raise NoSatelliteDataError(f"Scene {scene.get('id')} is missing required band {band}.")

    result = calculate_ndvi(
        red_url=scene["B04_url"],
        nir_url=scene["B08_url"],
        scl_url=scene["SCL_url"],
        aoi=aoi,
    )

    map_url = _render_map(result["ndvi"], "ndvi_map.png", "NDVI Map", "NDVI", "RdYlGn")

    return {
        "scene_id": scene.get("id"),
        "datetime": scene.get("datetime"),
        "statistics": result["statistics"],
        "width": result["width"],
        "height": result["height"],
        "map_url": map_url,
    }


@router.post(
    "/maps/ndwi",
    summary="NDWI statistics plus a rendered NDWI map image",
    responses={
        200: {"description": "NDWI computed and map rendered."},
        404: {"description": "No suitable Sentinel-2 scene found."},
        503: {"description": "Pipeline unavailable, or MOCK_GEO is enabled."},
    },
)
async def ndwi_map(body: PointAnalysisRequest) -> dict:
    """Compute NDWI around a point and render it as a PNG colour map."""
    _require_real_mode()

    from app.services.geo.ndwi_service import calculate_ndwi
    from app.services.geo.stac_service import get_best_sentinel_scene

    aoi = _aoi_around(body.longitude, body.latitude, body.buffer)

    scene = get_best_sentinel_scene(
        aoi=aoi,
        start_date=body.start_date,
        end_date=body.end_date,
        max_cloud_cover=body.max_cloud_cover,
    )
    if scene is None:
        raise NoSatelliteDataError("No suitable Sentinel-2 scene found for the given area and dates.")

    for band in ("B03_url", "B08_url"):
        if not scene.get(band):
            raise NoSatelliteDataError(f"Scene {scene.get('id')} is missing required band {band}.")

    result = calculate_ndwi(
        green_url=scene["B03_url"],
        nir_url=scene["B08_url"],
        aoi=aoi,
    )

    map_url = _render_map(result["ndwi"], "ndwi_map.png", "NDWI Map", "NDWI", "Blues")

    return {
        "scene_id": scene.get("id"),
        "datetime": scene.get("datetime"),
        "statistics": result["statistics"],
        "width": result["width"],
        "height": result["height"],
        "map_url": map_url,
    }


@router.post(
    "/maps/change-detection",
    summary="Two-period NDVI change detection around a point",
    responses={
        200: {"description": "Change detection completed."},
        404: {"description": "No suitable scene found for one or both periods."},
        503: {"description": "Pipeline unavailable, or MOCK_GEO is enabled."},
    },
)
async def change_detection(body: ChangeDetectionRequest) -> dict:
    """Compare mean NDVI between two time windows over the same point."""
    _require_real_mode()

    from app.services.geo.change_detection_service import detect_ndvi_change

    result = detect_ndvi_change(
        latitude=body.latitude,
        longitude=body.longitude,
        first_start_date=body.first_start_date,
        first_end_date=body.first_end_date,
        second_start_date=body.second_start_date,
        second_end_date=body.second_end_date,
        max_cloud_cover=body.max_cloud_cover,
    )

    if result is None:
        raise NoSatelliteDataError(
            "Change detection could not be completed: no comparable scenes were found "
            "for both periods, or the required bands were unavailable."
        )

    return {"status": "success", "result": result}
