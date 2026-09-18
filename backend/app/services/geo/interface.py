"""
GeoQueryAI — Geospatial Analysis Interface.

════════════════════════════════════════════════════════════════════
  PLUG-IN POINT: STAC search + Sentinel-2 COG analysis pipeline
════════════════════════════════════════════════════════════════════

This interface wraps the real STAC and raster services.
It is the authoritative boundary between the API layer and
geospatial computation. Route handlers call only this function.

Pipeline (per call):
  1. Search Planetary Computer for the best Sentinel-2 scene covering bbox.
  2. Read the relevant COG bands for the requested analysis.
  3. Compute the requested index / metric (NDVI, NDWI, water extent, etc.).
  4. Return an AnalysisResult with metrics and evidence.

The compute functions (NDVI, NDWI, change detection, etc.) are the
primary extension points for the geospatial engineer.
"""

from typing import Optional
import numpy as np

from app.schemas.analysis import AnalysisResult, SceneMetadata
from app.core.errors import GeoAnalysisError, NoSatelliteDataError
from app.services.geo.stac_service import get_best_sentinel_scene
from app.services.geo.raster_service import read_geometry_from_cog


def analyze(
    bbox: list[float],
    analysis: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> AnalysisResult:
    """
    Run the requested geospatial analysis over the given bounding box.

    ════════════════════════════════════════════════════════════════════
      PLUG-IN POINT: STAC + COG analysis pipeline
    ════════════════════════════════════════════════════════════════════

    To add a new analysis type:
      1. Add the type to the Literal in schemas/analysis.py.
      2. Implement a private `_compute_<type>` function below.
      3. Add a dispatch branch in the `_dispatch` dict.

    Args:
        bbox:       [min_lon, min_lat, max_lon, max_lat] from a SemanticCandidate.
                    These are validated tile coordinates — never raw LLM output.
        analysis:   Analysis type key (matches StructuredQuery.analysis).
        start_date: ISO 8601 date string for the STAC temporal window start.
        end_date:   ISO 8601 date string for the STAC temporal window end.

    Returns:
        AnalysisResult with scene metadata, numeric metrics, and evidence.

    Raises:
        NoSatelliteDataError: If no usable Sentinel-2 scene is found.
        GeoAnalysisError:     If the COG read or computation fails.
    """
    # ── 1. Find best Sentinel-2 scene ─────────────────────────────────────────
    try:
        scene_data = get_best_sentinel_scene(
            aoi=bbox,
            start_date=start_date or "",
            end_date=end_date or "",
        )
    except Exception as exc:
        raise GeoAnalysisError(f"STAC scene search failed: {exc}") from exc

    if scene_data is None:
        raise NoSatelliteDataError(
            f"No Sentinel-2 scene found for bbox={bbox}, "
            f"dates={start_date}–{end_date}"
        )

    scene_meta = SceneMetadata(
        scene_id=scene_data["id"],
        datetime=scene_data.get("datetime"),
        cloud_cover=scene_data.get("cloud_cover"),
        valid_pixel_fraction=scene_data.get("valid_pixel_fraction"),
        bbox=scene_data.get("bbox"),
    )

    # ── 2. Dispatch to analysis function ──────────────────────────────────────
    _dispatch = {
        "discovery":          _compute_discovery,
        "ndvi":               _compute_ndvi,
        "ndwi":               _compute_ndwi,
        "water_extent":       _compute_water_extent,
        "change":             _compute_change,
        "water_extent_change": _compute_water_extent_change,
    }

    compute_fn = _dispatch.get(analysis)
    if compute_fn is None:
        raise GeoAnalysisError(f"Unknown analysis type: '{analysis}'")

    try:
        metrics, evidence = compute_fn(bbox, scene_data)
    except (NoSatelliteDataError, GeoAnalysisError):
        raise
    except Exception as exc:
        raise GeoAnalysisError(f"Analysis '{analysis}' failed: {exc}") from exc

    return AnalysisResult(
        analysis=analysis,
        bbox=bbox,
        scene=scene_meta,
        metrics=metrics,
        evidence=evidence,
        status="ok",
    )


# ── Private analysis compute functions ────────────────────────────────────────
# Each function reads the required COG bands for the bbox and returns
# (metrics: dict, evidence: dict).
#
# These are the extension points for geospatial computation.
# ─────────────────────────────────────────────────────────────────────────────

def _compute_discovery(bbox: list[float], scene: dict) -> tuple[dict, dict]:
    """
    Discovery mode — returns scene metadata without pixel computation.
    Useful for verifying that a scene exists and is cloud-free.
    """
    return (
        {"scene_found": True},
        {"scene_id": scene["id"], "datetime": scene.get("datetime")},
    )


def _compute_ndvi(bbox: list[float], scene: dict) -> tuple[dict, dict]:
    """
    Compute mean NDVI over the AOI using Sentinel-2 B04 (Red) and B08 (NIR).

    NDVI = (NIR - Red) / (NIR + Red)

    ════════════════════════════════════════════════════════
      EXTEND HERE: Add per-pixel NDVI map, histogram, etc.
    ════════════════════════════════════════════════════════
    """
    b04 = read_geometry_from_cog(scene["B04_url"], bbox)
    b08 = read_geometry_from_cog(scene["B08_url"], bbox)

    red = b04["data"].astype(np.float32)
    nir = b08["data"].astype(np.float32)
    inside = b04["inside_mask"]

    denom = nir + red
    ndvi = np.where(denom != 0, (nir - red) / denom, np.nan)
    valid_pixels = ndvi[inside & ~np.isnan(ndvi)]

    return (
        {
            "mean_ndvi": round(float(np.mean(valid_pixels)), 4) if valid_pixels.size > 0 else None,
            "min_ndvi":  round(float(np.min(valid_pixels)), 4)  if valid_pixels.size > 0 else None,
            "max_ndvi":  round(float(np.max(valid_pixels)), 4)  if valid_pixels.size > 0 else None,
            "pixel_count": int(valid_pixels.size),
        },
        {"band_urls": {"B04": scene.get("B04_url"), "B08": scene.get("B08_url")}},
    )


def _compute_ndwi(bbox: list[float], scene: dict) -> tuple[dict, dict]:
    """
    Compute mean NDWI over the AOI using Sentinel-2 B03 (Green) and B08 (NIR).

    NDWI = (Green - NIR) / (Green + NIR)

    ════════════════════════════════════════════════════════
      EXTEND HERE: Add water body delineation, thresholding.
    ════════════════════════════════════════════════════════
    """
    b03 = read_geometry_from_cog(scene["B03_url"], bbox)
    b08 = read_geometry_from_cog(scene["B08_url"], bbox)

    green = b03["data"].astype(np.float32)
    nir   = b08["data"].astype(np.float32)
    inside = b03["inside_mask"]

    denom = green + nir
    ndwi = np.where(denom != 0, (green - nir) / denom, np.nan)
    valid_pixels = ndwi[inside & ~np.isnan(ndwi)]

    return (
        {
            "mean_ndwi": round(float(np.mean(valid_pixels)), 4) if valid_pixels.size > 0 else None,
            "pixel_count": int(valid_pixels.size),
        },
        {"band_urls": {"B03": scene.get("B03_url"), "B08": scene.get("B08_url")}},
    )


def _compute_water_extent(bbox: list[float], scene: dict) -> tuple[dict, dict]:
    """
    Estimate open-water extent using NDWI thresholding (NDWI > 0 → water).

    ════════════════════════════════════════════════════════════════
      EXTEND HERE: calibrate threshold, add area_km2 calculation,
      export water mask as GeoTIFF.
    ════════════════════════════════════════════════════════════════
    """
    b03 = read_geometry_from_cog(scene["B03_url"], bbox)
    b08 = read_geometry_from_cog(scene["B08_url"], bbox)

    green = b03["data"].astype(np.float32)
    nir   = b08["data"].astype(np.float32)
    inside = b03["inside_mask"]

    denom = green + nir
    ndwi = np.where(denom != 0, (green - nir) / denom, np.nan)

    valid = inside & ~np.isnan(ndwi)
    water_pixels  = int(np.sum((ndwi > 0) & valid))
    total_pixels  = int(np.sum(valid))
    water_fraction = round(water_pixels / total_pixels, 4) if total_pixels > 0 else 0.0

    # Pixel size at Sentinel-2 10m resolution → area in km²
    # (approximate; proper calculation needs the CRS-aware pixel size)
    pixel_area_km2 = (10 / 1000) ** 2
    water_area_km2 = round(water_pixels * pixel_area_km2, 4)

    return (
        {
            "water_pixels": water_pixels,
            "total_pixels": total_pixels,
            "water_fraction": water_fraction,
            "water_area_km2": water_area_km2,
        },
        {"band_urls": {"B03": scene.get("B03_url"), "B08": scene.get("B08_url")}},
    )


def _compute_change(bbox: list[float], scene: dict) -> tuple[dict, dict]:
    """
    Placeholder for multi-temporal change detection.

    ════════════════════════════════════════════════════════════════
      EXTEND HERE: Requires a second (reference) scene.
      Implement band differencing or NDVI change between two dates.
    ════════════════════════════════════════════════════════════════
    """
    return (
        {"status": "not_implemented", "message": "Change detection requires two scenes; plug in your implementation."},
        {},
    )


def _compute_water_extent_change(bbox: list[float], scene: dict) -> tuple[dict, dict]:
    """
    Placeholder for water-extent change detection between two dates.

    ════════════════════════════════════════════════════════════════
      EXTEND HERE: Compute water extent for start_date and end_date
      scenes separately, then diff the results.
    ════════════════════════════════════════════════════════════════
    """
    return (
        {"status": "not_implemented", "message": "Water extent change requires two scenes; plug in your implementation."},
        {},
    )
