"""
GeoQueryAI — Geospatial Analysis Interface.

════════════════════════════════════════════════════════════════════
  PLUG-IN POINT: STAC scene search + COG raster analysis
════════════════════════════════════════════════════════════════════

Contract:
    Input  : bbox [min_lon, min_lat, max_lon, max_lat] (EPSG:4326),
             analysis type, optional start/end dates
    Output : AnalysisResult (validated Pydantic model)

This is the single dispatch point for every raster computation. Route
handlers and the orchestration pipeline call `analyze()` and nothing else;
the individual services (ndvi_service, ndwi_service, …) stay independently
testable behind it.

IMPORTANT — Geographic design rule:
  The bbox passed in MUST originate from a validated SemanticCandidate.
  Coordinates produced by the LLM are never used as an AOI.

────────────────────────────────────────────────────────────────────
Mock mode:
  Controlled by the MOCK_GEO env var (default: true).

  When mock_geo=True  → deterministic synthetic metrics, no network.
                        Values are derived from the bbox so repeated runs
                        agree, but they are NOT real measurements — every
                        mock result carries status message "[MOCK]".
  When mock_geo=False → live Planetary Computer STAC + COG reads.
────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from app.schemas.analysis import AnalysisResult, SceneMetadata
from app.core.config import settings
from app.core.errors import GeoAnalysisError, InvalidAOIError, NoSatelliteDataError

logger = logging.getLogger("geoquery.geo")

# Analyses this interface knows how to run.
SUPPORTED_ANALYSES = (
    "discovery",
    "road_discovery",
    "water_body_discovery",
    "agricultural_area_discovery",
    "vegetation_health",
    "ndvi",
    "ndwi",
    "water_extent",
    "change",
    "water_extent_change",
    "clarification_needed",
)

# NDWI values above this threshold are classified as open water.
WATER_NDWI_THRESHOLD = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(value: Optional[str], fallback: date) -> date:
    """Parse a YYYY-MM-DD string, falling back to a default on None/invalid."""
    if not value:
        return fallback
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        logger.warning("Unparseable date %r — using fallback %s", value, fallback)
        return fallback


def _resolve_window(start_date: Optional[str], end_date: Optional[str]) -> tuple[date, date]:
    """
    Resolve the temporal search window, defaulting to the last 12 months.
    Guarantees start <= end.
    """
    today = date.today()
    end = _parse_date(end_date, today)
    start = _parse_date(start_date, end - timedelta(days=365))
    if start > end:
        start, end = end, start
    return start, end


def _split_window(start: date, end: date) -> tuple[tuple[date, date], tuple[date, date]]:
    """
    Split a window into two halves for change detection.

    Returns ((early_start, early_end), (late_start, late_end)).
    """
    midpoint = start + (end - start) / 2
    return (start, midpoint), (midpoint, end)


# ─────────────────────────────────────────────────────────────────────────────
# Mock metric generation (deterministic, derived from the bbox)
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_seed(bbox: List[float]) -> float:
    """
    Derive a stable value in [0, 1) from a bbox.

    Used only to make mock metrics deterministic per AOI — this is not a
    measurement and carries no physical meaning.
    """
    total = sum(abs(float(v)) * (i + 1) for i, v in enumerate(bbox))
    return (total * 1000.0) % 1000.0 / 1000.0


def _bbox_area_km2(bbox: List[float]) -> float:
    """
    Approximate bbox area in km², adequate for mock output at small extents.

    Uses 111.32 km per degree of latitude and cos(lat)-corrected longitude.
    """
    import math

    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    mid_lat_rad = math.radians((min_lat + max_lat) / 2.0)
    height_km = abs(max_lat - min_lat) * 111.32
    width_km = abs(max_lon - min_lon) * 111.32 * math.cos(mid_lat_rad)
    return round(height_km * width_km, 6)


def _mock_result(
    bbox: List[float],
    analysis: str,
    start: date,
    end: date,
) -> AnalysisResult:
    """Build a deterministic synthetic AnalysisResult for development use."""
    seed = _bbox_seed(bbox)
    area_km2 = _bbox_area_km2(bbox)

    scene = SceneMetadata(
        scene_id=f"MOCK_S2_{int(seed * 1e6):06d}",
        datetime=f"{end.isoformat()}T05:30:00Z",
        cloud_cover=round(seed * 15.0, 2),
        valid_pixel_fraction=round(0.80 + seed * 0.19, 4),
        bbox=list(bbox),
    )

    metrics: Dict[str, Any] = {}

    if analysis in ("discovery", "road_discovery", "water_body_discovery", "agricultural_area_discovery", "clarification_needed"):
        metrics = {"aoi_area_km2": area_km2}

    elif analysis in ("ndvi", "vegetation_health"):
        mean_ndvi = round(0.15 + seed * 0.55, 4)
        metrics = {
            "mean_ndvi": mean_ndvi,
            "min_ndvi": round(mean_ndvi - 0.35, 4),
            "max_ndvi": round(min(mean_ndvi + 0.30, 1.0), 4),
            "median_ndvi": round(mean_ndvi + 0.01, 4),
            "valid_pixel_count": 10000,
        }

    elif analysis == "ndwi":
        mean_ndwi = round(-0.30 + seed * 0.70, 4)
        metrics = {
            "mean_ndwi": mean_ndwi,
            "min_ndwi": round(max(mean_ndwi - 0.40, -1.0), 4),
            "max_ndwi": round(min(mean_ndwi + 0.45, 1.0), 4),
            "median_ndwi": round(mean_ndwi - 0.01, 4),
            "valid_pixel_count": 10000,
        }

    elif analysis == "water_extent":
        water_fraction = round(0.05 + seed * 0.50, 4)
        metrics = {
            "water_fraction": water_fraction,
            "water_area_km2": round(area_km2 * water_fraction, 6),
            "aoi_area_km2": area_km2,
            "water_pixel_count": int(10000 * water_fraction),
            "valid_pixel_count": 10000,
            "ndwi_threshold": WATER_NDWI_THRESHOLD,
        }

    elif analysis == "change":
        early = round(0.20 + seed * 0.40, 4)
        late = round(early + (seed - 0.5) * 0.20, 4)
        metrics = {
            "early_mean_ndvi": early,
            "late_mean_ndvi": late,
            "mean_ndvi_change": round(late - early, 4),
            "change_metrics": {
                "mean_ndvi_change": round(late - early, 4),
                "direction": "increase" if late >= early else "decrease",
            },
        }

    elif analysis == "water_extent_change":
        early_fraction = round(0.10 + seed * 0.40, 4)
        late_fraction = round(max(early_fraction + (seed - 0.5) * 0.15, 0.0), 4)
        early_area = round(area_km2 * early_fraction, 6)
        late_area = round(area_km2 * late_fraction, 6)
        metrics = {
            "early_water_area_km2": early_area,
            "late_water_area_km2": late_area,
            "aoi_area_km2": area_km2,
            "change_metrics": {
                "water_area_change_km2": round(late_area - early_area, 6),
                "water_area_change_percent": (
                    round((late_area - early_area) / early_area * 100.0, 2)
                    if early_area > 0 else None
                ),
                "direction": "increase" if late_area >= early_area else "decrease",
            },
        }

    return AnalysisResult(
        analysis=analysis,
        bbox=list(bbox),
        scene=scene,
        metrics=metrics,
        evidence={"mode": "mock", "window": f"{start.isoformat()}/{end.isoformat()}"},
        status="ok",
        message=(
            "[MOCK] Synthetic metrics generated without satellite data. "
            "Set MOCK_GEO=false for real Sentinel-2 analysis."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Real-mode raster helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pixel_area_km2(transform: Any) -> float:
    """
    Derive the ground area of one pixel, in km², from a rasterio transform.

    Sentinel-2 COGs are in UTM, so transform units are metres.
    """
    pixel_width_m = abs(float(transform.a))
    pixel_height_m = abs(float(transform.e))
    return (pixel_width_m * pixel_height_m) / 1_000_000.0


def _scene_to_metadata(scene: Dict[str, Any]) -> SceneMetadata:
    """Adapt a stac_service scene dict into the SceneMetadata schema."""
    return SceneMetadata(
        scene_id=scene.get("id", "unknown"),
        datetime=scene.get("datetime"),
        cloud_cover=scene.get("cloud_cover"),
        valid_pixel_fraction=scene.get("valid_pixel_fraction"),
        bbox=scene.get("bbox"),
    )


def _require_bands(scene: Dict[str, Any], bands: List[str]) -> None:
    """Raise NoSatelliteDataError if the scene is missing any required band."""
    missing = [b for b in bands if not scene.get(b)]
    if missing:
        raise NoSatelliteDataError(
            f"Scene {scene.get('id', 'unknown')} is missing required band(s): {', '.join(missing)}"
        )


def _find_scene(bbox: List[float], start: date, end: date) -> Dict[str, Any]:
    """Search for the best Sentinel-2 scene covering the AOI, or raise."""
    from app.services.geo.stac_service import get_best_sentinel_scene

    scene = get_best_sentinel_scene(
        aoi=list(bbox),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        max_cloud_cover=settings.stac_max_cloud_cover,
    )
    if scene is None:
        raise NoSatelliteDataError(
            f"No Sentinel-2 scene found for bbox {bbox} between {start} and {end}."
        )
    return scene


def _water_metrics(ndwi_result: Dict[str, Any]) -> Dict[str, Any]:
    """Classify open water from an NDWI array and measure its extent."""
    import numpy as np

    ndwi = ndwi_result["ndwi"]
    valid = np.isfinite(ndwi)
    valid_count = int(valid.sum())

    if valid_count == 0:
        return {
            "water_fraction": None,
            "water_area_km2": None,
            "water_pixel_count": 0,
            "valid_pixel_count": 0,
            "ndwi_threshold": WATER_NDWI_THRESHOLD,
        }

    water = valid & (ndwi > WATER_NDWI_THRESHOLD)
    water_count = int(water.sum())
    pixel_km2 = _pixel_area_km2(ndwi_result["transform"])

    return {
        "water_fraction": round(water_count / valid_count, 6),
        "water_area_km2": round(water_count * pixel_km2, 6),
        "aoi_area_km2": round(valid_count * pixel_km2, 6),
        "water_pixel_count": water_count,
        "valid_pixel_count": valid_count,
        "ndwi_threshold": WATER_NDWI_THRESHOLD,
    }


def _run_ndvi(bbox: List[float], scene: Dict[str, Any]) -> Dict[str, Any]:
    """Compute NDVI statistics for the AOI from a scene's COG bands."""
    from app.services.geo.ndvi_service import calculate_ndvi

    _require_bands(scene, ["B04_url", "B08_url", "SCL_url"])
    result = calculate_ndvi(
        red_url=scene["B04_url"],
        nir_url=scene["B08_url"],
        scl_url=scene["SCL_url"],
        aoi=list(bbox),
    )
    stats = result["statistics"]
    return {
        "mean_ndvi": stats["mean"],
        "min_ndvi": stats["min"],
        "max_ndvi": stats["max"],
        "median_ndvi": stats["median"],
        "valid_pixel_count": stats["valid_pixel_count"],
    }


def _run_ndwi(bbox: List[float], scene: Dict[str, Any]) -> Dict[str, Any]:
    """Compute NDWI for the AOI, returning both the array result and statistics."""
    from app.services.geo.ndwi_service import calculate_ndwi

    _require_bands(scene, ["B03_url", "B08_url"])
    return calculate_ndwi(
        green_url=scene["B03_url"],
        nir_url=scene["B08_url"],
        aoi=list(bbox),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze(
    bbox: List[float],
    analysis: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> AnalysisResult:
    """
    Run the requested geospatial analysis over the given AOI.

    Args:
        bbox:       [min_lon, min_lat, max_lon, max_lat] in EPSG:4326. Must come
                    from a validated SemanticCandidate, never from raw LLM output.
        analysis:   One of SUPPORTED_ANALYSES.
        start_date: ISO date (YYYY-MM-DD). Defaults to 12 months before end_date.
        end_date:   ISO date (YYYY-MM-DD). Defaults to today.

    Returns:
        AnalysisResult with scene metadata and analysis-specific metrics.

    Raises:
        GeoAnalysisError:     Invalid input, or the raster pipeline failed.
        NoSatelliteDataError: No suitable Sentinel-2 scene covers the AOI.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not bbox or len(bbox) != 4:
        raise InvalidAOIError(f"bbox must contain exactly 4 values, got: {bbox}")

    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    if min_lon >= max_lon or min_lat >= max_lat:
        raise InvalidAOIError(
            f"Invalid bbox ordering {bbox}; expected [min_lon, min_lat, max_lon, max_lat]."
        )
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise InvalidAOIError(f"Longitude out of range in bbox {bbox}.")
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise InvalidAOIError(f"Latitude out of range in bbox {bbox}.")

    if analysis not in SUPPORTED_ANALYSES:
        raise InvalidAOIError(
            f"Unsupported analysis {analysis!r}. Expected one of: {', '.join(SUPPORTED_ANALYSES)}."
        )

    start, end = _resolve_window(start_date, end_date)

    logger.info(
        "ANALYSIS_STARTED bbox=%s analysis=%s window=%s/%s mode=%s",
        bbox, analysis, start, end, "MOCK" if settings.mock_geo else "REAL",
    )

    # ── MOCK MODE ─────────────────────────────────────────────────────────────
    if settings.mock_geo:
        result = _mock_result(list(bbox), analysis, start, end)
        logger.info("ANALYSIS_COMPLETED [MOCK] analysis=%s metrics=%s", analysis, list(result.metrics))
        return result
    # ── END MOCK ──────────────────────────────────────────────────────────────

    # ── REAL MODE — Planetary Computer STAC + COG ─────────────────────────────
    try:
        aoi = list(bbox)

        # Single-date analyses -------------------------------------------------
        if analysis in ("discovery", "road_discovery", "water_body_discovery", "agricultural_area_discovery", "clarification_needed", "ndvi", "vegetation_health", "ndwi", "water_extent"):
            scene = _find_scene(aoi, start, end)

            if analysis in ("discovery", "road_discovery", "water_body_discovery", "agricultural_area_discovery", "clarification_needed"):
                metrics: Dict[str, Any] = {"aoi_area_km2": _bbox_area_km2(aoi)}
            elif analysis in ("ndvi", "vegetation_health"):
                metrics = _run_ndvi(aoi, scene)
            elif analysis == "ndwi":
                ndwi_result = _run_ndwi(aoi, scene)
                stats = ndwi_result["statistics"]
                metrics = {
                    "mean_ndwi": stats["mean"],
                    "min_ndwi": stats["min"],
                    "max_ndwi": stats["max"],
                    "median_ndwi": stats["median"],
                    "valid_pixel_count": stats["valid_pixel_count"],
                }
            else:  # water_extent
                metrics = _water_metrics(_run_ndwi(aoi, scene))

            return AnalysisResult(
                analysis=analysis,
                bbox=aoi,
                scene=_scene_to_metadata(scene),
                metrics=metrics,
                evidence={"mode": "real", "window": f"{start.isoformat()}/{end.isoformat()}"},
                status="ok",
                message=None,
            )

        # Two-date change analyses --------------------------------------------
        (early_start, early_end), (late_start, late_end) = _split_window(start, end)
        early_scene = _find_scene(aoi, early_start, early_end)
        late_scene = _find_scene(aoi, late_start, late_end)

        if analysis == "change":
            early = _run_ndvi(aoi, early_scene)
            late = _run_ndvi(aoi, late_scene)
            delta = (
                round(late["mean_ndvi"] - early["mean_ndvi"], 6)
                if early["mean_ndvi"] is not None and late["mean_ndvi"] is not None
                else None
            )
            metrics = {
                "early_mean_ndvi": early["mean_ndvi"],
                "late_mean_ndvi": late["mean_ndvi"],
                "mean_ndvi_change": delta,
                "change_metrics": {
                    "mean_ndvi_change": delta,
                    "early_scene_id": early_scene.get("id"),
                    "late_scene_id": late_scene.get("id"),
                    "direction": (
                        None if delta is None else ("increase" if delta >= 0 else "decrease")
                    ),
                },
            }
        else:  # water_extent_change
            early = _water_metrics(_run_ndwi(aoi, early_scene))
            late = _water_metrics(_run_ndwi(aoi, late_scene))
            early_area = early["water_area_km2"]
            late_area = late["water_area_km2"]
            delta = (
                round(late_area - early_area, 6)
                if early_area is not None and late_area is not None
                else None
            )
            metrics = {
                "early_water_area_km2": early_area,
                "late_water_area_km2": late_area,
                "aoi_area_km2": late.get("aoi_area_km2"),
                "change_metrics": {
                    "water_area_change_km2": delta,
                    "water_area_change_percent": (
                        round(delta / early_area * 100.0, 2)
                        if delta is not None and early_area else None
                    ),
                    "early_scene_id": early_scene.get("id"),
                    "late_scene_id": late_scene.get("id"),
                    "direction": (
                        None if delta is None else ("increase" if delta >= 0 else "decrease")
                    ),
                },
            }

        return AnalysisResult(
            analysis=analysis,
            bbox=aoi,
            scene=_scene_to_metadata(late_scene),
            metrics=metrics,
            evidence={
                "mode": "real",
                "early_window": f"{early_start.isoformat()}/{early_end.isoformat()}",
                "late_window": f"{late_start.isoformat()}/{late_end.isoformat()}",
            },
            status="ok",
            message=None,
        )

    except (NoSatelliteDataError, GeoAnalysisError, InvalidAOIError):
        # Already carry the right HTTP status — let them through untouched.
        raise
    except Exception as exc:
        logger.exception("Geospatial analysis failed for bbox=%s analysis=%s", bbox, analysis)
        raise GeoAnalysisError(f"Geospatial analysis failed: {exc}") from exc
