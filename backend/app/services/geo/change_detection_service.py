from typing import Optional, Dict, Any
import numpy as np

from app.services.geo.stac_service import (
    get_best_sentinel_scene
)

from app.services.geo.raster_service import (
    read_geometry_from_cog
)


def calculate_ndvi(red_band, nir_band):
    """
    Calculate NDVI from Red and NIR bands.
    """

    red = red_band.astype(float)
    nir = nir_band.astype(float)

    denominator = nir + red

    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0
    )

    return ndvi


def detect_ndvi_change(
    latitude: float,
    longitude: float,
    first_start_date: str,
    first_end_date: str,
    second_start_date: str,
    second_end_date: str,
    max_cloud_cover: float = 100.0,
    buffer: float = 0.02
) -> Optional[Dict[str, Any]]:
    """
    Compare NDVI between two time periods.
    """

    # Get the first satellite scene
    first_scene = get_best_sentinel_scene(
        latitude=latitude,
        longitude=longitude,
        start_date=first_start_date,
        end_date=first_end_date,
        max_cloud_cover=max_cloud_cover,
        buffer=buffer
    )

    # Get the second satellite scene
    second_scene = get_best_sentinel_scene(
        latitude=latitude,
        longitude=longitude,
        start_date=second_start_date,
        end_date=second_end_date,
        max_cloud_cover=max_cloud_cover,
        buffer=buffer
    )

    if first_scene is None or second_scene is None:
        return None

    required_bands = [
        "B04_url",
        "B08_url"
    ]

    for band in required_bands:
        if not first_scene.get(band) or not second_scene.get(band):
            return None

    # Read first-period bands
    first_red = read_geometry_from_cog(
        first_scene["B04_url"],
        [
            longitude - buffer,
            latitude - buffer,
            longitude + buffer,
            latitude + buffer
        ]
    )["data"]

    first_nir = read_geometry_from_cog(
        first_scene["B08_url"],
        [
            longitude - buffer,
            latitude - buffer,
            longitude + buffer,
            latitude + buffer
        ]
    )["data"]

    # Read second-period bands
    second_red = read_geometry_from_cog(
        second_scene["B04_url"],
        [
            longitude - buffer,
            latitude - buffer,
            longitude + buffer,
            latitude + buffer
        ]
    )["data"]

    second_nir = read_geometry_from_cog(
        second_scene["B08_url"],
        [
            longitude - buffer,
            latitude - buffer,
            longitude + buffer,
            latitude + buffer
        ]
    )["data"]

    # Ensure matching dimensions
    if (
        first_red.shape != first_nir.shape
        or second_red.shape != second_nir.shape
        or first_red.shape != second_red.shape
    ):
        return None

    # Calculate NDVI
    first_ndvi = calculate_ndvi(first_red, first_nir)
    second_ndvi = calculate_ndvi(second_red, second_nir)

    # Calculate change
    ndvi_difference = second_ndvi - first_ndvi

    return {
        "first_scene_id": first_scene["id"],
        "second_scene_id": second_scene["id"],
        "first_ndvi_mean": float(np.mean(first_ndvi)),
        "second_ndvi_mean": float(np.mean(second_ndvi)),
        "mean_ndvi_change": float(np.mean(ndvi_difference)),
        "minimum_change": float(np.min(ndvi_difference)),
        "maximum_change": float(np.max(ndvi_difference))
    }