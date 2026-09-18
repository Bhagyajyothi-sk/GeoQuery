
from typing import Dict, Any
import numpy as np

from backend.app.services.geoquery.raster_service import (
    read_geometry_from_cog
)


def calculate_ndwi(
    green_url: str,
    nir_url: str,
    aoi
) -> Dict[str, Any]:

    # Read Green band (B03)
    green_result = read_geometry_from_cog(
        cog_url=green_url,
        aoi=aoi
    )

    green = green_result["data"].astype(np.float32)
    target_shape = green.shape

    # Read NIR band (B08) at the same resolution
    nir_result = read_geometry_from_cog(
        cog_url=nir_url,
        aoi=aoi,
        output_shape=target_shape
    )

    nir = nir_result["data"].astype(np.float32)

    # Combine AOI masks
    inside_mask = (
        green_result["inside_mask"]
        & nir_result["inside_mask"]
    )

    # NDWI formula
    denominator = green + nir

    ndwi = np.full(
        target_shape,
        np.nan,
        dtype=np.float32
    )

    valid_denominator = denominator != 0

    valid_mask = (
        inside_mask
        & valid_denominator
    )

    ndwi[valid_mask] = (
        (green[valid_mask] - nir[valid_mask])
        / denominator[valid_mask]
    )

    # Keep values within valid NDWI range
    ndwi[valid_mask] = np.clip(
        ndwi[valid_mask],
        -1.0,
        1.0
    )

    # Calculate statistics
    valid_values = ndwi[np.isfinite(ndwi)]

    if valid_values.size > 0:
        statistics = {
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "mean": float(np.mean(valid_values)),
            "median": float(np.median(valid_values)),
            "valid_pixel_count": int(valid_values.size)
        }
    else:
        statistics = {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "valid_pixel_count": 0
        }

    return {
        "ndwi": ndwi,
        "statistics": statistics,
        "width": int(ndwi.shape[1]),
        "height": int(ndwi.shape[0]),
        "transform": green_result["transform"],
        "crs": green_result["crs"],
        "inside_mask": inside_mask
    }