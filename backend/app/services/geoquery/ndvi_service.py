
from typing import Dict, Any

import numpy as np
from rasterio.enums import Resampling

from backend.app.services.geoquery.raster_service import (
    read_geometry_from_cog
)


INVALID_SCL_CLASSES = {
    0, 1, 3, 8, 9, 10, 11
}


def calculate_ndvi(
    red_url: str,
    nir_url: str,
    scl_url: str,
    aoi
) -> Dict[str, Any]:
    """
    Calculate NDVI using Sentinel-2 B04 (Red)
    and B08 (NIR) bands.

    The SCL band is resampled to match the
    Red band's dimensions.
    """

    # Read Red band (10 m)
    red_result = read_geometry_from_cog(
        cog_url=red_url,
        aoi=aoi
    )

    red = red_result["data"].astype(
        np.float32
    )

    target_shape = red.shape

    # Read NIR band at the Red band's shape
    nir_result = read_geometry_from_cog(
        cog_url=nir_url,
        aoi=aoi,
        output_shape=target_shape
    )

    # Read SCL band (20 m) and resample to Red shape
    scl_result = read_geometry_from_cog(
        cog_url=scl_url,
        aoi=aoi,
        output_shape=target_shape
    )

    nir = nir_result["data"].astype(
        np.float32
    )

    scl = scl_result["data"]

    # Resize masks to the same shape if necessary
    red_mask = red_result["inside_mask"]

    nir_mask = nir_result["inside_mask"]

    scl_mask = scl_result["inside_mask"]

    inside_mask = (
        red_mask
        & nir_mask
        & scl_mask
    )

    # NDVI formula
    denominator = nir + red

    ndvi = np.full(
        target_shape,
        np.nan,
        dtype=np.float32
    )

    valid_denominator = denominator != 0

    valid_mask = (
        inside_mask
        & valid_denominator
        & ~np.isin(
            scl,
            list(INVALID_SCL_CLASSES)
        )
    )

    ndvi[valid_mask] = (
        (nir[valid_mask] - red[valid_mask])
        / denominator[valid_mask]
    )

    # Limit NDVI to -1 to 1
    ndvi[valid_mask] = np.clip(
        ndvi[valid_mask],
        -1.0,
        1.0
    )

    # Calculate statistics
    valid_values = ndvi[
        np.isfinite(ndvi)
    ]

    if valid_values.size > 0:

        statistics = {
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "mean": float(np.mean(valid_values)),
            "median": float(np.median(valid_values)),
            "valid_pixel_count": int(
                valid_values.size
            )
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
        "ndvi": ndvi,
        "statistics": statistics,
        "width": int(ndvi.shape[1]),
        "height": int(ndvi.shape[0]),
        "transform": red_result["transform"],
        "crs": red_result["crs"],
        "inside_mask": inside_mask
    }