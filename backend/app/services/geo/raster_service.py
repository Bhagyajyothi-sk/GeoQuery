from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path

import rasterio
import numpy as np

from rasterio.windows import from_bounds
from rasterio.warp import transform_geom
from rasterio.enums import Resampling
from rasterio.features import geometry_mask

import shapely.geometry
from shapely.geometry import shape, box, mapping


GeometryInput = Union[
    Dict[str, Any],
    shapely.geometry.base.BaseGeometry,
    List[float],
    Tuple[float, float, float, float]
]


def normalize_geometry(
    geometry_or_bbox: GeometryInput
) -> Dict[str, Any]:
    """
    Convert bounding boxes, Shapely geometries, or GeoJSON
    into a GeoJSON geometry mapping.
    """

    if isinstance(geometry_or_bbox, (list, tuple)):
        if len(geometry_or_bbox) != 4:
            raise ValueError(
                "Bounding box must contain 4 values."
            )

        min_lon, min_lat, max_lon, max_lat = geometry_or_bbox

        return mapping(
            box(min_lon, min_lat, max_lon, max_lat)
        )

    elif isinstance(
        geometry_or_bbox,
        shapely.geometry.base.BaseGeometry
    ):
        return mapping(geometry_or_bbox)

    elif isinstance(geometry_or_bbox, dict):

        if geometry_or_bbox.get("type") == "Feature":
            return geometry_or_bbox.get("geometry", {})

        elif geometry_or_bbox.get("type") == "FeatureCollection":
            features = geometry_or_bbox.get("features", [])

            if features:
                return features[0].get("geometry", {})

            raise ValueError(
                "FeatureCollection contains no features."
            )

        return geometry_or_bbox

    else:
        raise ValueError(
            f"Unsupported geometry format: {type(geometry_or_bbox)}"
        )


def read_geometry_from_cog(
    cog_url: str,
    aoi: GeometryInput,
    output_shape: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Read only the minimum raster window covering the AOI
    from a remote Cloud Optimized GeoTIFF.

    AOI coordinates must use EPSG:4326.
    """

    geojson_geom = normalize_geometry(aoi)

    is_polygon = geojson_geom.get("type") in (
        "Polygon",
        "MultiPolygon"
    )

    with rasterio.open(cog_url) as src:

        if src.crs is None:
            raise ValueError(
                "Raster does not contain a coordinate reference system."
            )

        transformed_geom = transform_geom(
            "EPSG:4326",
            src.crs,
            geojson_geom
        )

        geom_shape = shape(transformed_geom)

        minx, miny, maxx, maxy = geom_shape.bounds

        window = from_bounds(
            minx,
            miny,
            maxx,
            maxy,
            transform=src.transform
        )

        window = (
            window
            .round_offsets()
            .round_lengths()
        )

        win_transform = src.window_transform(window)

        if output_shape:

            data = src.read(
                1,
                window=window,
                out_shape=output_shape,
                resampling=Resampling.nearest
            )

        else:

            data = src.read(
                1,
                window=window
            )

        if is_polygon:

            inside_mask = geometry_mask(
                [transformed_geom],
                out_shape=data.shape,
                transform=win_transform,
                invert=True
            )

        else:

            inside_mask = np.ones(
                data.shape,
                dtype=bool
            )

        return {
            "data": data,
            "inside_mask": inside_mask,
            "width": int(data.shape[1]),
            "height": int(data.shape[0]),
            "crs": src.crs,
            "crs_str": str(src.crs),
            "transform": win_transform,
            "window": {
                "col_off": int(window.col_off),
                "row_off": int(window.row_off),
                "width": int(window.width),
                "height": int(window.height)
            },
            "bounds": [
                minx,
                miny,
                maxx,
                maxy
            ]
        }


def read_aoi_from_cog(
    cog_url: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    output_shape: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Backward-compatible helper for bounding box coordinates.
    """

    return read_geometry_from_cog(
        cog_url=cog_url,
        aoi=[
            min_lon,
            min_lat,
            max_lon,
            max_lat
        ],
        output_shape=output_shape
    )


def save_geotiff(
    data: np.ndarray,
    transform: Any,
    crs: Any,
    output_path: Union[str, Path],
    nodata: float = -9999.0
) -> None:
    """
    Save a 2D NumPy array as a georeferenced GeoTIFF.
    """

    export_data = np.nan_to_num(
        data,
        nan=nodata
    ).astype(np.float32)

    with rasterio.open(
        str(output_path),
        "w",
        driver="GTiff",
        height=export_data.shape[0],
        width=export_data.shape[1],
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate"
    ) as dst:

        dst.write(export_data, 1)