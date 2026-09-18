from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import rasterio
import numpy as np
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, transform_geom
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
import shapely.geometry
from shapely.geometry import shape, box, mapping


def normalize_geometry(
    geometry_or_bbox: Union[Dict[str, Any], shapely.geometry.base.BaseGeometry, List[float], Tuple[float, float, float, float]]
) -> Dict[str, Any]:
    """
    Normalizes bounding boxes, Shapely geometries, or GeoJSON dicts into a GeoJSON mapping.
    """
    if isinstance(geometry_or_bbox, (list, tuple)) and len(geometry_or_bbox) == 4:
        # [min_lon, min_lat, max_lon, max_lat]
        min_lon, min_lat, max_lon, max_lat = geometry_or_bbox
        return mapping(box(min_lon, min_lat, max_lon, max_lat))
    elif isinstance(geometry_or_bbox, shapely.geometry.base.BaseGeometry):
        return mapping(geometry_or_bbox)
    elif isinstance(geometry_or_bbox, dict):
        if geometry_or_bbox.get("type") == "Feature":
            return geometry_or_bbox.get("geometry", {})
        elif geometry_or_bbox.get("type") == "FeatureCollection":
            features = geometry_or_bbox.get("features", [])
            if features:
                return features[0].get("geometry", {})
        return geometry_or_bbox
    else:
        raise ValueError(f"Unsupported geometry format: {type(geometry_or_bbox)}")


def read_geometry_from_cog(
    cog_url: str,
    aoi: Union[Dict[str, Any], shapely.geometry.base.BaseGeometry, List[float], Tuple[float, float, float, float]],
    output_shape: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Reads ONLY the minimal enclosing window of the given AOI geometry from a remote COG.
    Supports GeoJSON Polygons, MultiPolygons, and Bounding Boxes in EPSG:4326.
    
    Creates an exact polygon mask so pixels outside the polygon can be filtered.
    Never downloads the entire Sentinel-2 scene.
    """
    geojson_geom = normalize_geometry(aoi)
    is_polygon = geojson_geom.get("type") in ("Polygon", "MultiPolygon")

    with rasterio.open(cog_url) as src:
        # Reproject WGS84 GeoJSON geometry into the native raster CRS
        transformed_geom = transform_geom(
            "EPSG:4326",
            src.crs,
            geojson_geom
        )
        geom_shape = shape(transformed_geom)
        minx, miny, maxx, maxy = geom_shape.bounds

        # Calculate exact pixel window from transformed bounding box
        window = from_bounds(
            minx, miny, maxx, maxy,
            transform=src.transform
        )

        # Round window values to discrete pixel boundaries
        window = window.round_offsets().round_lengths()
        win_transform = src.window_transform(window)

        # Read ONLY the window from the remote COG via HTTP range requests
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

        # Create exact boolean mask for polygon boundary (True = inside polygon)
        if is_polygon:
            # geometry_mask returns True for masked (outside) pixels when invert=False;
            # with invert=True, True represents pixels INSIDE the geometry
            inside_mask = geometry_mask(
                [transformed_geom],
                out_shape=data.shape,
                transform=win_transform,
                invert=True
            )
        else:
            inside_mask = np.ones(data.shape, dtype=bool)

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
            "bounds": [minx, miny, maxx, maxy]
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
        aoi=[min_lon, min_lat, max_lon, max_lat],
        output_shape=output_shape
    )


def save_geotiff(
    data: np.ndarray,
    transform: Any,
    crs: Any,
    output_path: str | Path,
    nodata: float = -9999.0
) -> None:
    """
    Saves a 2D numpy array as a standard georeferenced GeoTIFF (.tif) file
    compatible with QGIS, ArcGIS, GDAL, rasterio, and GIS applications.
    """
    export_data = np.nan_to_num(data, nan=nodata).astype(np.float32)

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
