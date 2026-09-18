from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from fastapi.staticfiles import StaticFiles

from backend.app.services.geoquery.visualization_service import (
    save_index_map
)

from backend.app.services.geoquery.stac_service import (
    get_best_sentinel_scene
)

from backend.app.services.geoquery.ndvi_service import (
    calculate_ndvi
)

from backend.app.services.geoquery.ndwi_service import (
    calculate_ndwi
)


app = FastAPI(
    title="GeoQuery AI",
    description="Natural Language Gateway for Earth Observation Data",
    version="0.1.0"
)

BASE_DIR = Path(__file__).resolve().parent

MAPS_DIR = BASE_DIR / "static" / "maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/maps",
    StaticFiles(directory=str(MAPS_DIR)),
    name="maps"
)

class AnalysisRequest(BaseModel):
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    max_cloud_cover: float = 100


@app.get("/")
def root():
    return {
        "message": "GeoQuery AI is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/ndvi")
def ndvi_endpoint(request: AnalysisRequest):

    aoi = [
        request.longitude - 0.01,
        request.latitude - 0.01,
        request.longitude + 0.01,
        request.latitude + 0.01
    ]

    scene = get_best_sentinel_scene(
        latitude=request.latitude,
        longitude=request.longitude,
        start_date=request.start_date,
        end_date=request.end_date,
        max_cloud_cover=request.max_cloud_cover
    )

    if scene is None:
        raise HTTPException(
            status_code=404,
            detail="No suitable satellite scene found."
        )

    result = calculate_ndvi(
        red_url=scene["B04_url"],
        nir_url=scene["B08_url"],
        scl_url=scene["SCL_url"],
        aoi=aoi
    )
    map_path = MAPS_DIR / "ndvi_map.png"

    save_index_map(
        data=result["ndvi"],
        output_path=map_path,
        title="NDVI Map",
        colorbar_label="NDVI",
        colormap="RdYlGn"
    )

    return {
    "scene_id": scene.get("id"),
    "statistics": result["statistics"],
    "width": result["width"],
    "height": result["height"],
    "map_url": "/maps/ndvi_map.png"
}


@app.post("/ndwi")
def ndwi_endpoint(request: AnalysisRequest):

    aoi = [
        request.longitude - 0.01,
        request.latitude - 0.01,
        request.longitude + 0.01,
        request.latitude + 0.01
    ]

    scene = get_best_sentinel_scene(
        latitude=request.latitude,
        longitude=request.longitude,
        start_date=request.start_date,
        end_date=request.end_date,
        max_cloud_cover=request.max_cloud_cover
    )

    if scene is None:
        raise HTTPException(
            status_code=404,
            detail="No suitable satellite scene found."
        )

    result = calculate_ndwi(
        green_url=scene["B03_url"],
        nir_url=scene["B08_url"],
        aoi=aoi
    )
    map_path = MAPS_DIR / "ndwi_map.png"

    save_index_map(
        data=result["ndwi"],
        output_path=map_path,
        title="NDWI Map",
        colorbar_label="NDWI",
        colormap="Blues"
    )

    return {
        "scene_id": scene.get("id"),
        "statistics": result["statistics"],
        "width": result["width"],
        "height": result["height"],
        "map_url": "/maps/ndwi_map.png"
    }