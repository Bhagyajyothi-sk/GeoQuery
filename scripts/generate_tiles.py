#!/usr/bin/env python
"""
GeoQueryAI — Satellite Tile Generation Script.

Queries Microsoft Planetary Computer STAC API for cloud-free Sentinel-2 L2A scenes,
extracts 100x100 RGB (B04, B03, B02) 10 m resolution tile windows around target coordinates,
performs percentile contrast stretching, saves PNGs to data/satellite_tiles/,
and updates data/satellite_tiles/metadata.json.

Target Categories:
  1. Lakes / Water Bodies (5-6)
  2. Dense Urban Areas (5-6)
  3. Vegetation / Parks / Forest (4-5)
  4. Agricultural Land (3-4)
  5. Open / Barren Land (3-4)
  6. Mixed Urban + Vegetation (2-3)
"""

import os
import json
import logging
import numpy as np
import rasterio
import rasterio.windows
import rasterio.transform
import rasterio.warp
import planetary_computer
from pystac_client import Client
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("geoquery.tile_generator")

# Target dataset locations (~24 new tiles + 2 existing = 26 total)
TARGET_TILES = [
    # ── Category 1: Lakes / Water Bodies ─────────────────────────────────────
    {
        "tile_id": "bellandur_001",
        "category": "lake_water",
        "lat": 12.9365,
        "lon": 77.6644,
        "desc": "Bellandur Lake, Bengaluru",
    },
    {
        "tile_id": "varthur_001",
        "category": "lake_water",
        "lat": 12.9422,
        "lon": 77.7460,
        "desc": "Varthur Lake, Bengaluru",
    },
    {
        "tile_id": "hebbal_001",
        "category": "lake_water",
        "lat": 13.0360,
        "lon": 77.5880,
        "desc": "Hebbal Lake, Bengaluru",
    },
    {
        "tile_id": "sankey_001",
        "category": "lake_water",
        "lat": 13.0065,
        "lon": 77.5750,
        "desc": "Sankey Tank, Bengaluru",
    },
    {
        "tile_id": "hussain_sagar_001",
        "category": "lake_water",
        "lat": 17.4239,
        "lon": 78.4738,
        "desc": "Hussain Sagar, Hyderabad",
    },
    # ── Category 2: Dense Urban Areas ─────────────────────────────────────────
    {
        "tile_id": "urban_indiranagar",
        "category": "dense_urban",
        "lat": 12.9784,
        "lon": 77.6408,
        "desc": "Indiranagar Commercial/Residential, Bengaluru",
    },
    {
        "tile_id": "urban_electronic_city",
        "category": "dense_urban",
        "lat": 12.8452,
        "lon": 77.6602,
        "desc": "Electronic City Tech Hub, Bengaluru",
    },
    {
        "tile_id": "urban_whitefield",
        "category": "dense_urban",
        "lat": 12.9698,
        "lon": 77.7499,
        "desc": "Whitefield IT Park, Bengaluru",
    },
    {
        "tile_id": "urban_hitech_city",
        "category": "dense_urban",
        "lat": 17.4435,
        "lon": 78.3772,
        "desc": "HITEC City, Hyderabad",
    },
    {
        "tile_id": "urban_mumbai_bkc",
        "category": "dense_urban",
        "lat": 19.0657,
        "lon": 72.8686,
        "desc": "Bandra Kurla Complex, Mumbai",
    },
    # ── Category 3: Vegetation / Parks / Forest ──────────────────────────────
    {
        "tile_id": "veg_cubbon_park",
        "category": "vegetation_park",
        "lat": 12.9763,
        "lon": 77.5929,
        "desc": "Cubbon Park canopy, Bengaluru",
    },
    {
        "tile_id": "veg_lalbagh",
        "category": "vegetation_park",
        "lat": 12.9507,
        "lon": 77.5848,
        "desc": "Lalbagh Botanical Garden, Bengaluru",
    },
    {
        "tile_id": "veg_bannerghatta",
        "category": "vegetation_park",
        "lat": 12.8000,
        "lon": 77.5770,
        "desc": "Bannerghatta National Park forest, Bengaluru",
    },
    {
        "tile_id": "veg_turahalli",
        "category": "vegetation_park",
        "lat": 12.8872,
        "lon": 77.5255,
        "desc": "Turahalli Reserve Forest, Bengaluru",
    },
    {
        "tile_id": "veg_nandi_hills",
        "category": "vegetation_park",
        "lat": 13.3702,
        "lon": 77.6835,
        "desc": "Nandi Hills Forested Slopes, Karnataka",
    },
    # ── Category 4: Agricultural Land ────────────────────────────────────────
    {
        "tile_id": "agri_mandya_001",
        "category": "agricultural_land",
        "lat": 12.5218,
        "lon": 76.8951,
        "desc": "Mandya irrigated paddy & sugarcane fields",
    },
    {
        "tile_id": "agri_kolar_001",
        "category": "agricultural_land",
        "lat": 13.1367,
        "lon": 78.1292,
        "desc": "Kolar farmland plots",
    },
    {
        "tile_id": "agri_tumakuru_001",
        "category": "agricultural_land",
        "lat": 13.3409,
        "lon": 77.1010,
        "desc": "Tumakuru crop fields",
    },
    {
        "tile_id": "agri_doddaballapur",
        "category": "agricultural_land",
        "lat": 13.2925,
        "lon": 77.5422,
        "desc": "Doddaballapur agricultural fields",
    },
    # ── Category 5: Open / Barren Land ───────────────────────────────────────
    {
        "tile_id": "barren_quarry_001",
        "category": "open_barren_land",
        "lat": 13.2100,
        "lon": 77.6300,
        "desc": "Granite quarry / open ground north Bengaluru",
    },
    {
        "tile_id": "barren_chitradurga",
        "category": "open_barren_land",
        "lat": 14.2250,
        "lon": 76.3980,
        "desc": "Chitradurga dry barren rocky terrain",
    },
    {
        "tile_id": "barren_devanahalli",
        "category": "open_barren_land",
        "lat": 13.2450,
        "lon": 77.7120,
        "desc": "Devanahalli open cleared ground",
    },
    {
        "tile_id": "barren_ramanagara",
        "category": "open_barren_land",
        "lat": 12.7150,
        "lon": 77.2810,
        "desc": "Ramanagara rocky barren hill outcrop",
    },
    # ── Category 6: Mixed Urban + Vegetation ─────────────────────────────────
    {
        "tile_id": "mixed_iisc_campus",
        "category": "mixed_urban_vegetation",
        "lat": 13.0184,
        "lon": 77.5684,
        "desc": "IISc Campus green trees with academic buildings",
    },
    {
        "tile_id": "mixed_jayanagar",
        "category": "mixed_urban_vegetation",
        "lat": 12.9250,
        "lon": 77.5830,
        "desc": "Jayanagar tree-lined residential avenues",
    },
    {
        "tile_id": "mixed_koramangala",
        "category": "mixed_urban_vegetation",
        "lat": 12.9350,
        "lon": 77.6240,
        "desc": "Koramangala residential blocks & park patches",
    },
]


def stretch(band: np.ndarray) -> np.ndarray:
    """Apply 2-98 percentile contrast stretch to a 2D float band array."""
    valid = band[~np.isnan(band) & (band > 0)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    if hi <= lo:
        return np.zeros_like(band, dtype=np.uint8)
    stretched = np.clip((band - lo) / (hi - lo) * 255.0, 0, 255)
    return stretched.astype(np.uint8)


def extract_tile(
    catalog: Client,
    target: dict,
    output_dir: str = "data/satellite_tiles",
    size_px: int = 100,
) -> dict | None:
    """
    Query STAC, extract 100x100 B04/B03/B02 RGB window around target coordinates,
    save PNG, and return metadata dict.
    """
    tile_id = target["tile_id"]
    lat = target["lat"]
    lon = target["lon"]

    logger.info("Extracting tile '%s' (%s) at lat=%.4f, lon=%.4f...", tile_id, target["desc"], lat, lon)

    # Search Sentinel-2 L2A for low cloud scene
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05],
        datetime="2025-01-01/2026-09-18",
        query={"eo:cloud_cover": {"lt": 15.0}},
    )

    items = list(search.items())
    if not items:
        logger.warning("No scenes with cloud_cover < 15%% found for %s. Relaxing to 30%%.", tile_id)
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=[lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05],
            datetime="2024-01-01/2026-09-18",
            query={"eo:cloud_cover": {"lt": 30.0}},
        )
        items = list(search.items())

    if not items:
        logger.error("FAILED: No usable Sentinel-2 scene found for %s", tile_id)
        return None

    # Sort by cloud cover ascending
    items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100.0))
    selected_item = planetary_computer.sign(items[0])

    # Open B04 (Red) band to find index & transform
    b04_url = selected_item.assets["B04"].href
    with rasterio.open(b04_url) as src:
        crs_str = str(src.crs)
        # Transform WGS84 (lon, lat) to raster native CRS coordinates
        xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
        x_utm, y_utm = xs[0], ys[0]

        row, col = src.index(x_utm, y_utm)
        half = size_px // 2
        window = rasterio.windows.Window(col - half, row - half, size_px, size_px)

        win_transform = src.window_transform(window)
        utm_bounds = rasterio.transform.array_bounds(size_px, size_px, win_transform)
        wgs84_bounds = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *utm_bounds)

    # Read B04, B03, B02 bands
    bands = {}
    for b_name in ["B04", "B03", "B02"]:
        b_url = selected_item.assets[b_name].href
        with rasterio.open(b_url) as src:
            bands[b_name] = src.read(1, window=window).astype(np.float32)

    # Verify tile array size
    if bands["B04"].shape != (size_px, size_px):
        logger.error("Tile shape mismatch for %s: %s", tile_id, bands["B04"].shape)
        return None

    # Stack RGB and stretch
    r = stretch(bands["B04"])
    g = stretch(bands["B03"])
    b = stretch(bands["B02"])
    rgb = np.stack([r, g, b], axis=-1)

    # Basic visual check: reject if mean brightness < 5 (completely black)
    mean_val = float(np.mean(rgb))
    if mean_val < 5.0:
        logger.warning("REJECTED %s: Image mean brightness %.2f is too dark", tile_id, mean_val)
        return None

    # Save PNG
    os.makedirs(output_dir, exist_ok=True)
    img_filename = f"{tile_id}.png"
    rel_img_path = f"{output_dir}/{img_filename}".replace("\\", "/")
    img_path = os.path.join(output_dir, img_filename)

    Image.fromarray(rgb).save(img_path)
    logger.info("SAVED: %s (shape: %s, mean_val: %.1f)", img_path, rgb.shape, mean_val)

    # Build metadata record
    min_lon, min_lat, max_lon, max_lat = wgs84_bounds
    c_lat = round((min_lat + max_lat) / 2.0, 6)
    c_lon = round((min_lon + max_lon) / 2.0, 6)

    return {
        "tile_id": tile_id,
        "image_path": rel_img_path,
        "scene_id": selected_item.id,
        "center_lat": c_lat,
        "center_lon": c_lon,
        "crs": crs_str,
        "pixel_size_m": 10,
        "width_px": size_px,
        "height_px": size_px,
        "bbox": [
            round(min_lon, 6),
            round(min_lat, 6),
            round(max_lon, 6),
            round(max_lat, 6),
        ],
    }


def main():
    logger.info("Opening Planetary Computer STAC Catalog...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    metadata_path = "data/satellite_tiles/metadata.json"
    existing_meta = []
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)

    existing_ids = {item["tile_id"] for item in existing_meta}
    logger.info("Found %d existing tile records (%s)", len(existing_meta), list(existing_ids))

    updated_meta = list(existing_meta)
    generated_count = 0
    rejected_count = 0

    for target in TARGET_TILES:
        tile_id = target["tile_id"]
        if tile_id in existing_ids:
            logger.info("Skipping existing tile '%s'", tile_id)
            continue

        try:
            record = extract_tile(catalog, target)
            if record:
                updated_meta.append(record)
                generated_count += 1
            else:
                rejected_count += 1
        except Exception as exc:
            logger.error("Failed to extract %s: %s", tile_id, exc)
            rejected_count += 1

    # Save updated metadata.json
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(updated_meta, f, indent=4)

    logger.info("==================================================")
    logger.info("Tile Generation Complete!")
    logger.info("Total tiles in metadata.json: %d", len(updated_meta))
    logger.info("New tiles generated         : %d", generated_count)
    logger.info("Tiles rejected/failed       : %d", rejected_count)
    logger.info("Saved metadata to %s", metadata_path)
    logger.info("==================================================")


if __name__ == "__main__":
    main()
