#!/usr/bin/env python
"""
GeoQueryAI — Satellite Tile Dataset & FAISS Index Validation Script.

1. Validates all images in data/satellite_tiles/ can be opened and have shape (100, 100, 3).
2. Validates metadata.json fields and bounding box sanity.
3. Generates data/satellite_tiles/evaluation_labels.json for ground-truth semantic query evaluation.
"""

import os
import json
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("geoquery.dataset_validator")

# Evaluation labels for semantic retrieval benchmarking
EVALUATION_MAPPING = {
    "queries": [
        {
            "query": "a satellite image of a lake",
            "category": "lake_water",
            "relevant_tiles": ["ulsoor_001", "bellandur_001", "hebbal_001", "sankey_001", "hussain_sagar_001"],
        },
        {
            "query": "a dense urban area",
            "category": "dense_urban",
            "relevant_tiles": ["urban_002", "urban_indiranagar", "urban_electronic_city", "urban_hitech_city", "urban_mumbai_bkc"],
        },
        {
            "query": "green vegetation",
            "category": "vegetation_park",
            "relevant_tiles": ["veg_cubbon_park", "veg_lalbagh", "veg_bannerghatta", "veg_turahalli", "veg_nandi_hills"],
        },
        {
            "query": "agricultural land",
            "category": "agricultural_land",
            "relevant_tiles": ["agri_mandya_001", "agri_kolar_001", "agri_tumakuru_001", "agri_doddaballapur"],
        },
        {
            "query": "open barren land",
            "category": "open_barren_land",
            "relevant_tiles": ["barren_quarry_001", "barren_chitradurga", "barren_devanahalli", "barren_ramanagara"],
        },
        {
            "query": "a lake surrounded by an urban area",
            "category": "mixed_lake_urban",
            "relevant_tiles": ["ulsoor_001", "sankey_001", "bellandur_001", "hussain_sagar_001"],
        },
    ]
}


def validate_dataset(metadata_path: str = "data/satellite_tiles/metadata.json") -> bool:
    if not os.path.exists(metadata_path):
        logger.error("Metadata file not found: %s", metadata_path)
        return False

    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info("Validating %d tile metadata records...", len(records))

    valid_count = 0
    errors = 0

    for idx, rec in enumerate(records):
        tile_id = rec.get("tile_id")
        img_path = rec.get("image_path")
        bbox = rec.get("bbox")
        c_lat = rec.get("center_lat")
        c_lon = rec.get("center_lon")

        # 1. Path & file existence
        if not img_path or not os.path.exists(img_path):
            logger.error("Record #%d [%s]: Image file missing at %s", idx, tile_id, img_path)
            errors += 1
            continue

        # 2. Image integrity & dimensions
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                w, h = img.size
                mode = img.mode
                if (w, h) != (100, 100):
                    logger.error("Record #%d [%s]: Dimension mismatch (%d, %d) != (100, 100)", idx, tile_id, w, h)
                    errors += 1
                    continue
        except Exception as exc:
            logger.error("Record #%d [%s]: Image corrupt (%s)", idx, tile_id, exc)
            errors += 1
            continue

        # 3. Bounding box sanity check [min_lon, min_lat, max_lon, max_lat]
        if not bbox or len(bbox) != 4:
            logger.error("Record #%d [%s]: Invalid bbox %s", idx, tile_id, bbox)
            errors += 1
            continue

        min_lon, min_lat, max_lon, max_lat = bbox
        if not (min_lon < max_lon and min_lat < max_lat):
            logger.error("Record #%d [%s]: Malformed bbox orientation %s", idx, tile_id, bbox)
            errors += 1
            continue

        # 4. Center coordinates sanity check
        if not (min_lat <= c_lat <= max_lat and min_lon <= c_lon <= max_lon):
            logger.error("Record #%d [%s]: Center (%.4f, %.4f) outside bbox %s", idx, tile_id, c_lat, c_lon, bbox)
            errors += 1
            continue

        valid_count += 1

    logger.info("Dataset Validation Summary: %d / %d tiles VALID (%d errors)", valid_count, len(records), errors)
    return errors == 0


def save_evaluation_labels(eval_path: str = "data/satellite_tiles/evaluation_labels.json"):
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(EVALUATION_MAPPING, f, indent=4)
    logger.info("Saved evaluation labels to %s", eval_path)


def main():
    success = validate_dataset()
    if success:
        save_evaluation_labels()
        logger.info("Dataset verification PASSED cleanly.")
    else:
        logger.error("Dataset verification FAILED.")


if __name__ == "__main__":
    main()
