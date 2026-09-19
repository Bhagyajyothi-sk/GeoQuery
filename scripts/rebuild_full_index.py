#!/usr/bin/env python
"""
GeoQueryAI — Full FAISS Index Rebuild Script.

Generates accurate EPSG:4326 bbox for every satellite tile PNG present in
data/satellite_tiles/, writes a complete metadata.json, then rebuilds
indexes/satellite.index and indexes/tile_metadata.json using RemoteCLIP
image embeddings.

All bbox coordinates are computed from the known target lat/lon and the
100-pixel window at 10 m/pixel resolution (~0.009° per side).

Usage (run from project root with .venv active):
    python scripts/rebuild_full_index.py

Output:
    data/satellite_tiles/metadata.json   — updated with all 29 tiles
    indexes/satellite.index             — FAISS IndexFlatIP(512) binary
    indexes/tile_metadata.json          — ordered list matching FAISS positions
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

TILES_DIR = PROJECT_ROOT / "data" / "satellite_tiles"
METADATA_FILE = TILES_DIR / "metadata.json"
INDEX_DIR = PROJECT_ROOT / "indexes"
INDEX_FILE = INDEX_DIR / "satellite.index"
INDEX_METADATA_FILE = INDEX_DIR / "tile_metadata.json"
EMBEDDING_DIM = 512

# ── Target tile registry (lat/lon from generate_tiles.py) ─────────────────────
# bbox is derived: 100 px at 10 m/px = 1000 m window each side from centre.
# 1° lat ≈ 111 320 m; 1° lon ≈ 111 320 * cos(lat) m.
# Half-window in degrees = 500 m / (111 320 * correction)

TARGET_TILES = [
    # Category 1: Lakes / Water Bodies
    {"tile_id": "ulsoor_001",        "lat": 12.9810, "lon": 77.5946,
     "scene_id": "S2C_MSIL2A_20260914T050651_R019_T43PGQ_20260914T100610",
     "category": "lake_water",      "desc": "Ulsoor Lake, Bengaluru"},
    {"tile_id": "bellandur_001",     "lat": 12.9365, "lon": 77.6644,
     "scene_id": None,
     "category": "lake_water",      "desc": "Bellandur Lake, Bengaluru"},
    {"tile_id": "hebbal_001",        "lat": 13.0360, "lon": 77.5880,
     "scene_id": None,
     "category": "lake_water",      "desc": "Hebbal Lake, Bengaluru"},
    {"tile_id": "sankey_001",        "lat": 13.0065, "lon": 77.5750,
     "scene_id": None,
     "category": "lake_water",      "desc": "Sankey Tank, Bengaluru"},
    {"tile_id": "hussain_sagar_001", "lat": 17.4239, "lon": 78.4738,
     "scene_id": None,
     "category": "lake_water",      "desc": "Hussain Sagar, Hyderabad"},
    # Category 2: Dense Urban Areas
    {"tile_id": "urban_001",         "lat": 12.9716, "lon": 77.5946,
     "scene_id": "S2C_MSIL2A_20260914T050651_R019_T43PGQ_20260914T100610",
     "category": "dense_urban",     "desc": "Central Bengaluru urban"},
    {"tile_id": "urban_002",         "lat": 12.9716, "lon": 77.5946,
     "scene_id": "S2C_MSIL2A_20260616T050651_R019_T43PGQ_20260616T100816",
     "category": "dense_urban",     "desc": "Central Bengaluru urban (June)"},
    {"tile_id": "urban_indiranagar", "lat": 12.9784, "lon": 77.6408,
     "scene_id": None,
     "category": "dense_urban",     "desc": "Indiranagar, Bengaluru"},
    {"tile_id": "urban_electronic_city", "lat": 12.8452, "lon": 77.6602,
     "scene_id": None,
     "category": "dense_urban",     "desc": "Electronic City, Bengaluru"},
    {"tile_id": "urban_hitech_city", "lat": 17.4435, "lon": 78.3772,
     "scene_id": None,
     "category": "dense_urban",     "desc": "HITEC City, Hyderabad"},
    {"tile_id": "urban_mumbai_bkc",  "lat": 19.0657, "lon": 72.8686,
     "scene_id": None,
     "category": "dense_urban",     "desc": "Bandra Kurla Complex, Mumbai"},
    {"tile_id": "blr_001",           "lat": 12.9716, "lon": 77.5946,
     "scene_id": None,
     "category": "dense_urban",     "desc": "Bengaluru city overview"},
    {"tile_id": "majestic_001",      "lat": 12.9766, "lon": 77.5713,
     "scene_id": None,
     "category": "dense_urban",     "desc": "Majestic / Krantivira Station area"},
    # Category 3: Vegetation / Parks / Forest
    {"tile_id": "veg_cubbon_park",   "lat": 12.9763, "lon": 77.5929,
     "scene_id": None,
     "category": "vegetation_park", "desc": "Cubbon Park, Bengaluru"},
    {"tile_id": "veg_lalbagh",       "lat": 12.9507, "lon": 77.5848,
     "scene_id": None,
     "category": "vegetation_park", "desc": "Lalbagh Botanical Garden, Bengaluru"},
    {"tile_id": "veg_bannerghatta",  "lat": 12.8000, "lon": 77.5770,
     "scene_id": None,
     "category": "vegetation_park", "desc": "Bannerghatta National Park"},
    {"tile_id": "veg_turahalli",     "lat": 12.8872, "lon": 77.5255,
     "scene_id": None,
     "category": "vegetation_park", "desc": "Turahalli Reserve Forest"},
    {"tile_id": "veg_nandi_hills",   "lat": 13.3702, "lon": 77.6835,
     "scene_id": None,
     "category": "vegetation_park", "desc": "Nandi Hills Forested Slopes"},
    # Category 4: Agricultural Land
    {"tile_id": "agri_mandya_001",   "lat": 12.5218, "lon": 76.8951,
     "scene_id": None,
     "category": "agricultural_land","desc": "Mandya irrigated fields"},
    {"tile_id": "agri_kolar_001",    "lat": 13.1367, "lon": 78.1292,
     "scene_id": None,
     "category": "agricultural_land","desc": "Kolar farmland"},
    {"tile_id": "agri_tumakuru_001", "lat": 13.3409, "lon": 77.1010,
     "scene_id": None,
     "category": "agricultural_land","desc": "Tumakuru crop fields"},
    {"tile_id": "agri_doddaballapur","lat": 13.2925, "lon": 77.5422,
     "scene_id": None,
     "category": "agricultural_land","desc": "Doddaballapur agricultural fields"},
    # Category 5: Open / Barren Land
    {"tile_id": "barren_quarry_001", "lat": 13.2100, "lon": 77.6300,
     "scene_id": None,
     "category": "open_barren_land","desc": "Granite quarry, north Bengaluru"},
    {"tile_id": "barren_chitradurga","lat": 14.2250, "lon": 76.3980,
     "scene_id": None,
     "category": "open_barren_land","desc": "Chitradurga dry barren terrain"},
    {"tile_id": "barren_devanahalli","lat": 13.2450, "lon": 77.7120,
     "scene_id": None,
     "category": "open_barren_land","desc": "Devanahalli open cleared ground"},
    {"tile_id": "barren_ramanagara", "lat": 12.7150, "lon": 77.2810,
     "scene_id": None,
     "category": "open_barren_land","desc": "Ramanagara rocky barren hill"},
    # Category 6: Mixed Urban + Vegetation
    {"tile_id": "mixed_iisc_campus", "lat": 13.0184, "lon": 77.5684,
     "scene_id": None,
     "category": "mixed_urban_vegetation","desc": "IISc Campus, Bengaluru"},
    {"tile_id": "mixed_jayanagar",   "lat": 12.9250, "lon": 77.5830,
     "scene_id": None,
     "category": "mixed_urban_vegetation","desc": "Jayanagar, Bengaluru"},
    {"tile_id": "mixed_koramangala", "lat": 12.9350, "lon": 77.6240,
     "scene_id": None,
     "category": "mixed_urban_vegetation","desc": "Koramangala, Bengaluru"},
]

# ── Build a dict for quick lookup by tile_id ──────────────────────────────────
_TARGET_DICT = {t["tile_id"]: t for t in TARGET_TILES}


def _compute_bbox(lat: float, lon: float, size_px: int = 100, pixel_size_m: float = 10.0) -> list[float]:
    """
    Compute approximate EPSG:4326 bbox for a tile centred at (lat, lon).
    Half-window = (size_px/2) * pixel_size_m = 500 m from centre.
    """
    half_m = (size_px / 2) * pixel_size_m          # 500 m
    half_lat = half_m / 111_320.0                   # degrees latitude
    half_lon = half_m / (111_320.0 * math.cos(math.radians(lat)))  # degrees lon
    return [
        round(lon - half_lon, 8),
        round(lat - half_lat, 8),
        round(lon + half_lon, 8),
        round(lat + half_lat, 8),
    ]


def build_metadata() -> list[dict]:
    """
    Scan data/satellite_tiles/*.png, match against TARGET_DICT for lat/lon,
    compute bbox, return complete metadata list.
    """
    records = []
    png_files = sorted(TILES_DIR.glob("*.png"))
    print(f"\nFound {len(png_files)} PNG files in {TILES_DIR}")

    for png in png_files:
        tid = png.stem  # filename without extension → tile_id
        img_path = f"data/satellite_tiles/{png.name}"

        if tid in _TARGET_DICT:
            t = _TARGET_DICT[tid]
            lat, lon = t["lat"], t["lon"]
        else:
            print(f"  [WARN] No target record for '{tid}' — skipping (cannot compute bbox).")
            continue

        bbox = _compute_bbox(lat, lon)
        record = {
            "tile_id": tid,
            "image_path": img_path,
            "scene_id": t.get("scene_id"),
            "center_lat": round(lat, 6),
            "center_lon": round(lon, 6),
            "crs": "EPSG:4326",
            "pixel_size_m": 10,
            "width_px": 100,
            "height_px": 100,
            "category": t.get("category", "unknown"),
            "description": t.get("desc", ""),
            "bbox": bbox,
        }
        records.append(record)
        print(f"  [OK]   {tid:30s}  bbox={[round(v,4) for v in bbox]}")

    return records


def build_faiss_index(metadata: list[dict]) -> int:
    """
    Load RemoteCLIP, embed all tile images, write FAISS index + metadata.
    Returns number of vectors indexed.
    """
    try:
        import faiss
        import numpy as np
    except ImportError as e:
        print(f"[ERROR] {e} — run: pip install faiss-cpu numpy")
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("[ERROR] Pillow not installed — run: pip install Pillow")
        sys.exit(1)

    print("\nLoading RemoteCLIP encoder (ViT-B-32)...")
    try:
        from app.services.search.remoteclip import RemoteCLIPEncoder
        encoder = RemoteCLIPEncoder.instance()
    except Exception as e:
        print(f"[ERROR] RemoteCLIP load failed: {e}")
        sys.exit(1)
    print("RemoteCLIP ready.\n")

    print("Embedding tiles...")
    vectors: list = []
    indexed_meta: list[dict] = []

    for i, tile in enumerate(metadata):
        img_path = PROJECT_ROOT / tile["image_path"]
        tid = tile["tile_id"]
        if not img_path.exists():
            print(f"  [{i+1}/{len(metadata)}] SKIP  {tid} — image not found")
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            emb = encoder.encode_image([img])   # (1, 512) normalized
            vectors.append(emb[0])
            indexed_meta.append(tile)
            print(f"  [{i+1}/{len(metadata)}] OK    {tid}")
        except Exception as e:
            print(f"  [{i+1}/{len(metadata)}] ERROR {tid} — {e}")

    n_embedded = len(vectors)
    print(f"\nSuccessfully embedded: {n_embedded} / {len(metadata)} tiles")

    if not vectors:
        print("[ERROR] No tiles embedded — aborting.")
        sys.exit(1)

    print("\nBuilding FAISS IndexFlatIP (dim=512)...")
    matrix = np.stack(vectors, axis=0).astype(np.float32)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(matrix)
    print(f"FAISS index size: {index.ntotal} vectors, dim={index.d}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"Saved FAISS index : {INDEX_FILE}")

    with open(INDEX_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(indexed_meta, f, indent=2, ensure_ascii=False)
    print(f"Saved tile metadata: {INDEX_METADATA_FILE}")

    return n_embedded


def verify_retrieval(top_k: int = 5) -> None:
    """
    Reload the freshly built index and run the benchmark water query to confirm
    retrieval works and returns Top-K semantically diverse results.
    """
    print("\n" + "="*60)
    print("RETRIEVAL VERIFICATION")
    print("Query: 'Find a lake around Bengaluru and calculate its water extent'")
    print("="*60)

    try:
        import faiss, json, numpy as np
        from app.services.search.remoteclip import RemoteCLIPEncoder

        # Force reload singletons
        from app.services.search.remoteclip import RemoteCLIPEncoder
        from app.services.search.faiss_store import FAISSStore
        FAISSStore.reset()

        store = FAISSStore.instance(
            index_path=str(INDEX_FILE),
            metadata_path=str(INDEX_METADATA_FILE),
        )
        encoder = RemoteCLIPEncoder.instance()

        visual_query = "water body lake"
        vec = encoder.encode_text([visual_query])  # (1, 512)
        candidates = store.search(vec, top_k=top_k)

        print(f"\nTop-{top_k} retrieval results for visual_query='{visual_query}':\n")
        print(f"  {'Rank':<5} {'tile_id':<30} {'score':>8}  category")
        print(f"  {'-'*4}  {'-'*29}  {'-'*8}  {'-'*25}")

        with open(INDEX_METADATA_FILE) as f:
            all_meta = {m["tile_id"]: m for m in json.load(f)}

        for rank, c in enumerate(candidates, 1):
            meta = all_meta.get(c["tile_id"], {})
            cat = meta.get("category", "?")
            print(f"  {rank:<5} {c['tile_id']:<30} {c['score']:>8.4f}  {cat}")

        print()
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        import traceback; traceback.print_exc()


def main() -> None:
    print("=" * 60)
    print("GeoQueryAI — Full FAISS Index Rebuild")
    print("=" * 60)

    # Step 1: Build comprehensive metadata for all discovered tiles
    print("\n[STEP 1] Building metadata for all satellite tiles...")
    metadata = build_metadata()

    tiles_found = len(list(TILES_DIR.glob("*.png")))
    print(f"\n  PNG files found : {tiles_found}")
    print(f"  Metadata records: {len(metadata)}")

    # Save updated data/satellite_tiles/metadata.json
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"\nSaved: {METADATA_FILE}")

    # Step 2: Build FAISS index
    print("\n[STEP 2] Building FAISS index with RemoteCLIP image embeddings...")
    n_indexed = build_faiss_index(metadata)

    # Step 3: Verify retrieval
    print("\n[STEP 3] Verifying retrieval on water query...")
    verify_retrieval(top_k=5)

    # Summary
    print("=" * 60)
    print("REBUILD COMPLETE")
    print(f"  PNG tiles found       : {tiles_found}")
    print(f"  Metadata records      : {len(metadata)}")
    print(f"  Tiles embedded        : {n_indexed}")
    print(f"  FAISS vectors (ntotal): {n_indexed}")
    print(f"  Embedding dimension   : {EMBEDDING_DIM}")
    print(f"  Index file            : {INDEX_FILE}")
    print(f"  Metadata file         : {INDEX_METADATA_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
