#!/usr/bin/env python
"""
GeoQueryAI — FAISS Index Builder.

Reads satellite tile metadata, generates RemoteCLIP image embeddings,
and saves the FAISS index + ordered metadata to indexes/.

Usage (run from the project root with your .venv active):
    python scripts/build_faiss_index.py

Output:
    indexes/satellite.index      — faiss.IndexFlatIP(512) binary
    indexes/tile_metadata.json   — ordered list of tile dicts

The two output files must stay in sync: FAISS position i maps to
tile_metadata.json[i]. Never edit tile_metadata.json by hand after building.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Make sure the backend package is importable ───────────────────────────────
# This script lives at <project_root>/scripts/build_faiss_index.py
# The backend package lives at <project_root>/backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# ── Paths (relative to project root) ─────────────────────────────────────────
METADATA_FILE = PROJECT_ROOT / "data" / "satellite_tiles" / "metadata.json"
INDEX_DIR = PROJECT_ROOT / "indexes"
INDEX_FILE = INDEX_DIR / "satellite.index"
INDEX_METADATA_FILE = INDEX_DIR / "tile_metadata.json"
EMBEDDING_DIM = 512


def main() -> None:
    print("=" * 60)
    print("GeoQueryAI — FAISS Index Builder")
    print("=" * 60)

    # ── Import dependencies ────────────────────────────────────────────────────
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        print(f"\n[ERROR] Missing dependency: {exc}")
        print("  Run: pip install faiss-cpu numpy")
        sys.exit(1)

    # ── Load tile metadata ─────────────────────────────────────────────────────
    print(f"\nReading tile metadata from: {METADATA_FILE}")
    if not METADATA_FILE.exists():
        print(f"[ERROR] metadata.json not found at {METADATA_FILE}")
        print("  Create data/satellite_tiles/metadata.json with your tile list.")
        sys.exit(1)

    with open(METADATA_FILE, encoding="utf-8") as f:
        all_tiles: list[dict] = json.load(f)

    print(f"Found {len(all_tiles)} tile(s) in metadata.")

    if not all_tiles:
        print("\n[WARNING] metadata.json is empty — no tiles to index.")
        print(
            "  Add entries to data/satellite_tiles/metadata.json and re-run.\n"
            "  See data/satellite_tiles/README.md for the schema."
        )
        # Write empty index + metadata so the FAISSStore loads cleanly
        _write_empty_index(faiss)
        return

    # ── Load RemoteCLIP encoder ────────────────────────────────────────────────
    print("\nLoading RemoteCLIP encoder...")
    try:
        from app.services.search.remoteclip import RemoteCLIPEncoder
        encoder = RemoteCLIPEncoder.instance()
    except Exception as exc:
        print(f"[ERROR] Failed to load RemoteCLIP: {exc}")
        sys.exit(1)
    print("RemoteCLIP ready.\n")

    # ── Encode tiles ───────────────────────────────────────────────────────────
    print("Encoding tiles...")
    try:
        from PIL import Image
    except ImportError:
        print("[ERROR] Pillow is not installed. Run: pip install Pillow")
        sys.exit(1)

    vectors = []
    indexed_metadata = []

    for i, tile in enumerate(all_tiles):
        tile_id = tile.get("tile_id", f"tile_{i:04d}")
        img_path_raw = tile.get("image_path", tile.get("path", ""))

        img_path = Path(img_path_raw)
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path_raw

        if not img_path.exists():
            print(f"  [{i+1}/{len(all_tiles)}] SKIP  {tile_id} — image not found: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            embedding = encoder.encode_image([img])   # (1, 512), normalized
            vectors.append(embedding[0])
            indexed_metadata.append(tile)
            print(f"  [{i+1}/{len(all_tiles)}] OK    {tile_id}")
        except Exception as exc:
            print(f"  [{i+1}/{len(all_tiles)}] ERROR {tile_id} — {exc}")

    print(f"\nSuccessfully encoded {len(vectors)} / {len(all_tiles)} tile(s).")

    if not vectors:
        print(
            "\n[WARNING] No tiles were encoded (all images missing or errored).\n"
            "  Check that image paths in metadata.json are correct."
        )
        _write_empty_index(faiss)
        return

    # ── Build FAISS index ──────────────────────────────────────────────────────
    print("\nBuilding FAISS index (IndexFlatIP, dim=512)...")
    import numpy as np  # noqa: F811 (already imported, safe)
    matrix = np.stack(vectors, axis=0).astype(np.float32)  # (N, 512)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(matrix)
    print(f"Index size: {index.ntotal} vectors")

    # ── Save outputs ───────────────────────────────────────────────────────────
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))
    print(f"Saved: {INDEX_FILE}")

    with open(INDEX_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(indexed_metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved: {INDEX_METADATA_FILE}")

    print("\n" + "=" * 60)
    print("Index build complete.")
    print(f"  Tiles indexed : {index.ntotal}")
    print(f"  Index file    : {INDEX_FILE}")
    print(f"  Metadata file : {INDEX_METADATA_FILE}")
    print("=" * 60)
    print("\nTo enable real search, set MOCK_SEARCH=false in your .env file.")


def _write_empty_index(faiss) -> None:
    """Write an empty FAISS index + empty metadata so FAISSStore can load."""
    import numpy as np

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    faiss.write_index(index, str(INDEX_FILE))

    with open(INDEX_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

    print(f"Saved empty index: {INDEX_FILE}")
    print(f"Saved empty metadata: {INDEX_METADATA_FILE}")
    print("\nReal retrieval is BLOCKED — add tile images and re-run this script.")


if __name__ == "__main__":
    main()
