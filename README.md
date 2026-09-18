# GeoQueryAI — Phase 3: RemoteCLIP + FAISS Semantic Retrieval

GeoQueryAI is an AI-powered geospatial analysis platform for natural-language satellite imagery queries, visual similarity search, tile indexing, and geospatial processing.

---

## Architecture Overview

```
User Query (Text) ─────────► RemoteCLIP Encoder (ViT-B-32)
                                   │
                                   ▼ [1, 512] Normalized Vector
                                   │
                                   ▼
                         FAISS FlatIP Index (indexes/satellite.index)
                                   │
                                   ▼ Top-K Cosine Similarity
                         Candidate Satellite Tiles (BBox, Tile ID, Score)
```

---

## Key Features

1. **RemoteCLIP Encoder (`app/services/search/remoteclip.py`)**:
   - Singleton loaded with `models/RemoteCLIP-ViT-B-32.pt`.
   - Generates L2-normalized 512-dimensional text and image embeddings.
   - Supports prompt ensemble averaging across domain-specific templates.
2. **FAISS Index Manager (`app/services/search/faiss_store.py`)**:
   - Manages inner-product (`IndexFlatIP`) vector retrieval.
   - Synchronizes vector IDs with satellite tile metadata (`indexes/tile_metadata.json`).
3. **Index Builder CLI (`scripts/build_faiss_index.py`)**:
   - Generates vector embeddings for images in `data/satellite_tiles/` and saves index binaries.
4. **Smoke Test CLI (`scripts/test_semantic_search.py`)**:
   - Validates end-to-end RemoteCLIP + FAISS search against the active index.
5. **Deterministic Mock Fallback (`MOCK_SEARCH=true`)**:
   - Guarantees complete operational fallback when real tile index data is not yet present.

---

## Quick Start

### 1. Requirements & Setup

Ensure Python 3.11 and the required dependencies are installed in `.venv`:

```bash
# Verify environment
.venv\Scripts\python.exe -m pip list
```

### 2. Running Unit & Integration Tests

```bash
# Run full RemoteCLIP & FAISS test suite (19 tests)
.venv\Scripts\python.exe -m pytest backend/tests/test_remoteclip.py -v
```

### 3. Building the FAISS Index

Add satellite tile images to `data/satellite_tiles/` and list them in `data/satellite_tiles/metadata.json`:

```json
[
  {
    "tile_id": "tile_001",
    "path": "data/satellite_tiles/tile_001.png",
    "bbox": [77.5946, 12.9716, 77.6046, 12.9816],
    "crs": "EPSG:4326",
    "resolution_m": 10.0,
    "captured_at": "2024-01-15T10:30:00Z"
  }
]
```

Build the index:

```bash
.venv\Scripts\python.exe scripts/build_faiss_index.py
```

### 4. Testing Semantic Search

```bash
.venv\Scripts\python.exe scripts/test_semantic_search.py "large water bodies and lakes"
```

### 5. Running the API Server

```bash
# Set MOCK_SEARCH=false in .env to use real RemoteCLIP + FAISS
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
```

---

## Configuration (`backend/app/core/config.py`)

| Setting | Default | Description |
|---|---|---|
| `MOCK_SEARCH` | `true` | `true` for mock candidates, `false` for real RemoteCLIP + FAISS |
| `REMOTECLIP_MODEL_PATH` | `models/RemoteCLIP-ViT-B-32.pt` | Path to RemoteCLIP PyTorch checkpoint |
| `FAISS_INDEX_PATH` | `indexes/satellite.index` | Path to FAISS binary index |
| `TILE_METADATA_PATH` | `indexes/tile_metadata.json` | Path to metadata index |
| `PROMPT_ENSEMBLE` | `false` | Enable multi-prompt template averaging |
