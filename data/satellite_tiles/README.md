# Satellite Tiles

This directory holds the RGB satellite tile images used for semantic retrieval.

## Metadata Schema

Each tile must be registered in `metadata.json` (this directory):

```json
[
  {
    "tile_id": "tile_001",
    "path": "data/satellite_tiles/tile_001.png",
    "bbox": [77.50, 12.90, 77.65, 13.05],
    "source": "sentinel-2",
    "date": "2025-06-15"
  }
]
```

### Required fields

| Field     | Type           | Description                                               |
|-----------|----------------|-----------------------------------------------------------|
| `tile_id` | `str`          | Unique identifier (used in API responses)                 |
| `path`    | `str`          | Path to the image file, relative to the project root      |
| `bbox`    | `[float × 4]`  | `[min_lon, min_lat, max_lon, max_lat]` in EPSG:4326       |

### Optional fields

| Field    | Type   | Description                  |
|----------|--------|------------------------------|
| `source` | `str`  | Data source (e.g. sentinel-2)|
| `date`   | `str`  | Acquisition date (YYYY-MM-DD)|

## Supported image formats

- PNG (recommended)
- JPEG / JPG
- TIFF (single-band or 3-band RGB; multi-spectral not yet supported)

## Adding tiles

1. Copy your satellite tile images into this directory.
2. Add an entry for each tile to `metadata.json`.
3. Run the index builder from the project root:

```powershell
python scripts/build_faiss_index.py
```

4. Set `MOCK_SEARCH=false` in your `.env` file.
5. Restart the backend server.

## Current status

`metadata.json` is currently empty. Real semantic retrieval requires real tile files.
