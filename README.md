# GeoQueryAI

Natural-language Earth-observation search and analysis. Submit a free-text query,
get semantically matched Sentinel-2 tiles plus computed geospatial metrics
(NDVI, NDWI, water extent, change detection) and a grounded explanation.

**The project runs end-to-end out of the box in mock mode** — no API keys, no
model weights, no network access.

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run the test suite
python -m pytest

# 4. Start the server
uvicorn app.main:app --reload --app-dir backend
```

Then open <http://127.0.0.1:8000/docs> for interactive API documentation.

```bash
# Smoke-test the pipeline
curl -X POST http://127.0.0.1:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me water bodies around Bengaluru"}'
```

Configuration is optional: `cp .env.example .env` and edit. Every setting has a
working default.

---

## Pipeline

```
POST /api/query
   │
   ├─ 1. parse_query()          Gemini → StructuredQuery
   │                            (visual_query, location, analysis, dates)
   ├─ 2. geocode()              place name → lat/lon/bbox
   ├─ 3. semantic_search()      RemoteCLIP text embedding → FAISS top-K tiles
   ├─ 4. validate_candidates()  spatial consistency filter → selected tile
   ├─ 5. analyze()              STAC scene search → COG raster computation
   ├─ 6. Evidence layer         only values actually computed
   └─ 7. explain_evidence()     Gemini, constrained strictly to the Evidence
```

**Geographic design rule:** coordinates from the LLM are never used as an AOI.
The only trusted coordinates are `SemanticCandidate.bbox` from the search layer,
after the validator confirms spatial consistency. Step 4 will pick a lower-ranked
tile over a higher-scoring one if only the former actually covers the location.

**Score semantics:** `SemanticCandidate.score` is cosine similarity in [0, 1].
It is not a probability, confidence, or accuracy figure.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/query` | Full end-to-end pipeline |
| `POST` | `/api/search` | Semantic retrieval in isolation |
| `POST` | `/api/analyze` | Raster analysis on a known bbox |
| `POST` | `/api/maps/ndvi` | NDVI stats + rendered PNG map |
| `POST` | `/api/maps/ndwi` | NDWI stats + rendered PNG map |
| `POST` | `/api/maps/change-detection` | Two-period NDVI change |
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/docs` | OpenAPI UI |

Rendered maps are served from `/maps/<file>.png`. The `/api/maps/*` endpoints
have no mock path — they need `MOCK_GEO=false` and network access, and return
503 with a clear message otherwise.

Analysis types: `discovery`, `ndvi`, `ndwi`, `water_extent`, `change`,
`water_extent_change`.

---

## Mock vs real mode

Each stage switches independently, so you can enable one real service at a time.

| Flag | Default | `false` requires |
|---|---|---|
| `MOCK_AI` | `true` | `GEMINI_API_KEY`, `pip install google-generativeai` |
| `MOCK_SEARCH` | `true` | RemoteCLIP weights + built FAISS index, `pip install torch open-clip-torch faiss-cpu` |
| `MOCK_GEO` | `true` | Network access (Nominatim + Planetary Computer) |

What mock mode gives you:

- **`MOCK_AI`** — a fixed `StructuredQuery`, and an explanation templated from
  the real Evidence values.
- **`MOCK_SEARCH`** — five deterministic Bengaluru-area tiles.
- **`MOCK_GEO`** — a small built-in gazetteer (Bengaluru, Ulsoor Lake, Mumbai,
  Chennai, Delhi, Hyderabad, Cauvery delta), and synthetic analysis metrics
  derived from the bbox so repeated runs agree.

> Mock metrics are **not measurements**. Every mock `AnalysisResult` carries a
> `message` beginning with `[MOCK]`. An unrecognised place name returns
> `grounded: false`, which the validator treats as "no spatial constraint"
> rather than an error.

---

## Enabling real semantic search

Real mode needs three assets that are **not** in this repository:

1. **Model weights** — download `RemoteCLIP-ViT-B-32.pt` into `models/`.
   Not distributable via PyPI; get it from the RemoteCLIP authors' release.
2. **Tile images** — put satellite tiles in `data/satellite_tiles/` and list them
   in `data/satellite_tiles/metadata.json`. To cut a large scene into
   224 px tiles with 20 % overlap:
   ```bash
   python -m app.services.search.tiling path/to/scene.png --output-dir data/satellite_tiles
   ```
3. **FAISS index** — build it, then verify:
   ```bash
   python scripts/build_faiss_index.py
   python scripts/test_semantic_search.py "large water bodies and lakes"
   ```

`indexes/tile_metadata.json` ships with two sample entries, but
`indexes/satellite.index` and the tile PNGs are absent, so `MOCK_SEARCH=false`
will fail until you build them. The FAISS index and metadata are positional:
entry *i* must match vector *i*. Never hand-edit the metadata after building.

Then set `MOCK_SEARCH=false` and restart. The encoder and index are pre-warmed
at startup, so the first request has no cold-start cost.

---

## Project layout

```
backend/app/
  main.py                    FastAPI app, CORS, error handlers, static mount
  core/                      config (env/.env) and typed HTTP errors
  schemas/                   Pydantic contracts between every stage
  api/routes/                query · search · analysis · maps
  services/
    ai/interface.py          parse_query() · explain_evidence()      [Gemini]
    geo/
      interface.py           analyze()  ← single raster dispatch point
      geocoding_interface.py geocode()  ← single geocoding entry point
      stac_service.py        Planetary Computer scene search + cloud screening
      raster_service.py      windowed COG reads, AOI masking, GeoTIFF export
      ndvi_service.py        NDVI + SCL cloud masking
      ndwi_service.py        NDWI
      change_detection_service.py   two-period NDVI comparison
      visualization_service.py      PNG colour maps
    search/
      interface.py           semantic_search()  ← single retrieval entry point
      remoteclip.py          RemoteCLIP ViT-B-32 encoder (singleton)
      faiss_store.py         FAISS IndexFlatIP + tile metadata (singleton)
      validator.py           spatial consistency filter
      tiling.py              scene → 224 px overlapping tiles
scripts/                     index builder · search smoke test · checkpoint check
backend/tests/               pipeline + RemoteCLIP/FAISS test suites
```

The `*/interface.py` modules are the plug-in points. Swap an implementation
there and the routes need no changes.

---

## Testing

```bash
python -m pytest                              # full suite
python -m pytest -m "not integration"         # skip checkpoint-dependent tests
```

Tests force all three mock flags on, so they need no network, keys, or weights.
Tests requiring `faiss-cpu` self-skip when it is not installed; those requiring
the RemoteCLIP checkpoint self-skip when `models/RemoteCLIP-ViT-B-32.pt` is absent.
