# FIXES

What was changed to make this project run. Grouped by severity.

---

## Blocking — the app could not start

**1. Two imported modules did not exist.**
`backend/app/api/routes/query.py` imported:

```python
from app.services.geo.geocoding_interface import geocode
from app.services.geo.interface import analyze
```

Neither file existed, so importing `app.main` raised `ModuleNotFoundError` and
the server could not boot at all. Both were written:

- `geo/geocoding_interface.py` — `geocode(location) -> GeographicResult`.
  Mock mode resolves against a built-in gazetteer; real mode wraps the existing
  Nominatim `geocoding_service`, adapting its dict bbox into the schema's
  `[min_lon, min_lat, max_lon, max_lat]` list form.
- `geo/interface.py` — `analyze(bbox, analysis, start_date, end_date) -> AnalysisResult`,
  the single dispatch point for all six analysis types, with both a mock path and
  a real STAC/COG path built on your existing ndvi/ndwi/stac/change services.

**2. Every geo service imported a package that does not exist.**
`stac_service.py`, `ndvi_service.py`, `ndwi_service.py` and
`change_detection_service.py` all did:

```python
from backend.app.services.geoquery.raster_service import ...
```

The directory is `geo`, not `geoquery`, and the `backend.` prefix is wrong
because `backend/` is the package root, not a package itself. Corrected to
`from app.services.geo.raster_service import ...`.

**3. `app/services/main.py` was a second, dead FastAPI app.**
An older standalone app carrying the same broken `geoquery` imports, competing
with `app/main.py`. Rather than deleting the work, its three endpoints were
ported into a proper router at `app/api/routes/maps.py`
(`/api/maps/ndvi`, `/api/maps/ndwi`, `/api/maps/change-detection`), mounted on
the real app, with the `/maps` static directory mounted in `main.py`. There is
now exactly one application object.

---

## Latent — would fail at runtime

**4. `create_tiles()` in `geocoding_service.py` was dead code.**
It used `Path` and `Image`, neither imported — an immediate `NameError` on call.
Its `__main__` block called `search_sentinel_scene()` and `create_rgb_composite()`,
two functions that exist nowhere in the project. Tile-cutting also has nothing
to do with geocoding. Moved to `app/services/search/tiling.py` with correct
imports, input validation, a real CLI, and verified working against Pillow.

**5. `requirements.txt` was missing four packages the code imports:**
`matplotlib` (visualization_service), `pillow` (tiling), `pytest`, and `httpx`
(required by `fastapi.testclient`). Installing from the old file and starting the
server would fail. Optional ML extras are now separated and documented.

**6. Malformed input returned 503 instead of 422.**
A bad bbox is a client error, not a service outage — and `analysis.py` already
documented 422. Added `InvalidAOIError` (422) in `core/errors.py`, used for bbox
shape/ordering/range and unknown analysis type. `query.py` catches it so a bad
candidate bbox degrades gracefully instead of failing the whole pipeline.

---

## Test suite

**7. Root `test_remoteclip.py` broke collection.** A scratch script doing
`import torch` at module level, which pytest collected as a test and errored on.
Its filename also collided with the real `backend/tests/test_remoteclip.py`.
Moved to `scripts/verify_remoteclip_checkpoint.py` and documented as a manual
diagnostic.

**8. FAISS tests had no skip guard.** The `TestFAISSStoreErrors` and
`TestFAISSStoreSearch` classes use `faiss` directly, so the suite errored out
when the optional `faiss-cpu` was absent. Both now carry a `@faiss_required`
skip marker. (The RemoteCLIP integration tests were already guarded on the
checkpoint's presence.)

**9. `conftest.py` assumed `app` was already importable.** Now bootstraps
`backend/` onto `sys.path`, so pytest works from any directory. Added
`pytest.ini` with `testpaths`, `pythonpath`, and registration of the
`integration` marker (previously an unknown-mark warning).

---

## Robustness

**10. `.env` discovery was CWD-dependent.** `config.py` used a bare `.env`, so
behaviour changed depending on where you launched. Now searches project root,
then `backend/`, then CWD.

**11. `CORS_ORIGINS` could not be set as a plain list.** `cors_origins: list[str]`
made pydantic-settings JSON-decode the value, so
`CORS_ORIGINS=http://localhost:3000,http://localhost:5173` crashed startup.
A `field_validator` now accepts both JSON arrays and comma-separated strings.

**12. Housekeeping.** Removed the bundled 
Windows `venv/` (the whole 360 KB of source was buried under it) and `.git/`,
stripped stale Python 3.14 `.pyc` files, normalised CRLF to LF, added
`.env.example`, and rewrote the README (it documented Windows-only
`.venv\Scripts\python.exe` commands and omitted the map endpoints).

---

## Not fixed — missing assets

Real mode still cannot run, and no code change can fix this. The repository has
no `models/` directory (RemoteCLIP weights), no `indexes/satellite.index`, and
no tile PNGs in `data/satellite_tiles/` — only two sample metadata entries.
`MOCK_SEARCH=false` will fail until you supply the weights and build the index;
see the README. Mock mode is fully functional in the meantime.
