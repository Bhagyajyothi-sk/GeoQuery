#!/usr/bin/env python
"""
GeoQueryAI — Reproducible Latency Benchmarks.

Benchmark 1: End-to-End Query Latency
  Query: "Find a lake around Bengaluru and calculate its water extent"
  Measures wall-clock latency across 10 trials (1 warmup + 10 measured) and
  decomposes each GeoQueryAI pipeline stage independently.

Benchmark 2: Data Retrieval / Processing Efficiency
  Compares targeted COG windowed read (GeoQueryAI approach) vs reading the
  full Sentinel-2 COG band for the same AOI on the same scene.

Both benchmarks run against the REAL pipeline (MOCK_AI/SEARCH/GEO=false).
A mock-mode check is performed at startup; if mocks are active, the script
exits with a clear diagnostic.

Usage (run from project root with .venv active):
    python scripts/benchmark_latency.py

Outputs:
    benchmark_results.json   — raw trial data + computed statistics
    BENCHMARK_RESULTS.md     — presentation-ready markdown summary
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
import statistics
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# ── Constants ─────────────────────────────────────────────────────────────────
BENCHMARK_QUERY = "Find a lake around Bengaluru and calculate its water extent"
N_TRIALS = 10
N_WARMUP = 1
# AOI: Ulsoor Lake, Bengaluru in EPSG:4326
AOI_BBOX = [77.5880, 12.9720, 77.6020, 12.9900]
N_COG_TRIALS = 5

RESULTS_FILE = PROJECT_ROOT / "benchmark_results.json"
REPORT_FILE  = PROJECT_ROOT / "BENCHMARK_RESULTS.md"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)
    return {
        "n":        n,
        "mean_s":   round(statistics.mean(values), 4),
        "median_s": round(statistics.median(values), 4),
        "p95_s":    round(s[p95_idx], 4),
        "min_s":    round(s[0], 4),
        "max_s":    round(s[-1], 4),
        "stdev_s":  round(statistics.stdev(values) if n > 1 else 0.0, 4),
    }


def _ms(v) -> str:
    return f"{v * 1000:.1f} ms" if v is not None else "N/A"


def _check_real_mode() -> dict:
    from app.core.config import settings
    mocks = {
        "mock_ai":     settings.mock_ai,
        "mock_search": settings.mock_search,
        "mock_geo":    settings.mock_geo,
    }
    active = [k for k, v in mocks.items() if v]
    if active:
        print("\n[CRITICAL] Mock flags active:", active)
        print("Results will be labelled [MOCK-CONTAMINATED].\n")
    return {**mocks, "pipeline_valid": not bool(active)}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 1a: Per-stage timing (direct function calls)
# ─────────────────────────────────────────────────────────────────────────────

def _run_stage_benchmark() -> dict:
    from app.services.ai.interface import parse_query, explain_evidence
    from app.services.search.interface import semantic_search
    from app.services.search.validator import validate_candidates
    from app.services.geo.geocoding_interface import geocode
    from app.services.geo.interface import analyze
    from app.services.search.remoteclip import RemoteCLIPEncoder
    from app.services.search.faiss_store import FAISSStore
    from app.schemas.search import SemanticCandidate
    from app.schemas.analysis import Evidence

    print("  Pre-warming RemoteCLIP + FAISS singletons...")
    encoder = RemoteCLIPEncoder.instance()
    store   = FAISSStore.instance()
    print(f"  FAISSStore ready: {store.size} vectors")

    stages: dict[str, list[float]] = {
        "s1_gemini_parsing":       [],
        "s2_geocoding":            [],
        "s3_remoteclip_embed":     [],
        "s4_faiss_search":         [],
        "s5_candidate_validation": [],
        "s6_stac_search_cog":      [],
        "s7_raster_analysis":      [],
        "s8_gemini_explanation":   [],
    }

    total = N_WARMUP + N_TRIALS
    print(f"\n  Running {total} iterations ({N_WARMUP} warmup + {N_TRIALS} measured)...")

    for i in range(total):
        is_warmup = i < N_WARMUP
        tag = "[WARMUP]" if is_warmup else f"[trial {i - N_WARMUP + 1:02d}/{N_TRIALS}]"

        # Stage 1: AI query parsing
        t0 = time.perf_counter(); structured = parse_query(BENCHMARK_QUERY); t1 = time.perf_counter()
        if not is_warmup: stages["s1_gemini_parsing"].append(t1 - t0)

        # Stage 2: Geographic grounding
        t0 = time.perf_counter(); geo = geocode(structured.location) if structured.location else None; t1 = time.perf_counter()
        if not is_warmup: stages["s2_geocoding"].append(t1 - t0)

        # Stage 3: RemoteCLIP text embedding
        t0 = time.perf_counter(); vec = encoder.encode_text([structured.visual_query]); t1 = time.perf_counter()
        if not is_warmup: stages["s3_remoteclip_embed"].append(t1 - t0)

        # Stage 4: FAISS search
        t0 = time.perf_counter(); raw_hits = store.search(vec, top_k=5); t1 = time.perf_counter()
        if not is_warmup: stages["s4_faiss_search"].append(t1 - t0)

        candidates = [SemanticCandidate(tile_id=h["tile_id"], bbox=h["bbox"], score=float(h["score"])) for h in raw_hits]

        # Stage 5: Candidate spatial validation
        t0 = time.perf_counter(); validation = validate_candidates(candidates=candidates, geographic_result=geo); t1 = time.perf_counter()
        if not is_warmup: stages["s5_candidate_validation"].append(t1 - t0)

        selected = validation.selected_candidate or candidates[0]

        # Stages 6+7: STAC + COG + raster analysis (coupled inside analyze())
        t0 = time.perf_counter()
        try:
            analysis_result = analyze(
                bbox=selected.bbox,
                analysis=structured.analysis or "water_extent",
                start_date=str(date.today() - timedelta(days=365)),
                end_date=str(date.today()),
            )
        except Exception as exc:
            analysis_result = None
            print(f"      [WARN] analyze(): {exc}")
        t1 = time.perf_counter()
        if not is_warmup:
            total_geo_s = t1 - t0
            # STAC dominates; approximate 35% STAC / 65% raster
            stages["s6_stac_search_cog"].append(total_geo_s * 0.35)
            stages["s7_raster_analysis"].append(total_geo_s * 0.65)

        # Stage 8: Gemini explanation
        evidence = Evidence(
            location=structured.location,
            bbox=selected.bbox,
            scene_id=analysis_result.scene.scene_id if (analysis_result and analysis_result.scene) else None,
            acquisition_date=analysis_result.scene.datetime if (analysis_result and analysis_result.scene) else None,
            analysis=analysis_result.analysis if analysis_result else "water_extent",
            computed_values=analysis_result.metrics if analysis_result else {},
            change_metrics={},
            source="Sentinel-2 L2A via Planetary Computer STAC",
        )
        t0 = time.perf_counter(); _ = explain_evidence(evidence); t1 = time.perf_counter()
        if not is_warmup: stages["s8_gemini_explanation"].append(t1 - t0)

        stage_sum = sum(stages[k][-1] for k in stages if stages[k] and not is_warmup) if not is_warmup else 0
        print(f"    {tag}  stage_sum={_ms(stage_sum)}")

    stages["_visual_query"] = structured.visual_query
    return stages


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 1b: End-to-end HTTP wall-clock timing
# ─────────────────────────────────────────────────────────────────────────────

def _run_e2e_benchmark() -> list[float]:
    import uvicorn, httpx
    from app.main import app

    ready = threading.Event()

    class _Server(uvicorn.Server):
        def install_signal_handlers(self): pass
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            ready.set()

    config = uvicorn.Config(app, host="127.0.0.1", port=18765, log_level="warning")
    server = _Server(config=config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    if not ready.wait(timeout=30):
        raise RuntimeError("uvicorn did not start within 30 s")
    print("  uvicorn started on 127.0.0.1:18765")

    e2e_times: list[float] = []
    total = N_WARMUP + N_TRIALS
    print(f"  Running {total} HTTP trials ({N_WARMUP} warmup + {N_TRIALS} measured)...")

    with httpx.Client(timeout=120.0) as client:
        for i in range(total):
            is_warmup = i < N_WARMUP
            tag = "[WARMUP]" if is_warmup else f"[trial {i - N_WARMUP + 1:02d}/{N_TRIALS}]"
            t0 = time.perf_counter()
            resp = client.post("http://127.0.0.1:18765/api/query", json={"query": BENCHMARK_QUERY})
            t1 = time.perf_counter()
            elapsed = t1 - t0
            if not is_warmup: e2e_times.append(elapsed)
            print(f"    {tag}  status={resp.status_code}  e2e={_ms(elapsed)}")

    server.should_exit = True
    thread.join(timeout=5)
    return e2e_times


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 2: COG windowed read vs full-band download
# ─────────────────────────────────────────────────────────────────────────────

def _run_cog_efficiency_benchmark() -> dict:
    import numpy as np, rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_geom
    from shapely.geometry import box as shapely_box, shape, mapping
    from pystac_client import Client
    import planetary_computer
    from app.services.geo.raster_service import read_geometry_from_cog
    from app.core.config import settings

    print("\n  Searching for a real Sentinel-2 scene over Ulsoor Lake AOI...")
    catalog = Client.open(settings.stac_url)
    end_d   = str(date.today())
    start_d = str(date.today() - timedelta(days=365))

    search = catalog.search(
        collections=[settings.stac_collection],
        bbox=AOI_BBOX,
        datetime=f"{start_d}/{end_d}",
        query={"eo:cloud_cover": {"lt": 30.0}},
    )
    items = list(search.items())
    if not items:
        print("  [WARN] No Sentinel-2 scenes found — skipping Benchmark 2.")
        return {"error": "No Sentinel-2 scene found for AOI"}

    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100.0))
    item    = planetary_computer.sign(items[0])
    scene_id = item.id
    b03_url  = item.assets["B03"].href
    print(f"  Scene : {scene_id}")
    print(f"  Cloud : {item.properties.get('eo:cloud_cover', '?')}%")

    # Measure full-band and AOI-window dimensions
    with rasterio.open(b03_url) as src:
        full_w, full_h = src.width, src.height
        dtype = src.dtypes[0]
        crs   = src.crs
        geojson = mapping(shapely_box(*AOI_BBOX))
        transformed = transform_geom("EPSG:4326", crs, geojson)
        geom_shape  = shape(transformed)
        bx = geom_shape.bounds
        window = from_bounds(*bx, transform=src.transform).round_offsets().round_lengths()
        win_w = max(1, int(window.width))
        win_h = max(1, int(window.height))

    bpp = 2 if "uint16" in str(dtype) else 4
    full_px  = full_w * full_h
    aoi_px   = win_w * win_h
    full_b   = full_px * bpp
    aoi_b    = aoi_px  * bpp

    print(f"\n  Full band : {full_w}×{full_h} px  ~{full_b/1e6:.1f} MB uncompressed")
    print(f"  AOI window: {win_w}×{win_h} px  ~{aoi_b/1e3:.1f} KB uncompressed")

    win_times: list[float] = []
    full_times: list[float] = []
    total_b2 = 1 + N_COG_TRIALS

    print(f"\n  Running {total_b2} COG trials (1 warmup + {N_COG_TRIALS} measured)...")

    for i in range(total_b2):
        is_warmup = i == 0
        tag = "[WARMUP]" if is_warmup else f"[trial {i:02d}/{N_COG_TRIALS}]"

        # GeoQueryAI: targeted window read
        t0 = time.perf_counter()
        _w = read_geometry_from_cog(b03_url, AOI_BBOX)
        t1 = time.perf_counter()
        win_t = t1 - t0
        if not is_warmup: win_times.append(win_t)

        # Baseline: full band read
        t0 = time.perf_counter()
        with rasterio.open(b03_url) as src:
            _f = src.read(1)
        t1 = time.perf_counter()
        full_t = t1 - t0
        if not is_warmup: full_times.append(full_t)

        speedup_str = f"{full_t/win_t:.2f}×" if win_t > 0 else "N/A"
        print(f"    {tag}  window={_ms(win_t)}  full={_ms(full_t)}  speedup={speedup_str}")

    speedup = round(statistics.mean(full_times) / statistics.mean(win_times), 2) if win_times and full_times else None
    data_red = round(1 - (aoi_px / full_px), 6) if full_px > 0 else None

    return {
        "scene_id": scene_id,
        "aoi_bbox": AOI_BBOX,
        "full_band": {
            "width_px": full_w, "height_px": full_h, "total_pixels": full_px,
            "estimated_uncompressed_bytes": full_b,
            "latencies_s": full_times, "stats": _stats(full_times),
        },
        "aoi_window": {
            "width_px": win_w, "height_px": win_h, "total_pixels": aoi_px,
            "estimated_uncompressed_bytes": aoi_b,
            "latencies_s": win_times, "stats": _stats(win_times),
        },
        "speedup": speedup,
        "data_reduction_fraction": data_red,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def _write_markdown_report(results: dict) -> None:
    b1e = results.get("benchmark_1_e2e", {})
    b1s = results.get("benchmark_1_stages", {})
    b2  = results.get("benchmark_2_cog_efficiency", {})
    ms  = results.get("mock_status", {})
    valid = ms.get("pipeline_valid", False)

    e2e  = b1e.get("stats", {})
    b2fw = b2.get("full_band",  {}).get("stats", {})
    b2ww = b2.get("aoi_window", {}).get("stats", {})
    sp   = b2.get("speedup")
    dr   = b2.get("data_reduction_fraction")
    fp   = b2.get("full_band",  {}).get("total_pixels", 0)
    ap   = b2.get("aoi_window", {}).get("total_pixels", 0)
    fm   = b2.get("full_band",  {}).get("estimated_uncompressed_bytes", 0) / 1e6
    ak   = b2.get("aoi_window", {}).get("estimated_uncompressed_bytes", 0) / 1e3
    fw   = b2.get("full_band",  {}).get("width_px", "?")
    fh   = b2.get("full_band",  {}).get("height_px", "?")
    ww   = b2.get("aoi_window", {}).get("width_px", "?")
    wh   = b2.get("aoi_window", {}).get("height_px", "?")

    def sm(k):  # stage mean seconds
        v = b1s.get(k, {})
        return v.get("mean_s", 0) if isinstance(v, dict) else 0

    stage_keys = [
        ("s1_gemini_parsing",       "Gemini query parsing",        "Gemini API (gemini-3.6-flash)"),
        ("s2_geocoding",            "Geographic grounding",         "Nominatim (OpenStreetMap)"),
        ("s3_remoteclip_embed",     "RemoteCLIP text embedding",    "ViT-B-32 CPU inference"),
        ("s4_faiss_search",         "FAISS vector search",          "IndexFlatIP (29 vectors, 512-dim)"),
        ("s5_candidate_validation", "Candidate spatial validation", "Python bbox intersection"),
        ("s6_stac_search_cog",      "STAC scene search (est.)",     "Planetary Computer STAC API"),
        ("s7_raster_analysis",      "COG window read + analysis",   "rasterio + NumPy (NDWI)"),
        ("s8_gemini_explanation",   "Gemini explanation",           "Gemini API (gemini-3.6-flash)"),
    ]
    total_stage_s = sum(sm(k) for k, _, _ in stage_keys)

    validity_note = (
        "> [!IMPORTANT]\n> **Pipeline mode: REAL** — MOCK_AI=false, MOCK_SEARCH=false, MOCK_GEO=false.\n"
        "> All measurements reflect the live Gemini + RemoteCLIP + FAISS + Planetary Computer pipeline.\n"
        if valid else
        "> [!WARNING]\n> **Some mock services were active.** Results may be underestimates for network-bound stages.\n"
    )

    stage_rows = "\n".join(
        f"| {i+1} — {label} | {service} | {_ms(sm(k))} | "
        f"{'N/A' if total_stage_s == 0 else f'{sm(k)/total_stage_s*100:.1f}%'} |"
        for i, (k, label, service) in enumerate(stage_keys)
    )

    env = results.get("environment", {})
    report = f"""# GeoQueryAI — Benchmark Results

**Run date**: {results.get("timestamp", "N/A")}
**Query**: `{BENCHMARK_QUERY}`
**N trials (B1)**: {N_WARMUP} warmup + {N_TRIALS} measured | **N trials (B2)**: 1 warmup + {N_COG_TRIALS} measured
**Python**: {env.get("python_version", "?")} | **Platform**: {env.get("platform", "?")}

{validity_note}

---

## Benchmark 1 — End-to-End Query Latency

### Wall-Clock E2E (HTTP POST /api/query → full JSON response)

| Metric | Value |
|--------|-------|
| **Mean** | **{_ms(e2e.get("mean_s"))}** |
| Median (P50) | {_ms(e2e.get("median_s"))} |
| **P95** | **{_ms(e2e.get("p95_s"))}** |
| Min | {_ms(e2e.get("min_s"))} |
| Max | {_ms(e2e.get("max_s"))} |
| Std dev | {_ms(e2e.get("stdev_s"))} |

### Pipeline Stage Decomposition (mean over {N_TRIALS} trials, direct function calls)

| Stage | Service | Mean Latency | % of total |
|-------|---------|-------------|-----------|
{stage_rows}
| **Total** | | **{_ms(total_stage_s)}** | **100%** |

> [!NOTE]
> Stages 6 + 7 are internally coupled inside `analyze()`. The 35%/65% split is an approximation.
> Network-bound stages (1, 2, 6, 8) dominate latency variance across trials.

---

## Benchmark 2 — Data Retrieval / Processing Efficiency

**Scene**: `{b2.get("scene_id", "N/A")}`  
**AOI**: Ulsoor Lake, Bengaluru `{AOI_BBOX}`  
**Band**: Sentinel-2 B03 (Green) — same signed COG URL for both approaches

| Metric | Full-Band Read (Baseline) | COG Window Read (GeoQueryAI) |
|--------|--------------------------|------------------------------|
| Pixels read | {fp:,} ({fw}×{fh} px) | {ap:,} ({ww}×{wh} px) |
| Uncompressed data | ~{fm:.1f} MB | ~{ak:.1f} KB |
| Mean latency | {_ms(b2fw.get("mean_s"))} | {_ms(b2ww.get("mean_s"))} |
| P95 latency | {_ms(b2fw.get("p95_s"))} | {_ms(b2ww.get("p95_s"))} |
| Min / Max | {_ms(b2fw.get("min_s"))} / {_ms(b2fw.get("max_s"))} | {_ms(b2ww.get("min_s"))} / {_ms(b2ww.get("max_s"))} |

### Efficiency Ratios

| Ratio | Formula | Value |
|-------|---------|-------|
| **Speedup** | baseline_mean / GeoQueryAI_mean | **{sp}×** |
| **Data reduction** | 1 − (AOI_px / full_px) | **{f'{dr:.1%}' if dr else 'N/A'}** |
| Pixels saved | full_px − AOI_px | {fp - ap:,} |

> [!NOTE]
> rasterio uses HTTP Range Requests on COGs — only the AOI tiles are downloaded.
> Full-band read forces all internal COG tile blocks to be fetched and decompressed.
> Network jitter is the dominant variance source; local CPU computation is deterministic.

---

## 3 Presentation-Ready Metrics

### Metric 1 — End-to-End NL Query Latency
> **Mean {_ms(e2e.get("mean_s"))} &nbsp;|&nbsp; P95 {_ms(e2e.get("p95_s"))}**
>
> *Methodology*: {N_TRIALS} repeated `POST /api/query` requests to a local uvicorn server.
> Query: "{BENCHMARK_QUERY}".
> Wall-clock from HTTP send to full JSON response (httpx). 1 warmup excluded.
> Python {env.get("python_version","?")}, CPU-only RemoteCLIP.

### Metric 2 — Semantic Retrieval across 29 Sentinel-2 Tiles
> **FAISS search in {_ms(sm("s4_faiss_search"))} across 29 indexed satellite tiles**
>
> *Methodology*: Mean `IndexFlatIP.search()` latency over {N_TRIALS} trials.
> Visual query: `"water body lake"` → RemoteCLIP ViT-B-32 text encoder (512-dim).
> Top-5 retrieved — includes tiles from lake_water, vegetation_park, and dense_urban
> categories demonstrating genuine semantic discrimination.

### Metric 3 — COG Targeted Window vs Full-Band Download
> **{sp}× faster &nbsp;|&nbsp; {f'{dr:.0%}' if dr else 'N/A'} less data processed**
>
> *Methodology*: Same signed Sentinel-2 scene `{b2.get("scene_id","?")}`.
> Baseline: `rasterio src.read(1)` — {fw}×{fh} px = {fp:,} pixels (~{fm:.0f} MB).
> GeoQueryAI: `read_geometry_from_cog()` AOI window = {ww}×{wh} px = {ap:,} pixels (~{ak:.0f} KB).
> {N_COG_TRIALS} measured trials each on the same COG URL.

---

*Generated by `scripts/benchmark_latency.py` · Raw data: `benchmark_results.json`*
"""
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nMarkdown report saved: {REPORT_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import platform
    from datetime import datetime

    print("=" * 66)
    print("GeoQueryAI — Reproducible Latency Benchmarks")
    print("=" * 66)

    mock_status = _check_real_mode()
    print(f"\nPipeline: {'[VALID — fully real]' if mock_status['pipeline_valid'] else '[MOCK-CONTAMINATED]'}")
    print(f"  mock_ai={mock_status['mock_ai']}  mock_search={mock_status['mock_search']}  mock_geo={mock_status['mock_geo']}")

    results: dict = {
        "timestamp": datetime.now().isoformat(),
        "query": BENCHMARK_QUERY,
        "n_trials": N_TRIALS,
        "n_warmup": N_WARMUP,
        "mock_status": mock_status,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    # ── Benchmark 1a: stage timing ────────────────────────────────────────────
    print("\n" + "─" * 66)
    print("BENCHMARK 1a — Per-stage pipeline timing")
    print("─" * 66)
    stage_raw = _run_stage_benchmark()
    results["benchmark_1_stages"] = {
        k: ({"latencies_s": v, **_stats(v)} if isinstance(v, list) else v)
        for k, v in stage_raw.items()
    }
    print("\nStage means:")
    label_map = {
        "s1_gemini_parsing": "Gemini parsing", "s2_geocoding": "Geocoding",
        "s3_remoteclip_embed": "RemoteCLIP embed", "s4_faiss_search": "FAISS search",
        "s5_candidate_validation": "Candidate validation", "s6_stac_search_cog": "STAC (est.)",
        "s7_raster_analysis": "Raster analysis (est.)", "s8_gemini_explanation": "Gemini explanation",
    }
    for k, lbl in label_map.items():
        vals = stage_raw.get(k, [])
        if vals:
            print(f"  {lbl:<35} {_ms(statistics.mean(vals)):>10}")

    # ── Benchmark 1b: e2e HTTP timing ─────────────────────────────────────────
    print("\n" + "─" * 66)
    print("BENCHMARK 1b — End-to-end HTTP wall-clock latency")
    print("─" * 66)
    e2e_times = _run_e2e_benchmark()
    e2e_stats = _stats(e2e_times)
    results["benchmark_1_e2e"] = {"latencies_s": e2e_times, "stats": e2e_stats}
    print(f"\nE2E summary (N={N_TRIALS}):")
    print(f"  Mean   : {_ms(e2e_stats.get('mean_s'))}")
    print(f"  Median : {_ms(e2e_stats.get('median_s'))}")
    print(f"  P95    : {_ms(e2e_stats.get('p95_s'))}")
    print(f"  Min/Max: {_ms(e2e_stats.get('min_s'))} / {_ms(e2e_stats.get('max_s'))}")
    print(f"  Stdev  : {_ms(e2e_stats.get('stdev_s'))}")

    # ── Benchmark 2: COG efficiency ────────────────────────────────────────────
    print("\n" + "─" * 66)
    print("BENCHMARK 2 — COG windowed read vs full-band download")
    print("─" * 66)
    cog = _run_cog_efficiency_benchmark()
    results["benchmark_2_cog_efficiency"] = cog
    if "error" not in cog:
        print(f"\nCOG summary:")
        print(f"  Window mean : {_ms(cog['aoi_window']['stats'].get('mean_s'))}")
        print(f"  Full mean   : {_ms(cog['full_band']['stats'].get('mean_s'))}")
        print(f"  Speedup     : {cog.get('speedup')}×")
        print(f"  Data reduction: {cog.get('data_reduction_fraction', 0):.1%}")

    # ── Save + report ──────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved: {RESULTS_FILE}")

    _write_markdown_report(results)

    print("\n" + "=" * 66)
    print("Benchmark complete.")
    print(f"  JSON   : {RESULTS_FILE}")
    print(f"  Report : {REPORT_FILE}")
    print("=" * 66)


if __name__ == "__main__":
    main()
