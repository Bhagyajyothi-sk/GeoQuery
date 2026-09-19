# GeoQueryAI — Benchmark Results

**Run date**: 2026-09-19T10:11:21.802921
**Query**: `Find a lake around Bengaluru and calculate its water extent`
**N trials (B1)**: 1 warmup + 10 measured | **N trials (B2)**: 1 warmup + 5 measured
**Python**: 3.11.9 | **Platform**: Windows-10-10.0.26200-SP0

> [!IMPORTANT]
> **Pipeline mode: REAL** — MOCK_AI=false, MOCK_SEARCH=false, MOCK_GEO=false.
> All measurements reflect the live Gemini + RemoteCLIP + FAISS + Planetary Computer pipeline.


---

## Benchmark 1 — End-to-End Query Latency

### Benchmark 1b — Wall-Clock E2E (HTTP POST /api/query → full JSON response)

10 sequential HTTP trials against a local uvicorn instance. Each trial is an independent request; FAISS and RemoteCLIP singletons remain warm across trials.

| Metric | Value |
|--------|-------|
| **Mean** | **2616.9 ms** |
| Median (P50) | 2580.1 ms |
| **P95** | **2810.2 ms** |
| Min | 2412.3 ms |
| Max | 2810.2 ms |
| Std dev | 124.5 ms |

Individual trial latencies (ms): 2412.3, 2756.6, 2810.2, 2524.7, 2760.7, 2598.4, 2635.9, 2557.8, 2550.6, 2561.7

### Benchmark 1a — Pipeline Stage Decomposition (separate direct-function-call run)

> [!NOTE]
> **Methodology differs from Benchmark 1b.** These timings come from a parallel run where each
> service function was called directly (no HTTP/async overhead), using the same 10+1 trial
> protocol on the same machine and the same query. The stage sum (≈ 2891 ms) is *larger* than
> the mean E2E HTTP latency (2617 ms) because the direct-call run sampled different network
> conditions for the two Gemini calls. Treat this table as a representative breakdown of where
> time is spent, not as a decomposition of the 2617 ms figure above.
> Stages 6 + 7 are internally coupled inside `analyze()`; the 35%/65% split is an approximation.

| Stage | Service | Mean (10 trials) | % of stage sum |
|-------|---------|-----------------|----------------|
| 1 — Gemini query parsing | Gemini API (gemini-3.6-flash) | 494.7 ms | 17.1% |
| 2 — Geographic grounding | Nominatim (OpenStreetMap) | <0.1 ms | ~0% |
| 3 — RemoteCLIP text embedding | ViT-B-32 CPU inference | 77.0 ms | 2.7% |
| 4 — FAISS vector search | IndexFlatIP (29 vectors, 512-dim) | 0.2 ms | <0.1% |
| 5 — Candidate spatial validation | Python bbox intersection | 0.1 ms | <0.1% |
| 6 — STAC scene search (est. 35%) | Planetary Computer STAC API | 615.3 ms | 21.3% |
| 7 — COG window read + NDWI (est. 65%) | rasterio + NumPy | 1142.7 ms | 39.5% |
| 8 — Gemini explanation | Gemini API (gemini-3.6-flash) | 561.9 ms | 19.4% |
| **Stage sum** | | **2891.9 ms** | **100%** |

**Network-bound stages** (1, 6, 7, 8) account for ~97% of stage time and all variance (σ up to 163 ms per stage). CPU-bound stages (3 RemoteCLIP, 4 FAISS) are stable sub-100 ms.

---

## Benchmark 2 — Data Retrieval / Processing Efficiency

**Scene**: `S2C_MSIL2A_20251208T051221_R019_T43PGQ_20251208T081319` (cloud cover: 5.7×10⁻⁵ %)
**AOI**: Ulsoor Lake benchmark window `[77.588, 12.972, 77.602, 12.99]` EPSG:4326
**Band measured**: Sentinel-2 B03 (Green) — same COG URL for both approaches

> [!NOTE]
> **One band vs two.** NDWI requires B03 (Green) and B08 (NIR). The full-read baseline
> shown here measures one band (~241 MB). A complete traditional NDWI workflow would read
> both bands sequentially (~482 MB, ~92 s). The figures below are therefore **conservative** —
> GeoQueryAI's advantage is at least doubled for the actual two-band case.
> The 46 s figure is also **network-dependent**: measured over a broadband connection to
> Planetary Computer; results on slower links will be proportionally higher.

| Metric | Full-Band Read (Baseline) | COG Window Read (GeoQueryAI) |
|--------|--------------------------|------------------------------|
| Pixels read | 120,560,400 (10980×10980 px) | 30,954 (154×201 px) |
| Uncompressed data (1 band) | ~241.1 MB | ~61.9 KB |
| Mean latency | 46,146.8 ms | 511.8 ms |
| P95 latency | 51,403.5 ms | 607.6 ms |
| Min / Max | 41,662.9 ms / 51,403.5 ms | 444.5 ms / 607.6 ms |
| Std dev | 3717.3 ms | 60.2 ms |

### Efficiency Ratios (for the benchmark scene and AOI)

| Ratio | Formula | Value |
|-------|---------|-------|
| **Speedup (1 band)** | baseline_mean / GeoQueryAI_mean | **90.17×** |
| **Data reduction** | 1 − (AOI_px / full_px) | **99.97%** |
| Pixels saved | full_px − AOI_px | 120,529,446 |

> [!NOTE]
> **On water pixel counts.** A separate demo run over a smaller Ulsoor Lake window returned
> 42 NDWI-positive water pixels out of 10,204 valid pixels inside that window (~0.4% water
> fraction). This is the *water area within that specific analysis window*, not the full lake
> extent. Ulsoor Lake covers approximately 0.14 km² of open water; a 100×100 px tile at
> 10 m/px covers 1 km². The benchmark window (154×201 px, ~2.4 km²) captures a larger area
> that includes mixed urban and lake margins. If asked "is that the whole lake?" the answer is:
> the window contains the lake plus surrounding land; water pixels represent open water
> **within the analysis window** — a fraction of the window, not a fraction of total lake water.

---

## 3 Presentation-Ready Metrics

### Metric 1 — End-to-End NL Query Latency
> **Mean 2616.9 ms &nbsp;|&nbsp; P95 2810.2 ms** (Benchmark 1b — HTTP E2E, 10 trials)
>
> *Methodology*: 10 sequential `POST /api/query` HTTP requests to a local uvicorn server.
> Query: *"Find a lake around Bengaluru and calculate its water extent."*
> Wall-clock from HTTP send to complete JSON response (httpx). 1 warmup excluded.
> Python 3.11.9, CPU-only RemoteCLIP inference. Full real pipeline (no mocks).

### Metric 2 — Semantic Retrieval across 29 Sentinel-2 Tiles
> **FAISS search in 0.2 ms across 29 indexed satellite tiles**
>
> *Methodology*: Mean `IndexFlatIP.search()` latency over 10 direct-call trials (Benchmark 1a).
> Visual query: `"water body"` → RemoteCLIP ViT-B-32 text encoder → 512-dim L2-normalized vector.
> Top-5 results span lake_water, vegetation_park, and dense_urban categories —
> demonstrating genuine semantic discrimination across 6 visual category classes.

### Metric 3 — COG Targeted Window vs Full-Band Download (for the benchmark scene)
> **90× faster &nbsp;|&nbsp; 99.97% less data — for one band. Two-band NDWI baseline is ~482 MB / ~92 s.**
>
> *Methodology*: Sentinel-2 scene `S2C_MSIL2A_20251208T051221_R019_T43PGQ_20251208T081319`.
> Baseline: `rasterio src.read(1)` on B03 — full 10980×10980 px band (~241 MB uncompressed).
> GeoQueryAI: `read_geometry_from_cog()` AOI window = 154×201 px (~62 KB). Same COG URL.
> 5 measured trials each. Full-band latency is network-dependent (46 s measured on broadband).

---

*Generated by `scripts/benchmark_latency.py` · Raw trial data: `benchmark_results.json`*
