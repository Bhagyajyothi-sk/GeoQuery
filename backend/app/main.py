"""
GeoQueryAI — FastAPI application entry point.

Run from the backend/ directory:
    uvicorn app.main:app --reload

The app is structured around three clean API groups:
  /api/query    — full NL orchestration pipeline
  /api/search   — semantic search in isolation
  /api/analyze  — geospatial analysis in isolation
"""

import logging
from contextlib import asynccontextmanager

import sys
from pathlib import Path

# Ensure backend/ directory is on sys.path so imports work regardless of launch directory
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes import query, search, analysis, maps
from app.core.errors import (
    AIServiceError,
    SearchServiceError,
    NoCandidatesError,
    GeoAnalysisError,
    NoSatelliteDataError,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("geoquery.main")


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup / shutdown hook.

    When MOCK_SEARCH=false, pre-loads the RemoteCLIP encoder and FAISS index
    so that the first real request has no cold-start latency.

    ════════════════════════════════════════════════════════════════════
      PLUG-IN POINT: Model initialisation at startup
    ════════════════════════════════════════════════════════════════════
      RemoteCLIP and FAISS are loaded here when MOCK_SEARCH=false.
      To force a specific checkpoint path at startup:

        RemoteCLIPEncoder.instance("/abs/path/to/weights.pt")
        FAISSStore.instance("/abs/path/to/index", "/abs/path/to/meta.json")
    ════════════════════════════════════════════════════════════════════
    """
    logger.info("GeoQueryAI %s starting up", settings.app_version)
    logger.info(
        "AI service    : %s", "MOCK" if settings.mock_ai else "REAL (Gemini)"
    )
    logger.info(
        "Search service: %s", "MOCK" if settings.mock_search else "REAL (RemoteCLIP + FAISS)"
    )
    logger.info(
        "Geo service   : %s", "MOCK" if settings.mock_geo else "REAL (Nominatim)"
    )

    # ── Pre-warm RemoteCLIP + FAISS (real mode only) ──────────────────────────
    if not settings.mock_search:
        logger.info("Search: warming up RemoteCLIP encoder and FAISS store...")
        try:
            from app.services.search.remoteclip import RemoteCLIPEncoder
            from app.services.search.faiss_store import FAISSStore

            RemoteCLIPEncoder.instance()   # loads checkpoint once
            FAISSStore.instance()          # loads FAISS index once

            logger.info("Search: RemoteCLIP and FAISS ready.")
        except Exception as exc:
            # Non-fatal at startup — requests will fail with SearchServiceError
            logger.error(
                "Search: warm-up failed (%s). "
                "Set MOCK_SEARCH=true for development without tile data.",
                exc,
            )

    yield
    logger.info("GeoQueryAI shutting down")



# ── App factory ────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "**GeoQueryAI** — Natural Language Earth Observation Search and Analysis.\n\n"
        "Submit a free-text query to retrieve semantically matched Sentinel-2 satellite "
        "tiles and run geospatial analyses (NDVI, NDWI, water extent, change detection)."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
origins = settings.cors_origins.copy()
if settings.frontend_url and settings.frontend_url not in origins:
    origins.append(settings.frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handlers ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal error occurred."},
    )


# ── Routers ────────────────────────────────────────────────────────────────────
API_PREFIX = "/api"

app.include_router(query.router,    prefix=API_PREFIX, tags=["Pipeline"])
app.include_router(search.router,   prefix=API_PREFIX, tags=["Semantic Search"])
app.include_router(analysis.router, prefix=API_PREFIX, tags=["Geospatial Analysis"])
app.include_router(maps.router,     prefix=API_PREFIX, tags=["Rendered Maps"])


# ── Static map output ──────────────────────────────────────────────────────────
# Rendered NDVI/NDWI PNGs written by the /api/maps/* endpoints are served here.
MAPS_DIR = Path(__file__).resolve().parent / "static" / "maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/maps", StaticFiles(directory=str(MAPS_DIR)), name="maps")

# ── Static tiles output ────────────────────────────────────────────────────────
# Actual satellite tile images from data/satellite_tiles/
TILES_DIR = _BACKEND_DIR.parent / "data" / "satellite_tiles"
app.mount("/tiles", StaticFiles(directory=str(TILES_DIR)), name="tiles")


# ── Health probe ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"], summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok", "service": "GeoQuery AI", "version": settings.app_version}
