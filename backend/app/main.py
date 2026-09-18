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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.routes import query, search, analysis
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

    Startup:
      - Log service mode (mock vs real).
      - Future: initialise RemoteCLIP model and FAISS index here so they
        are loaded once and reused across requests (not per-call).

    ════════════════════════════════════════════════════════════════════
      PLUG-IN POINT: Model initialisation at startup
    ════════════════════════════════════════════════════════════════════
      Load RemoteCLIP weights and FAISS index here:

        from app.services.search.interface import load_models
        load_models()   # call your initialiser

      This avoids reloading 1–2 GB models on every request.
    ════════════════════════════════════════════════════════════════════
    """
    logger.info("GeoQueryAI %s starting up", settings.app_version)
    logger.info("AI service   : MOCK (replace services/ai/interface.py)")
    logger.info("Search service: MOCK (replace services/search/interface.py)")
    logger.info("Geo service  : REAL (STAC + COG via Planetary Computer)")
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
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


# ── Health probe ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"], summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}
