"""
GeoQueryAI — application configuration.

Settings are loaded from environment variables (or a .env file).
All values have sane defaults so the server starts without any configuration.
"""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# This file lives at <project_root>/backend/app/core/config.py
# parents: [0]=core, [1]=app, [2]=backend, [3]=project_root
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # A .env is looked for in the project root, then backend/, then the current
    # working directory, so the server behaves the same however it is launched.
    model_config = SettingsConfigDict(
        env_file=(_PROJECT_ROOT / ".env", _BACKEND_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = "GeoQueryAI"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── CORS ───────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins, e.g. "http://localhost:3000"
    cors_origins: list[str] = ["*"]

    # ── Geocoding ──────────────────────────────────────────────────────────────
    nominatim_url: str = "https://nominatim.openstreetmap.org/search"
    nominatim_user_agent: str = "GeoQueryAI-GeocodingService/1.0"

    # ── STAC / Planetary Computer ──────────────────────────────────────────────
    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    stac_collection: str = "sentinel-2-l2a"
    stac_max_cloud_cover: float = 20.0

    # ── Semantic search defaults ───────────────────────────────────────────────
    default_top_k: int = 5

    # ──────────────────────────────────────────────────────────────────────────
    # Mock mode flags
    # Set to False once the real implementation is plugged in.
    # Controlled via environment variables or .env file:
    #   MOCK_AI=false      → use real Gemini
    #   MOCK_SEARCH=false  → use real RemoteCLIP + FAISS
    #   MOCK_GEO=false     → use real Nominatim geocoder
    # ──────────────────────────────────────────────────────────────────────────
    mock_ai: bool = True      # Gemini query parser   — set False when Gemini is connected
    mock_search: bool = True  # RemoteCLIP + FAISS    — set False when models are loaded
    mock_geo: bool = True     # Nominatim geocoder    — set False for live geocoding

    # ──────────────────────────────────────────────────────────────────────────
    # AI/ML plug-in keys (populated by the AI/ML engineer)
    # ──────────────────────────────────────────────────────────────────────────
    gemini_api_key: str = ""           # ← Gemini query parser

    # RemoteCLIP + FAISS paths (relative to project root)
    remoteclip_model_path: str = "models/RemoteCLIP-ViT-B-32.pt"  # ← RemoteCLIP weights
    faiss_index_path: str = "indexes/satellite.index"             # ← FAISS index
    tile_metadata_path: str = "indexes/tile_metadata.json"        # ← FAISS ↔ tile mapping

    # Prompt ensemble — encode multiple phrasings of the query, then average.
    # Improves zero-shot recall at the cost of ~4× encoding time.
    # Set PROMPT_ENSEMBLE=true in .env to enable.
    prompt_ensemble: bool = False

    # ── Validators ─────────────────────────────────────────────────────────────
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value):
        """
        Accept CORS_ORIGINS as either a JSON array or a comma-separated string.

        Without this, CORS_ORIGINS=http://localhost:3000,http://localhost:5173
        in a .env file fails to parse as JSON and the app refuses to start.
        """
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ["*"]
            if text.startswith("["):
                return value  # leave JSON for pydantic to decode
            return [origin.strip() for origin in text.split(",") if origin.strip()]
        return value


settings = Settings()
