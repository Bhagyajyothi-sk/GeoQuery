"""
GeoQueryAI — application configuration.

Settings are loaded from environment variables (or a .env file).
All values have sane defaults so the server starts without any configuration.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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
    remoteclip_model_path: str = ""    # ← RemoteCLIP weights path
    faiss_index_path: str = ""         # ← FAISS index file path


settings = Settings()
