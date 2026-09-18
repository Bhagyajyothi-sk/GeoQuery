"""
GeoQueryAI — AI Service Interface.

════════════════════════════════════════════════════════════════════
  PLUG-IN POINT: Gemini natural-language query parser
════════════════════════════════════════════════════════════════════

Replace the body of `parse_query()` with your Gemini implementation.

Contract:
    Input  : free-text user query (str)
    Output : StructuredQuery  (validated Pydantic model)

The StructuredQuery drives the entire downstream pipeline:
  - visual_query  → RemoteCLIP embedding
  - location      → geocoder (context only, never used as raw coordinates)
  - analysis      → STAC + COG pipeline dispatch
  - start_date /
    end_date      → STAC temporal filter

Do NOT embed coordinates in StructuredQuery — coordinates come exclusively
from the SemanticCandidate.bbox returned by the search layer.

────────────────────────────────────────────────────────────────────
Mock mode:
  Controlled by MOCK_AI env var (default: true).
  Set MOCK_AI=false once Gemini is connected.

  When mock_ai=True  → returns deterministic StructuredQuery.
  When mock_ai=False → calls the real Gemini implementation below.
────────────────────────────────────────────────────────────────────
"""

import logging

from app.schemas.query import StructuredQuery
from app.core.config import settings
from app.core.errors import AIServiceError

logger = logging.getLogger("geoquery.ai")


def parse_query(text: str) -> StructuredQuery:
    """
    Parse a natural-language user query into a StructuredQuery.

    ┌─────────────────────────────────────────────────────────────────┐
    │  MOCK MODE  (MOCK_AI=true, the default)                         │
    │  Returns a deterministic StructuredQuery for development.       │
    │  Set MOCK_AI=false and implement the real block below.          │
    └─────────────────────────────────────────────────────────────────┘

    Gemini integration guide:
      1. Send `text` to the Gemini API with a system prompt instructing
         it to return JSON conforming to the StructuredQuery schema.
      2. Parse the JSON response into a StructuredQuery instance.
      3. Raise AIServiceError on any Gemini API failure.
      4. Set MOCK_AI=false in your .env file.

    Args:
        text: The raw natural-language query string from the user.

    Returns:
        StructuredQuery: Validated structured representation of the query.

    Raises:
        AIServiceError: If the AI service is unavailable or returns invalid output.
    """
    if settings.mock_ai:
        # ── MOCK: deterministic response for development/testing ───────────────
        result = StructuredQuery(
            visual_query="large water body",
            location="Bengaluru",
            analysis="water_extent",
            start_date="2025-09-01",
            end_date="2026-09-01",
        )
        logger.info(
            "QUERY_PARSED [MOCK] visual_query=%r location=%r analysis=%s",
            result.visual_query,
            result.location,
            result.analysis,
        )
        return result
        # ── END MOCK ───────────────────────────────────────────────────────────

    # ═══════════════════════════════════════════════════════════════════════════
    #  REAL IMPLEMENTATION — insert your Gemini code here
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Step-by-step:
    #
    #   import json
    #   import google.generativeai as genai
    #
    #   SYSTEM_PROMPT = """
    #   You are a geospatial query parser. Extract the following fields from
    #   the user query and return ONLY valid JSON with these keys:
    #     - visual_query  (str): a visual description for satellite image retrieval
    #     - location      (str | null): a place name (NOT coordinates)
    #     - analysis      (str): one of discovery|ndvi|ndwi|water_extent|change|water_extent_change
    #     - start_date    (str | null): YYYY-MM-DD
    #     - end_date      (str | null): YYYY-MM-DD
    #   """
    #
    #   try:
    #       genai.configure(api_key=settings.gemini_api_key)
    #       model = genai.GenerativeModel("gemini-2.5-pro")
    #       response = model.generate_content(
    #           f"{SYSTEM_PROMPT}\n\nUser query: {text}"
    #       )
    #       data = json.loads(response.text)
    #       result = StructuredQuery(**data)
    #       logger.info(
    #           "QUERY_PARSED visual_query=%r location=%r analysis=%s",
    #           result.visual_query, result.location, result.analysis,
    #       )
    #       return result
    #   except Exception as exc:
    #       raise AIServiceError(f"Gemini query parsing failed: {exc}") from exc
    #
    # ═══════════════════════════════════════════════════════════════════════════

    raise AIServiceError(
        "Gemini implementation not yet connected. "
        "Set MOCK_AI=true to use mock mode, or plug in your Gemini code above."
    )
