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

import json
import os
import re
import logging
from typing import Any, Union
from pydantic import BaseModel

from app.schemas.query import StructuredQuery
from app.schemas.analysis import Evidence
from app.core.config import settings
from app.core.errors import AIServiceError

logger = logging.getLogger("geoquery.ai")


def parse_query(text: str) -> StructuredQuery:
    """
    Parse a natural-language user query into a StructuredQuery.

    When MOCK_AI=true:
      Returns a deterministic StructuredQuery for development.

    When MOCK_AI=false:
      Calls the Gemini API to parse the natural-language text into a structured JSON
      object and validates it against StructuredQuery schema.

    Args:
        text: The raw natural-language query string from the user.

    Returns:
        StructuredQuery: Validated structured representation of the query.

    Raises:
        AIServiceError: If the Gemini API is unavailable or returns invalid JSON.
    """
    if settings.mock_ai:
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

    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise AIServiceError("Gemini API key not configured. Set GEMINI_API_KEY in .env")

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.6-flash")

        system_prompt = (
            "You are a geospatial query parser for GeoQueryAI.\n"
            "Extract structured query parameters from the user's natural language input.\n"
            "Return ONLY valid JSON with no markdown formatting or code fences:\n"
            "{\n"
            '  "visual_query": "visual description for satellite image search (e.g. large water body, dense forest)",\n'
            '  "location": "place name string or null",\n'
            '  "analysis": "one of: discovery | ndvi | ndwi | water_extent | change | water_extent_change",\n'
            '  "start_date": "YYYY-MM-DD or null",\n'
            '  "end_date": "YYYY-MM-DD or null"\n'
            "}\n\n"
            "Rules:\n"
            "- visual_query must be a visual description suitable for RemoteCLIP image matching (no coordinates).\n"
            "- location must be a free-text place name string only, or null if unstated.\n"
            "- analysis MUST strictly be one of: 'discovery', 'ndvi', 'ndwi', 'water_extent', 'change', 'water_extent_change'. If unspecified, default to 'water_extent'."
        )

        response = model.generate_content(f"{system_prompt}\n\nUser query: {text}")
        content = response.text.strip()

        # Clean markdown code fences if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        data = json.loads(content)
        result = StructuredQuery(**data)
        logger.info(
            "QUERY_PARSED visual_query=%r location=%r analysis=%s start=%s end=%s",
            result.visual_query,
            result.location,
            result.analysis,
            result.start_date,
            result.end_date,
        )
        return result
    except Exception as exc:
        logger.error("Gemini query parsing failed: %s", exc)
        raise AIServiceError(f"Gemini query parsing failed: {exc}") from exc


def explain_evidence(evidence: Union[Evidence, dict[str, Any]]) -> str:
    """
    Generate a grounded natural-language explanation based STRICTLY on the Evidence layer.

    When MOCK_AI=true:
      Returns a deterministic grounded explanation derived from evidence metrics.

    When MOCK_AI=false:
      Sends Evidence JSON to Gemini API with strict instructions not to invent facts or dates.

    Args:
        evidence: Evidence Pydantic model or dict containing computed metrics & metadata.

    Returns:
        str: Concise grounded explanation.
    """
    if isinstance(evidence, BaseModel):
        evidence_dict = evidence.model_dump()
    else:
        evidence_dict = dict(evidence)

    if settings.mock_ai:
        analysis_type = evidence_dict.get("analysis", "geospatial")
        scene_id = evidence_dict.get("scene_id") or "Sentinel-2 scene"
        date = evidence_dict.get("acquisition_date") or "recent date"
        metrics = evidence_dict.get("computed_values", {})

        metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items()) if metrics else "no numeric metrics calculated"
        return (
            f"Based on Sentinel-2 satellite observation ({scene_id} acquired on {date}), "
            f"the {analysis_type} analysis yielded: {metrics_str}."
        )

    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        metrics = evidence_dict.get("computed_values", {})
        metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items()) if metrics else "no numeric metrics calculated"
        return (
            f"Based on Sentinel-2 satellite observation ({evidence_dict.get('scene_id', 'scene')}), "
            f"the {evidence_dict.get('analysis', 'analysis')} analysis yielded: {metrics_str}."
        )

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.6-flash")

        system_prompt = (
            "You are a scientific geospatial assistant for GeoQueryAI.\n"
            "You will receive a JSON object representing the Evidence layer of a satellite image analysis.\n"
            "Your task is to write a concise, clear natural language summary explaining the findings to the user.\n"
            "CRITICAL RULES:\n"
            "1. Rely STRICTLY on the values present in the provided Evidence JSON.\n"
            "2. Do NOT invent measurements, dates, satellite observations, or locations not present in the evidence.\n"
            "3. If a value is missing or null, do NOT speculate about it.\n"
            "4. Keep the explanation professional and concise (2-4 sentences)."
        )

        evidence_json = json.dumps(evidence_dict, indent=2)
        response = model.generate_content(f"{system_prompt}\n\nEvidence JSON:\n{evidence_json}")

        explanation = response.text.strip()
        logger.info("EXPLANATION_GENERATED evidence_keys=%s len=%d", list(evidence_dict.keys()), len(explanation))
        return explanation
    except Exception as exc:
        logger.error("Gemini evidence explanation failed (%s) — using fallback explanation", exc)
        metrics = evidence_dict.get("computed_values", {})
        metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items()) if metrics else "no numeric metrics calculated"
        return (
            f"Based on Sentinel-2 satellite observation ({evidence_dict.get('scene_id', 'scene')}), "
            f"the {evidence_dict.get('analysis', 'analysis')} analysis yielded: {metrics_str}."
        )

