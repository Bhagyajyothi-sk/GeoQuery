"""
GeoQueryAI — Semantic Search Interface.

════════════════════════════════════════════════════════════════════
  PLUG-IN POINT: RemoteCLIP image embedding + FAISS Top-K retrieval
════════════════════════════════════════════════════════════════════

Replace the body of `semantic_search()` with your RemoteCLIP + FAISS
implementation.

Contract:
    Input  : visual_query (str), optional location hint (str), top_k (int)
    Output : CandidateResponse  (validated Pydantic model)

Pipeline expected inside this function:
  1. Encode visual_query using RemoteCLIP text encoder.
  2. Query the FAISS index for the top_k nearest tile embeddings.
  3. (Optionally) filter/re-rank candidates using the location hint.
  4. Return as a CandidateResponse with SemanticCandidate objects.

IMPORTANT — Geographic design rule:
  The location string is a HINT only (used for optional geographic
  filtering of FAISS results). Candidate bbox values from FAISS are the
  ONLY coordinates trusted downstream. Never pass LLM-generated
  coordinates directly into the geospatial pipeline.

IMPORTANT — Score semantics:
  SemanticCandidate.score is a cosine similarity retrieval score in [0, 1].
  It is NOT a probability, confidence percentage, or accuracy measure.
  Do not label it as such in any downstream output.

────────────────────────────────────────────────────────────────────
Mock mode:
  Controlled by MOCK_SEARCH env var (default: true).
  Set MOCK_SEARCH=false once RemoteCLIP + FAISS are connected.

  When mock_search=True  → returns deterministic Bengaluru-area tiles.
  When mock_search=False → calls the real RemoteCLIP + FAISS implementation.
────────────────────────────────────────────────────────────────────
"""

import uuid
import logging
from typing import Optional

from app.schemas.search import CandidateResponse, SemanticCandidate
from app.core.config import settings
from app.core.errors import SearchServiceError

logger = logging.getLogger("geoquery.search")

# ── Mock tile database (Bengaluru area, five representative tiles) ─────────────
# These tiles are only returned when mock_search=True.
_MOCK_TILES: list[SemanticCandidate] = [
    SemanticCandidate(tile_id="blr_001", bbox=[77.58999082, 12.96710844, 77.59920918, 12.97609156], score=0.87, image_url="/tiles/blr_001.png"),
    SemanticCandidate(tile_id="bellandur_001", bbox=[77.65979147, 12.93200844, 77.66900853, 12.94099156], score=0.81, image_url="/tiles/bellandur_001.png"),
    SemanticCandidate(tile_id="hebbal_001", bbox=[77.58338963, 13.03150844, 77.59261037, 13.04049156], score=0.76, image_url="/tiles/hebbal_001.png"),
    SemanticCandidate(tile_id="majestic_001", bbox=[77.56669073, 12.97210844, 77.57590927, 12.98109156], score=0.71, image_url="/tiles/majestic_001.png"),
    SemanticCandidate(tile_id="ulsoor_001", bbox=[77.58999065, 12.97650844, 77.59920935, 12.98549156], score=0.65, image_url="/tiles/ulsoor_001.png"),
]


def semantic_search(
    visual_query: str,
    location: Optional[str] = None,
    top_k: int = 5,
) -> CandidateResponse:
    """
    Retrieve the top-K semantically matching satellite tile candidates.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  MOCK MODE  (MOCK_SEARCH=true, the default)                          │
    │  Returns deterministic Bengaluru-area tiles for development.         │
    │  Set MOCK_SEARCH=false and build the FAISS index to enable real mode.│
    └──────────────────────────────────────────────────────────────────────┘

    Real mode pipeline:
      1. Encode visual_query via RemoteCLIP text encoder
         (optionally using prompt ensemble if PROMPT_ENSEMBLE=true).
      2. Search FAISS index for top_k nearest tile embeddings.
      3. Map FAISS positions → tile metadata (tile_id, bbox).
      4. (Optional) location is available for post-hoc filtering if needed.
      5. Return CandidateResponse.

    Args:
        visual_query: RemoteCLIP text embedding input (from StructuredQuery.visual_query).
        location:     Optional location hint. Available for geographic filtering;
                      never used as raw coordinates in geospatial analysis.
        top_k:        Maximum number of candidates to return.

    Returns:
        CandidateResponse with ranked SemanticCandidate list.
        Note: SemanticCandidate.score is a cosine similarity score, NOT a probability.

    Raises:
        SearchServiceError: If the embedding or FAISS search fails.
    """
    logger.info(
        "SEMANTIC_SEARCH_STARTED visual_query=%r location=%r top_k=%d mode=%s",
        visual_query,
        location,
        top_k,
        "MOCK" if settings.mock_search else "REAL",
    )

    if settings.mock_search:
        # ── MOCK: slice and return pre-defined tiles ───────────────────────────
        candidates = _MOCK_TILES[:min(top_k, len(_MOCK_TILES))]
        result = CandidateResponse(
            query_id=str(uuid.uuid4()),
            visual_query=visual_query,
            location=location,
            candidates=candidates,
        )
        logger.info(
            "CANDIDATES_RETRIEVED [MOCK] count=%d top_score=%.3f",
            len(result.candidates),
            result.candidates[0].score if result.candidates else 0.0,
        )
        return result
        # ── END MOCK ───────────────────────────────────────────────────────────

    # ═══════════════════════════════════════════════════════════════════════════
    #  REAL IMPLEMENTATION — RemoteCLIP + FAISS
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Pipeline:
    #   text → RemoteCLIPEncoder → (1, 512) normalized vector
    #        → FAISSStore.search(top_k) → [{tile_id, bbox, score}, ...]
    #        → SemanticCandidate list → CandidateResponse
    #
    # location is available here for optional post-retrieval geographic
    # filtering, but is NOT passed to FAISS (FAISS is text-only retrieval).
    # ═══════════════════════════════════════════════════════════════════════════

    try:
        from app.services.search.remoteclip import RemoteCLIPEncoder
        from app.services.search.faiss_store import FAISSStore

        encoder = RemoteCLIPEncoder.instance()
        store = FAISSStore.instance()

        # ── Encode query ───────────────────────────────────────────────────────
        if settings.prompt_ensemble:
            # Average over multiple prompt phrasings for better zero-shot recall
            query_vector = encoder.encode_text_ensemble(visual_query)  # (1, 512)
            logger.debug("SEMANTIC_SEARCH using prompt ensemble")
        else:
            query_vector = encoder.encode_text([visual_query])         # (1, 512)

        # ── FAISS retrieval ────────────────────────────────────────────────────
        raw_candidates = store.search(query_vector, top_k=top_k)

        # ── Build response ─────────────────────────────────────────────────────
        candidates = [
            SemanticCandidate(
                tile_id=c["tile_id"],
                bbox=c["bbox"],
                score=float(c["score"]),
                image_url=f"/tiles/{c['tile_id']}.png"
            )
            for c in raw_candidates
        ]

        result = CandidateResponse(
            query_id=str(uuid.uuid4()),
            visual_query=visual_query,
            location=location,
            candidates=candidates,
        )

        logger.info(
            "CANDIDATES_RETRIEVED count=%d top_score=%.3f",
            len(result.candidates),
            result.candidates[0].score if result.candidates else 0.0,
        )
        return result

    except SearchServiceError:
        # Re-raise app-level errors as-is (they have correct HTTP status codes)
        raise
    except Exception as exc:
        raise SearchServiceError(
            f"Semantic search failed unexpectedly: {exc}"
        ) from exc
    # ═══════════════════════════════════════════════════════════════════════════
