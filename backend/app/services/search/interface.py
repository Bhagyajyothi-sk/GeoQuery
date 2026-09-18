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
    SemanticCandidate(tile_id="tile_001", bbox=[77.50, 12.90, 77.65, 13.05], score=0.87),
    SemanticCandidate(tile_id="tile_002", bbox=[77.55, 12.85, 77.70, 13.00], score=0.81),
    SemanticCandidate(tile_id="tile_003", bbox=[77.45, 12.95, 77.60, 13.10], score=0.76),
    SemanticCandidate(tile_id="tile_004", bbox=[77.60, 12.80, 77.75, 12.95], score=0.71),
    SemanticCandidate(tile_id="tile_005", bbox=[77.40, 13.00, 77.55, 13.15], score=0.65),
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
    │  Set MOCK_SEARCH=false and implement the real block below.           │
    └──────────────────────────────────────────────────────────────────────┘

    RemoteCLIP + FAISS integration guide:
      1. Load RemoteCLIP model at startup (not per-call!) via lifespan hook in main.py.
      2. Encode visual_query:  embedding = remoteclip_model.encode_text(visual_query)
      3. Search FAISS index:   distances, indices = faiss_index.search(embedding, top_k)
      4. Map indices → tile metadata (tile_id, bbox) from your tile registry.
      5. Optionally filter results by location bbox from the geocoding stage.
      6. Return CandidateResponse.

    Args:
        visual_query: RemoteCLIP text embedding input. Comes from StructuredQuery.visual_query.
        location:     Optional location hint for geographic pre-filtering. Never used as
                      raw coordinates — it is a geocoded name only.
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
    #  REAL IMPLEMENTATION — insert your RemoteCLIP + FAISS code here
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # Prerequisites (set these up in the lifespan hook in main.py):
    #   remoteclip_model  — loaded RemoteCLIP model instance
    #   faiss_index       — loaded faiss.Index instance
    #   tile_db           — dict/list mapping FAISS index position → tile metadata
    #
    # try:
    #     embedding = remoteclip_model.encode_text(visual_query)          # step 2
    #     distances, indices = faiss_index.search(embedding, top_k)       # step 3
    #     candidates = [
    #         SemanticCandidate(
    #             tile_id=tile_db[idx]["tile_id"],
    #             bbox=tile_db[idx]["bbox"],
    #             score=float(distances[0][rank]),   # cosine similarity in [0, 1]
    #         )
    #         for rank, idx in enumerate(indices[0])
    #         if idx != -1   # FAISS returns -1 for unfilled slots
    #     ]
    #     result = CandidateResponse(
    #         query_id=str(uuid.uuid4()),
    #         visual_query=visual_query,
    #         location=location,
    #         candidates=candidates,
    #     )
    #     logger.info(
    #         "CANDIDATES_RETRIEVED count=%d top_score=%.3f",
    #         len(result.candidates),
    #         result.candidates[0].score if result.candidates else 0.0,
    #     )
    #     return result
    # except Exception as exc:
    #     raise SearchServiceError(f"Semantic search failed: {exc}") from exc
    #
    # ═══════════════════════════════════════════════════════════════════════════

    raise SearchServiceError(
        "RemoteCLIP + FAISS implementation not yet connected. "
        "Set MOCK_SEARCH=true to use mock mode, or plug in your implementation above."
    )
