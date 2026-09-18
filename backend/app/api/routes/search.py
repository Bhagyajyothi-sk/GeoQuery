"""
GeoQueryAI — POST /api/search

Semantic satellite-image search endpoint.
Directly exposes the RemoteCLIP + FAISS retrieval layer for standalone use
(e.g. debugging retrieval quality independently of the full pipeline).
"""

import logging

from fastapi import APIRouter

from app.schemas.search import SearchRequest, CandidateResponse
from app.services.search.interface import semantic_search
from app.core.config import settings

logger = logging.getLogger("geoquery.search")
router = APIRouter()


@router.post(
    "/search",
    response_model=CandidateResponse,
    summary="Semantic satellite-image search (RemoteCLIP + FAISS)",
    description=(
        "Encodes the visual_query using RemoteCLIP and retrieves the top-K most similar "
        "satellite tile candidates from the FAISS index. "
        "Currently returns a deterministic mock response until RemoteCLIP + FAISS are connected."
    ),
    responses={
        200: {"description": "Candidate tiles returned successfully."},
        404: {"description": "No matching tiles found."},
        503: {"description": "Semantic search service unavailable."},
    },
)
async def search(body: SearchRequest) -> CandidateResponse:
    """
    Semantic search endpoint.

    Directly calls the RemoteCLIP + FAISS interface — no AI parsing, no geo analysis.
    Use this to test retrieval quality in isolation.

    HTTP error codes:
      503 — search service failure
      404 — no candidates found
    """
    top_k = body.top_k if body.top_k is not None else settings.default_top_k
    logger.info("search visual_query=%r location=%r top_k=%d", body.visual_query, body.location, top_k)

    result = semantic_search(
        visual_query=body.visual_query,
        location=body.location,
        top_k=top_k,
    )

    logger.info("search candidates_returned=%d", len(result.candidates))
    return result
