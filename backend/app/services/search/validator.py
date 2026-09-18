"""
GeoQueryAI — Candidate Validation Service.

Module Role in GeoQueryAI Architecture:
  1. RemoteCLIP           = Semantic candidate generation (visual concept matching)
  2. Candidate Validator  = Spatial consistency & data integrity filter (verifies spatial location)
  3. STAC Service         = Satellite scene data availability verification
  4. Geospatial Engine    = Actual physical pixel measurement (NDVI, NDWI, extent)

This service ensures RemoteCLIP similarity scores are NOT treated as geographic certainty.
It validates Top-K semantic search results against spatial bounds and location hints.
"""

import logging
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field

from app.schemas.search import SemanticCandidate
from app.schemas.query import GeographicResult

logger = logging.getLogger("geoquery.validator")


def is_spatial_match(
    candidate_bbox: List[float],
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    loc_bbox: Optional[List[float]] = None,
    buffer_deg: float = 0.002,
) -> Tuple[bool, bool, str]:
    """
    Check if a geographic point (lat, lon) or location bbox falls inside or reasonably
    near a candidate tile bbox [min_lon, min_lat, max_lon, max_lat].

    Args:
        candidate_bbox: [min_lon, min_lat, max_lon, max_lat] of the candidate tile.
        lat: Latitude of the geocoded location centroid.
        lon: Longitude of the geocoded location centroid.
        loc_bbox: [min_lon, min_lat, max_lon, max_lat] of the geocoded location.
        buffer_deg: Spatial buffer margin in degrees (~0.002° ≈ 220 m).

    Returns:
        Tuple[bool, bool, str]: (is_exact_match, is_buffered_match, description_reason)
    """
    if not candidate_bbox or len(candidate_bbox) != 4:
        return False, False, f"Invalid candidate bbox format: {candidate_bbox}"

    min_lon, min_lat, max_lon, max_lat = candidate_bbox

    # 1. Centroid point check
    if lat is not None and lon is not None:
        exact = (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)
        if exact:
            return True, True, f"Location ({lat:.4f}, {lon:.4f}) is inside candidate bbox {candidate_bbox}"

        buf_min_lon = min_lon - buffer_deg
        buf_max_lon = max_lon + buffer_deg
        buf_min_lat = min_lat - buffer_deg
        buf_max_lat = max_lat + buffer_deg
        buffered = (buf_min_lon <= lon <= buf_max_lon) and (buf_min_lat <= lat <= buf_max_lat)

        if buffered:
            return False, True, f"Location ({lat:.4f}, {lon:.4f}) is near candidate bbox within {buffer_deg}° buffer"
        else:
            return False, False, f"Location ({lat:.4f}, {lon:.4f}) is outside candidate tile bbox {candidate_bbox}"

    # 2. Location bbox intersection check if centroid unavailable
    if loc_bbox and len(loc_bbox) == 4:
        l_min_lon, l_min_lat, l_max_lon, l_max_lat = loc_bbox
        exact_intersects = not (
            l_max_lon < min_lon or
            l_min_lon > max_lon or
            l_max_lat < min_lat or
            l_min_lat > max_lat
        )
        if exact_intersects:
            return True, True, f"Location bbox intersects candidate tile bbox {candidate_bbox}"

        buffered_intersects = not (
            l_max_lon < min_lon - buffer_deg or
            l_min_lon > max_lon + buffer_deg or
            l_max_lat < min_lat - buffer_deg or
            l_min_lat > max_lat + buffer_deg
        )
        if buffered_intersects:
            return False, True, f"Location bbox intersects candidate bbox within buffer"
        else:
            return False, False, f"Location bbox does not intersect candidate tile bbox {candidate_bbox}"

    # 3. No coordinates supplied
    return True, True, "No location coordinates specified — spatial check passed by default"


class CandidateValidationDetail(BaseModel):
    """Inspectable validation log for a single candidate tile."""

    tile_id: str
    score: float
    spatial_match: bool
    passed: bool
    reason: str


class CandidateValidationResult(BaseModel):
    """
    Summary of candidate validation and tile selection.
    """

    selected_candidate: Optional[SemanticCandidate] = None
    validation_status: str = Field(
        ...,
        description="One of: 'passed' (valid candidate selected), 'rejected' (candidates failed spatial check), 'no_candidates' (empty candidate list).",
    )
    validation_reason: str = Field(..., description="Human-readable rationale for selection or rejection.")
    details: List[CandidateValidationDetail] = Field(default_factory=list, description="Per-candidate validation logs.")


def validate_candidates(
    candidates: List[SemanticCandidate],
    geographic_result: Optional[GeographicResult] = None,
    min_score_threshold: float = 0.0,
    buffer_deg: float = 0.002,
) -> CandidateValidationResult:
    """
    Validate Top-K RemoteCLIP + FAISS candidates against geographic constraints.

    Selection strategy:
      - Iterate candidates in rank order (highest similarity score first).
      - If geographic_result is present and grounded:
        Check spatial match. Prioritizes exact spatial containment over buffered matches.
      - If no location was supplied:
        Accept candidates based on semantic similarity rank.
      - Select the candidate with spatial consistency.
      - If no candidate satisfies spatial consistency, set selected_candidate=None
        and validation_status='rejected'.

    Args:
        candidates: List of SemanticCandidate tiles.
        geographic_result: Optional geocoding result.
        min_score_threshold: Minimum similarity threshold.
        buffer_deg: Spatial buffer margin in degrees (~0.002° ≈ 220 m).

    Returns:
        CandidateValidationResult: Evaluated selection result.
    """
    if not candidates:
        return CandidateValidationResult(
            selected_candidate=None,
            validation_status="no_candidates",
            validation_reason="No candidate tiles were returned by the retrieval layer.",
            details=[],
        )

    has_grounded_location = (
        geographic_result is not None
        and geographic_result.grounded
        and (geographic_result.latitude is not None or geographic_result.bbox is not None)
    )

    details: List[CandidateValidationDetail] = []
    exact_match_candidate: Optional[SemanticCandidate] = None
    buffered_match_candidate: Optional[SemanticCandidate] = None

    for candidate in candidates:
        if candidate.score < min_score_threshold:
            details.append(
                CandidateValidationDetail(
                    tile_id=candidate.tile_id,
                    score=candidate.score,
                    spatial_match=False,
                    passed=False,
                    reason=f"Similarity score {candidate.score:.3f} below minimum threshold {min_score_threshold}",
                )
            )
            continue

        if has_grounded_location:
            exact, buffered, reason = is_spatial_match(
                candidate_bbox=candidate.bbox,
                lat=geographic_result.latitude if geographic_result else None,
                lon=geographic_result.longitude if geographic_result else None,
                loc_bbox=geographic_result.bbox if geographic_result else None,
                buffer_deg=buffer_deg,
            )
        else:
            exact, buffered, reason = True, True, "No grounded location supplied — spatial check passed by default"

        match = exact or buffered
        detail = CandidateValidationDetail(
            tile_id=candidate.tile_id,
            score=candidate.score,
            spatial_match=match,
            passed=match,
            reason=reason,
        )
        details.append(detail)

        if exact and exact_match_candidate is None:
            exact_match_candidate = candidate
        elif buffered and buffered_match_candidate is None:
            buffered_match_candidate = candidate

    selected_candidate = exact_match_candidate or buffered_match_candidate

    if selected_candidate is not None:
        status = "passed"
        loc_name = geographic_result.name if geographic_result else "unspecified location"
        summary_reason = (
            f"Selected candidate tile '{selected_candidate.tile_id}' (score: {selected_candidate.score:.3f}) "
            f"matching geographic location '{loc_name}'."
        )
    else:
        status = "rejected"
        loc_name = geographic_result.name if geographic_result else "specified location"
        summary_reason = f"None of the {len(candidates)} candidate tiles matched geographic constraints for '{loc_name}'."

    return CandidateValidationResult(
        selected_candidate=selected_candidate,
        validation_status=status,
        validation_reason=summary_reason,
        details=details,
    )
