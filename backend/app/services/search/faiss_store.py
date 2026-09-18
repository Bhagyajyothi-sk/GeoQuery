"""
GeoQueryAI — FAISS Index Store.

Manages the FAISS flat inner-product index and the tile metadata mapping.
Separated from remoteclip.py so the index can be replaced or reloaded
independently of the encoder.

Index format:
  indexes/satellite.index     — faiss.IndexFlatIP(512)
  indexes/tile_metadata.json  — list of {tile_id, bbox, path, ...}
                                ordered to match FAISS vector positions

FAISS IDs are positional: FAISS position i → tile_metadata[i].
The build script (scripts/build_faiss_index.py) guarantees this ordering.

Score semantics:
  The raw FAISS inner-product score on L2-normalized vectors equals
  cosine similarity ∈ [0, 1].  It is NOT a probability or confidence.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("geoquery.faiss_store")

EMBEDDING_DIM = 512


class FAISSStore:
    """
    Singleton managing the FAISS index and tile metadata.

    Usage:
        store = FAISSStore.instance()
        results = store.search(query_vector, top_k=5)
        # results: [{"tile_id": ..., "bbox": [...], "score": ...}, ...]
    """

    _lock: threading.Lock = threading.Lock()
    _store: Optional["FAISSStore"] = None

    def __init__(self, index, metadata: list[dict]) -> None:
        self._index = index
        self._metadata = metadata

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal

    # ── Singleton ──────────────────────────────────────────────────────────────

    @classmethod
    def instance(
        cls,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> "FAISSStore":
        """
        Return the singleton store, loading the index on first call.

        Args:
            index_path:    Path to the FAISS index file (overrides settings).
            metadata_path: Path to the tile metadata JSON (overrides settings).

        Raises:
            SearchServiceError: If the index or metadata file is missing/corrupt.
        """
        if cls._store is not None:
            return cls._store

        with cls._lock:
            if cls._store is not None:
                return cls._store
            cls._store = cls._load(index_path, metadata_path)
            return cls._store

    @classmethod
    def _load(
        cls,
        index_path: Optional[str],
        metadata_path: Optional[str],
    ) -> "FAISSStore":
        """Load FAISS index and tile metadata. Called exactly once."""
        from app.core.config import settings
        from app.core.errors import SearchServiceError

        try:
            import faiss  # noqa: PLC0415
        except ImportError as exc:
            raise SearchServiceError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from exc

        # ── Resolve project root ───────────────────────────────────────────────
        # This file: <project_root>/backend/app/services/search/faiss_store.py
        # parents: [0]=search, [1]=services, [2]=app, [3]=backend, [4]=project_root
        project_root = Path(__file__).resolve().parents[4]

        # ── Resolve index path ─────────────────────────────────────────────────
        raw_idx = index_path or settings.faiss_index_path
        if not raw_idx:
            raise SearchServiceError(
                "faiss_index_path is not configured. "
                "Set FAISS_INDEX_PATH in your .env file or run build_faiss_index.py."
            )
        idx_file = Path(raw_idx)
        if not idx_file.is_absolute():
            idx_file = project_root / raw_idx

        if not idx_file.exists():
            raise SearchServiceError(
                f"FAISS index not found: {idx_file}. "
                "Run scripts/build_faiss_index.py to build the index first."
            )

        # ── Resolve metadata path ──────────────────────────────────────────────
        raw_meta = metadata_path or settings.tile_metadata_path
        if not raw_meta:
            raise SearchServiceError(
                "tile_metadata_path is not configured. "
                "Set TILE_METADATA_PATH in your .env file."
            )
        meta_file = Path(raw_meta)
        if not meta_file.is_absolute():
            meta_file = project_root / raw_meta

        if not meta_file.exists():
            raise SearchServiceError(
                f"Tile metadata not found: {meta_file}. "
                "Run scripts/build_faiss_index.py to build the index first."
            )

        # ── Load ───────────────────────────────────────────────────────────────
        logger.info("FAISSStore: loading index from %s ...", idx_file)
        index = faiss.read_index(str(idx_file))

        logger.info("FAISSStore: loading metadata from %s ...", meta_file)
        with open(meta_file, encoding="utf-8") as f:
            metadata: list[dict] = json.load(f)

        # ── Sanity check ───────────────────────────────────────────────────────
        if index.ntotal != len(metadata):
            raise SearchServiceError(
                f"FAISS index size ({index.ntotal}) does not match "
                f"metadata length ({len(metadata)}). "
                "Re-run scripts/build_faiss_index.py."
            )

        logger.info(
            "FAISSStore: ready | tiles=%d | dim=%d",
            index.ntotal,
            index.d,
        )
        return cls(index, metadata)

    @classmethod
    def reset(cls) -> None:
        """Release the singleton (used in tests to force reload)."""
        with cls._lock:
            cls._store = None

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        """
        Search the FAISS index for the top-K nearest tile embeddings.

        Args:
            query_vector: float32 numpy array of shape (1, 512), L2-normalized.
            top_k:        Number of nearest neighbors to retrieve.

        Returns:
            List of dicts (ordered by descending score):
            [{"tile_id": str, "bbox": [float, ...], "score": float}, ...]

        Notes:
            - score is cosine similarity (inner product of normalized vectors) ∈ [0, 1].
            - If top_k > index size, returns all available candidates.
            - FAISS returns index -1 for unfilled slots; these are skipped.
        """
        from app.core.errors import SearchServiceError

        if self._index.ntotal == 0:
            raise SearchServiceError(
                "FAISS index is empty — no satellite tiles have been indexed. "
                "Add tiles to data/satellite_tiles/ and run build_faiss_index.py."
            )

        # Clamp top_k to index size
        effective_k = min(top_k, self._index.ntotal)
        if effective_k < top_k:
            logger.warning(
                "FAISSStore: top_k=%d > index size %d; returning %d candidates.",
                top_k,
                self._index.ntotal,
                effective_k,
            )

        # FAISS expects float32 C-contiguous (1, dim)
        vec = np.ascontiguousarray(query_vector, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[np.newaxis, :]  # (512,) → (1, 512)

        scores, indices = self._index.search(vec, effective_k)

        candidates = []
        for rank in range(effective_k):
            idx = int(indices[0][rank])
            if idx == -1:
                continue  # FAISS sentinel for unfilled slot
            tile = self._metadata[idx]
            candidates.append(
                {
                    "tile_id": tile["tile_id"],
                    "bbox": tile["bbox"],
                    "score": float(scores[0][rank]),
                }
            )

        logger.debug(
            "FAISSStore: search returned %d candidates (top score=%.4f)",
            len(candidates),
            candidates[0]["score"] if candidates else 0.0,
        )
        return candidates
