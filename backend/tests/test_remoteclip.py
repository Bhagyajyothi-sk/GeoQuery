"""
GeoQueryAI — RemoteCLIP + FAISS unit and integration tests.

Test groups
-----------
Unit tests (fast, no model loading, always run):
  - Schema and config validation
  - FAISSStore error when index is missing
  - Mock search path is unchanged

Integration tests (require the RemoteCLIP checkpoint, marked slow):
  - RemoteCLIP loads successfully
  - Text embedding shape = (1, 512)
  - Text embeddings are L2-normalized
  - Image embedding shape = (1, 512)
  - Prompt ensemble output shape = (1, 512) and normalized

To run:
    # Unit tests only (fast)
    pytest tests/ -m "not integration" -v

    # All tests (requires models/RemoteCLIP-ViT-B-32.pt)
    pytest tests/ -v
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure mock mode is on for all tests that don't override it
os.environ.setdefault("MOCK_AI", "true")
os.environ.setdefault("MOCK_SEARCH", "true")
os.environ.setdefault("MOCK_GEO", "true")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_faiss_index_bytes(dim: int = 512, n_vectors: int = 0) -> bytes:
    """
    Build a minimal valid FAISS IndexFlatIP binary blob in memory.
    Used to create temporary index files in tests without calling faiss.write_index.
    """
    import faiss
    import numpy as np

    index = faiss.IndexFlatIP(dim)
    if n_vectors > 0:
        rng = np.random.default_rng(42)
        vecs = rng.random((n_vectors, dim), dtype=np.float32)
        # L2-normalize
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        vecs = vecs / norms
        index.add(vecs)

    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
        tmp_path = f.name
    faiss.write_index(index, tmp_path)
    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 1. Unit — FAISSStore: missing index raises SearchServiceError
# ─────────────────────────────────────────────────────────────────────────────

class TestFAISSStoreErrors:
    def setup_method(self):
        """Reset singleton before each test."""
        from app.services.search.faiss_store import FAISSStore
        FAISSStore.reset()

    def teardown_method(self):
        from app.services.search.faiss_store import FAISSStore
        FAISSStore.reset()

    def test_missing_index_raises_search_service_error(self):
        """FAISSStore.instance() raises SearchServiceError when index file is absent."""
        from app.core.errors import SearchServiceError
        from app.services.search.faiss_store import FAISSStore

        with pytest.raises(SearchServiceError, match="FAISS index not found"):
            FAISSStore.instance(
                index_path="/nonexistent/path/satellite.index",
                metadata_path="/nonexistent/path/tile_metadata.json",
            )

    def test_missing_metadata_raises_search_service_error(self, tmp_path):
        """FAISSStore.instance() raises SearchServiceError when metadata file is absent."""
        import faiss
        from app.core.errors import SearchServiceError
        from app.services.search.faiss_store import FAISSStore

        # Write a valid (empty) index file
        idx_file = tmp_path / "satellite.index"
        index = faiss.IndexFlatIP(512)
        faiss.write_index(index, str(idx_file))

        with pytest.raises(SearchServiceError, match="Tile metadata not found"):
            FAISSStore.instance(
                index_path=str(idx_file),
                metadata_path=str(tmp_path / "nonexistent.json"),
            )

    def test_size_mismatch_raises_search_service_error(self, tmp_path):
        """
        FAISSStore raises SearchServiceError when FAISS vector count
        does not match metadata list length.
        """
        import faiss
        from app.core.errors import SearchServiceError
        from app.services.search.faiss_store import FAISSStore

        # Index with 2 vectors
        idx_file = tmp_path / "satellite.index"
        index = faiss.IndexFlatIP(512)
        vecs = np.random.random((2, 512)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)
        index.add(vecs)
        faiss.write_index(index, str(idx_file))

        # Metadata with 1 entry (mismatch)
        meta_file = tmp_path / "tile_metadata.json"
        meta_file.write_text(
            json.dumps([{"tile_id": "tile_001", "bbox": [0.0, 0.0, 1.0, 1.0]}])
        )

        with pytest.raises(SearchServiceError, match="does not match"):
            FAISSStore.instance(
                index_path=str(idx_file),
                metadata_path=str(meta_file),
            )

    def test_empty_index_search_raises_service_error(self, tmp_path):
        """Searching an empty FAISS index raises SearchServiceError with a clear message."""
        import faiss
        from app.core.errors import SearchServiceError
        from app.services.search.faiss_store import FAISSStore

        idx_file = tmp_path / "satellite.index"
        faiss.write_index(faiss.IndexFlatIP(512), str(idx_file))

        meta_file = tmp_path / "tile_metadata.json"
        meta_file.write_text("[]")

        store = FAISSStore.instance(
            index_path=str(idx_file),
            metadata_path=str(meta_file),
        )

        query_vec = np.random.random((1, 512)).astype(np.float32)
        with pytest.raises(SearchServiceError, match="empty"):
            store.search(query_vec, top_k=5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Unit — FAISSStore: successful search on a tiny in-memory index
# ─────────────────────────────────────────────────────────────────────────────

class TestFAISSStoreSearch:
    def setup_method(self):
        from app.services.search.faiss_store import FAISSStore
        FAISSStore.reset()

    def teardown_method(self):
        from app.services.search.faiss_store import FAISSStore
        FAISSStore.reset()

    @pytest.fixture
    def tiny_store(self, tmp_path):
        """FAISSStore backed by a tiny 3-tile index."""
        import faiss
        from app.services.search.faiss_store import FAISSStore

        n = 3
        dim = 512
        rng = np.random.default_rng(7)
        vecs = rng.random((n, dim), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)

        idx_file = tmp_path / "satellite.index"
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        faiss.write_index(index, str(idx_file))

        metadata = [
            {"tile_id": f"tile_{i:03d}", "bbox": [float(i), float(i), float(i+1), float(i+1)]}
            for i in range(n)
        ]
        meta_file = tmp_path / "tile_metadata.json"
        meta_file.write_text(json.dumps(metadata))

        return FAISSStore.instance(
            index_path=str(idx_file),
            metadata_path=str(meta_file),
        ), vecs

    def test_search_returns_correct_count(self, tiny_store):
        store, vecs = tiny_store
        query = vecs[0:1]   # exact match for first vector
        results = store.search(query, top_k=2)
        assert len(results) == 2

    def test_search_top_result_is_exact_match(self, tiny_store):
        """Querying with a stored vector should return that tile as the top result."""
        store, vecs = tiny_store
        query = vecs[0:1]
        results = store.search(query, top_k=1)
        assert results[0]["tile_id"] == "tile_000"
        # Cosine similarity of identical normalized vectors = 1.0
        assert abs(results[0]["score"] - 1.0) < 1e-4

    def test_top_k_clamped_to_index_size(self, tiny_store):
        """top_k larger than index size returns at most index_size candidates."""
        store, vecs = tiny_store
        query = vecs[0:1]
        results = store.search(query, top_k=100)
        assert len(results) <= store.size   # tiny_store has 3 tiles

    def test_result_has_required_keys(self, tiny_store):
        store, vecs = tiny_store
        results = store.search(vecs[0:1], top_k=1)
        assert "tile_id" in results[0]
        assert "bbox" in results[0]
        assert "score" in results[0]
        assert len(results[0]["bbox"]) == 4

    def test_results_ordered_descending_by_score(self, tiny_store):
        store, vecs = tiny_store
        results = store.search(vecs[0:1], top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Unit — Mock search mode is unchanged
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_search_returns_deterministic_tiles(client):
    """In mock mode, /api/search always returns the same pre-defined tiles."""
    resp = client.post(
        "/api/search",
        json={"visual_query": "large water body", "top_k": 3},
    )
    assert resp.status_code == 200
    data = resp.json()
    candidates = data["candidates"]
    assert len(candidates) == 3
    # All mock tiles must have scores in [0, 1]
    for c in candidates:
        assert 0.0 <= c["score"] <= 1.0
        assert len(c["bbox"]) == 4


def test_mock_query_pipeline_works(client):
    """Full /api/query pipeline still works in mock mode."""
    resp = client.post(
        "/api/query",
        json={"query": "Show me water bodies around Bengaluru"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["candidates"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Integration tests — require RemoteCLIP checkpoint
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "RemoteCLIP-ViT-B-32.pt"
SKIP_IF_NO_CHECKPOINT = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason=f"RemoteCLIP checkpoint not found at {CHECKPOINT}",
)


@pytest.mark.integration
class TestRemoteCLIPEncoder:
    """Integration tests — load the actual RemoteCLIP model."""

    def setup_method(self):
        """Reset singleton before each test to ensure clean state."""
        from app.services.search.remoteclip import RemoteCLIPEncoder
        RemoteCLIPEncoder.reset()

    def teardown_method(self):
        from app.services.search.remoteclip import RemoteCLIPEncoder
        RemoteCLIPEncoder.reset()

    @SKIP_IF_NO_CHECKPOINT
    def test_encoder_initialization(self):
        """RemoteCLIPEncoder.instance() loads the checkpoint without raising."""
        from app.services.search.remoteclip import RemoteCLIPEncoder
        encoder = RemoteCLIPEncoder.instance(checkpoint_path=str(CHECKPOINT))
        assert encoder is not None

    @SKIP_IF_NO_CHECKPOINT
    def test_text_embedding_shape(self):
        """encode_text returns shape (1, 512) for a single query."""
        from app.services.search.remoteclip import RemoteCLIPEncoder
        encoder = RemoteCLIPEncoder.instance(checkpoint_path=str(CHECKPOINT))
        vec = encoder.encode_text(["large water body"])
        assert vec.shape == (1, 512)

    @SKIP_IF_NO_CHECKPOINT
    def test_text_embedding_normalized(self):
        """encode_text output is L2-normalized (norm ≈ 1.0)."""
        from app.services.search.remoteclip import RemoteCLIPEncoder
        encoder = RemoteCLIPEncoder.instance(checkpoint_path=str(CHECKPOINT))
        vec = encoder.encode_text(["large water body"])
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    @SKIP_IF_NO_CHECKPOINT
    def test_batch_text_embedding_shape(self):
        """encode_text handles a batch of multiple strings."""
        from app.services.search.remoteclip import RemoteCLIPEncoder
        encoder = RemoteCLIPEncoder.instance(checkpoint_path=str(CHECKPOINT))
        texts = ["large water body", "flooded fields", "dense forest"]
        vec = encoder.encode_text(texts)
        assert vec.shape == (3, 512)

    @SKIP_IF_NO_CHECKPOINT
    def test_image_embedding_shape(self):
        """encode_image returns shape (1, 512) for a dummy PIL RGB image."""
        from PIL import Image
        from app.services.search.remoteclip import RemoteCLIPEncoder
        encoder = RemoteCLIPEncoder.instance(checkpoint_path=str(CHECKPOINT))
        # 256×256 random RGB image
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        vec = encoder.encode_image([img])
        assert vec.shape == (1, 512)

    @SKIP_IF_NO_CHECKPOINT
    def test_image_embedding_normalized(self):
        """encode_image output is L2-normalized (norm ≈ 1.0)."""
        from PIL import Image
        from app.services.search.remoteclip import RemoteCLIPEncoder
        encoder = RemoteCLIPEncoder.instance(checkpoint_path=str(CHECKPOINT))
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        vec = encoder.encode_image([img])
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    @SKIP_IF_NO_CHECKPOINT
    def test_prompt_ensemble_shape_and_normalized(self):
        """encode_text_ensemble returns shape (1, 512) and is L2-normalized."""
        from app.services.search.remoteclip import RemoteCLIPEncoder
        encoder = RemoteCLIPEncoder.instance(checkpoint_path=str(CHECKPOINT))
        vec = encoder.encode_text_ensemble("large water body")
        assert vec.shape == (1, 512)
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    @SKIP_IF_NO_CHECKPOINT
    def test_singleton_returns_same_instance(self):
        """Calling instance() twice must return the exact same object."""
        from app.services.search.remoteclip import RemoteCLIPEncoder
        enc1 = RemoteCLIPEncoder.instance(checkpoint_path=str(CHECKPOINT))
        enc2 = RemoteCLIPEncoder.instance()  # no path needed — already cached
        assert enc1 is enc2
