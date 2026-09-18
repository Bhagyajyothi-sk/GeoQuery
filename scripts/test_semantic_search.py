#!/usr/bin/env python
"""
GeoQueryAI — Semantic Search Smoke Test.

Runs a real end-to-end semantic search query (RemoteCLIP → FAISS → candidates)
and prints results.  Does NOT use mock mode.

Usage (from project root with .venv active):
    python scripts/test_semantic_search.py

    # Custom query:
    python scripts/test_semantic_search.py "flooded agricultural land"

    # Custom top_k:
    python scripts/test_semantic_search.py "dense forest" --top_k 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Make the backend package importable ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

DEFAULT_QUERY = "large water body"


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoQueryAI semantic search smoke test")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="Visual query string")
    parser.add_argument("--top_k", type=int, default=5, help="Number of candidates (default: 5)")
    args = parser.parse_args()

    query: str = args.query
    top_k: int = args.top_k

    print("=" * 60)
    print("GeoQueryAI — Semantic Search Smoke Test")
    print("=" * 60)
    print(f"\nQuery  : {query!r}")
    print(f"Top-K  : {top_k}")
    print()

    # ── Load RemoteCLIP ────────────────────────────────────────────────────────
    print("Step 1: Loading RemoteCLIP encoder...")
    try:
        from app.services.search.remoteclip import RemoteCLIPEncoder
        encoder = RemoteCLIPEncoder.instance()
        print("        OK\n")
    except Exception as exc:
        print(f"        FAIL: {exc}\n")
        _report_blocked("RemoteCLIP checkpoint is missing or corrupt.")
        sys.exit(1)

    # ── Load FAISS ─────────────────────────────────────────────────────────────
    print("Step 2: Loading FAISS index...")
    try:
        from app.services.search.faiss_store import FAISSStore
        store = FAISSStore.instance()
        print(f"        OK — {store.size} tile(s) indexed\n")
    except Exception as exc:
        print(f"        FAIL: {exc}\n")
        _report_blocked(
            "FAISS index is missing. "
            "Run: python scripts/build_faiss_index.py"
        )
        sys.exit(1)

    if store.size == 0:
        _report_blocked(
            "FAISS index exists but contains 0 tiles.\n"
            "  Add tiles to data/satellite_tiles/ and re-run build_faiss_index.py."
        )
        sys.exit(0)

    # ── Encode query ───────────────────────────────────────────────────────────
    print("Step 3: Encoding query with RemoteCLIP...")
    try:
        query_vector = encoder.encode_text([query])   # (1, 512)
        norm = float((query_vector ** 2).sum() ** 0.5)
        print(f"        OK — shape={query_vector.shape}, norm={norm:.4f}\n")
    except Exception as exc:
        print(f"        FAIL: {exc}\n")
        sys.exit(1)

    # ── FAISS search ───────────────────────────────────────────────────────────
    print("Step 4: Searching FAISS index...")
    try:
        candidates = store.search(query_vector, top_k=top_k)
        print(f"        OK — {len(candidates)} candidate(s) returned\n")
    except Exception as exc:
        print(f"        FAIL: {exc}\n")
        sys.exit(1)

    # ── Print results ──────────────────────────────────────────────────────────
    print("-" * 60)
    print(f"Query: {query}\n")
    print("Top candidates:")
    for i, c in enumerate(candidates, 1):
        print(f"\n  {i}. {c['tile_id']}")
        print(f"     score : {c['score']:.4f}  (cosine similarity, NOT probability)")
        print(f"     bbox  : {c['bbox']}")

    print("\n" + "=" * 60)
    print("Semantic search: PASS")
    print("=" * 60)


def _report_blocked(reason: str) -> None:
    print("=" * 60)
    print("Real satellite retrieval: BLOCKED BY MISSING TILE DATA")
    print(f"\nReason: {reason}")
    print("\nTo enable real retrieval:")
    print("  1. Add satellite tile images to data/satellite_tiles/")
    print("  2. Register them in data/satellite_tiles/metadata.json")
    print("  3. Run: python scripts/build_faiss_index.py")
    print("  4. Set MOCK_SEARCH=false in .env")
    print("=" * 60)


if __name__ == "__main__":
    main()
