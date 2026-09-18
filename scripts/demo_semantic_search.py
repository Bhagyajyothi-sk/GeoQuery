#!/usr/bin/env python
"""
GeoQueryAI — Semantic Retrieval Demonstration Script.

Runs real RemoteCLIP + FAISS queries against the expanded 26-tile satellite index.
Displays Top-K candidates, similarity scores, category, and location metadata.
"""

import os
import sys
import json
import logging

# Ensure backend directory is in sys.path
backend_dir = os.path.join(os.path.dirname(__file__), "..", "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(backend_dir))

from app.services.search.remoteclip import RemoteCLIPEncoder
from app.services.search.faiss_store import FAISSStore


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("geoquery.demo_search")

QUERIES = [
    "a satellite image of a lake",
    "a dense urban area",
    "green vegetation",
    "agricultural land",
    "open barren land",
    "a lake surrounded by an urban area",
]


def run_demo(top_k: int = 5):
    logger.info("============================================================")
    logger.info("GeoQueryAI — RemoteCLIP + FAISS Semantic Search Demo")
    logger.info("============================================================")

    # 1. Load RemoteCLIP & FAISS
    encoder = RemoteCLIPEncoder.instance()
    store = FAISSStore.instance()

    logger.info("Active FAISS index contains %d tiles.", store.size)
    print("\n" + "=" * 70)

    for q_idx, query_str in enumerate(QUERIES, 1):
        print(f"\nQUERY [{q_idx}/{len(QUERIES)}]: '{query_str}'")
        print("-" * 70)

        # Encode text query
        query_vec = encoder.encode_text([query_str])

        # FAISS search
        candidates = store.search(query_vector=query_vec, top_k=top_k)


        for rank, item in enumerate(candidates, 1):
            tile_id = item["tile_id"]
            score = item["score"]
            bbox = item["bbox"]
            center_lat = item.get("center_lat", (bbox[1] + bbox[3]) / 2)
            center_lon = item.get("center_lon", (bbox[0] + bbox[2]) / 2)

            print(
                f"  Rank #{rank} | Tile: {tile_id:<22} | Score: {score:.4f} | Center: ({center_lat:.4f}, {center_lon:.4f})"
            )

    print("\n" + "=" * 70)
    print("Semantic search demonstration complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
