"""
GeoQueryAI — Satellite Tile Generator.

Cuts a large satellite scene composite into fixed-size overlapping tiles
suitable for RemoteCLIP image embedding.

This previously lived (unreachable) inside geocoding_service.py, where its
`Path` and `Image` dependencies were never imported. It belongs with the
search layer: its output feeds scripts/build_faiss_index.py.

Usage:
    python -m app.services.search.tiling path/to/composite.png

Tile geometry:
    tile size : 224 x 224 px  (RemoteCLIP ViT-B-32 input resolution)
    overlap   : 20 %          (so features near tile edges are not lost)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

from PIL import Image

logger = logging.getLogger("geoquery.tiling")

TILE_SIZE = 224
OVERLAP = 0.20
DEFAULT_OUTPUT_DIR = Path("data/tiles")


def create_tiles(
    image_path: Union[str, Path],
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    tile_size: int = TILE_SIZE,
    overlap: float = OVERLAP,
) -> List[Path]:
    """
    Split an image into overlapping square tiles and write them to disk.

    Args:
        image_path: Source composite image (any format Pillow can read).
        output_dir: Directory for the generated tiles; created if absent.
        tile_size:  Edge length of each square tile, in pixels.
        overlap:    Fractional overlap between neighbouring tiles, in [0, 1).

    Returns:
        Paths of the tiles written, in generation order.

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError:        If tile_size or overlap is out of range.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    tiles_dir = Path(output_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    if width < tile_size or height < tile_size:
        raise ValueError(
            f"Image {width}x{height} is smaller than the tile size {tile_size}x{tile_size}."
        )

    step = max(1, int(tile_size * (1 - overlap)))
    written: List[Path] = []

    for y in range(0, height - tile_size + 1, step):
        for x in range(0, width - tile_size + 1, step):
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            tile_path = tiles_dir / f"tile_{len(written):04d}.png"
            tile.save(tile_path)
            written.append(tile_path)

    logger.info("Created %d tiles from %s into %s", len(written), image_path, tiles_dir)
    return written


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Cut a satellite composite into tiles")
    parser.add_argument("image", help="Path to the source composite image")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    parser.add_argument("--overlap", type=float, default=OVERLAP)
    args = parser.parse_args()

    paths = create_tiles(
        image_path=args.image,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )
    print(f"Created {len(paths)} tiles in {args.output_dir}")
