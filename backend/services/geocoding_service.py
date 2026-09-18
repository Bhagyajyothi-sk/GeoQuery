from typing import Optional, Dict, Any
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {
    "User-Agent": "GeoQuery-AI-GeocodingService/1.0"
}

_GEOCODE_CACHE: Dict[str, Dict[str, Any]] = {}


def geocode_location(location: str) -> Optional[Dict[str, Any]]:
    """
    Convert a place name into latitude, longitude, and bounding box
    using the OpenStreetMap Nominatim geocoding service with in-memory caching.
    """
    if not location or not location.strip():
        return None

    clean_loc = location.strip().lower()
    if clean_loc in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[clean_loc]

    params = {
        "q": location,
        "format": "json",
        "limit": 1
    }

    try:
        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        results = response.json()

        if not results:
            return None

        result = results[0]
        lat = float(result["lat"])
        lon = float(result["lon"])

        # bbox from nominatim: [min_lat, max_lat, min_lon, max_lon]
        raw_bbox = result.get("boundingbox")
        bbox = None
        if raw_bbox and len(raw_bbox) == 4:
            bbox = {
                "min_lat": float(raw_bbox[0]),
                "max_lat": float(raw_bbox[1]),
                "min_lon": float(raw_bbox[2]),
                "max_lon": float(raw_bbox[3])
            }

        resolved = {
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "display_name": result.get("display_name", location),
            "bbox": bbox
        }
        _GEOCODE_CACHE[clean_loc] = resolved
        return resolved

    except Exception as e:
        # Graceful failure handling
        return None


def create_tiles(image_path):
    tiles_dir = Path("data/tiles")
    tiles_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    tile_size = 224
    overlap = 0.20
    step = int(tile_size * (1 - overlap))

    tile_count = 0

    for y in range(0, height - tile_size + 1, step):
        for x in range(0, width - tile_size + 1, step):
            tile = image.crop(
                (x, y, x + tile_size, y + tile_size)
            )

            tile_path = tiles_dir / f"tile_{tile_count:04d}.png"
            tile.save(tile_path)

            tile_count += 1

    print(f"Created {tile_count} tiles.")


if __name__ == "__main__":
    scene = search_sentinel_scene()
    composite_path = create_rgb_composite(scene)
    create_tiles(composite_path)
