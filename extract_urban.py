import rasterio
import planetary_computer
import numpy as np
from pystac_client import Client
from PIL import Image

# Find the Sentinel-2 scene
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[77.58, 12.95, 77.62, 13.01],
    datetime="2026-09-14/2026-09-15"
)

item = next(iter(search.items()))

# Bengaluru city center
lon, lat = 77.5946, 12.9716

# Get pixel location from B04
with rasterio.open(item.assets["B04"].href) as src:
    row, col = src.index(
        781481.0202947541,
        1435426.4069104537
    )

    size = 100
    half = size // 2

    window = rasterio.windows.Window(
        col - half,
        row - half,
        size,
        size
    )

    red = src.read(1, window=window).astype(np.float32)

# Read Green and Blue
with rasterio.open(item.assets["B03"].href) as src:
    green = src.read(1, window=window).astype(np.float32)

with rasterio.open(item.assets["B02"].href) as src:
    blue = src.read(1, window=window).astype(np.float32)

# Percentile stretch
def stretch(band):
    lo, hi = np.percentile(band, [2, 98])
    return np.clip((band - lo) / (hi - lo) * 255, 0, 255)

rgb = np.stack([
    stretch(red),
    stretch(green),
    stretch(blue)
], axis=-1).astype(np.uint8)

Image.fromarray(rgb).save(
    "data/satellite_tiles/urban_001.png"
)

print("CREATED: data/satellite_tiles/urban_001.png")
print("SHAPE:", rgb.shape)
print("CENTER PIXEL:", row, col)
