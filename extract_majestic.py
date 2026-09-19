import rasterio
import planetary_computer
import numpy as np
from pystac_client import Client
from PIL import Image

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[77.56, 12.96, 77.59, 12.99],
    datetime="2026-09-14/2026-09-15"
)

item = next(iter(search.items()))

row, col = 6408, 7912
size = 100
half = size // 2

window = rasterio.windows.Window(
    col - half,
    row - half,
    size,
    size
)

bands = {}

for name in ["B04", "B03", "B02"]:
    with rasterio.open(item.assets[name].href) as src:
        bands[name] = src.read(1, window=window).astype(np.float32)

def stretch(band):
    lo, hi = np.percentile(band, [2, 98])
    return np.clip((band - lo) / (hi - lo) * 255, 0, 255)

rgb = np.stack([
    stretch(bands["B04"]),
    stretch(bands["B03"]),
    stretch(bands["B02"])
], axis=-1).astype(np.uint8)

Image.fromarray(rgb).save(
    "data/satellite_tiles/majestic_001.png"
)

print("CREATED: data/satellite_tiles/majestic_001.png")
print("SHAPE:", rgb.shape)
print("CENTER PIXEL:", row, col)
