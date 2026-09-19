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
    bbox=[77.56, 12.96, 77.62, 13.01],
    datetime="2026-06-16/2026-06-17"
)

item = next(iter(search.items()))

# Central Bengaluru UTM coordinates
x = 781481.0202947541
y = 1435426.4069104537

with rasterio.open(item.assets["B04"].href) as src:
    row, col = src.index(x, y)

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
    return np.clip(
        (band - lo) / (hi - lo) * 255,
        0,
        255
    )

rgb = np.stack([
    stretch(bands["B04"]),
    stretch(bands["B03"]),
    stretch(bands["B02"])
], axis=-1).astype(np.uint8)

Image.fromarray(rgb).save(
    "data/satellite_tiles/urban_002.png"
)

print("SCENE:", item.id)
print("CREATED: data/satellite_tiles/urban_002.png")
print("SHAPE:", rgb.shape)
print("CENTER PIXEL:", row, col)