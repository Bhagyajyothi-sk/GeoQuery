import rasterio
import planetary_computer
from pyproj import Transformer

url = """PASTE_B04_URL_HERE"""

lon, lat = 77.5946, 12.9716

t = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32643",
    always_xy=True
)

x, y = t.transform(lon, lat)

print("UTM:", x, y)

with rasterio.open(planetary_computer.sign(url)) as src:
    print("BOUNDS:", src.bounds)
    row, col = src.index(x, y)
    print("PIXEL:", row, col)
    print(
        "INSIDE:",
        0 <= row < src.height and
        0 <= col < src.width
    )
