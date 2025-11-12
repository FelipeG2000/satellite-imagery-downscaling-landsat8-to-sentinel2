"""
One-click georeferencing: wrap a super-resolved PNG into a GeoTIFF
by copying CRS and (scaled) transform from a reference GeoTIFF.

Edit the DEFAULTS section and run:
  python scripts/attach_georef.py
"""

from pathlib import Path
from src.downscaling.georef import wrap_png_with_georef

# ---------- DEFAULTS (edit here) ----------
REFERENCE_TIF = "data/raw/L8_reference.tif"           # <-- set your reference L8 GeoTIFF path
SR_PNG        = "data/processed/l8_rgb_swinir_x4.png" # <-- result from your SR pipeline (PNG)
OUTPUT_TIF    = "data/processed/l8_rgb_swinir_x4_georef.tif"
UPSCALE       = 4
# -----------------------------------------

def main():
    root = Path(__file__).resolve().parents[1]
    ref = root / REFERENCE_TIF
    png = root / SR_PNG
    out = root / OUTPUT_TIF

    if not ref.exists():
        raise FileNotFoundError(f"Reference GeoTIFF not found: {ref}")
    if not png.exists():
        raise FileNotFoundError(f"SR PNG not found: {png}")

    wrap_png_with_georef(ref, png, out, UPSCALE)
    print(f"✅ Saved georeferenced GeoTIFF: {out}")

if __name__ == "__main__":
    main()