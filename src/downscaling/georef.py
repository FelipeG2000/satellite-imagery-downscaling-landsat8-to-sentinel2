"""
Georeferencing helpers to wrap super-resolved images (e.g., PNG) into GeoTIFF
using metadata from a reference GeoTIFF (CRS + transform), adjusting the pixel
size for a given upscale factor (e.g., x4).

Assumptions:
- The super-resolved image is aligned with the original UL corner (no crop/shift).
- The only change is pixel size: new_size = original_size / scale.
- Use this when your SR output is an RGB PNG (H x W x 3, uint8) or a single-band array.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from pathlib import Path
from PIL import Image
import rasterio
from rasterio.transform import Affine


def load_png_rgb_as_chw(png_path: str | Path) -> np.ndarray:
    """
    Loads a PNG as CHW (3, H, W) uint8.

    Args:
        png_path: path to an 8-bit RGB PNG.

    Returns:
        np.ndarray with shape (3, H, W), dtype=uint8
    """
    arr = np.array(Image.open(png_path))  # H x W x 3
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA PNG: got shape {arr.shape}")
    if arr.shape[2] == 4:  # drop alpha
        arr = arr[:, :, :3]
    return np.moveaxis(arr, 2, 0).astype(np.uint8)  # -> (3, H, W)


def compute_scaled_transform(transform: Affine, scale: int | float) -> Affine:
    """
    Returns a new Affine with pixel size divided by 'scale'.
    (i.e., higher spatial resolution)

    Original:
        x = c + a*col + b*row
        y = f + d*col + e*row
    We keep (b, c, d, f) and divide (a, e) by 'scale'.

    Args:
        transform: original Affine from reference GeoTIFF.
        scale: upscale factor (e.g., 4).

    Returns:
        new Affine with a/scale and e/scale.
    """
    return Affine(transform.a / scale, transform.b, transform.c,
                  transform.d, transform.e / scale, transform.f)


def write_geotiff_like(
    reference_tif: str | Path,
    out_array: np.ndarray,
    out_tif: str | Path,
    compress: str = "LZW",
    photometric: Optional[str] = None
) -> None:
    """
    Writes 'out_array' to GeoTIFF using CRS/transform from 'reference_tif'.
    NOTE: Assumes 'out_array' already has the desired shape (C, H, W) and that
    its transform has been pre-adjusted if resolution changed.

    Args:
        reference_tif: path to the original georeferenced GeoTIFF.
        out_array: array shaped (C, H, W), dtype typically uint8/float32.
        out_tif: output GeoTIFF path.
        compress: GDAL compression (e.g., LZW, DEFLATE).
        photometric: e.g. "RGB" if count=3 and dtype=uint8; None lets GDAL infer.

    Behavior:
        - Copies CRS from reference.
        - Uses provided out_array.shape for size and band count.
        - Requires transform to be provided by caller via profile["transform"].
    """
    if out_array.ndim == 2:
        out_array = out_array[np.newaxis, :, :]

    C, H, W = out_array.shape
    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()

    profile.update({
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": C,
        "compress": compress,
        # 'transform' must be set by caller if resolution changed.
        # We keep whatever 'transform' remains in profile; caller should override.
    })
    if photometric:
        profile["photometric"] = photometric

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(out_array)


def wrap_png_with_georef(
    reference_tif: str | Path,
    png_path: str | Path,
    out_tif: str | Path,
    scale: int | float,
    compress: str = "LZW"
) -> None:
    """
    Convenience function:
    - Loads PNG (RGB) → CHW
    - Reads CRS + transform from reference_tif
    - Scales transform by 'scale'
    - Writes GeoTIFF out

    Args:
        reference_tif: original georeferenced image (GeoTIFF).
        png_path: SR PNG produced by ESRGAN/SwinIR (H x W x 3).
        out_tif: output GeoTIFF path.
        scale: upscale factor used to produce the PNG (e.g., 4).
        compress: GDAL compression.
    """
    rgb = load_png_rgb_as_chw(png_path)  # (3, H, W)

    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()
        new_transform = compute_scaled_transform(ref.transform, scale)

    profile.update({
        "driver": "GTiff",
        "height": rgb.shape[1],
        "width": rgb.shape[2],
        "count": 3,
        "dtype": rgb.dtype,
        "transform": new_transform,
        "compress": compress,
        "photometric": "RGB",
    })

    Path(out_tif).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(rgb)