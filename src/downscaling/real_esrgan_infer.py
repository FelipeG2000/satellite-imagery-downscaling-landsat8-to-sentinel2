"""
Real-ESRGAN inference (x4) — minimal, production-ready wrapper.

Usage:
  python -m downscaling.real_esrgan_infer \
    --input data/interim/l8_rgb.png \
    --output data/processed/l8_rgb_realesrgan_x4.png \
    --weights models/weights/RealESRGAN_x4plus.pth \
    --tile 512 --tile-pad 16

Requirements:
  - torch, torchvision, basicsr, realesrgan, opencv-python, numpy
"""

from __future__ import annotations
import argparse
import os
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Real-ESRGAN x4 inference")
    p.add_argument("--input", required=True, help="Input image path (RGB/BGR).")
    p.add_argument("--output", required=True, help="Output image path.")
    p.add_argument("--weights", default="models/weights/RealESRGAN_x4plus.pth",
                   help="Path to RealESRGAN_x4plus.pth.")
    p.add_argument("--tile", type=int, default=512, help="Tile size to control VRAM/RAM usage.")
    p.add_argument("--tile-pad", type=int, default=16, help="Padding for tile borders.")
    p.add_argument("--half", action="store_true",
                   help="Use FP16 if CUDA is available (overrides auto).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Basic checks
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    # Define ESRGAN architecture used by the x4 model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
        scale=4,
        model_path=args.weights,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=0,
        half=(args.half or torch.cuda.is_available())
    )

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {args.input}")

    out, _ = upsampler.enhance(img, outscale=4)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, out)
    print(f"✅ Saved: {args.output}")


if __name__ == "__main__":
    main()