"""
SwinIR (Transformer) inference (classical SR x4) — minimal, production-ready wrapper.

Usage:
  python -m downscaling.swinir_infer \
    --input data/interim/l8_rgb.png \
    --output data/processed/l8_rgb_swinir_x4.png \
    --weights models/weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth \
    --tile 512 --tile-overlap 32

Notes:
  - This script expects the SwinIR model definition to be importable as:
      from models.network_swinir import SwinIR
    If you keep the original SwinIR repo, add it to PYTHONPATH:
      export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/SwinIR
  - Model here is configured for SwinIR-M (embed_dim=180). For SwinIR-S, change embed_dim to 60
    and use the corresponding lightweight weights.
"""

from __future__ import annotations
import argparse
import os
from typing import Dict

import cv2
import numpy as np
import torch

try:
    # prefer third_party path if available
    from models.network_swinir import SwinIR as Net  # provided by the SwinIR repo
except Exception as e:
    raise ImportError(
        "Cannot import 'models.network_swinir'. "
        "Make sure the SwinIR repository is available and added to PYTHONPATH, e.g.\n"
        "  git submodule add https://github.com/JingyunLiang/SwinIR third_party/SwinIR\n"
        "  export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/SwinIR"
    ) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("SwinIR classical x4 inference")
    p.add_argument("--input", required=True, help="Input image path (RGB).")
    p.add_argument("--output", required=True, help="Output image path.")
    p.add_argument("--weights", required=True,
                   help="Path to SwinIR-M x4 .pth (e.g., 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth).")
    p.add_argument("--tile", type=int, default=512, help="Tile size (input space).")
    p.add_argument("--tile-overlap", type=int, default=32, help="Tile overlap (pixels).")
    # Architecture knobs (defaults for SwinIR-M x4)
    p.add_argument("--embed-dim", type=int, default=180, help="180 for SwinIR-M, 60 for SwinIR-S.")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--window-size", type=int, default=8)
    return p.parse_args()


def build_model(cfg: Dict) -> torch.nn.Module:
    model = Net(
        upscale=4,
        in_chans=3,
        img_size=cfg["img_size"],
        window_size=cfg["window_size"],
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=cfg["embed_dim"],
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    sd = torch.load(cfg["weights"], map_location="cpu")
    sd = sd.get("params_ema", sd.get("params", sd))
    model.load_state_dict(sd, strict=True)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def tiled_infer(model: torch.nn.Module, img_rgb: np.ndarray, scale: int,
                tile: int, overlap: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    out = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    cnt = np.zeros_like(out)
    step = tile - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            patch = img_rgb[y:min(y + tile, h), x:min(x + tile, w)]
            ph, pw = patch.shape[:2]
            pad = ((0, tile - ph), (0, tile - pw), (0, 0))
            patch_p = np.pad(patch, pad, mode="reflect")

            t = torch.from_numpy(patch_p.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            if torch.cuda.is_available():
                t = t.cuda()

            with torch.no_grad():
                sr = model(t).clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
            sr = sr[: ph * scale, : pw * scale]

            ys, xs = y * scale, x * scale
            out[ys:ys + sr.shape[0], xs:xs + sr.shape[1]] += sr
            cnt[ys:ys + sr.shape[0], xs:xs + sr.shape[1]] += 1

    out /= np.maximum(cnt, 1e-6)
    return (out * 255.0).round().astype(np.uint8)


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    cfg = {
        "weights": args.weights,
        "embed_dim": args.embed_dim,
        "img_size": args.img_size,
        "window_size": args.window_size,
    }

    img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {args.input}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    model = build_model(cfg)
    sr_rgb = tiled_infer(model, img_rgb, scale=4, tile=args.tile, overlap=args.tile_overlap)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved: {args.output}")


if __name__ == "__main__":
    main()
