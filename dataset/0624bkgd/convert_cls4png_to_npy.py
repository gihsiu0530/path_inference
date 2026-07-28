import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


PALETTE4 = {
    0: (128, 64, 128),   # road
    1: (220, 20, 60),    # person
    2: (0, 0, 142),      # movable
    3: (70, 70, 70),     # static
}


def convert_png_to_cls_id(png_path: Path, strict: bool = True) -> np.ndarray:
    rgb = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.uint8)
    cls = np.full(rgb.shape[:2], 255, dtype=np.uint8)

    for class_id, color in PALETTE4.items():
        color_arr = np.asarray(color, dtype=np.uint8)
        mask = np.all(rgb == color_arr, axis=-1)
        cls[mask] = class_id

    unknown = cls == 255
    if unknown.any():
        unknown_count = int(unknown.sum())
        if strict:
            colors = np.unique(rgb[unknown].reshape(-1, 3), axis=0)
            preview = colors[:10].tolist()
            raise ValueError(
                f"{png_path} has {unknown_count} pixels with colors outside PALETTE4. "
                f"First unknown colors: {preview}"
            )

        # Non-strict fallback: assign each unknown pixel to the nearest palette color.
        colors = np.asarray([PALETTE4[i] for i in range(len(PALETTE4))], dtype=np.int16)
        unknown_rgb = rgb[unknown].astype(np.int16)
        dist = ((unknown_rgb[:, None, :] - colors[None, :, :]) ** 2).sum(axis=-1)
        cls[unknown] = dist.argmin(axis=1).astype(np.uint8)

    return cls


def main():
    parser = argparse.ArgumentParser(
        description="Convert *_cls4_224.png color maps to uint8 *_cls4_224.npy class-id maps."
    )
    parser.add_argument(
        "--root",
        default="/home/cyc/dataset/nuscenes/trainval/seg_cl4_png",
        help="Root directory containing seq_id subdirectories with *_cls4_224.png files.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing .npy files.")
    parser.add_argument(
        "--non_strict",
        action="store_true",
        help="Map unknown colors to nearest palette color instead of raising an error.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    png_files = sorted(root.rglob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No *_cls4_224.png files found under {root}")

    converted = 0
    skipped = 0
    for png_path in tqdm(png_files, desc="cls4 png -> npy"):
        npy_path = png_path.with_suffix(".npy")
        if npy_path.exists() and not args.force:
            skipped += 1
            continue

        cls = convert_png_to_cls_id(png_path, strict=not args.non_strict)
        tmp_path = npy_path.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(tmp_path, cls)
        Path(tmp_path).replace(npy_path)
        converted += 1

    print(f"[DONE] converted={converted} skipped={skipped} total_png={len(png_files)} root={root}")


if __name__ == "__main__":
    main()
