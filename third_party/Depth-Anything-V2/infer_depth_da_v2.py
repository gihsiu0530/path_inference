#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True  # 讓 cuDNN 自動選最快 kernel

from depth_anything_v2.dpt import DepthAnythingV2

# python infer_depth_da_v2.py   --input_dir /home/cyc/dataset/nuscenes/trainval/samples/CAM_FRONT   --out_dir   /home/cyc/dataset/nuscenes/trainval/depths/CAM_FRONT/   --ckpt depth_anything_v2_vitl.pth   --device    cuda   --dtype_fp16   --save_vis

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def list_images(d: Path):
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])

def load_bgr(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"讀圖失敗: {path}")
    return img

def to_batched_tensor_bgr(imgs_bgr, device):
    """
    將 BGR numpy 圖片陣列轉為 (B,3,H,W) 的 torch.float32/float16 Tensor。
    這裡不改尺寸，交給模型內部處理；若模型需要固定尺寸，可在這裡先 resize。
    """
    # 轉 BGR->RGB，縮放到 [0,1]
    tensors = []
    for im in imgs_bgr:
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        tensors.append(t)
    x = torch.stack(tensors, dim=0).to(device)  # (B,3,H,W)
    return x

def save_depth(out_dir: Path, stem: str, depth_hw: np.ndarray, as_fp16: bool):
    if as_fp16:
        depth_hw = depth_hw.astype(np.float16)
    np.save(out_dir / f"{stem}.npy", depth_hw)

def save_vis(vis_dir: Path, stem: str, depth_hw: np.ndarray):
    d = depth_hw.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    d_img = (d * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(vis_dir / f"{stem}_depth224.png"), d_img)

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser("Depth Anything V2 (vitl) 批次輸出 depth 224x224 .npy")
    ap.add_argument("--input_dir", required=True,
                    help="例如 /home/cyc/dataset/nuscenes/trainval/samples/CAM_FRONT")
    ap.add_argument("--out_dir", default="/home/cyc/dataset/nuscenes/depths/CAM_FRONT/npy",
                    help="輸出 .npy 目錄")
    ap.add_argument("--ckpt", required=True,
                    help="checkpoints/depth_anything_v2_vitl.pth 的本地路徑")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch_size", type=int, default=32, help="批次大小（依顯存調整）")
    ap.add_argument("--workers", type=int, default=4, help="I/O 載圖執行緒數")
    ap.add_argument("--dtype_fp16", action="store_true", help="以 float16 存檔（建議開）")
    ap.add_argument("--amp", action="store_true", help="啟用 autocast 混合精度（建議開）")
    ap.add_argument("--save_vis", action="store_true", help="另外輸出可視化 PNG")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir   = Path(args.out_dir)
    vis_dir   = out_dir.parent / "vis_png"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # 載模型（Large / vitl）
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)

    paths = list_images(input_dir)
    if not paths:
        print(f"[警告] {input_dir} 沒有影像（支援：{sorted(list(IMG_EXTS))}）")
        return

    # 先把所有影像載到 RAM（或使用預取併行 I/O）
    # 為了節省記憶體，這裡採「邊載邊跑」+ thread 預取
    idx = 0
    total = len(paths)
    pbar = tqdm(total=total, desc="Inferring depth (batched)")

    def read_many(start, end):
        # 回傳 [(stem, img_bgr), ...]
        pairs = []
        for p in paths[start:end]:
            pairs.append((p.stem, load_bgr(p)))
        return pairs

    while idx < total:
        j = min(idx + args.batch_size, total)
        # 併行載圖
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut = ex.submit(read_many, idx, j)
            stem_imgs = fut.result()  # [(stem, bgr), ...]

        stems = [s for s, _ in stem_imgs]
        imgs_bgr = [im for _, im in stem_imgs]

        # 嘗試「真正的 batch 前向」
        batch_supported = True
        depth_batch = None

        try:
            x = to_batched_tensor_bgr(imgs_bgr, device)  # (B,3,H,W)
            if args.amp and device == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = model(x)  # 嘗試直接 batch 前向
            else:
                pred = model(x)

            # pred 可能是 torch.Tensor 或 dict；嘗試取出深度
            if isinstance(pred, dict) and "depth" in pred:
                depth_batch = pred["depth"]
            else:
                depth_batch = pred

            if isinstance(depth_batch, torch.Tensor) and depth_batch.dim() in (3, 4):
                # 期待 (B,1,H,W) 或 (B,H,W)
                if depth_batch.dim() == 4:
                    depth_batch = depth_batch.squeeze(1)  # -> (B,H,W)
                depth_batch = depth_batch.detach().float().cpu()
            else:
                # 如果不是 tensor，就當作不支援批次
                batch_supported = False

        except Exception:
            batch_supported = False

        if not batch_supported:
            # fallback：逐張推（但仍與批次流程共用輸出邏輯）
            depth_batch = []
            for im in imgs_bgr:
                if args.amp and device == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        d = model.infer_image(im)
                else:
                    d = model.infer_image(im)
                if isinstance(d, torch.Tensor):
                    d = d.detach().float().cpu().numpy()
                else:
                    d = d.astype(np.float32)
                depth_batch.append(torch.from_numpy(d))
            depth_batch = torch.stack(depth_batch, dim=0)  # (B,H,W)

        # 統一縮到 224x224、儲存
        for k in range(depth_batch.shape[0]):
            depth_hw = depth_batch[k].numpy()
            depth_224 = cv2.resize(depth_hw, (224, 224), interpolation=cv2.INTER_LINEAR)
            save_depth(out_dir, stems[k], depth_224, as_fp16=args.dtype_fp16)
            if args.save_vis:
                save_vis(vis_dir, stems[k], depth_224)

        pbar.update(j - idx)
        idx = j

    pbar.close()
    print(f"完成！.npy 在：{out_dir}")
    if args.save_vis:
        print(f"可視化 PNG 在：{vis_dir}")

if __name__ == "__main__":
    main()
