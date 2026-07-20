#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
離線 bag → 資料轉檔工具。

直接讀取原始 rosbag，對 RGB 影像做 SegFormer-B2 語意分割 + resize，
一次輸出 img/ 與 seg/ 的 224x224 PNG，以及 odom.csv，
輸出結構與 extract_bag_data.py 完全相同，可直接接 resample / 推論。

相較「rosbag play → seg_real_time.py 節點 → rosbag record → extract_bag_data.py」
四步流程，本工具不需 roscore / play / record / 中間 bag，一支腳本搞定。

執行環境：需同時具備 rosbag + torch + transformers + cv2，
ROS noetic 的 python3 已齊備：
    source /opt/ros/noetic/setup.bash
    python3 bag_to_data.py --root <root> --start 1 --end N
"""

import argparse
from contextlib import ExitStack
import csv
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import rosbag
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# 重用既有邏輯，確保分割結果與存檔格式與現行流程逐字一致
from seg_real_time import (
    rosimg_to_rgb_numpy,
    resize_keep_ratio_center_crop_rgb,
    logits19_to_cls4,
    resize_keep_ratio_center_crop_uint8,
    colorize_cls4_rgb,
)
from extract_bag_data import find_bag, msg_stamp_to_ns, write_odom_row


MODEL_NAME = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

# 與 extract_bag_data.py 的 odom.csv 欄位一致
ODOM_FIELDNAMES = [
    "timestep",
    "position_x",
    "position_y",
    "position_z",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "orientation_w",
]


def load_model(device):
    """載入 SegFormer 模型（邏輯同 seg_real_time.py 的 __init__）。"""
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device).eval()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    return processor, model


@torch.inference_mode()
def segment_rgb(rgb, processor, model, device, use_fp16=True):
    """對整張 RGB 影像做語意分割，回傳 (224,224,3) uint8 RGB 的 4 類彩色圖。

    流程完全對齊 seg_real_time.py 的 cb_image。
    """
    inputs = processor(images=[rgb], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(device == "cuda" and use_fp16),
    ):
        out = model(pixel_values=inputs["pixel_values"])

    logits = F.interpolate(
        out.logits,
        size=inputs["pixel_values"].shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).float()

    cls4 = logits19_to_cls4(logits).unsqueeze(1)
    cls4_224 = resize_keep_ratio_center_crop_uint8(cls4)[0, 0]
    return colorize_cls4_rgb(cls4_224.cpu().numpy())


def save_rgb_png(rgb, out_path):
    """RGB → BGR 後用 cv2 存 PNG。

    與 extract_bag_data.py 對 rgb8 影像的處理一致，確保磁碟 PNG 的 RGB 值
    與 PALETTE4 相符，convert_cls4png_to_npy.py（以 PIL 讀 RGB）才不會遇到未知色。
    """
    ok = cv2.imwrite(str(out_path), rgb[:, :, ::-1])
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")


def process_video(video_dir, rgb_topic, odom_topic, processor, model, device, overwrite):
    """處理單一 video 資料夾內的原始 bag，輸出 img/ seg/ 與 odom.csv。"""
    bag_path = find_bag(video_dir)
    img_dir = video_dir / "img"
    seg_dir = video_dir / "seg"
    img_dir.mkdir(exist_ok=True)
    seg_dir.mkdir(exist_ok=True)
    odom_path = video_dir / "odom.csv"

    img_count = 0
    odom_count = 0

    with ExitStack() as stack:
        bag = stack.enter_context(rosbag.Bag(str(bag_path), "r"))

        odom_writer = None
        if overwrite or not odom_path.exists():
            csv_file = stack.enter_context(
                odom_path.open("w", newline="", encoding="utf-8-sig")
            )
            csv_file.write("sep=,\n")
            odom_writer = csv.DictWriter(csv_file, fieldnames=ODOM_FIELDNAMES)
            odom_writer.writeheader()

        for topic, msg, bag_time in bag.read_messages(topics=[rgb_topic, odom_topic]):
            if topic == odom_topic:
                if odom_writer is not None:
                    write_odom_row(odom_writer, msg, bag_time)
                    odom_count += 1
                continue

            # rgb_topic：同一張影像同時輸出 img 與 seg，用同一時間戳命名以確保對齊
            timestamp_ns = msg_stamp_to_ns(msg, bag_time)
            img_path = img_dir / f"{timestamp_ns}.png"
            seg_path = seg_dir / f"{timestamp_ns}.png"
            if img_path.exists() and seg_path.exists() and not overwrite:
                continue

            rgb = rosimg_to_rgb_numpy(msg)
            img224 = resize_keep_ratio_center_crop_rgb(rgb)
            seg224 = segment_rgb(rgb, processor, model, device)

            save_rgb_png(img224, img_path)
            save_rgb_png(seg224, seg_path)
            img_count += 1

    return bag_path, img_count, odom_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="離線讀取原始 rosbag，做 SegFormer 語意分割 + resize，"
        "輸出 img/ seg/ PNG 與 odom.csv（結構同 extract_bag_data.py）。"
    )
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="含 video1 ... videoN 的根目錄。"
    )
    parser.add_argument("--start", type=int, default=1, help="起始 video 編號。")
    parser.add_argument("--end", type=int, default=9, help="結束 video 編號。")
    parser.add_argument(
        "--rgb-topic",
        default="/zed2i/zed_node/right_raw/image_raw_color",
        help="原始 RGB 影像 topic。",
    )
    parser.add_argument("--odom-topic", default="/odom", help="里程計 topic。")
    parser.add_argument("--overwrite", action="store_true", help="覆寫既有輸出檔。")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bag_to_data] device={device}，載入模型 {MODEL_NAME} ...")
    processor, model = load_model(device)

    root = args.root.resolve()
    for index in range(args.start, args.end + 1):
        video_dir = root / f"video{index}"
        if not video_dir.is_dir():
            raise RuntimeError(f"找不到資料夾: {video_dir}")

        bag_path, img_count, odom_count = process_video(
            video_dir,
            args.rgb_topic,
            args.odom_topic,
            processor,
            model,
            device,
            args.overwrite,
        )
        print(
            f"{video_dir.name}: {bag_path.name} -> "
            f"img/seg={img_count}, odom={odom_count}"
        )


if __name__ == "__main__":
    main()