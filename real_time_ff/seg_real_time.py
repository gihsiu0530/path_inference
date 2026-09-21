#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Realtime SegFormer-B2 semantic segmentation (RGB output only)
- 完全對齊 precompute_segformer_b2_cls4png.py 的輸出語意
- 不使用 CvBridge
- 輸出：224x224 RGB 四類 segmentation（road / person / movable / static）
"""

import time
import numpy as np

import rospy
from sensor_msgs.msg import Image as RosImage

import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# ===============================
# 19-class → 4-class grouping
# ===============================
GROUPS = {
    "road":    [0],
    "person":  [11],
    "movable": [12, 13, 14, 15, 16, 17, 18],
    "static":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

# ===============================
# Color palette (RGB) — MUST match offline
# ===============================
PALETTE4 = {
    0: (128, 64, 128),   # road
    1: (220, 20, 60),    # person
    2: (0, 0, 142),      # movable
    3: (70, 70, 70),     # static
}


# ===============================
# Utils (完全複製離線語意)
# ===============================
@torch.inference_mode()
def logits19_to_cls4(logits_19: torch.Tensor) -> torch.Tensor:
    """
    logits_19: (B,19,H,W)
    return:    (B,H,W) uint8 in {0,1,2,3}
    """
    road   = torch.logsumexp(logits_19[:, GROUPS["road"]], dim=1)
    person = torch.logsumexp(logits_19[:, GROUPS["person"]], dim=1)
    move   = torch.logsumexp(logits_19[:, GROUPS["movable"]], dim=1)
    stat   = torch.logsumexp(logits_19[:, GROUPS["static"]], dim=1)
    four   = torch.stack([road, person, move, stat], dim=1)
    return four.argmax(dim=1).to(torch.uint8)


def resize_keep_ratio_center_crop_uint8(label_hw: torch.Tensor, target=224) -> torch.Tensor:
    """
    label_hw: (B,1,H,W) uint8
    return:   (B,1,224,224) uint8
    """
    B, C, H, W = label_hw.shape
    short = min(H, W)
    scale = float(target) / float(short)
    newH, newW = int(round(H * scale)), int(round(W * scale))

    x = label_hw.float()
    x = F.interpolate(x, size=(newH, newW), mode="nearest")
    top  = (newH - target) // 2
    left = (newW - target) // 2
    x = x[:, :, top:top+target, left:left+target]
    return x.to(torch.uint8)

def resize_keep_ratio_center_crop_rgb(rgb: np.ndarray, target=224) -> np.ndarray:
    """
    rgb: (H,W,3) uint8
    return: (224,224,3) uint8
    """
    H, W, _ = rgb.shape
    short = min(H, W)
    scale = float(target) / float(short)
    newH, newW = int(round(H * scale)), int(round(W * scale))

    # resize
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float()
    x = F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)

    # center crop
    top  = (newH - target) // 2
    left = (newW - target) // 2
    x = x[:, :, top:top+target, left:left+target]

    return x[0].permute(1, 2, 0).byte().cpu().numpy()


def resize_keep_ratio_center_crop_depth(depth: np.ndarray, target=224) -> np.ndarray:
    """
    depth: (H,W) numeric
    return: (224,224) same dtype
    """
    H, W = depth.shape
    short = min(H, W)
    scale = float(target) / float(short)
    newH, newW = int(round(H * scale)), int(round(W * scale))

    x = torch.from_numpy(depth.copy()).unsqueeze(0).unsqueeze(0).float()
    x = F.interpolate(x, size=(newH, newW), mode="nearest")

    top  = (newH - target) // 2
    left = (newW - target) // 2
    x = x[:, :, top:top+target, left:left+target]

    return x[0, 0].cpu().numpy().astype(depth.dtype, copy=False)



def colorize_cls4_rgb(cls4_hw: np.ndarray) -> np.ndarray:
    """
    cls4_hw: (224,224) uint8
    return:  (224,224,3) uint8 RGB
    """
    h, w = cls4_hw.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, (r, g, b) in PALETTE4.items():
        rgb[cls4_hw == k] = (r, g, b)
    return rgb


def rosimg_to_rgb_numpy(msg: RosImage) -> np.ndarray:
    """
    支援: rgb8 / bgr8 / mono8 / rgba8 / bgra8
    """
    enc = msg.encoding.lower()
    h, w, step = msg.height, msg.width, msg.step
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        return buf.reshape(h, step)[:, :w*3].reshape(h, w, 3)

    if enc == "bgr8":
        bgr = buf.reshape(h, step)[:, :w*3].reshape(h, w, 3)
        return bgr[..., ::-1]

    if enc == "rgba8":
        rgba = buf.reshape(h, step)[:, :w*4].reshape(h, w, 4)
        return rgba[..., :3]

    if enc == "bgra8":
        bgra = buf.reshape(h, step)[:, :w*4].reshape(h, w, 4)
        return bgra[..., :3][..., ::-1]

    if enc == "mono8":
        gray = buf.reshape(h, step)[:, :w]
        return np.repeat(gray[..., None], 3, axis=2)

    raise RuntimeError(f"Unsupported encoding: {msg.encoding}")


def rosimg_to_depth_numpy(msg: RosImage) -> np.ndarray:
    """
    支援常見 depth encoding: 16UC1 / mono16 / 32FC1 / 32SC1
    """
    enc = msg.encoding.lower()
    h, w, step = msg.height, msg.width, msg.step

    dtype_by_encoding = {
        "16uc1": np.uint16,
        "mono16": np.uint16,
        "32fc1": np.float32,
        "32sc1": np.int32,
    }
    if enc not in dtype_by_encoding:
        raise RuntimeError(f"Unsupported depth encoding: {msg.encoding}")

    dtype = np.dtype(dtype_by_encoding[enc])
    if msg.is_bigendian:
        dtype = dtype.newbyteorder(">")

    row_bytes = w * dtype.itemsize
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)
    depth = np.frombuffer(buf[:, :row_bytes].tobytes(), dtype=dtype).reshape(h, w)
    return depth.astype(dtype_by_encoding[enc], copy=False)


def rgb_numpy_to_rosimg(rgb: np.ndarray, header) -> RosImage:
    msg = RosImage()
    msg.header = header
    msg.height, msg.width = rgb.shape[:2]
    msg.encoding = "rgb8"
    msg.step = msg.width * 3
    msg.is_bigendian = 0
    msg.data = rgb.tobytes()
    return msg


def depth_numpy_to_rosimg(depth: np.ndarray, header, encoding: str) -> RosImage:
    msg = RosImage()
    msg.header = header
    msg.height, msg.width = depth.shape[:2]
    msg.encoding = encoding
    msg.step = msg.width * depth.dtype.itemsize
    msg.is_bigendian = 0
    msg.data = np.ascontiguousarray(depth).tobytes()
    return msg


# ===============================
# ROS Node
# ===============================
class SegFormerRealtimeNode:
    def __init__(self):
        # self.in_topic  = rospy.get_param("~in_topic", "/zed2i/zed_node/rgb_raw/image_raw_color") //平常改成rgb
        self.in_topic  = rospy.get_param("~in_topic", "/zed2i/zed_node/rgb_raw/image_raw_color")
        # self.in_topic  = rospy.get_param("~in_topic", "/zed2i/zed_node/stereo_raw/image_raw_color")
        self.depth_topic = rospy.get_param("~depth_topic", "/zed2i/zed_node/depth/depth_registered")
        self.out_topic = rospy.get_param("~out_topic", "/seg_cls4_224")
        self.depth_out_topic = rospy.get_param("~depth_out_topic", "/depth_224")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = True

        rospy.loginfo(f"[seg_node] device={self.device}")

        self.processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        ).to(self.device).eval()

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        self.pub = rospy.Publisher(self.out_topic, RosImage, queue_size=1)
        self.pub_img224 = rospy.Publisher("/image_224", RosImage, queue_size=1)
        self.pub_depth224 = rospy.Publisher(self.depth_out_topic, RosImage, queue_size=1)

        self.sub = rospy.Subscriber(
            self.in_topic, RosImage, self.cb_image,
            queue_size=1, buff_size=2**24
        )
        self.sub_depth = rospy.Subscriber(
            self.depth_topic, RosImage, self.cb_depth,
            queue_size=1, buff_size=2**24
        )

        self._busy = False
        rospy.loginfo(f"[seg_node] subscribe {self.in_topic}")
        rospy.loginfo(f"[seg_node] subscribe depth {self.depth_topic}")
        rospy.loginfo(f"[seg_node] publish   {self.out_topic}")
        rospy.loginfo(f"[seg_node] publish depth {self.depth_out_topic}")

    def cb_depth(self, msg: RosImage):
        depth = rosimg_to_depth_numpy(msg)
        depth_224 = resize_keep_ratio_center_crop_depth(depth)
        depth_msg = depth_numpy_to_rosimg(depth_224, msg.header, msg.encoding)
        self.pub_depth224.publish(depth_msg)

    def cb_image(self, msg: RosImage):
        if self._busy:
            return
        self._busy = True

        t0 = time.perf_counter()

        # ---------- decode ----------
        rgb = rosimg_to_rgb_numpy(msg)

        # ---------- resize image (same crop as seg) ----------
        rgb_224_img = resize_keep_ratio_center_crop_rgb(rgb)


        # ---------- preprocess ----------
        inputs = self.processor(images=[rgb], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # ---------- forward ----------
        if self.device == "cuda":
            torch.cuda.synchronize()
        t_fwd0 = time.perf_counter()

        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(self.device == "cuda" and self.use_fp16)
        ):
            out = self.model(pixel_values=inputs["pixel_values"])

        if self.device == "cuda":
            torch.cuda.synchronize()
        t_fwd1 = time.perf_counter()

        # ---------- postprocess ----------
        logits = F.interpolate(
            out.logits,
            size=inputs["pixel_values"].shape[-2:],
            mode="bilinear",
            align_corners=False
        ).float()

        cls4 = logits19_to_cls4(logits).unsqueeze(1)
        cls4_224 = resize_keep_ratio_center_crop_uint8(cls4)[0, 0]
        rgb_224 = colorize_cls4_rgb(cls4_224.cpu().numpy())

        # ---------- publish ----------
        seg_msg = rgb_numpy_to_rosimg(rgb_224, msg.header)
        img_msg = rgb_numpy_to_rosimg(rgb_224_img, msg.header)

        self.pub.publish(seg_msg)
        self.pub_img224.publish(img_msg)

        t1 = time.perf_counter()
        rospy.loginfo_throttle(
            1.0,
            f"[seg_node] forward={1000*(t_fwd1-t_fwd0):.1f} ms | total={1000*(t1-t0):.1f} ms"
        )

        self._busy = False


def main():
    rospy.init_node("segformer_realtime_node", anonymous=False)
    SegFormerRealtimeNode()
    rospy.spin()


if __name__ == "__main__":
    main()
