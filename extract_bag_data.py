#!/usr/bin/env python3
import argparse
from contextlib import ExitStack
import csv
from pathlib import Path

import cv2
import numpy as np
import rosbag


DEFAULT_TOPICS = {
    "img": "/image_224",
    "seg": "/seg_cls4_224",
    "depth": "/depth_224",
    "odom": "/odom",
}


def stamp_to_ns(stamp):
    return stamp.secs * 1_000_000_000 + stamp.nsecs


def msg_stamp_to_ns(msg, bag_time):
    if hasattr(msg, "header") and msg.header.stamp:
        return stamp_to_ns(msg.header.stamp)
    return stamp_to_ns(bag_time)


def image_msg_to_array(msg):
    encoding = msg.encoding.lower()
    dtype, channels = image_dtype_and_channels(encoding)

    arr = np.frombuffer(msg.data, dtype=dtype)
    if msg.is_bigendian != (arr.dtype.byteorder == ">"):
        arr = arr.byteswap().newbyteorder()

    if channels == 1:
        arr = arr.reshape(msg.height, msg.step // dtype().nbytes)
        return arr[:, : msg.width].copy()

    arr = arr.reshape(msg.height, msg.step // dtype().nbytes)
    arr = arr[:, : msg.width * channels]
    arr = arr.reshape(msg.height, msg.width, channels).copy()

    if encoding in ("rgb8", "rgba8"):
        code = cv2.COLOR_RGB2BGR if channels == 3 else cv2.COLOR_RGBA2BGRA
        arr = cv2.cvtColor(arr, code)

    return arr


def image_dtype_and_channels(encoding):
    explicit = {
        "mono8": (np.uint8, 1),
        "8uc1": (np.uint8, 1),
        "8uc3": (np.uint8, 3),
        "bgr8": (np.uint8, 3),
        "rgb8": (np.uint8, 3),
        "rgba8": (np.uint8, 4),
        "bgra8": (np.uint8, 4),
        "mono16": (np.uint16, 1),
        "16uc1": (np.uint16, 1),
        "32fc1": (np.float32, 1),
    }
    if encoding in explicit:
        return explicit[encoding]

    raise ValueError(f"Unsupported image encoding: {encoding}")


def depth_to_png_array(arr):
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr * 1000.0, 0, np.iinfo(np.uint16).max)
        return arr.astype(np.uint16)
    return arr


def write_image(msg, bag_time, out_dir, kind, overwrite):
    timestamp_ns = msg_stamp_to_ns(msg, bag_time)
    out_path = out_dir / f"{timestamp_ns}.png"
    if out_path.exists() and not overwrite:
        return False

    arr = image_msg_to_array(msg)
    if kind == "depth":
        arr = depth_to_png_array(arr)

    ok = cv2.imwrite(str(out_path), arr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {out_path}")
    return True


def write_odom_row(writer, msg, bag_time):
    timestamp_ns = msg_stamp_to_ns(msg, bag_time)
    pose = msg.pose.pose
    position = pose.position
    orientation = pose.orientation
    writer.writerow(
        {
            "timestep": timestamp_ns,
            "position_x": position.x,
            "position_y": position.y,
            "position_z": position.z,
            "orientation_x": orientation.x,
            "orientation_y": orientation.y,
            "orientation_z": orientation.z,
            "orientation_w": orientation.w,
        }
    )


def find_bag(video_dir):
    bags = sorted(video_dir.glob("*.bag"))
    if len(bags) != 1:
        raise RuntimeError(f"{video_dir} should contain exactly one .bag file, found {len(bags)}")
    return bags[0]


def extract_video(video_dir, topics, overwrite):
    bag_path = find_bag(video_dir)
    img_dir = video_dir / "img"
    seg_dir = video_dir / "seg"
    depth_dir = video_dir / "depth"
    for out_dir in (img_dir, seg_dir, depth_dir):
        out_dir.mkdir(exist_ok=True)

    odom_path = video_dir / "odom.csv"
    image_counts = {"img": 0, "seg": 0, "depth": 0}
    odom_count = 0

    with ExitStack() as stack:
        bag = stack.enter_context(rosbag.Bag(str(bag_path), "r"))
        fieldnames = [
            "timestep",
            "position_x",
            "position_y",
            "position_z",
            "orientation_x",
            "orientation_y",
            "orientation_z",
            "orientation_w",
        ]
        writer = None
        if overwrite or not odom_path.exists():
            csv_file = stack.enter_context(odom_path.open("w", newline="", encoding="utf-8-sig"))
            csv_file.write("sep=,\n")
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

        topic_to_kind = {
            topics["img"]: "img",
            topics["seg"]: "seg",
            topics["depth"]: "depth",
            topics["odom"]: "odom",
        }
        selected_topics = list(topic_to_kind.keys())

        for topic, msg, bag_time in bag.read_messages(topics=selected_topics):
            kind = topic_to_kind[topic]
            if kind == "odom":
                if writer is not None:
                    write_odom_row(writer, msg, bag_time)
                    odom_count += 1
            elif kind == "img":
                image_counts["img"] += write_image(msg, bag_time, img_dir, kind, overwrite)
            elif kind == "seg":
                image_counts["seg"] += write_image(msg, bag_time, seg_dir, kind, overwrite)
            elif kind == "depth":
                image_counts["depth"] += write_image(msg, bag_time, depth_dir, kind, overwrite)

    return bag_path, image_counts, odom_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract img, segmentation, depth images and odometry CSV from video*/ROS bag folders."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Root folder containing video1 ... video9.")
    parser.add_argument("--start", type=int, default=1, help="First video index.")
    parser.add_argument("--end", type=int, default=9, help="Last video index.")
    parser.add_argument("--img-topic", default=DEFAULT_TOPICS["img"], help="RGB image topic.")
    parser.add_argument("--seg-topic", default=DEFAULT_TOPICS["seg"], help="Segmentation image topic.")
    parser.add_argument("--depth-topic", default=DEFAULT_TOPICS["depth"], help="Depth image topic.")
    parser.add_argument("--odom-topic", default=DEFAULT_TOPICS["odom"], help="Odometry topic.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite extracted files.")
    return parser.parse_args()


def main():
    args = parse_args()
    topics = {
        "img": args.img_topic,
        "seg": args.seg_topic,
        "depth": args.depth_topic,
        "odom": args.odom_topic,
    }

    root = args.root.resolve()
    for index in range(args.start, args.end + 1):
        video_dir = root / f"video{index}"
        if not video_dir.is_dir():
            raise RuntimeError(f"Missing folder: {video_dir}")

        bag_path, image_counts, odom_count = extract_video(video_dir, topics, args.overwrite)
        print(
            f"{video_dir.name}: {bag_path.name} -> "
            f"img={image_counts['img']}, seg={image_counts['seg']}, "
            f"depth={image_counts['depth']}, odom={odom_count}"
        )


if __name__ == "__main__":
    main()
