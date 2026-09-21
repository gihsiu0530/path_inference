#!/usr/bin/env python3
import argparse
import csv
import shutil
from bisect import bisect_left
from pathlib import Path


HALF_SECOND_NS = 500_000_000
CSV_ENCODING = "utf-8"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create no-bag resampled video folders from extracted img/seg/depth/odom data."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Root folder containing video1 ... video9.")
    parser.add_argument("--output", type=Path, default=Path("resample"), help="Output folder.")
    parser.add_argument("--start", type=int, default=1, help="First video index.")
    parser.add_argument("--end", type=int, default=9, help="Last video index.")
    parser.add_argument(
        "--interval-ns",
        type=int,
        default=HALF_SECOND_NS,
        help="Resample interval in nanoseconds. Default is 0.5 second.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output folders.")
    parser.add_argument(
        "--poseimu-csv-name",
        default="poseimu_zero.csv",
        help="PoseIMU CSV filename inside each video folder.",
    )
    return parser.parse_args()


def image_timestamps(folder):
    files = sorted(folder.glob("*.png"), key=lambda path: int(path.stem))
    if not files:
        raise RuntimeError(f"No PNG files found in {folder}")
    return [(int(path.stem), path) for path in files]


def read_csv_rows(path, timestamp_field, label):
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        first_line = csv_file.readline()
        if not first_line.startswith("sep="):
            csv_file.seek(0)
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise RuntimeError(f"No {label} rows found in {path}")
    if timestamp_field not in rows[0]:
        raise RuntimeError(f"Missing {timestamp_field!r} column in {path}")
    rows.sort(key=lambda row: int(row[timestamp_field]))
    return rows


def nearest_by_timestamp(items, target):
    timestamps = [item[0] for item in items]
    index = bisect_left(timestamps, target)
    if index == 0:
        return items[0]
    if index == len(items):
        return items[-1]

    before = items[index - 1]
    after = items[index]
    if target - before[0] <= after[0] - target:
        return before
    return after


def nearest_csv_row(rows, timestamp_field, target):
    timestamps = [int(row[timestamp_field]) for row in rows]
    index = bisect_left(timestamps, target)
    if index == 0:
        return rows[0]
    if index == len(rows):
        return rows[-1]

    before = rows[index - 1]
    after = rows[index]
    before_ts = int(before[timestamp_field])
    after_ts = int(after[timestamp_field])
    if target - before_ts <= after_ts - target:
        return before
    return after


def copy_selected(src, dst_dir):
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def prepare_output_dir(path, overwrite):
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"Output folder already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    (path / "img").mkdir(parents=True)
    (path / "seg").mkdir()
    (path / "depth").mkdir()


def resample_video(video_dir, output_dir, interval_ns, overwrite, poseimu_csv_name):
    prepare_output_dir(output_dir, overwrite)

    img_items = image_timestamps(video_dir / "img")
    seg_items = image_timestamps(video_dir / "seg")
    depth_items = image_timestamps(video_dir / "depth")
    odom_rows = read_csv_rows(video_dir / "odom.csv", "timestep", "odom")
    poseimu_rows = read_csv_rows(video_dir / poseimu_csv_name, "timestamp(ns)", "poseimu")

    start_ts = max(
        img_items[0][0],
        seg_items[0][0],
        depth_items[0][0],
        int(odom_rows[0]["timestep"]),
        int(poseimu_rows[0]["timestamp(ns)"]),
    )
    end_ts = min(
        img_items[-1][0],
        seg_items[-1][0],
        depth_items[-1][0],
        int(odom_rows[-1]["timestep"]),
        int(poseimu_rows[-1]["timestamp(ns)"]),
    )
    if start_ts > end_ts:
        raise RuntimeError(f"No overlapping time range for {video_dir}")

    selected_keys = set()
    odom_fieldnames = list(odom_rows[0].keys())
    poseimu_fieldnames = list(poseimu_rows[0].keys())
    index_fieldnames = [
        "target_timestep",
        "img_timestep",
        "seg_timestep",
        "depth_timestep",
        "odom_timestep",
        "poseimu_timestep",
    ]

    count = 0
    with (output_dir / "odom.csv").open("w", newline="", encoding=CSV_ENCODING) as odom_file, (
        output_dir / poseimu_csv_name
    ).open("w", newline="", encoding=CSV_ENCODING) as poseimu_file, (
        output_dir / "resample_index.csv"
    ).open("w", newline="", encoding=CSV_ENCODING) as index_file:
        odom_writer = csv.DictWriter(odom_file, fieldnames=odom_fieldnames)
        poseimu_writer = csv.DictWriter(poseimu_file, fieldnames=poseimu_fieldnames)
        index_writer = csv.DictWriter(index_file, fieldnames=index_fieldnames)
        odom_writer.writeheader()
        poseimu_writer.writeheader()
        index_writer.writeheader()

        target = start_ts
        while target <= end_ts:
            img_ts, img_path = nearest_by_timestamp(img_items, target)
            seg_ts, seg_path = nearest_by_timestamp(seg_items, img_ts)
            depth_ts, depth_path = nearest_by_timestamp(depth_items, img_ts)
            odom_row = nearest_csv_row(odom_rows, "timestep", target)
            odom_ts = int(odom_row["timestep"])
            poseimu_row = nearest_csv_row(poseimu_rows, "timestamp(ns)", target)
            poseimu_ts = int(poseimu_row["timestamp(ns)"])

            key = (img_ts, seg_ts, depth_ts, odom_ts, poseimu_ts)
            if key not in selected_keys:
                copy_selected(img_path, output_dir / "img")
                copy_selected(seg_path, output_dir / "seg")
                copy_selected(depth_path, output_dir / "depth")
                odom_writer.writerow(odom_row)
                poseimu_writer.writerow(poseimu_row)
                index_writer.writerow(
                    {
                        "target_timestep": target,
                        "img_timestep": img_ts,
                        "seg_timestep": seg_ts,
                        "depth_timestep": depth_ts,
                        "odom_timestep": odom_ts,
                        "poseimu_timestep": poseimu_ts,
                    }
                )
                selected_keys.add(key)
                count += 1

            target += interval_ns

    return count


def main():
    args = parse_args()
    root = args.root.resolve()
    output_root = args.output if args.output.is_absolute() else root / args.output
    output_root.mkdir(exist_ok=True)

    for index in range(args.start, args.end + 1):
        video_dir = root / f"video{index}"
        output_dir = output_root / f"video{index}"
        if not video_dir.is_dir():
            raise RuntimeError(f"Missing folder: {video_dir}")

        count = resample_video(video_dir, output_dir, args.interval_ns, args.overwrite, args.poseimu_csv_name)
        print(f"video{index}: {count} samples -> {output_dir}")


if __name__ == "__main__":
    main()
