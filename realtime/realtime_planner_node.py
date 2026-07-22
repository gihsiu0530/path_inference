#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Realtime ST-P3 trajectory planning node.

Single-process pipeline that replaces the offline four-step flow
(bag_to_data -> resample -> convert_cls4png_to_npy -> park_L2_ASAP):

    camera + /odom  ->  SegFormer-B2 (4-class seg)  ->  ST-P3  ->  nav_msgs/Path

Runs on the ROS noetic python3 (3.8), which already provides rospy, torch,
transformers, pytorch_lightning, pandas and pyquaternion. Do NOT run this in
the `stp3_env` conda env: it has neither rospy nor transformers.

The batch fed to the model reproduces the contract of the offline loader
(stp3/data_0512_graduate/NuscenesData_0624_ASAP.py) exactly; see the notes on
SEG_PALETTE and the coordinate swap below.
"""

import os
import sys
import time
import pathlib
import datetime
from collections import deque

# Must be set before park_L2_ASAP pulls in matplotlib.pyplot (headless node).
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
from pyquaternion import Quaternion

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import rospy
from sensor_msgs.msg import Image as RosImage
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float64MultiArray

import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from nuscenes.eval.common.utils import quaternion_yaw

from seg_real_time import (
    logits19_to_cls4,
    resize_keep_ratio_center_crop_rgb,
    resize_keep_ratio_center_crop_uint8,
    rosimg_to_rgb_numpy,
    rgb_numpy_to_rosimg,
    colorize_cls4_rgb,
)
from park_L2_ASAP import (
    _load_trainer_for_eval,
    _call_model_forward,
    _call_model_planning,
    _prepare_l2_labels,
    save_inference_plot,
    _input_history_from_egomotion,
)
from stp3.utils.geometry import mat2pose_vec


# ===============================
# Constants — must match the checkpoint's hyper_parameters
# ===============================
SAMPLE_INTERVAL = 0.5      # seconds; the model hard-assumes this cadence
TIME_RECEPTIVE_FIELD = 3   # image frames fed to the model
ADMLP_PAST_FRAMES = 4      # past poses for the AD-MLP feature (+ t0 => 5 poses)
N_FUTURE_FRAMES = 6        # predicted trajectory points

COMMAND_TO_ONEHOT = {
    "LEFT":    [1.0, 0.0, 0.0],
    "FORWARD": [0.0, 1.0, 0.0],
    "RIGHT":   [0.0, 0.0, 1.0],
}

# Copied verbatim from NuscenesData_0624_ASAP.py:30-35.
#
# WARNING: this palette is indexed directly by the 4-class seg id, and it is
# offset by one relative to convert_cls4png_to_npy.py's PALETTE4 (id 0 = road
# becomes black here, id 3 = static becomes green). That offset is what the
# checkpoint was trained on, so it must be reproduced, not "fixed".
SEG_PALETTE = np.array([
    [0, 0, 0],
    [128, 64, 128],
    [220, 20, 60],
    [0, 142, 0],
], dtype=np.uint8)

SEGFORMER_NAME = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
DEFAULT_CHECKPOINT = os.path.join(
    _REPO_ROOT, "model", "best-box-col-epoch=24-epoch_val_plan_obj_box_col=0.0054.ckpt"
)


def pose_matrix_from_odom(msg: Odometry) -> np.ndarray:
    """
    nav_msgs/Odometry -> 4x4 pose matrix.
    Mirrors NuscenesData_0624_ASAP.get_pose_matrix (:187-205), which builds the
    matrix straight from the odom pose with no axis correction.
    """
    p = msg.pose.pose.position
    o = msg.pose.pose.orientation
    q = Quaternion(o.w, o.x, o.y, o.z)  # pyquaternion order: (w, x, y, z)

    mat = np.eye(4)
    mat[:3, :3] = q.rotation_matrix
    mat[:3, 3] = np.array([p.x, p.y, p.z])
    return mat


def relative_xy_yaw(pose_curr_inv: np.ndarray, pose_other: np.ndarray):
    """
    Pose of `pose_other` in the current planning frame, as (x_left, y_front, yaw).

    The raw body frame is (x_forward, y_left); the model uses (x_left, y_front),
    so the translation is swapped. The yaw is NOT swapped — same as the loader
    (:378-395, :333-354).
    """
    rel = pose_curr_inv @ pose_other
    x_forward = rel[0, 3]
    y_left = rel[1, 3]
    yaw = quaternion_yaw(Quaternion(matrix=rel))
    return [y_left, x_forward, yaw]


class RealtimeSequenceBuffer:
    """
    Ring buffers holding the past observations the model needs, sampled on a
    strict SAMPLE_INTERVAL cadence.

    The effective history horizon is driven by the poses (ADMLP_PAST_FRAMES + 1
    = 5 samples = 2.5 s), which is longer than the image window (3 samples), so
    the node stays in warm-up until both are full.
    """

    def __init__(self):
        self.rgb = deque(maxlen=TIME_RECEPTIVE_FIELD)     # (224,224,3) uint8
        self.seg_id = deque(maxlen=TIME_RECEPTIVE_FIELD)  # (224,224)   uint8
        self.poses = deque(maxlen=ADMLP_PAST_FRAMES + 1)  # 4x4 matrices

    def push(self, rgb_224, seg_id_224, pose):
        self.rgb.append(rgb_224)
        self.seg_id.append(seg_id_224)
        self.poses.append(pose)

    @property
    def ready(self) -> bool:
        return (len(self.rgb) == self.rgb.maxlen
                and len(self.poses) == self.poses.maxlen)

    def status(self) -> str:
        return (f"images {len(self.rgb)}/{self.rgb.maxlen}, "
                f"poses {len(self.poses)}/{self.poses.maxlen}")

    # ---------- model inputs derived from the pose history ----------

    def admlp_past_trajectory(self) -> np.ndarray:
        """(4,3) past poses in the t0 planning frame — loader :378-395."""
        pose_curr_inv = np.linalg.inv(self.poses[-1])
        past = [relative_xy_yaw(pose_curr_inv, self.poses[i])
                for i in range(len(self.poses) - 1)]
        return np.asarray(past, dtype=np.float32)

    def estimate_current_motion(self, past: np.ndarray):
        """
        Causal polynomial least-squares fit over the past poses + t0 — loader
        :420-476. With the checkpoint's default degree of 1, acceleration is
        always exactly zero, but the fit is kept in full to stay faithful.
        """
        dt = max(float(SAMPLE_INTERVAL), 1e-6)
        current = np.zeros((1, 3), dtype=np.float64)
        poses = np.concatenate([past.astype(np.float64), current], axis=0)

        times = np.arange(-ADMLP_PAST_FRAMES, 1, dtype=np.float64) * dt
        # yaw must be unwrapped along time so +/-pi wraps do not break the fit.
        poses[:, 2] = np.unwrap(poses[:, 2])

        design = np.vander(times, N=2, increasing=True)  # ADMLP_FIT_DEGREE = 1
        xy_coefficients, _, _, _ = np.linalg.lstsq(design, poses[:, :2], rcond=None)
        yaw_coefficients, _, _, _ = np.linalg.lstsq(design, poses[:, 2], rcond=None)

        velocity = np.asarray(
            [xy_coefficients[1, 0], xy_coefficients[1, 1], yaw_coefficients[1]],
            dtype=np.float64,
        )
        acceleration = np.zeros(3, dtype=np.float64)  # degree 1 => no acceleration
        return velocity.astype(np.float32), acceleration.astype(np.float32)

    def build_admlp_input(self, command: str) -> np.ndarray:
        """21-dim feature — loader :478-494."""
        past = self.admlp_past_trajectory()
        velocity, acceleration = self.estimate_current_motion(past)
        command_onehot = np.asarray(COMMAND_TO_ONEHOT[command], dtype=np.float32)
        feature = np.concatenate(
            [past.reshape(-1), velocity, acceleration, command_onehot], axis=0
        ).astype(np.float32)
        if feature.shape != (21,) or not np.isfinite(feature).all():
            raise ValueError(f"Invalid AD-MLP feature: shape={feature.shape}")
        return feature

    def build_future_egomotion(self) -> np.ndarray:
        """
        (3,6) 6-DoF motion t->t+1 over the image window — loader :267-291.

        Entry 2 is derived from a future frame offline, but the model only ever
        reads entries 0 and 1 (codex_pure_ASAP.py:653-666), so it is left zero.
        """
        image_poses = list(self.poses)[-TIME_RECEPTIVE_FIELD:]
        out = np.zeros((TIME_RECEPTIVE_FIELD, 6), dtype=np.float32)
        for i in range(TIME_RECEPTIVE_FIELD - 1):
            egomotion = np.linalg.inv(image_poses[i + 1]) @ image_poses[i]
            egomotion[3, :3] = 0.0
            egomotion[3, 3] = 1.0
            vec = mat2pose_vec(torch.from_numpy(egomotion).float().unsqueeze(0))
            out[i] = vec.squeeze(0).numpy()
        return out


class RealtimePlannerNode:
    def __init__(self):
        self.in_topic = rospy.get_param(
            "~in_topic", "/zed2i/zed_node/rgb_raw/image_raw_color")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.command_topic = rospy.get_param("~command_topic", "/senpai/command")
        self.path_topic = rospy.get_param("~path_topic", "/senpai/path")
        self.path_global_topic = rospy.get_param(
            "~path_global_topic", "/senpai/path_global")
        # Same global trajectory as path_global, but flattened to [x0,y0,x1,y1,...]
        # as std_msgs/Float64MultiArray so the MPC chain (local_path -> mpc) can
        # consume it unchanged in place of global_path's CSV route (array_topic).
        self.array_topic = rospy.get_param("~array_topic", "array_topic")
        self.seg_topic = rospy.get_param("~seg_topic", "/senpai/seg_cls4_224")
        self.frame_id = rospy.get_param("~frame_id", "base_link")
        self.checkpoint = rospy.get_param("~checkpoint", DEFAULT_CHECKPOINT)
        self.sample_interval = float(rospy.get_param("~sample_interval", SAMPLE_INTERVAL))
        # The checkpoint's command channel is inverted: feeding "LEFT" steers the
        # path to the right and vice-versa (the model's dir_loss disagrees with
        # the loader's LEFT/RIGHT labels; verified by a same-scene A/B test).
        # Swapping LEFT<->RIGHT before the model makes the /senpai/command topic
        # match physical intent. Set false to feed the raw label through.
        self.flip_command = bool(rospy.get_param("~flip_command", True))
        # Save an offline-style inference plot (camera image + trajectory panel)
        # per inference cycle. Off by default so normal runs stay lightweight.
        self.save_plots = bool(rospy.get_param("~save_plots", False))

        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = rospy.get_param("~device", default_device)
        self.use_fp16 = bool(rospy.get_param("~use_fp16", True))

        rospy.loginfo(f"[planner] device={self.device}")

        # ---------- SegFormer (same as seg_real_time.py:227-232) ----------
        self.processor = SegformerImageProcessor.from_pretrained(SEGFORMER_NAME)
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_NAME
        ).to(self.device).eval()

        # ---------- ST-P3 ----------
        rospy.loginfo(f"[planner] loading checkpoint {self.checkpoint}")
        try:
            trainer = _load_trainer_for_eval(self.checkpoint, strict=True)
        except RuntimeError as exc:
            # A checkpoint of a different (smaller) model shows up here as a wall
            # of missing keys. checkpoint/last.ckpt is the usual culprit: it holds
            # a pure AD-MLP baseline (10 tensors, no model.vlm.*), so translate
            # that into something actionable instead of 600 lines of key names.
            if "Missing key(s)" in str(exc) and "model.vlm." in str(exc):
                raise RuntimeError(
                    f"{self.checkpoint} is not a full ST-P3 checkpoint: it is missing the "
                    "model.vlm.* visual weights (633 tensors).\n"
                    "checkpoint/last.ckpt holds only a pure AD-MLP baseline (10 tensors) and "
                    "cannot drive realtime inference.\n"
                    "Use model/best-box-col-*.ckpt instead."
                ) from exc
            raise
        self.model = trainer.model.to(self.device).eval()
        self.n_present = getattr(self.model, "receptive_field", TIME_RECEPTIVE_FIELD)

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        self.buffer = RealtimeSequenceBuffer()
        self.command = "FORWARD"
        self.last_odom = None
        self.last_sample_time = None
        self._busy = False
        # Lazily created on the first saved plot (avoids leaving an empty dir).
        self._plot_dir = None
        self._plot_seq = 0

        self.pub_path = rospy.Publisher(self.path_topic, Path, queue_size=1)
        self.pub_path_global = rospy.Publisher(
            self.path_global_topic, Path, queue_size=1)
        self.pub_array = rospy.Publisher(
            self.array_topic, Float64MultiArray, queue_size=1)
        self.pub_seg = rospy.Publisher(self.seg_topic, RosImage, queue_size=1)

        self.sub_odom = rospy.Subscriber(
            self.odom_topic, Odometry, self.cb_odom, queue_size=1)
        self.sub_command = rospy.Subscriber(
            self.command_topic, String, self.cb_command, queue_size=1)
        self.sub_image = rospy.Subscriber(
            self.in_topic, RosImage, self.cb_image, queue_size=1, buff_size=2**24)

        rospy.loginfo(f"[planner] subscribe image   {self.in_topic}")
        rospy.loginfo(f"[planner] subscribe odom    {self.odom_topic}")
        rospy.loginfo(f"[planner] subscribe command {self.command_topic}")
        rospy.loginfo(f"[planner] publish   path    {self.path_topic} ({self.frame_id})")
        rospy.loginfo(f"[planner] publish   path    {self.path_global_topic} (odom, global)")
        rospy.loginfo(f"[planner] publish   array   {self.array_topic} (Float64MultiArray, for MPC)")
        rospy.loginfo(f"[planner] flip_command={self.flip_command} "
                      f"(LEFT/RIGHT swapped before the model)")
        if self.save_plots:
            rospy.loginfo("[planner] save_plots=true: inference plots -> "
                          "realtime/inference/<ts>/inference_plots/")

    # ---------- callbacks ----------

    def cb_odom(self, msg: Odometry):
        self.last_odom = msg

    def cb_command(self, msg: String):
        command = msg.data.strip().upper()
        if command not in COMMAND_TO_ONEHOT:
            rospy.logwarn_throttle(
                5.0, f"[planner] ignoring unknown command {msg.data!r}; "
                     f"expected one of {sorted(COMMAND_TO_ONEHOT)}")
            return
        if command != self.command:
            rospy.loginfo(f"[planner] command {self.command} -> {command}")
        self.command = command

    def cb_image(self, msg: RosImage):
        if self._busy:
            return

        now = msg.header.stamp.to_sec()

        # A backwards time jump means the clock restarted (rosbag replay/loop,
        # sim-time reset). The buffered history belongs to the old timeline, so
        # drop it — otherwise every later frame fails the cadence check below
        # and the node stalls for good.
        if self.last_sample_time is not None and now < self.last_sample_time:
            rospy.logwarn(f"[planner] time jumped backwards by "
                          f"{self.last_sample_time - now:.1f}s; resetting history")
            self.buffer = RealtimeSequenceBuffer()
            self.last_sample_time = None
            # Treat a clock restart as a new run: start a fresh plot folder so
            # each bag replay lands in its own realtime/inference/<ts>/ dir.
            self._plot_dir = None
            self._plot_seq = 0

        # Enforce the model's 0.5 s cadence; everything else is dropped before
        # the segmentation runs.
        if self.last_sample_time is not None and (now - self.last_sample_time) < self.sample_interval:
            return
        if self.last_odom is None:
            rospy.loginfo_throttle(2.0, f"[planner] waiting for {self.odom_topic}")
            return

        self._busy = True
        try:
            self.process(msg)
            self.last_sample_time = now
        except Exception:
            rospy.logerr(f"[planner] inference failed:\n{__import__('traceback').format_exc()}")
        finally:
            self._busy = False

    # ---------- pipeline ----------

    def process(self, msg: RosImage):
        t0 = time.perf_counter()

        rgb = rosimg_to_rgb_numpy(msg)
        rgb_224 = resize_keep_ratio_center_crop_rgb(rgb)
        seg_id_224 = self.segment(rgb)
        t_seg = time.perf_counter()

        self.buffer.push(rgb_224, seg_id_224, pose_matrix_from_odom(self.last_odom))

        self.pub_seg.publish(rgb_numpy_to_rosimg(colorize_cls4_rgb(seg_id_224), msg.header))

        if not self.buffer.ready:
            rospy.loginfo_throttle(1.0, f"[planner] warming up: {self.buffer.status()}")
            return

        final_traj = self.plan()
        t_plan = time.perf_counter()

        self.pub_path.publish(self.build_path(final_traj, msg.header.stamp))
        self.pub_path_global.publish(
            self.build_path_global(final_traj, msg.header.stamp, self.last_odom))
        self.pub_array.publish(
            self.build_array_topic(final_traj, self.last_odom))

        if self.save_plots:
            self.save_plot(final_traj, msg.header.stamp)

        rospy.loginfo_throttle(
            1.0,
            f"[planner] seg={1000*(t_seg-t0):.1f} ms | plan={1000*(t_plan-t_seg):.1f} ms "
            f"| total={1000*(t_plan-t0):.1f} ms | command={self.command}"
        )

    @torch.inference_mode()
    def segment(self, rgb: np.ndarray) -> np.ndarray:
        """
        Full-resolution RGB -> (224,224) uint8 class ids in PALETTE4 semantics
        (0=road, 1=person, 2=movable, 3=static). Same path as seg_real_time.py.
        """
        inputs = self.processor(images=[rgb], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(self.device == "cuda" and self.use_fp16)):
            out = self.segformer(pixel_values=pixel_values)

        logits = F.interpolate(
            out.logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
        ).float()
        cls4 = logits19_to_cls4(logits).unsqueeze(1)
        return resize_keep_ratio_center_crop_uint8(cls4)[0, 0].cpu().numpy()

    def model_command(self) -> str:
        """Command actually fed to the model — see ~flip_command in __init__."""
        if self.flip_command and self.command in ("LEFT", "RIGHT"):
            return "RIGHT" if self.command == "LEFT" else "LEFT"
        return self.command

    def build_batch(self) -> dict:
        """Reproduces the offline loader's __getitem__ contract, with B=1."""
        rgb_seq = np.stack(list(self.buffer.rgb), axis=0)        # (3,224,224,3)
        seg_id_seq = np.stack(list(self.buffer.seg_id), axis=0)  # (3,224,224)
        seg_rgb_seq = SEG_PALETTE[seg_id_seq]                    # (3,224,224,3)

        command = self.model_command()  # LEFT/RIGHT swapped if flip_command
        empty = torch.empty(0)
        batch = {
            'rgb_224_seq': torch.from_numpy(rgb_seq).unsqueeze(0),
            'seg_224_seq': torch.from_numpy(seg_rgb_seq).unsqueeze(0),
            'seg_id_224_seq': torch.from_numpy(seg_id_seq.astype(np.int64)).unsqueeze(0),
            'future_egomotion': torch.from_numpy(self.buffer.build_future_egomotion()).unsqueeze(0),
            'admlp_input': torch.from_numpy(self.buffer.build_admlp_input(command)).unsqueeze(0),
            'command': [command],
            'target_point': torch.zeros(1, 2, dtype=torch.float32),
            # gt_trajectory never reaches the prediction: codex_pure_ASAP.py:756-770
            # uses it only for `device` and the training losses.
            'gt_trajectory': torch.zeros(1, N_FUTURE_FRAMES + 1, 3, dtype=torch.float32),
            # Dummy labels; _build_valid_occupancy rejects them (shape[-1] <= 1)
            # and returns None, exactly as it does offline.
            'segmentation': torch.zeros(1, TIME_RECEPTIVE_FIELD, 1, 1, 1, dtype=torch.long),
            'pedestrian': torch.zeros(1, TIME_RECEPTIVE_FIELD, 1, 1, 1, dtype=torch.long),
            # Unused by this image-space model (codex_pure_ASAP.py:622).
            'image': empty,
            'intrinsics': empty,
            'extrinsics': empty,
            'sample_trajectory': empty,
        }
        return batch

    def plan(self) -> np.ndarray:
        batch = self.build_batch()
        labels = _prepare_l2_labels(batch)

        # forward must run first: planning asserts on the caches it populates
        # (codex_pure_ASAP.py:757-759).
        output, is_vlm_gen = _call_model_forward(self.model, batch, self.device)
        _, final_traj = _call_model_planning(
            self.model, output, labels, batch, self.n_present, self.device, is_vlm_gen)
        return final_traj[0].detach().float().cpu().numpy()  # (6,3)

    def build_path(self, traj: np.ndarray, stamp) -> Path:
        """
        Model output is (x_left, y_front, yaw); ROS REP-103 base_link is x
        forward, y left — so the xy pair is swapped back. The yaw needs no
        swap: the loader never swapped it either.
        """
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.frame_id

        points = np.vstack([np.zeros((1, 3), dtype=np.float32), traj])  # prepend t0
        for i, (x_left, y_front, yaw) in enumerate(points):
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.header.stamp = stamp + rospy.Duration(i * self.sample_interval)
            pose.pose.position.x = float(y_front)
            pose.pose.position.y = float(x_left)
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = float(np.sin(yaw / 2.0))
            pose.pose.orientation.w = float(np.cos(yaw / 2.0))
            path.poses.append(pose)
        return path

    def _global_points(self, traj: np.ndarray, odom_msg: Odometry) -> np.ndarray:
        """
        The trajectory (t0 start + 6 future points) expressed in the global odom
        frame, as (7, 3) rows of (gx, gy, gyaw). The start point sits at the
        robot's current /odom (x, y); future points are rotated/translated by the
        current pose. The xy swap back to base_link (x forward, y left) matches
        build_path; the global rotation mirrors visualize.py:40-46
        (base_link_to_global). Shared by build_path_global and build_array_topic.
        """
        p = odom_msg.pose.pose.position
        o = odom_msg.pose.pose.orientation
        rx, ry = float(p.x), float(p.y)
        ryaw = quaternion_yaw(Quaternion(o.w, o.x, o.y, o.z))
        cos_r, sin_r = np.cos(ryaw), np.sin(ryaw)

        points = np.vstack([np.zeros((1, 3), dtype=np.float32), traj])  # prepend t0
        out = np.empty((points.shape[0], 3), dtype=np.float64)
        for i, (x_left, y_front, yaw) in enumerate(points):
            x_forward, y_left = y_front, x_left  # model (x_left, y_front) -> base_link
            out[i, 0] = rx + cos_r * x_forward - sin_r * y_left
            out[i, 1] = ry + sin_r * x_forward + cos_r * y_left
            out[i, 2] = ryaw + yaw
        return out

    def build_path_global(self, traj: np.ndarray, stamp, odom_msg: Odometry) -> Path:
        """
        Same trajectory as build_path, but expressed in the global odom frame
        (see _global_points). frame_id follows the /odom message frame.
        """
        gpts = self._global_points(traj, odom_msg)

        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = odom_msg.header.frame_id or "odom"

        for i, (gx, gy, gyaw) in enumerate(gpts):
            pose = PoseStamped()
            pose.header.frame_id = path.header.frame_id
            pose.header.stamp = stamp + rospy.Duration(i * self.sample_interval)
            pose.pose.position.x = float(gx)
            pose.pose.position.y = float(gy)
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = float(np.sin(gyaw / 2.0))
            pose.pose.orientation.w = float(np.cos(gyaw / 2.0))
            path.poses.append(pose)
        return path

    def build_array_topic(self, traj: np.ndarray, odom_msg: Odometry) -> Float64MultiArray:
        """
        The same global-frame points as build_path_global, flattened to
        [x0, y0, x1, y1, ...] (7 points = 14 values) as std_msgs/Float64MultiArray.
        This is the exact layout local_path.cpp::callbackorgwp expects on
        array_topic, so the MPC chain consumes the inferred path in place of
        global_path's CSV route without any controller-side change.
        """
        gpts = self._global_points(traj, odom_msg)
        msg = Float64MultiArray()
        msg.data = gpts[:, :2].reshape(-1).tolist()  # [x0,y0,x1,y1,...]
        return msg

    def _ensure_plot_dir(self) -> pathlib.Path:
        """
        realtime/inference/<MM_DD_HH_MM_SS>/inference_plots — mirrors the offline
        park_L2_ASAP.mk_save_dir layout, rooted under realtime/ instead. Created
        once, on the first saved plot.
        """
        if self._plot_dir is None:
            now = datetime.datetime.now()
            stamp = "_".join("%02d" % v for v in
                             (now.month, now.day, now.hour, now.minute, now.second))
            self._plot_dir = (pathlib.Path(_REPO_ROOT) / "realtime" / "inference"
                              / stamp / "inference_plots")
            self._plot_dir.mkdir(parents=True, exist_ok=True)
            rospy.loginfo(f"[planner] saving inference plots -> {self._plot_dir}")
        return self._plot_dir

    def save_plot(self, final_traj: np.ndarray, stamp) -> None:
        """
        Offline-style combo plot (camera image + trajectory panel), reusing
        park_L2_ASAP.save_inference_plot. Realtime has no future GT, so gt/l2 are
        empty (the panel then omits the blue GT track and the L2 text). The pred x
        sign is flipped exactly like the offline path (park_L2_ASAP.py:718) so the
        panel orientation matches; this copy never touches the published paths.
        """
        rgb_224 = self.buffer.rgb[-1]

        pred = np.asarray(final_traj, dtype=np.float64).copy()
        if pred.shape[1] == 2:
            pred = np.concatenate([pred, np.zeros((pred.shape[0], 1))], axis=1)
        pred[:, 0] *= -1.0

        fego = torch.from_numpy(self.buffer.build_future_egomotion()).float()
        input_xy, input_yaw = _input_history_from_egomotion(fego)

        gt = np.zeros((0, 3), dtype=np.float32)
        l2 = np.zeros((0,), dtype=np.float32)

        save_inference_plot(
            rgb_224=rgb_224,
            pred=pred,
            gt=gt,
            l2=l2,
            t_ref=stamp.to_nsec(),
            seq_idx=self._plot_seq,
            out_dir=self._ensure_plot_dir(),
            input_xy=input_xy,
            input_yaw=input_yaw,
        )
        self._plot_seq += 1


def main():
    rospy.init_node("realtime_planner_node", anonymous=False)
    RealtimePlannerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
