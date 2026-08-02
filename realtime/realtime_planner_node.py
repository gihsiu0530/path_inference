#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Realtime ST-P3 trajectory planning node.

Single-process pipeline that replaces the offline four-step flow
(bag_to_data -> resample -> convert_cls4png_to_npy -> park_L2_ASAP):

    camera + /odom  ->  SegFormer-B2 (4-class seg)  ->  ST-P3  ->  nav_msgs/Path
                        Depth-Anything-V2 (rel. depth)  ^

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
    _trajectory_xy_error,
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

# Depth-Anything-V2 — the depth channel the ST-P3 checkpoint was trained on.
#
# The offline dataset's depth_infer/*.npy came from tools/Depth-Anything-V2/
# infer_depth_da_v2.py run over the 224x224 crops, which takes that script's
# *batch* branch (`model(x)`): BGR->RGB, /255, NO ImageNet normalisation, and the
# 224x224 image straight into the network (224 = 16*14, so DINOv2's patch grid
# divides evenly). infer_image()'s 518-resize + normalisation is the other branch
# and produces different numbers — do not use it here.
#
# Both the package and the 1.3 GB weights are kept in-tree rather than referenced
# from /home/cyc/dataset/: that tree is volatile (several dataset folders,
# including the second Depth-Anything-V2 checkout, disappeared during a single
# session), and a vanished default would stop the node from starting. The weights
# are excluded from git by the *.pth rule, same as the ST-P3 *.ckpt.
# Both stay overridable by rosparam.
DEFAULT_DA_V2_REPO = os.path.join(_REPO_ROOT, "third_party", "Depth-Anything-V2")
DEFAULT_DA_V2_CKPT = os.path.join(_REPO_ROOT, "model", "depth_anything_v2_vitl.pth")
# vitl — the encoder the training depth was produced with. A smaller encoder
# shifts the whole output distribution and the ST-P3 checkpoint has never seen it.
DA_V2_ENCODER = "vitl"
DA_V2_FEATURES = 256
DA_V2_OUT_CHANNELS = [256, 512, 1024, 1024]


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


def load_depth_anything_v2(repo_dir: str, ckpt_path: str, device: str):
    """
    Build the Depth-Anything-V2 vitl model used to produce the depth channel.

    The import is done here rather than at module scope so that ~use_depth:=false
    still starts on a machine where the package or the weights are missing.
    """
    if not os.path.isdir(repo_dir):
        raise RuntimeError(
            f"Depth-Anything-V2 repo not found: {repo_dir}\n"
            "Point ~da_v2_repo at a checkout of it, or run with _use_depth:=false "
            "(which feeds the model zero depth — not what the checkpoint was trained on)."
        )
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(
            f"Depth-Anything-V2 weights not found: {ckpt_path}\n"
            f"Point ~da_v2_ckpt at depth_anything_v2_{DA_V2_ENCODER}.pth."
        )
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(encoder=DA_V2_ENCODER, features=DA_V2_FEATURES,
                            out_channels=DA_V2_OUT_CHANNELS)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval().to(device)


class RealtimeSequenceBuffer:
    """
    Ring buffers holding the past observations the model needs, sampled on a
    strict SAMPLE_INTERVAL cadence.

    The effective history horizon is driven by the poses (ADMLP_PAST_FRAMES + 1
    = 5 samples = 2.5 s), which is longer than the image window (3 samples), so
    the node stays in warm-up until both are full.
    """

    def __init__(self, use_depth: bool = False):
        self.rgb = deque(maxlen=TIME_RECEPTIVE_FIELD)     # (224,224,3) uint8
        self.seg_id = deque(maxlen=TIME_RECEPTIVE_FIELD)  # (224,224)   uint8
        self.poses = deque(maxlen=ADMLP_PAST_FRAMES + 1)  # 4x4 matrices
        # None when ~use_depth is off, which is also what tells build_batch to
        # leave 'depth_224_seq' out of the batch entirely.
        self.depth = deque(maxlen=TIME_RECEPTIVE_FIELD) if use_depth else None

    def push(self, rgb_224, seg_id_224, pose, depth_224=None):
        self.rgb.append(rgb_224)
        self.seg_id.append(seg_id_224)
        self.poses.append(pose)
        if self.depth is not None:
            self.depth.append(depth_224)

    @property
    def ready(self) -> bool:
        if self.depth is not None and len(self.depth) != self.depth.maxlen:
            return False
        return (len(self.rgb) == self.rgb.maxlen
                and len(self.poses) == self.poses.maxlen)

    def status(self) -> str:
        depth = (f", depths {len(self.depth)}/{self.depth.maxlen}"
                 if self.depth is not None else "")
        return (f"images {len(self.rgb)}/{self.rgb.maxlen}, "
                f"poses {len(self.poses)}/{self.poses.maxlen}{depth}")

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

    @staticmethod
    def fixed_speed_history(speed: float):
        """
        Synthetic straight-line history at a constant speed, in the model's
        (x_left, y_front, yaw) frame: the ego drove `speed` m/s straight
        forward, so past pose k sits (k * SAMPLE_INTERVAL * speed) metres
        behind the origin.

        Uses the module-level SAMPLE_INTERVAL, not ~sample_interval: the
        cadence the checkpoint assumes is what the feature must encode.

        Feeding this through estimate_current_motion would fit exactly
        [0, speed, 0] anyway (a degree-1 fit of a perfectly linear track), so
        the velocity is written directly and the lstsq is skipped.
        """
        past = np.zeros((ADMLP_PAST_FRAMES, 3), dtype=np.float32)
        for i in range(ADMLP_PAST_FRAMES):
            past[i, 1] = -(ADMLP_PAST_FRAMES - i) * SAMPLE_INTERVAL * speed
        velocity = np.asarray([0.0, speed, 0.0], dtype=np.float32)
        acceleration = np.zeros(3, dtype=np.float32)
        return past, velocity, acceleration

    def build_admlp_input(self, command: str, fixed_speed: float = 0.0) -> np.ndarray:
        """21-dim feature — loader :478-494."""
        if fixed_speed > 0.0:
            past, velocity, acceleration = self.fixed_speed_history(fixed_speed)
        else:
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
        self.array_topic = rospy.get_param("~array_topic", "/senpai/array_topic")
        self.seg_topic = rospy.get_param("~seg_topic", "/senpai/seg_cls4_224")
        self.frame_id = rospy.get_param("~frame_id", "base_link")
        self.checkpoint = rospy.get_param("~checkpoint", DEFAULT_CHECKPOINT)
        self.sample_interval = float(rospy.get_param("~sample_interval", SAMPLE_INTERVAL))
        # Save an offline-style inference plot (camera image + trajectory panel)
        # per inference cycle. On by default; set false for a lightweight run.
        self.save_plots = bool(rospy.get_param("~save_plots", True))
        # Add the two segmentation panels (PALETTE4 + the model's own offset
        # palette) to that plot. False falls back to the two-panel layout.
        self.plot_seg = bool(rospy.get_param("~plot_seg", True))
        # Add the Depth-Anything-V2 panel (min-max normalised, like the offline
        # script's --save_vis). Ignored when ~use_depth is off.
        self.plot_depth = bool(rospy.get_param("~plot_depth", True))
        # Feed the model the Depth-Anything-V2 relative depth it was trained on.
        # Off means a zero depth map reaches the fixed depth channel of the
        # model's fusion layers — the channel cannot be removed, so "no depth"
        # is really "constant depth", which is not what the checkpoint saw.
        self.use_depth = bool(rospy.get_param("~use_depth", True))
        self.da_v2_repo = rospy.get_param("~da_v2_repo", DEFAULT_DA_V2_REPO)
        self.da_v2_ckpt = rospy.get_param("~da_v2_ckpt", DEFAULT_DA_V2_CKPT)
        # Feed the model a synthetic "driving straight at N m/s" ego state instead
        # of the velocity/past poses fitted from real odom. The model plans
        # noticeably better when it believes the vehicle is moving, so this keeps
        # the quality of a rolling start even at standstill. 0 (default) = use the
        # real history. Only admlp_input is affected; future_egomotion, the
        # published paths and the inference plots all stay on real odom.
        self.fixed_speed = float(rospy.get_param("~fixed_speed", 0.0))
        if self.fixed_speed < 0.0:
            rospy.logwarn(f"[planner] ignoring negative ~fixed_speed={self.fixed_speed}")
            self.fixed_speed = 0.0

        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = rospy.get_param("~device", default_device)
        self.use_fp16 = bool(rospy.get_param("~use_fp16", True))

        rospy.loginfo(f"[planner] device={self.device}")

        # ---------- SegFormer (same as seg_real_time.py:227-232) ----------
        self.processor = SegformerImageProcessor.from_pretrained(SEGFORMER_NAME)
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_NAME
        ).to(self.device).eval()

        # ---------- Depth-Anything-V2 ----------
        # Kept in fp32: the benchmark put it at 17.8 ms/frame against 13.4 ms
        # under autocast, and fp32 actually needs less memory (1.4 vs 2.1 GB)
        # because autocast keeps a cast buffer. 17.8 ms is 3.6% of the 0.5 s
        # cycle, so the faster path buys nothing worth the extra memory.
        self.depth_model = None
        if self.use_depth:
            if self.device == "cpu":
                rospy.logwarn("[planner] use_depth=true on CPU: vitl takes ~17.8 ms on a "
                              "4070 SUPER but seconds on CPU, which will not hold the "
                              f"{self.sample_interval:.1f} s cadence")
            rospy.loginfo(f"[planner] loading Depth-Anything-V2 {self.da_v2_ckpt}")
            self.depth_model = load_depth_anything_v2(
                self.da_v2_repo, self.da_v2_ckpt, self.device)

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

        self.buffer = RealtimeSequenceBuffer(self.use_depth)
        self.command = "FORWARD"
        self.last_odom = None
        self.last_sample_time = None
        self._busy = False
        # Lazily created on the first saved plot (avoids leaving an empty dir).
        self._plot_dir = None
        self._plot_seq = 0
        # Inferences waiting for their retrospective GT to fill in; at most
        # N_FUTURE_FRAMES entries (~400 kB each: 150 kB camera frame, 50 kB seg
        # ids, 200 kB depth when ~use_depth is on), so ~2.4 MB fully loaded.
        self._pending = deque()

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
        if self.use_depth:
            rospy.loginfo("[planner] use_depth=true: Depth-Anything-V2 relative depth "
                          f"({DA_V2_ENCODER}) -> depth_224_seq, as the checkpoint was trained")
        else:
            rospy.logwarn("[planner] use_depth=false: the model gets a ZERO depth map. "
                          "The checkpoint was trained with real Depth-Anything-V2 depth, "
                          "so this is a train/inference mismatch")
        if self.fixed_speed > 0.0:
            rospy.loginfo(f"[planner] fixed_speed={self.fixed_speed:.2f} m/s: admlp_input uses "
                          "a synthetic straight-line history (real odom still drives "
                          "future_egomotion and the published paths)")
        if self.save_plots:
            panels = ["camera"]
            if self.plot_seg:
                panels += ["seg PALETTE4", "seg model-input"]
            if self.plot_depth and self.use_depth:
                panels.append("depth DA-V2")
            layout = " | ".join(panels + ["trajectory"])
            rospy.loginfo("[planner] save_plots=true: inference plots -> "
                          "realtime/inference/<ts>/inference_plots/ "
                          f"[{layout}] "
                          f"(each plot lands {N_FUTURE_FRAMES * self.sample_interval:.1f}s late, "
                          "once its GT is known; _save_plots:=false to disable, "
                          "_plot_seg:=false to drop the seg panels)")

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
            self.buffer = RealtimeSequenceBuffer(self.use_depth)
            self.last_sample_time = None
            # Pending plots belong to the old timeline: write them out with the
            # GT collected so far, before _plot_dir moves on. Order matters —
            # flushing after the reset below would file them under the new dir.
            self._flush_pending(force=True)
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

        depth_224 = self.infer_depth(rgb_224) if self.use_depth else None
        t_depth = time.perf_counter()

        pose = pose_matrix_from_odom(self.last_odom)
        self.buffer.push(rgb_224, seg_id_224, pose, depth_224)

        # This pose is a *future* point for every inference already queued, so
        # feed it in before this cycle enqueues its own record below — that way a
        # record never consumes its own pose and each collects exactly t+1..t+6.
        self._advance_pending(pose)

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

        self._enqueue_plot(final_traj, msg.header.stamp, pose)

        depth_ms = (f"depth={1000*(t_depth-t_seg):.1f} ms | " if self.use_depth else "")
        rospy.loginfo_throttle(
            1.0,
            f"[planner] seg={1000*(t_seg-t0):.1f} ms | {depth_ms}"
            f"plan={1000*(t_plan-t_depth):.1f} ms "
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

    @torch.inference_mode()
    def infer_depth(self, rgb_224: np.ndarray) -> np.ndarray:
        """
        (224,224,3) uint8 RGB -> (224,224) float32 relative inverse depth, i.e.
        larger = closer, with a per-frame scale (measured drift across
        consecutive frames: ~2% of the frame maximum).

        Reproduces infer_depth_da_v2.py's batch branch bit for bit: /255 only —
        no ImageNet mean/std — and no resize, since the crop is already 224.
        The fp16 round-trip is what the offline `.npy` files went through
        (`--dtype_fp16`), reproduced so the model sees the same quantisation.

        Values land around 14..330 on this data while the model's own _depth()
        does clamp(0,80)/80, so roughly three quarters of the pixels saturate.
        That is what the checkpoint was trained on: rescaling to "use the range
        properly" would move the input off the trained distribution.
        """
        x = torch.from_numpy(np.ascontiguousarray(rgb_224))
        x = x.permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(self.device)
        depth = self.depth_model(x)[0].float().cpu().numpy()
        return depth.astype(np.float16).astype(np.float32)

    def build_batch(self) -> dict:
        """Reproduces the offline loader's __getitem__ contract, with B=1."""
        rgb_seq = np.stack(list(self.buffer.rgb), axis=0)        # (3,224,224,3)
        seg_id_seq = np.stack(list(self.buffer.seg_id), axis=0)  # (3,224,224)
        seg_rgb_seq = SEG_PALETTE[seg_id_seq]                    # (3,224,224,3)

        command = self.command
        empty = torch.empty(0)
        batch = {
            'rgb_224_seq': torch.from_numpy(rgb_seq).unsqueeze(0),
            'seg_224_seq': torch.from_numpy(seg_rgb_seq).unsqueeze(0),
            'seg_id_224_seq': torch.from_numpy(seg_id_seq.astype(np.int64)).unsqueeze(0),
            'future_egomotion': torch.from_numpy(self.buffer.build_future_egomotion()).unsqueeze(0),
            'admlp_input': torch.from_numpy(
                self.buffer.build_admlp_input(command, self.fixed_speed)).unsqueeze(0),
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
        # Only present when ~use_depth is on. Leaving the key out is what makes
        # _call_model_forward (park_L2_ASAP.py:486) skip the kwarg, which in turn
        # makes codex_pure_ASAP.forward substitute its zero-depth placeholder —
        # the same fallback the offline `--no-depth` run takes.
        if self.buffer.depth is not None:
            depth_seq = np.stack(list(self.buffer.depth), axis=0)  # (3,224,224)
            batch['depth_224_seq'] = torch.from_numpy(depth_seq).float().unsqueeze(0)
        return batch

    def plan(self) -> np.ndarray:
        """
        One inference, as (6, 3) rows of (x_left, y_front, yaw).

        The model's own lateral channel is +right, the opposite of the loader's
        gt_trajectory convention, so it is flipped here — once, for every
        consumer — exactly like the offline path (park_L2_ASAP.py:742). Verified
        against the offline dump: with the flip, all 177 turning samples
        (|gt_x| >= 1) match the GT sign and corr(gt_x, pred_x) is 0.88.
        """
        batch = self.build_batch()
        labels = _prepare_l2_labels(batch)

        # forward must run first: planning asserts on the caches it populates
        # (codex_pure_ASAP.py:757-759).
        output, is_vlm_gen = _call_model_forward(self.model, batch, self.device)
        _, final_traj = _call_model_planning(
            self.model, output, labels, batch, self.n_present, self.device, is_vlm_gen)
        # .copy() so the in-place flip never writes through to the model's own
        # tensor, which numpy() shares storage with on a CPU device.
        traj = final_traj[0].detach().float().cpu().numpy().copy()  # (6,3)
        traj[:, 0] *= -1.0  # model lateral is +right -> x_left
        return traj

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

    def _enqueue_plot(self, final_traj: np.ndarray, stamp, pose: np.ndarray) -> None:
        """
        Queue one inference for plotting once its GT is known.

        Everything the plot needs from the *current* observation is captured here,
        because the ring buffers will have moved on by the time the record is
        written out 3 s later. The trajectory arrives already in the plot's
        (x_left, y_front) convention — plan() does the one lateral flip — so this
        copy only pads the yaw column when the model returns bare xy.
        """
        if not self.save_plots:
            return

        pred = np.asarray(final_traj, dtype=np.float64).copy()
        if pred.shape[1] == 2:
            pred = np.concatenate([pred, np.zeros((pred.shape[0], 1))], axis=1)

        fego = torch.from_numpy(self.buffer.build_future_egomotion()).float()
        input_xy, input_yaw = _input_history_from_egomotion(fego)

        self._pending.append({
            'seq_idx': self._plot_seq,
            't_ref': stamp.to_nsec(),
            'rgb_224': np.array(self.buffer.rgb[-1], copy=True),
            # Same frame as rgb_224: process() pushes all three before enqueueing.
            'seg_id_224': np.array(self.buffer.seg_id[-1], copy=True),
            'depth_224': (np.array(self.buffer.depth[-1], copy=True)
                          if self.buffer.depth is not None else None),
            'pred': pred,
            'input_xy': input_xy,
            'input_yaw': input_yaw,
            'pose_inv': np.linalg.inv(pose),
            'future': [],   # driven path in this record's planning frame
        })
        self._plot_seq += 1

    def _advance_pending(self, pose: np.ndarray) -> None:
        """
        Record `pose` as the next driven waypoint of every queued inference, and
        write out the ones whose 3 s of GT is now complete. In steady state this
        writes exactly one plot per cycle, so the per-cycle cost stays flat.
        """
        for rec in self._pending:
            rec['future'].append(relative_xy_yaw(rec['pose_inv'], pose))

        while self._pending and len(self._pending[0]['future']) >= N_FUTURE_FRAMES:
            self._write_plot(self._pending.popleft())

    def _flush_pending(self, force: bool = False) -> None:
        """
        Drain the queue. With force=True the still-incomplete records are written
        using the GT collected so far (shorter blue track) — used on shutdown and
        on a clock restart, so the last few inferences of a run still get a plot.
        """
        while self._pending:
            rec = self._pending.popleft()
            if force or len(rec['future']) >= N_FUTURE_FRAMES:
                self._write_plot(rec)

    def _seg_panels(self, rec: dict) -> list:
        """
        The two segmentation panels for the combo plot, both derived from the
        same class-id map the model was fed:

        - PALETTE4     — colorize_cls4_rgb, i.e. exactly what /senpai/seg_cls4_224
                         and the offline dataset PNGs (bag_to_data.py) look like.
        - model-input  — SEG_PALETTE indexed the same way as build_batch, i.e. the
                         literal pixels that reach the checkpoint. Deliberately
                         *not* labelled by palette name: road reads black here
                         because of the documented one-class offset above, and
                         that is not a bug to fix.
        """
        seg_id = rec.get('seg_id_224')
        if not self.plot_seg or seg_id is None:
            return []
        return [
            ("SEG PALETTE4", colorize_cls4_rgb(seg_id)),
            ("SEG model-input", SEG_PALETTE[seg_id]),
        ]

    def _depth_panel(self, rec: dict) -> list:
        """
        The Depth-Anything-V2 panel, min-max normalised per frame exactly like
        infer_depth_da_v2.py's --save_vis. Bright = near (the raw values are
        inverse depth). This is a *visualisation* stretch only — the model is
        still fed the raw, unnormalised values.
        """
        depth = rec.get('depth_224')
        if not self.plot_depth or depth is None:
            return []
        d = depth.astype(np.float32)
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        gray = (d * 255.0).clip(0, 255).astype(np.uint8)
        return [("DEPTH DA-V2 (bright=near)", np.repeat(gray[..., None], 3, axis=2))]

    def _write_plot(self, rec: dict) -> None:
        """
        Offline-style combo plot (camera image + seg panels + trajectory panel),
        reusing park_L2_ASAP.save_inference_plot. GT is the path the robot drove,
        in the same (x_left, y_front, yaw) convention and same leading (0,0,0) row
        as the offline loader's gt_trajectory
        (NuscenesData_0624_ASAP.get_gt_trajectory), so it needs no sign flip.
        """
        future = rec['future']
        if future:
            gt = np.vstack([np.zeros((1, 3)), np.asarray(future, dtype=np.float64)])
            # _trajectory_xy_error clamps to min(T), which is what makes a
            # force-flushed record with partial GT work.
            xy_error = _trajectory_xy_error(
                torch.from_numpy(rec['pred']).float().unsqueeze(0),
                torch.from_numpy(gt).float().unsqueeze(0),
            )
            l2 = torch.linalg.norm(xy_error, dim=-1)[0].numpy()
        else:
            # Force-flushed before a single future pose arrived: fall back to the
            # input + pred panel rather than drawing a lone GT dot at the origin.
            gt = np.zeros((0, 3), dtype=np.float32)
            l2 = np.zeros((0,), dtype=np.float32)

        save_inference_plot(
            rgb_224=rec['rgb_224'],
            pred=rec['pred'],
            gt=gt,
            l2=l2,
            t_ref=rec['t_ref'],
            seq_idx=rec['seq_idx'],
            out_dir=self._ensure_plot_dir(),
            input_xy=rec['input_xy'],
            input_yaw=rec['input_yaw'],
            extra_panels=self._seg_panels(rec) + self._depth_panel(rec),
        )


def main():
    rospy.init_node("realtime_planner_node", anonymous=False)
    node = RealtimePlannerNode()
    # The last few inferences of a run never see their full 3 s of GT; write them
    # out with what was collected instead of dropping them.
    rospy.on_shutdown(lambda: node._flush_pending(force=True))
    rospy.spin()


if __name__ == "__main__":
    main()
