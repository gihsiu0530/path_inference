#!/usr/bin/env python3
"""ROS realtime planner with VIO history and CLIP trajectory correction.

The copied legacy node remains untouched. This entry point injects a new-model
runtime, fixes the semantic palette and calibration, and uses a low-latency
vision-language scorer to keep candidate trajectories away from wooden
boardwalks and pedestrian decking that the semantic model mistakes for road.
"""

from contextlib import nullcontext
from pathlib import Path
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
# ROS Noetic's rospy is already exposed through /opt/ros in this machine, but
# its pure-Python rospkg/catkin_pkg dependencies live here. Append (never
# prepend) after importing conda NumPy/Torch, otherwise Ubuntu's Python-3.8
# NumPy shadows the environment's Python-3.9 binary extension.
_SYSTEM_PYTHON_DIST = "/usr/lib/python3/dist-packages"
try:
    import catkin_pkg  # noqa: F401
    import rospkg  # noqa: F401
except ImportError:
    if _SYSTEM_PYTHON_DIST not in sys.path:
        sys.path.append(_SYSTEM_PYTHON_DIST)
        remove_system_dist = True
    else:
        remove_system_dist = False
    try:
        import catkin_pkg  # noqa: F401
        import rospkg  # noqa: F401
    finally:
        if remove_system_dist:
            sys.path.remove(_SYSTEM_PYTHON_DIST)
# Use only this self-contained FF tree for project-local imports.  The parent
# project is deliberately removed so a missing FF module cannot silently fall
# back to another copy under path_inference.
for local_path in (str(_THIS_DIR), str(_PROJECT_ROOT)):
    while local_path in sys.path:
        sys.path.remove(local_path)
sys.path.insert(0, str(_THIS_DIR))

# Do not silently reuse an stp3 package preloaded by a wrapper or launch file.
import stp3 as _stp3_package

_EXPECTED_STP3_INIT = (_THIS_DIR / "stp3" / "__init__.py").resolve()
_LOADED_STP3_INIT = getattr(_stp3_package, "__file__", None)
if _LOADED_STP3_INIT is None or Path(_LOADED_STP3_INIT).resolve() != _EXPECTED_STP3_INIT:
    raise RuntimeError(
        "realtime_planner_node_ff must use its private stp3 package: "
        f"expected {_EXPECTED_STP3_INIT}, loaded {_LOADED_STP3_INIT}"
    )

# The legacy node imports these names from park_L2_ASAP. Supply the small FF
# runtime module instead, avoiding the legacy workspace's incompatible stp3.
import realtime_runtime_ff as runtime_ff

sys.modules["park_L2_ASAP"] = runtime_ff
import realtime_planner_node as legacy
import twinlite_backend

from geometry_msgs.msg import PoseWithCovarianceStamped
from traj_smoothing import TrajectorySmoothingError, smooth_and_resample


STANDARD_SEG_PALETTE = np.asarray(
    [
        [128, 64, 128],  # road
        [220, 20, 60],   # person
        [0, 0, 142],     # movable
        [70, 70, 70],    # static
    ],
    dtype=np.uint8,
)

# The model's lateral output is intentionally amplified more at the far end of
# the horizon, where a stronger turn is useful, without making the first
# controller waypoint jump sideways.  The gain follows quadratic waypoint
# progress, so it grows gently near the vehicle and faster near the horizon.
LATERAL_GAIN_FIRST = 3.0
LATERAL_GAIN_LAST = 10.0

# YOLO26 semantic checkpoints without a dataset suffix use the standard 19
# Cityscapes train IDs. Convert their dense class map to the four classes the FF
# checkpoint was trained with: 0=road, 1=person, 2=movable, 3=static.
YOLO_SEM_CITYSCAPES_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)
YOLO_SEM_CITYSCAPES_TO_CLS4 = (
    0,  # road
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # static scene classes
    1,  # person
    2, 2, 2, 2, 2, 2, 2,  # rider and vehicles
)
DEFAULT_YOLO_SEM_MODEL = str(_THIS_DIR / "model" / "yolo26l-sem.pt")

# The only vision-language checkpoint already cached on the deployment
# machine. It is a full OpenAI CLIP ViT-L/14 dual encoder and can score all five
# candidate corridor images in one batch without autoregressive text decoding.
DEFAULT_VLM_CLIP_CHECKPOINT = (
    "/home/cyc/.cache/huggingface/hub/"
    "models--timm--vit_large_patch14_clip_224.openai/"
    "snapshots/18d0535469bb561bf468d76c1d73aa35156c922b/"
    "open_clip_model.safetensors"
)
VLM_CANDIDATE_OFFSETS_LEFT_M = np.asarray(
    [0.6, 0.3, 0.0, -0.3, -0.6], dtype=np.float32
)
VLM_SAFE_PROMPTS = (
    "a photo of a safe paved asphalt road for driving",
    "black asphalt pavement for vehicles",
    "a vehicle driving lane made of asphalt",
)
VLM_UNSAFE_PROMPTS = (
    "a photo of a wooden boardwalk with planks",
    "a pedestrian wooden walkway",
    "wooden decking unsafe for a vehicle",
    "a sidewalk or pedestrian path beside a road",
)
CLIP_RGB_MEAN = np.asarray([0.48145466, 0.45782750, 0.40821073], dtype=np.float32)
CLIP_RGB_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# Calibration copied from park_L2_ASAP_ff.py for the driver-rotated ZED2i
# image, OpenVINS cam0 intrinsics, assumed IMU roll, and 1.8 m IMU height.
K_224_FF = np.asarray(
    [
        [293.93628, 0.0, 112.05932],
        [0.0, 293.93628, 110.19693],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
CAMERA_TO_EGO_FF = np.asarray(
    [
        [0.00429783, -0.00244263, 0.9999878, 0.00209962],
        [-0.99997777, -0.00510304, 0.00428532, -0.02305081],
        [0.00509251, -0.99998397, -0.00246451, 1.8003294],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


# Patch only globals deliberately consumed by the copied implementation.
legacy.SEG_PALETTE = STANDARD_SEG_PALETTE
legacy.DEFAULT_CHECKPOINT = str(_THIS_DIR / "model" / "last.ckpt")
legacy.SEGFORMER_NAME = str(_THIS_DIR / "model" / "segformer-b2-cityscapes")
legacy.DEFAULT_DA_V2_REPO = str(_THIS_DIR / "third_party" / "Depth-Anything-V2")
legacy.DEFAULT_DA_V2_CKPT = str(
    _PROJECT_ROOT / "model" / "depth_anything_v2_vitl.pth"
)


class RealtimeSequenceBufferFF(legacy.RealtimeSequenceBuffer):
    """Legacy observation buffer plus FF's VIO ego-motion history."""

    def __init__(self, use_depth: bool = False):
        super().__init__(use_depth)
        self._next_history_pose = None

    def set_next_history_pose(self, pose: np.ndarray) -> None:
        """Select the VIO pose that belongs to the next sampled image."""
        self._next_history_pose = pose

    def push(self, rgb_224, seg_id_224, pose, depth_224=None):
        # The inherited pipeline passes its /odom pose here because that pose is
        # still needed for global path publication and GT plots. Only the pose
        # stored as model history is replaced with /ov_msckf/poseimu.
        if self._next_history_pose is None:
            raise RuntimeError("VIO history pose was not set before buffer.push().")
        history_pose = self._next_history_pose
        self._next_history_pose = None
        super().push(rgb_224, seg_id_224, history_pose, depth_224)

    @staticmethod
    def _stored_motion_between(pose_old: np.ndarray, pose_new: np.ndarray) -> np.ndarray:
        # Match NuscenesData.get_egomotion_between: inv(pose_t1) @ pose_t0.
        transform = np.linalg.inv(pose_new) @ pose_old
        transform[3, :3] = 0.0
        transform[3, 3] = 1.0
        vector = legacy.mat2pose_vec(torch.from_numpy(transform).float().unsqueeze(0))
        return vector.squeeze(0).numpy().astype(np.float32)

    def build_ego_history_egomotion(
        self, fixed_speed_mps: float = 0.0
    ) -> np.ndarray:
        """Return four stored increments plus current zero sentinel: (5,6)."""
        history = np.zeros((legacy.ADMLP_PAST_FRAMES + 1, 6), dtype=np.float32)
        if fixed_speed_mps > 0.0:
            history[:-1, 0] = -float(fixed_speed_mps) * legacy.SAMPLE_INTERVAL
            return history
        poses = list(self.poses)
        if len(poses) < legacy.ADMLP_PAST_FRAMES + 1:
            raise RuntimeError("FF ego history requested before five VIO poses are ready.")
        poses = poses[-(legacy.ADMLP_PAST_FRAMES + 1):]
        for index in range(legacy.ADMLP_PAST_FRAMES):
            history[index] = self._stored_motion_between(poses[index], poses[index + 1])
        return history

    @staticmethod
    def build_fixed_future_egomotion(fixed_speed_mps: float) -> np.ndarray:
        motion = np.zeros((legacy.TIME_RECEPTIVE_FIELD, 6), dtype=np.float32)
        motion[:-1, 0] = -float(fixed_speed_mps) * legacy.SAMPLE_INTERVAL
        return motion


# The base node creates and resets this class through its module global.
legacy.RealtimeSequenceBuffer = RealtimeSequenceBufferFF


class RealtimePlannerNodeFF(
    twinlite_backend.TwinLiteBackendMixin, legacy.RealtimePlannerNode
):
    def __init__(self):
        # Keep /odom for global path publication, but source every pose stored in
        # the model's temporal buffer from OpenVINS instead.
        self.vio_topic = legacy.rospy.get_param(
            "~vio_topic", "/ov_msckf/poseimu"
        )
        self.last_vio = None

        requested_segmentation_backend = str(
            legacy.rospy.get_param("~segmentation_backend", "segformer")
        ).strip().lower().replace("-", "_")
        if requested_segmentation_backend == "legacy":
            requested_segmentation_backend = "segformer"
        if requested_segmentation_backend in {
            "yolo26", "yolo_sem", "yolo26s", "yolo26s_sem",
            "yolo26m", "yolo26m_sem",
            "yolo26l", "yolo26l_sem",
        }:
            requested_segmentation_backend = "yolo26_sem"
        if requested_segmentation_backend in twinlite_backend.TWINLITE_BACKEND_ALIASES:
            requested_segmentation_backend = twinlite_backend.TWINLITE_BACKEND
        if requested_segmentation_backend not in {
            "segformer", "yolo26_sem", "twinlitenet",
        }:
            raise ValueError(
                "~segmentation_backend must be segformer (or legacy), "
                "yolo26_sem, or twinlitenet"
            )

        # The inherited constructor loads the legacy backend and starts ROS
        # subscribers. Keep callbacks on it until the new backend is fully ready.
        self.segmentation_backend = "segformer"
        self.yolo_sem = None
        self._twinlite_init_state()
        super().__init__()

        requested_mode = str(legacy.rospy.get_param("~ego_input_mode", "auto")).strip().lower()
        if requested_mode == "auto":
            requested_mode = "fixed_speed" if self.fixed_speed > 0.0 else "real_odom"
        if requested_mode not in {"real_odom", "fixed_speed"}:
            raise ValueError("~ego_input_mode must be auto, real_odom, or fixed_speed")
        default_fixed_speed = self.fixed_speed if self.fixed_speed > 0.0 else 1.0
        self.fixed_speed_mps = float(
            legacy.rospy.get_param("~fixed_speed_mps", default_fixed_speed)
        )
        if requested_mode == "fixed_speed" and self.fixed_speed_mps <= 0.0:
            raise ValueError("fixed_speed mode requires ~fixed_speed_mps > 0")
        self.ego_input_mode = requested_mode

        tag = str(getattr(self.model.cfg, "TAG", "")).strip().lower()
        self.model_variant = runtime_ff.SUPPORTED_TAGS.get(tag)
        if self.model_variant is None:
            raise RuntimeError(f"Loaded unsupported FF model TAG={getattr(self.model.cfg, 'TAG', None)!r}")
        expected = str(legacy.rospy.get_param("~expected_model_variant", "auto")).strip().lower()
        if expected not in {"auto", "all_admlp", "hybrid"}:
            raise ValueError("~expected_model_variant must be auto, all_admlp, or hybrid")
        if expected != "auto" and expected != self.model_variant:
            raise RuntimeError(
                f"Checkpoint auto-detected as {self.model_variant}, but "
                f"~expected_model_variant={expected}"
            )

        cfg_dt = float(getattr(self.model.cfg, "SAMPLE_INTERVAL", legacy.SAMPLE_INTERVAL))
        if abs(self.sample_interval - cfg_dt) > 1e-6:
            raise ValueError(
                f"Realtime sample interval {self.sample_interval} differs from checkpoint {cfg_dt}."
            )
        if int(self.n_present) != legacy.TIME_RECEPTIVE_FIELD:
            raise ValueError(
                f"Checkpoint receptive field={self.n_present}; runtime expects "
                f"{legacy.TIME_RECEPTIVE_FIELD}."
            )

        time_steps = int(self.n_present)
        self.intrinsics_ff = (
            torch.from_numpy(K_224_FF).view(1, 1, 1, 3, 3).repeat(1, time_steps, 1, 1, 1)
        )
        self.extrinsics_ff = (
            torch.from_numpy(CAMERA_TO_EGO_FF)
            .view(1, 1, 1, 4, 4)
            .repeat(1, time_steps, 1, 1, 1)
        )
        self.last_plan_inference_ms = float("nan")

        # Fit one smooth B-spline through the predicted waypoints and re-sample
        # it at a constant spacing before publishing. The raw points are spaced
        # by time, so their spacing (and the MPC's effective look-ahead) drifts
        # with speed, and the model's lateral noise reaches the controller as a
        # weave. False publishes the raw points exactly as the legacy node does.
        self.path_smoothing = bool(legacy.rospy.get_param("~path_smoothing", True))
        self.path_point_spacing_m = float(
            legacy.rospy.get_param("~path_point_spacing_m", 0.7)
        )
        # Metres of RMS deviation from the raw prediction the fit may spend:
        # larger is smoother and less faithful.
        self.path_smooth_sigma_m = float(
            legacy.rospy.get_param("~path_smooth_sigma_m", 0.2)
        )
        # local_path.cpp publishes /stop_end when fewer than four waypoints lie
        # ahead of the vehicle, so short predictions are extrapolated to at
        # least this many points.
        self.path_min_points = int(legacy.rospy.get_param("~path_min_points", 7))
        # (source trajectory, points, times) for the current cycle. The three
        # builders are called once each per cycle on the same trajectory, and
        # the smoothing must not run three times.
        self._published_cache = None

        # OpenVINS can briefly publish an all-zero or numerically tiny history
        # during initialization. A hybrid FORWARD coarse path extrapolated from
        # that history has no longitudinal direction, so visual residual noise
        # can dominate and make the prediction look mostly lateral. In that
        # specific case, feed a small synthetic forward history and enforce the
        # same minimum progress on the output waypoints.
        self.vio_small_motion_fallback = bool(
            legacy.rospy.get_param("~vio_small_motion_fallback", True)
        )
        self.vio_small_translation_m = float(
            legacy.rospy.get_param("~vio_small_translation_m", 0.03)
        )
        self.vio_small_yaw_deg = float(
            legacy.rospy.get_param("~vio_small_yaw_deg", 1.0)
        )
        self.vio_fallback_forward_step_m = float(
            legacy.rospy.get_param("~vio_fallback_forward_step_m", 0.30)
        )
        if self.vio_small_translation_m < 0.0 or self.vio_small_yaw_deg < 0.0:
            raise ValueError("VIO small-motion thresholds must be non-negative")
        if self.vio_fallback_forward_step_m <= 0.0:
            raise ValueError("~vio_fallback_forward_step_m must be positive")
        self.vio_fallback_active = False

        # ---------- vision-language candidate scorer ----------
        # Five trajectories are evaluated as a batch. Positive offset is public
        # / ROS left, matching build_path() after plan() performs its one axis
        # conversion. The VLM never emits a free-form number or text response.
        self.vlm_enabled = bool(legacy.rospy.get_param("~vlm_enabled", True))
        self.vlm_clip_checkpoint = str(
            Path(
                legacy.rospy.get_param(
                    "~vlm_clip_checkpoint", DEFAULT_VLM_CLIP_CHECKPOINT
                )
            ).expanduser()
        )
        self.vlm_max_latency_ms = float(
            legacy.rospy.get_param("~vlm_max_latency_ms", 200.0)
        )
        self.vlm_near_m = float(legacy.rospy.get_param("~vlm_near_m", 4.3))
        self.vlm_lookahead_m = float(
            legacy.rospy.get_param("~vlm_lookahead_m", 12.0)
        )
        self.vlm_corridor_half_width_m = float(
            legacy.rospy.get_param("~vlm_corridor_half_width_m", 0.40)
        )
        self.vlm_offset_ramp_m = float(
            legacy.rospy.get_param("~vlm_offset_ramp_m", 2.0)
        )
        self.vlm_offset_penalty = float(
            legacy.rospy.get_param("~vlm_offset_penalty", 0.002)
        )
        self.vlm_min_score_improvement = float(
            legacy.rospy.get_param("~vlm_min_score_improvement", 0.004)
        )
        self.vlm_min_safe_probability = float(
            legacy.rospy.get_param("~vlm_min_safe_probability", 0.55)
        )
        self.vlm_filter_alpha = float(
            legacy.rospy.get_param("~vlm_filter_alpha", 0.65)
        )
        self.vlm_max_offset_step_m = float(
            legacy.rospy.get_param("~vlm_max_offset_step_m", 0.50)
        )
        if self.vlm_near_m <= 0.0 or self.vlm_lookahead_m <= self.vlm_near_m:
            raise ValueError("VLM lookahead must satisfy 0 < ~vlm_near_m < ~vlm_lookahead_m")
        if self.vlm_corridor_half_width_m <= 0.0:
            raise ValueError("~vlm_corridor_half_width_m must be positive")
        if self.vlm_offset_ramp_m <= 0.0:
            raise ValueError("~vlm_offset_ramp_m must be positive")
        if not 0.0 < self.vlm_filter_alpha <= 1.0:
            raise ValueError("~vlm_filter_alpha must be in (0, 1]")
        if self.vlm_max_offset_step_m <= 0.0:
            raise ValueError("~vlm_max_offset_step_m must be positive")

        self.vlm_clip = None
        self.vlm_tokenizer = None
        self.vlm_safe_text_features = None
        self.vlm_unsafe_text_features = None
        self.vlm_clip_mean = None
        self.vlm_clip_std = None
        self.vlm_offset_state_m = 0.0
        self.last_vlm_latency_ms = float("nan")
        self.last_vlm_selected_offset_m = 0.0
        self.last_vlm_applied_offset_m = 0.0
        self.last_vlm_safe_probabilities = np.zeros(
            len(VLM_CANDIDATE_OFFSETS_LEFT_M), dtype=np.float32
        )
        self.last_vlm_semantic_margins = np.zeros(
            len(VLM_CANDIDATE_OFFSETS_LEFT_M), dtype=np.float32
        )
        self.last_vlm_debug_rgb = None
        self.last_vlm_debug_title = None
        self.last_vlm_candidate_images = None
        self.last_pre_vlm_trajectory = None

        yolo_sem_model = Path(
            legacy.rospy.get_param("~yolo_sem_model", DEFAULT_YOLO_SEM_MODEL)
        ).expanduser()
        if not yolo_sem_model.is_absolute():
            yolo_sem_model = _THIS_DIR / yolo_sem_model
        self.yolo_sem_model_path = str(yolo_sem_model.resolve())
        self.yolo_sem_imgsz = int(legacy.rospy.get_param("~yolo_sem_imgsz", 1024))
        if self.yolo_sem_imgsz <= 0:
            raise ValueError("~yolo_sem_imgsz must be positive")

        self._twinlite_read_params()

        if requested_segmentation_backend in {
            "yolo26_sem", twinlite_backend.TWINLITE_BACKEND,
        }:
            self._busy = True
            try:
                if requested_segmentation_backend == "yolo26_sem":
                    self._load_yolo_semantic_backend()
                else:
                    self._load_twinlite_backend()
                self.segmentation_backend = requested_segmentation_backend
                # Do not let a three-frame model input mix SegFormer and new
                # backend maps if images arrived while the backend was loading.
                self.buffer = RealtimeSequenceBufferFF(self.use_depth)
                self.last_sample_time = None
                # The base constructor necessarily loaded SegFormer. Release it
                # after the new backend is ready so new mode does not retain
                # both models.
                self.processor = None
                self.segformer = None
                if str(self.device).startswith("cuda"):
                    torch.cuda.empty_cache()
            finally:
                self._busy = False

        coarse_description = (
            "FORWARD=ego extrapolation, LEFT/RIGHT=AD-MLP"
            if self.model_variant == "hybrid"
            else "FORWARD/LEFT/RIGHT=AD-MLP"
        )
        ego_description = (
            f"VIO odometry ({self.vio_topic})"
            if self.ego_input_mode == "real_odom"
            else f"fixed straight {self.fixed_speed_mps:.2f} m/s"
        )
        legacy.rospy.loginfo(
            f"[FF planner] TAG={self.model.cfg.TAG} | variant={self.model_variant} | "
            f"coarse: {coarse_description}"
        )
        legacy.rospy.loginfo(f"[FF planner] ego input mode: {ego_description}")
        legacy.rospy.loginfo(
            f"[FF planner] lateral gain by waypoint: {LATERAL_GAIN_FIRST:.1f}x -> "
            f"{LATERAL_GAIN_LAST:.1f}x (quadratic by waypoint index)"
        )
        if self.segmentation_backend == "segformer":
            legacy.rospy.loginfo(
                "[FF planner] segmentation backend: legacy SegFormer (unchanged)"
            )
        elif self.segmentation_backend == "yolo26_sem":
            legacy.rospy.loginfo(
                f"[FF planner] segmentation backend: YOLO26 semantic | "
                f"model={self.yolo_sem_model_path} | imgsz={self.yolo_sem_imgsz}"
            )
        else:
            self._twinlite_log_startup()
        legacy.rospy.loginfo("[FF planner] semantic RGB: standard PALETTE4 (matches training)")
        legacy.rospy.loginfo(
            "[FF planner] camera calibration: park_L2_ASAP_ff ZED2i K/T, height=1.8 m"
        )
        if self.path_smoothing:
            legacy.rospy.loginfo(
                f"[FF planner] path smoothing on: B-spline sigma={self.path_smooth_sigma_m:.2f} m, "
                f"re-sampled every {self.path_point_spacing_m:.2f} m, extrapolated to "
                f">= {self.path_min_points} points (all three published topics; the "
                "inference plots keep the raw prediction)"
            )
        else:
            legacy.rospy.loginfo(
                "[FF planner] path smoothing off: publishing the raw "
                f"{legacy.N_FUTURE_FRAMES + 1} time-spaced points"
            )

        if self.vio_small_motion_fallback:
            legacy.rospy.loginfo(
                "[FF planner] tiny-VIO fallback on: when every recent step is "
                f"<= {self.vio_small_translation_m:.3f} m and yaw <= "
                f"{self.vio_small_yaw_deg:.1f} deg, synthesize/enforce "
                f"{self.vio_fallback_forward_step_m:.2f} m forward per waypoint"
            )

        if self.vlm_enabled:
            self._load_vlm_candidate_scorer()
            legacy.rospy.loginfo(
                "[FF planner] VLM correction on: CLIP ViT-L/14 scores "
                "LEFT+0.6/+0.3/CENTER/RIGHT-0.3/-0.6 m corridors | "
                f"lookahead={self.vlm_near_m:.1f}-{self.vlm_lookahead_m:.1f} m | "
                f"latency target={self.vlm_max_latency_ms:.0f} ms"
            )
        else:
            legacy.rospy.logwarn("[FF planner] VLM correction disabled")

        # Register this last so image callbacks cannot consume VIO and enter the
        # subclass pipeline before all FF-specific attributes are initialized.
        self.sub_vio = legacy.rospy.Subscriber(
            self.vio_topic, PoseWithCovarianceStamped, self.cb_vio, queue_size=1
        )
        legacy.rospy.loginfo(
            f"[FF planner] subscribe VIO history {self.vio_topic} "
            "(geometry_msgs/PoseWithCovarianceStamped)"
        )

    @property
    def _uses_fixed_ego(self) -> bool:
        return self.ego_input_mode == "fixed_speed"

    def cb_vio(self, msg) -> None:
        self.last_vio = msg

    def cb_image(self, msg) -> None:
        # This override is also the callback registered by the inherited
        # constructor. Do not advance the sampling clock until a VIO pose exists.
        now = msg.header.stamp.to_sec()
        if self.last_sample_time is not None and now < self.last_sample_time:
            self.vlm_offset_state_m = 0.0
            self.last_vlm_debug_rgb = None
            self.last_vlm_debug_title = None
            self.last_vlm_candidate_images = None
            self.last_pre_vlm_trajectory = None
            # The boardwalk vote is a temporal filter, so a clock restart makes
            # its history meaningless in exactly the same way.
            self._twinlite_reset()
        if self.last_vio is None:
            legacy.rospy.loginfo_throttle(
                2.0, f"[FF planner] waiting for VIO history {self.vio_topic}"
            )
            return
        self._twinlite_prepass(msg, now)
        super().cb_image(msg)

    def process(self, msg) -> None:
        # Capture one coherent VIO sample for this image. The legacy process()
        # continues to use /odom for path_global, array_topic, and GT plotting.
        vio_pose = legacy.pose_matrix_from_odom(self.last_vio)
        self.buffer.set_next_history_pose(vio_pose)
        # segment() receives only the resized RGB, so hand it the stamp it needs
        # to prove the cached TwinLiteNet mask belongs to this very frame, and
        # make sure this frame really was pushed -- the camera-rate throttle can
        # skip the frame the cadence gate goes on to select.
        stamp = msg.header.stamp.to_sec()
        self._current_image_stamp = stamp
        self._twinlite_prepass(msg, stamp, force=True)
        super().process(msg)

    def _load_vlm_candidate_scorer(self) -> None:
        # Keep the .safetensors symlink name: resolving it produces a hash-only
        # Hugging Face blob path, which makes open_clip mistake it for a pickle
        # checkpoint and hand it to torch.load().
        checkpoint = Path(self.vlm_clip_checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = _THIS_DIR / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"VLM CLIP checkpoint not found: {checkpoint}. "
                "Set ~vlm_clip_checkpoint or run with ~vlm_enabled:=false."
            )
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "VLM correction requires the open_clip package in the ROS Python environment"
            ) from exc

        precision = (
            "fp16"
            if str(self.device).startswith("cuda") and self.use_fp16
            else "fp32"
        )
        legacy.rospy.loginfo(
            f"[FF planner] loading VLM scorer {checkpoint} ({precision})"
        )
        clip_model = open_clip.create_model(
            "ViT-L-14",
            pretrained=str(checkpoint),
            device=self.device,
            precision=precision,
        ).eval()
        self.vlm_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        prompts = list(VLM_SAFE_PROMPTS) + list(VLM_UNSAFE_PROMPTS)
        tokens = self.vlm_tokenizer(prompts).to(self.device)
        with torch.inference_mode():
            text_features = clip_model.encode_text(tokens)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
        safe_count = len(VLM_SAFE_PROMPTS)
        self.vlm_safe_text_features = text_features[:safe_count]
        self.vlm_unsafe_text_features = text_features[safe_count:]
        self.vlm_logit_scale = float(
            clip_model.logit_scale.detach().float().exp().clamp(max=100.0).cpu()
        )

        # Text is encoded once. Retain only the vision tower so the unused text
        # transformer does not permanently consume deployment VRAM.
        self.vlm_clip = clip_model.visual.eval()
        del clip_model
        # OpenCLIP intentionally keeps some scalar/embedding parameters in
        # fp32 even under precision="fp16", while conv1 consumes fp16. Match
        # the actual image entry layer; using next(parameters()) breaks only on
        # CUDA with "Input type FloatTensor and weight type HalfTensor".
        vlm_dtype = self.vlm_clip.conv1.weight.dtype
        self.vlm_clip_mean = torch.as_tensor(
            CLIP_RGB_MEAN, device=self.device, dtype=vlm_dtype
        ).view(1, 3, 1, 1)
        self.vlm_clip_std = torch.as_tensor(
            CLIP_RGB_STD, device=self.device, dtype=vlm_dtype
        ).view(1, 3, 1, 1)

    @staticmethod
    def _smoothstep01(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    def _trajectory_with_lateral_offset(
        self, trajectory: np.ndarray, offset_left_m: float, update_yaw: bool = False
    ) -> np.ndarray:
        candidate = trajectory.copy()
        forward = np.maximum(candidate[:, 1], 0.0)
        ramp = self._smoothstep01(forward / self.vlm_offset_ramp_m)
        if not np.isfinite(ramp).all() or float(np.max(ramp)) < 1e-4:
            ramp = self._smoothstep01(
                np.linspace(1.0 / len(candidate), 1.0, len(candidate), dtype=np.float32)
            )
        candidate[:, 0] += float(offset_left_m) * ramp

        if update_yaw and len(candidate) > 0:
            points = np.vstack(
                [np.zeros((1, 2), dtype=candidate.dtype), candidate[:, :2]]
            )
            tangent = np.empty_like(points)
            tangent[0] = points[1] - points[0] if len(points) > 1 else (0.0, 1.0)
            tangent[-1] = points[-1] - points[-2] if len(points) > 1 else (0.0, 1.0)
            if len(points) > 2:
                tangent[1:-1] = points[2:] - points[:-2]
            candidate[:, 2] = np.arctan2(tangent[1:, 0], tangent[1:, 1])
        return candidate

    def _extended_candidate_centerline(
        self, trajectory: np.ndarray, offset_left_m: float
    ):
        candidate = self._trajectory_with_lateral_offset(
            trajectory, offset_left_m, update_yaw=False
        )
        forward = np.concatenate(
            [np.zeros(1, dtype=np.float32), candidate[:, 1].astype(np.float32)]
        )
        left = np.concatenate(
            [np.zeros(1, dtype=np.float32), candidate[:, 0].astype(np.float32)]
        )
        finite = np.isfinite(forward) & np.isfinite(left) & (forward >= 0.0)
        forward, left = forward[finite], left[finite]
        if len(forward) < 2:
            forward = np.asarray([0.0, self.vlm_near_m], dtype=np.float32)
            left = np.asarray([0.0, float(offset_left_m)], dtype=np.float32)

        order = np.argsort(forward)
        forward, left = forward[order], left[order]
        forward, unique_indices = np.unique(forward, return_index=True)
        left = left[unique_indices]
        if len(forward) < 2:
            forward = np.asarray([0.0, self.vlm_near_m], dtype=np.float32)
            left = np.asarray([0.0, float(offset_left_m)], dtype=np.float32)

        sample_forward = np.linspace(
            self.vlm_near_m, self.vlm_lookahead_m, 48, dtype=np.float32
        )
        clipped_forward = np.minimum(sample_forward, forward[-1])
        sample_left = np.interp(clipped_forward, forward, left).astype(np.float32)
        delta_forward = float(forward[-1] - forward[-2])
        terminal_slope = (
            float((left[-1] - left[-2]) / delta_forward)
            if delta_forward > 1e-3
            else 0.0
        )
        # A short/noisy model horizon should not fan the scoring corridor out of
        # the image. This is only a visual extension, not a published path.
        terminal_slope = float(np.clip(terminal_slope, -0.35, 0.35))
        extrapolated = sample_forward > forward[-1]
        sample_left[extrapolated] = (
            left[-1]
            + terminal_slope * (sample_forward[extrapolated] - forward[-1])
        )
        return sample_left, sample_forward

    @staticmethod
    def _project_ground_points(left: np.ndarray, forward: np.ndarray) -> np.ndarray:
        count = len(forward)
        ego_points = np.stack(
            [
                forward,
                left,
                np.zeros(count, dtype=np.float32),
                np.ones(count, dtype=np.float32),
            ],
            axis=1,
        )
        ego_to_camera = np.linalg.inv(CAMERA_TO_EGO_FF).astype(np.float32)
        camera_points = ego_points @ ego_to_camera.T
        depth = camera_points[:, 2]
        if bool(np.any(depth <= 0.05)):
            raise RuntimeError("VLM candidate corridor projects behind the camera")
        pixels_h = camera_points[:, :3] @ K_224_FF.T
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
        if not np.isfinite(pixels).all():
            raise RuntimeError("VLM candidate projection produced non-finite pixels")
        return pixels

    def _candidate_focus_image(
        self, rgb_224: np.ndarray, trajectory: np.ndarray, offset_left_m: float
    ) -> np.ndarray:
        center_left, forward = self._extended_candidate_centerline(
            trajectory, offset_left_m
        )
        half_width = self.vlm_corridor_half_width_m
        edge_a = self._project_ground_points(center_left + half_width, forward)
        edge_b = self._project_ground_points(center_left - half_width, forward)
        center = self._project_ground_points(center_left, forward)
        polygon = np.concatenate([edge_a, edge_b[::-1]], axis=0)
        polygon = np.rint(np.clip(polygon, -4096.0, 4096.0)).astype(np.int32)

        mask = np.zeros(rgb_224.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        if int(np.count_nonzero(mask)) < 120:
            raise RuntimeError("VLM candidate corridor is not visible enough in the RGB image")

        # Preserve the candidate corridor's real texture and suppress unrelated
        # scene content. A common green outline tells CLIP what region to judge
        # without letting a candidate-specific colour bias the five scores.
        focused = np.asarray(rgb_224, dtype=np.uint8).copy()
        dimmed = np.rint(focused.astype(np.float32) * 0.18).astype(np.uint8)
        dimmed[mask > 0] = focused[mask > 0]
        cv2.polylines(dimmed, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
        center_line = np.rint(np.clip(center, -4096.0, 4096.0)).astype(np.int32)
        cv2.polylines(dimmed, [center_line], False, (0, 255, 0), 1, cv2.LINE_AA)
        return dimmed

    def _score_vlm_candidates(
        self, rgb_224: np.ndarray, trajectory: np.ndarray
    ):
        started = time.perf_counter()
        focused_images = []
        valid_candidates = []
        fallback_image = np.rint(
            np.asarray(rgb_224, dtype=np.uint8).astype(np.float32) * 0.18
        ).astype(np.uint8)
        for offset in VLM_CANDIDATE_OFFSETS_LEFT_M:
            try:
                focused_images.append(
                    self._candidate_focus_image(
                        rgb_224, trajectory, float(offset)
                    )
                )
                valid_candidates.append(True)
            except RuntimeError:
                # A sharply turning candidate can leave the camera FOV. Keep
                # the batch shape fixed and make that candidate unselectable.
                focused_images.append(fallback_image.copy())
                valid_candidates.append(False)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack(focused_images, axis=0))
        ).permute(0, 3, 1, 2)
        image_tensor = image_tensor.to(
            device=self.device,
            dtype=self.vlm_clip_mean.dtype,
            non_blocking=True,
        ).div_(255.0)
        image_tensor = (image_tensor - self.vlm_clip_mean) / self.vlm_clip_std

        with torch.inference_mode():
            image_features = self.vlm_clip(image_tensor)
            if isinstance(image_features, (tuple, list)):
                image_features = image_features[0]
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            safe_similarity = (
                image_features @ self.vlm_safe_text_features.T
            ).mean(dim=1)
            unsafe_similarity = (
                image_features @ self.vlm_unsafe_text_features.T
            ).mean(dim=1)
            semantic_margin = safe_similarity - unsafe_similarity
            safe_probability = torch.softmax(
                torch.stack([safe_similarity, unsafe_similarity], dim=1)
                * self.vlm_logit_scale,
                dim=1,
            )[:, 0]
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(self.device))
        latency_ms = (time.perf_counter() - started) * 1000.0
        semantic_margin = semantic_margin.detach().float().cpu().numpy()
        safe_probability = safe_probability.detach().float().cpu().numpy()
        valid_candidates = np.asarray(valid_candidates, dtype=bool)
        semantic_margin[~valid_candidates] = -np.inf
        safe_probability[~valid_candidates] = 0.0
        return (
            semantic_margin,
            safe_probability,
            focused_images,
            valid_candidates,
            latency_ms,
        )

    def _vlm_correct_trajectory(
        self, rgb_224: np.ndarray, trajectory: np.ndarray
    ) -> np.ndarray:
        if not self.vlm_enabled:
            return trajectory
        try:
            margins, safe_probabilities, focused_images, valid, latency_ms = (
                self._score_vlm_candidates(rgb_224, trajectory)
            )
            center_index = int(np.flatnonzero(VLM_CANDIDATE_OFFSETS_LEFT_M == 0.0)[0])
            if not bool(valid[center_index]):
                raise RuntimeError("The unshifted VLM corridor is outside the camera FOV")
            combined = margins - self.vlm_offset_penalty * np.abs(
                VLM_CANDIDATE_OFFSETS_LEFT_M
            )
            best_index = int(np.argmax(combined))
            improvement = float(combined[best_index] - combined[center_index])
            confident = (
                best_index != center_index
                and improvement >= self.vlm_min_score_improvement
                and float(safe_probabilities[best_index])
                >= self.vlm_min_safe_probability
            )
            selected_offset = (
                float(VLM_CANDIDATE_OFFSETS_LEFT_M[best_index])
                if confident
                else 0.0
            )

            filtered_target = (
                self.vlm_offset_state_m
                + self.vlm_filter_alpha
                * (selected_offset - self.vlm_offset_state_m)
            )
            step = float(
                np.clip(
                    filtered_target - self.vlm_offset_state_m,
                    -self.vlm_max_offset_step_m,
                    self.vlm_max_offset_step_m,
                )
            )
            self.vlm_offset_state_m += step
            if selected_offset == 0.0 and abs(self.vlm_offset_state_m) < 0.03:
                self.vlm_offset_state_m = 0.0

            shown_index = best_index if confident else center_index
            self.last_vlm_latency_ms = latency_ms
            self.last_vlm_selected_offset_m = selected_offset
            self.last_vlm_applied_offset_m = self.vlm_offset_state_m
            self.last_vlm_safe_probabilities = safe_probabilities
            self.last_vlm_semantic_margins = margins
            self.last_vlm_debug_rgb = focused_images[shown_index]
            self.last_vlm_candidate_images = [
                np.array(image, copy=True) for image in focused_images
            ]
            direction = (
                "LEFT" if selected_offset > 0.0
                else "RIGHT" if selected_offset < 0.0
                else "CENTER"
            )
            self.last_vlm_debug_title = (
                f"VLM {direction} selected={selected_offset:+.1f}m "
                f"applied={self.vlm_offset_state_m:+.2f}m"
            )
            if latency_ms > self.vlm_max_latency_ms:
                legacy.rospy.logwarn_throttle(
                    2.0,
                    f"[FF planner] VLM latency {latency_ms:.1f} ms exceeds "
                    f"{self.vlm_max_latency_ms:.0f} ms target",
                )
            legacy.rospy.loginfo_throttle(
                1.0,
                f"[FF planner] VLM={latency_ms:.1f} ms | choose={selected_offset:+.1f} m "
                f"| applied={self.vlm_offset_state_m:+.2f} m | "
                f"improvement={improvement:.4f} | "
                f"safe={np.round(safe_probabilities, 2).tolist()}",
            )
            return self._trajectory_with_lateral_offset(
                trajectory, self.vlm_offset_state_m, update_yaw=True
            )
        except Exception:
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            # Do not make a one-frame scoring failure jerk the path back to the
            # uncorrected centre. Decay the last stable offset toward zero.
            self.vlm_offset_state_m *= 0.8
            if abs(self.vlm_offset_state_m) < 0.03:
                self.vlm_offset_state_m = 0.0
            self.last_vlm_selected_offset_m = 0.0
            self.last_vlm_applied_offset_m = self.vlm_offset_state_m
            self.last_vlm_candidate_images = None
            self.last_vlm_debug_rgb = None
            self.last_vlm_debug_title = None
            legacy.rospy.logerr_throttle(
                2.0,
                "[FF planner] VLM correction failed; decaying the last correction:\n"
                + __import__("traceback").format_exc(),
            )
            return self._trajectory_with_lateral_offset(
                trajectory, self.vlm_offset_state_m, update_yaw=True
            )

    def _load_yolo_semantic_backend(self) -> None:
        model_path = Path(self.yolo_sem_model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YOLO26 semantic checkpoint not found: {model_path}. "
                "Place yolo26l-sem.pt there or set ~yolo_sem_model:=/path/to/model.pt"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "~segmentation_backend=yolo26_sem requires the ultralytics package"
            ) from exc

        model = YOLO(str(model_path), task="semantic", verbose=False)
        names = model.names
        if isinstance(names, dict):
            names = tuple(str(names[index]).strip().lower() for index in range(len(names)))
        else:
            names = tuple(str(name).strip().lower() for name in names)
        if names != YOLO_SEM_CITYSCAPES_NAMES:
            raise RuntimeError(
                "YOLO semantic checkpoint must use the standard 19-class "
                f"Cityscapes labels; got {names}"
            )
        self.yolo_sem = model



    @torch.inference_mode()
    def segment(self, rgb: np.ndarray) -> np.ndarray:
        """Return any backend as identical (224,224) PALETTE4 class IDs."""
        if self.segmentation_backend == "segformer":
            # This inherited call is intentionally unchanged: old preprocessing,
            # autocast, logit grouping, resize, and crop all remain byte-for-byte.
            return super().segment(rgb)

        if self.segmentation_backend == "twinlitenet":
            # The network already ran in cb_image at camera rate; this only
            # applies the causal vote and the crop.
            return self._twinlite_segment(rgb)

        # Ultralytics assumes a NumPy input is OpenCV BGR, then converts it to
        # RGB internally. The planner already has RGB, so swap once here.
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        results = self.yolo_sem.predict(
            source=bgr,
            imgsz=self.yolo_sem_imgsz,
            device=self.device,
            quantize=(
                16
                if str(self.device).startswith("cuda") and self.use_fp16
                else None
            ),
            verbose=False,
            save=False,
        )
        if len(results) != 1 or results[0].semantic_mask is None:
            raise RuntimeError("YOLO26 semantic inference returned no dense class map")

        cityscapes = results[0].semantic_mask.data.long()
        if cityscapes.ndim != 2:
            raise RuntimeError(
                f"YOLO26 semantic class map must be (H,W), got {tuple(cityscapes.shape)}"
            )
        if (
            cityscapes.numel() == 0
            or bool(cityscapes.min() < 0)
            or bool(cityscapes.max() >= len(YOLO_SEM_CITYSCAPES_TO_CLS4))
        ):
            raise RuntimeError("YOLO26 semantic class map contains a non-Cityscapes class ID")

        cls4_lut = torch.tensor(
            YOLO_SEM_CITYSCAPES_TO_CLS4,
            dtype=torch.uint8,
            device=cityscapes.device,
        )
        cls4 = cls4_lut[cityscapes].unsqueeze(0).unsqueeze(0)
        cls4_224 = legacy.resize_keep_ratio_center_crop_uint8(cls4)
        return cls4_224[0, 0].cpu().numpy()

    def build_batch(self) -> dict:
        rgb_seq = np.stack(list(self.buffer.rgb), axis=0)
        seg_id_seq = np.stack(list(self.buffer.seg_id), axis=0)
        # One canonical palette for the published image and the model input.
        seg_rgb_seq = np.stack(
            [legacy.colorize_cls4_rgb(seg_id) for seg_id in seg_id_seq], axis=0
        )
        command = self.command
        synthetic_speed = self.fixed_speed_mps if self._uses_fixed_ego else 0.0
        if self._uses_fixed_ego:
            future_egomotion = self.buffer.build_fixed_future_egomotion(synthetic_speed)
            ego_history = self.buffer.build_ego_history_egomotion(synthetic_speed)
            self.vio_fallback_active = False
        else:
            future_egomotion = self.buffer.build_future_egomotion()
            ego_history = self.buffer.build_ego_history_egomotion()
            history_steps = ego_history[:-1]
            planar_steps = np.linalg.norm(history_steps[:, :2], axis=1)
            yaw_steps_deg = np.rad2deg(np.abs(history_steps[:, 5]))
            self.vio_fallback_active = bool(
                self.vio_small_motion_fallback
                and np.all(planar_steps <= self.vio_small_translation_m)
                and np.all(yaw_steps_deg <= self.vio_small_yaw_deg)
            )
            if self.vio_fallback_active:
                synthetic_speed = (
                    self.vio_fallback_forward_step_m
                    / max(float(self.sample_interval), 1e-6)
                )
                future_egomotion = self.buffer.build_fixed_future_egomotion(
                    synthetic_speed
                )
                ego_history = self.buffer.build_ego_history_egomotion(
                    synthetic_speed
                )
                legacy.rospy.logwarn_throttle(
                    2.0,
                    "[FF planner] VIO history is nearly zero; using synthetic "
                    f"{synthetic_speed:.2f} m/s forward history",
                )
        admlp_input = self.buffer.build_admlp_input(command, synthetic_speed)

        future_count = int(getattr(self.model, "n_future", legacy.N_FUTURE_FRAMES))
        batch = {
            "rgb_224_seq": torch.from_numpy(rgb_seq).unsqueeze(0),
            "seg_224_seq": torch.from_numpy(seg_rgb_seq).unsqueeze(0),
            "seg_id_224_seq": torch.from_numpy(seg_id_seq.astype(np.int64)).unsqueeze(0),
            "depth_224_seq": torch.zeros(
                1, legacy.TIME_RECEPTIVE_FIELD, 224, 224, dtype=torch.float32
            ),
            "future_egomotion": torch.from_numpy(future_egomotion).unsqueeze(0),
            "ego_history_egomotion": torch.from_numpy(ego_history).unsqueeze(0),
            "admlp_input": torch.from_numpy(admlp_input).unsqueeze(0),
            "command": [command],
            # No external navigation route in v1: FF route attention falls back to coarse.
            "target_point": torch.empty(1, 0, 2, dtype=torch.float32),
            "gt_trajectory": torch.zeros(1, future_count + 1, 3, dtype=torch.float32),
            "segmentation": torch.zeros(
                1, legacy.TIME_RECEPTIVE_FIELD, 1, 1, 1, dtype=torch.long
            ),
            "pedestrian": torch.zeros(
                1, legacy.TIME_RECEPTIVE_FIELD, 1, 1, 1, dtype=torch.long
            ),
            "image": torch.empty(1, 0),
            "intrinsics": self.intrinsics_ff.clone(),
            "extrinsics": self.extrinsics_ff.clone(),
            "sample_trajectory": torch.empty(1, 0),
        }
        if self.buffer.depth is not None:
            depth_seq = np.stack(list(self.buffer.depth), axis=0)
            batch["depth_224_seq"] = torch.from_numpy(depth_seq).float().unsqueeze(0)
        return batch

    def _enforce_tiny_vio_forward_progress(
        self, trajectory: np.ndarray
    ) -> np.ndarray:
        """Guarantee longitudinal ordering when the tiny-VIO fallback is active."""
        if not self.vio_fallback_active or len(trajectory) == 0:
            return trajectory
        corrected = trajectory.copy()
        step = self.vio_fallback_forward_step_m
        corrected[0, 1] = max(float(corrected[0, 1]), step)
        for index in range(1, len(corrected)):
            corrected[index, 1] = max(
                float(corrected[index, 1]),
                float(corrected[index - 1, 1]) + step,
            )
        legacy.rospy.loginfo_throttle(
            2.0,
            "[FF planner] tiny-VIO forward guard applied: waypoint forward "
            f"spacing >= {step:.2f} m",
        )
        return corrected

    def plan(self) -> np.ndarray:
        batch = self.build_batch()
        labels = runtime_ff._prepare_l2_labels(batch)
        use_cuda_amp = str(self.device).startswith("cuda") and self.use_fp16
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_cuda_amp
            else nullcontext()
        )
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(self.device))
        started = time.perf_counter()
        with torch.inference_mode(), amp_context:
            output, is_vlm_gen = runtime_ff._call_model_forward(
                self.model, batch, self.device
            )
            _, final_traj = runtime_ff._call_model_planning(
                self.model,
                output,
                labels,
                batch,
                self.n_present,
                self.device,
                is_vlm_gen,
            )
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(self.device))
        self.last_plan_inference_ms = (time.perf_counter() - started) * 1000.0

        trajectory = final_traj[0].detach().float().cpu().numpy().copy()
        waypoint_progress = np.linspace(
            0.0,
            1.0,
            num=trajectory.shape[0],
            dtype=trajectory.dtype,
        )
        lateral_gains = (
            LATERAL_GAIN_FIRST
            + (LATERAL_GAIN_LAST - LATERAL_GAIN_FIRST) * waypoint_progress**2
        )
        # The minus sign converts planner +right to public/ROS +left.  Increase
        # the gain quadratically by waypoint index so near points change gently
        # while far points retain the stronger lateral response.
        trajectory[:, 0] *= -lateral_gains
        trajectory = self._enforce_tiny_vio_forward_progress(trajectory)
        self.last_pre_vlm_trajectory = trajectory.copy()
        trajectory = self._vlm_correct_trajectory(self.buffer.rgb[-1], trajectory)
        bev_valid = bool(getattr(self.model.vlm.metric_bev, "last_valid", False))
        alpha = getattr(self.model.vlm, "last_alpha", None)
        alpha_mean = float(alpha.detach().float().mean().cpu()) if alpha is not None else float("nan")
        runtime_losses = getattr(self.model, "last_runtime_loss_dict_ff", {})
        safety_value = runtime_losses.get("ff_safety_scale_mean")
        safety_scale = (
            float(safety_value.detach().float().mean().cpu())
            if torch.is_tensor(safety_value)
            else float("nan")
        )
        legacy.rospy.loginfo_throttle(
            1.0,
            f"[FF planner] model inference={self.last_plan_inference_ms:.1f} ms | "
            f"variant={self.model_variant} | ego={self.ego_input_mode} | "
            f"BEV-valid={int(bev_valid)} | alpha={alpha_mean:.3f} | "
            f"safety-scale={safety_scale:.2f}",
        )
        return trajectory

    # ---------- published path: smoothed and distance-spaced ----------
    #
    # The legacy node publishes plan()'s six points with the t0 origin prepended
    # by each builder. Those points are spaced by time, and carry the model's
    # lateral noise unfiltered — visible as the kinked, weaving path in the
    # inference plots. plan() progressively amplifies later points with a
    # quadratic-by-waypoint gain. The overrides below replace that shared
    # (7, 3) array with a
    # B-spline fit re-sampled every ~path_point_spacing_m metres, so all three
    # topics publish one consistent smooth path.
    #
    # The inference plots are deliberately left on the raw prediction: plot_mode
    # with_gt compares it against six driven waypoints one sample_interval
    # apart, and a distance-spaced path cannot be aligned to those.

    def _published_path(self, traj: np.ndarray):
        """
        The path actually published for `traj`, as (points, times).

        `points` is (N, 3) of (x_left, y_front, yaw) *including* the t0 start
        point, so the builders below must not prepend an origin of their own.
        `times` is (N,) seconds from now; after re-sampling it no longer equals
        index * sample_interval. Memoised on the trajectory object because
        process() calls all three builders on the same one.

        A smoothing failure is not fatal: publishing the raw prediction for a
        cycle beats killing the planner on a spline edge case.
        """
        if self._published_cache is not None and self._published_cache[0] is traj:
            return self._published_cache[1], self._published_cache[2]

        points = np.vstack([np.zeros((1, 3), dtype=np.float64), traj])
        times = np.arange(len(points), dtype=np.float64) * self.sample_interval
        if self.path_smoothing:
            try:
                points, times = smooth_and_resample(
                    points[:, :2],
                    point_times=times,
                    spacing_m=self.path_point_spacing_m,
                    sigma_m=self.path_smooth_sigma_m,
                    min_points=self.path_min_points,
                )
            except TrajectorySmoothingError as exc:
                legacy.rospy.logwarn_throttle(
                    5.0, f"[FF planner] publishing the raw trajectory: {exc}"
                )
        self._published_cache = (traj, points, times)
        return points, times

    def build_path(self, traj: np.ndarray, stamp) -> legacy.Path:
        """Smoothed base_link path. Frame conventions follow the legacy builder."""
        points, times = self._published_path(traj)

        path = legacy.Path()
        path.header.stamp = stamp
        path.header.frame_id = self.frame_id
        for (x_left, y_front, yaw), t in zip(points, times):
            pose = legacy.PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.header.stamp = stamp + legacy.rospy.Duration(float(t))
            pose.pose.position.x = float(y_front)
            pose.pose.position.y = float(x_left)
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = float(np.sin(yaw / 2.0))
            pose.pose.orientation.w = float(np.cos(yaw / 2.0))
            path.poses.append(pose)
        return path

    def _global_points(self, traj: np.ndarray, odom_msg) -> np.ndarray:
        """
        The smoothed path in the global odom frame, (N, 3) of (gx, gy, gyaw).
        Same transform as the legacy builder, over the re-sampled points.
        Shared by build_path_global and the inherited build_array_topic.
        """
        points, _ = self._published_path(traj)

        p = odom_msg.pose.pose.position
        o = odom_msg.pose.pose.orientation
        rx, ry = float(p.x), float(p.y)
        ryaw = legacy.quaternion_yaw(legacy.Quaternion(o.w, o.x, o.y, o.z))
        cos_r, sin_r = np.cos(ryaw), np.sin(ryaw)

        out = np.empty((points.shape[0], 3), dtype=np.float64)
        for i, (x_left, y_front, yaw) in enumerate(points):
            x_forward, y_left = y_front, x_left  # model (x_left, y_front) -> base_link
            out[i, 0] = rx + cos_r * x_forward - sin_r * y_left
            out[i, 1] = ry + sin_r * x_forward + cos_r * y_left
            out[i, 2] = ryaw + yaw
        return out

    def build_path_global(self, traj: np.ndarray, stamp, odom_msg) -> legacy.Path:
        """Same points as build_array_topic, as a Path in the /odom frame."""
        _, times = self._published_path(traj)
        gpts = self._global_points(traj, odom_msg)

        path = legacy.Path()
        path.header.stamp = stamp
        path.header.frame_id = odom_msg.header.frame_id or "odom"
        for (gx, gy, gyaw), t in zip(gpts, times):
            pose = legacy.PoseStamped()
            pose.header.frame_id = path.header.frame_id
            pose.header.stamp = stamp + legacy.rospy.Duration(float(t))
            pose.pose.position.x = float(gx)
            pose.pose.position.y = float(gy)
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = float(np.sin(gyaw / 2.0))
            pose.pose.orientation.w = float(np.cos(gyaw / 2.0))
            path.poses.append(pose)
        return path

    def _handle_plot(self, final_traj: np.ndarray, stamp, pose: np.ndarray) -> None:
        """Legacy plot queue plus the inputs and trajectories from this VLM cycle."""
        if self.plot_mode == "off":
            return

        corrected = np.asarray(final_traj, dtype=np.float64).copy()
        if corrected.shape[1] == 2:
            corrected = np.concatenate(
                [corrected, np.zeros((corrected.shape[0], 1))], axis=1
            )
        raw = (
            np.asarray(self.last_pre_vlm_trajectory, dtype=np.float64).copy()
            if self.last_pre_vlm_trajectory is not None
            else corrected.copy()
        )
        fego = torch.from_numpy(self.buffer.build_future_egomotion()).float()
        input_xy, input_yaw = legacy._input_history_from_egomotion(fego)
        rec = {
            "seq_idx": self._plot_seq,
            "t_ref": stamp.to_nsec(),
            "rgb_224": np.array(self.buffer.rgb[-1], copy=True),
            "seg_id_224": np.array(self.buffer.seg_id[-1], copy=True),
            "depth_224": (
                np.array(self.buffer.depth[-1], copy=True)
                if self.buffer.depth is not None
                else None
            ),
            # Red in the plot: model output before CLIP correction.
            "pred": raw,
            # Blue in the plot: CLIP-corrected, before spline smoothing.
            "vlm_pred": corrected,
            "vlm_candidate_images": (
                [np.array(image, copy=True) for image in self.last_vlm_candidate_images]
                if self.last_vlm_candidate_images is not None
                else None
            ),
            "vlm_safe_probabilities": np.array(
                self.last_vlm_safe_probabilities, copy=True
            ),
            "vlm_selected_offset_m": float(self.last_vlm_selected_offset_m),
            "vlm_applied_offset_m": float(self.last_vlm_applied_offset_m),
            "input_xy": input_xy,
            "input_yaw": input_yaw,
            "include_gt": self.plot_mode == "with_gt",
            "pose_inv": np.linalg.inv(pose),
            "future": [],
        }
        self._plot_seq += 1
        if self.plot_mode == "realtime":
            self._write_plot(rec)
        else:
            self._pending.append(rec)

    @staticmethod
    def _vlm_candidate_panel(rec: dict) -> list:
        """Return a 3x2 contact sheet of the five exact RGB inputs sent to CLIP."""
        images = rec.get("vlm_candidate_images")
        if not images or len(images) != len(VLM_CANDIDATE_OFFSETS_LEFT_M):
            return []
        probabilities = np.asarray(
            rec.get("vlm_safe_probabilities", np.zeros(len(images))),
            dtype=np.float32,
        )
        selected = float(rec.get("vlm_selected_offset_m", 0.0))
        tile_size, columns, rows = 112, 3, 2
        panel = np.zeros((rows * tile_size, columns * tile_size, 3), dtype=np.uint8)
        for index, (image, offset) in enumerate(
            zip(images, VLM_CANDIDATE_OFFSETS_LEFT_M)
        ):
            row, column = divmod(index, columns)
            tile = cv2.resize(
                np.asarray(image, dtype=np.uint8),
                (tile_size, tile_size),
                interpolation=cv2.INTER_AREA,
            )
            label = (
                "CENTER" if abs(float(offset)) < 1e-6
                else f"{'L' if offset > 0 else 'R'}{abs(float(offset)):.1f}"
            )
            probability = float(probabilities[index]) if index < len(probabilities) else 0.0
            cv2.rectangle(tile, (0, 0), (tile_size - 1, 17), (0, 0, 0), -1)
            cv2.putText(
                tile,
                f"{label} p={probability:.2f}",
                (3, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if abs(float(offset) - selected) < 1e-6:
                cv2.rectangle(
                    tile, (1, 1), (tile_size - 2, tile_size - 2), (30, 110, 255), 3
                )
            y0, x0 = row * tile_size, column * tile_size
            panel[y0:y0 + tile_size, x0:x0 + tile_size] = tile
        cv2.putText(
            panel,
            f"applied {float(rec.get('vlm_applied_offset_m', 0.0)):+.2f}m",
            (2 * tile_size + 5, tile_size + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return [("VLM INPUTS: projected candidate corridors", panel)]

    def _seg_panels(self, rec: dict) -> list:
        panels = self._vlm_candidate_panel(rec) if self.vlm_enabled else []
        seg_id = rec.get("seg_id_224")
        if not self.plot_seg or seg_id is None:
            return panels
        panels.append(
            ("SEG model-input PALETTE4", legacy.colorize_cls4_rgb(seg_id))
        )
        return panels

    @staticmethod
    def _overlay_vlm_trajectory(
        output_path: Path,
        raw_pred: np.ndarray,
        corrected_pred: np.ndarray,
        gt: np.ndarray,
        history: np.ndarray,
        published: np.ndarray,
        lateral_limit_m,
    ) -> None:
        """Overlay the pre-spline VLM result in blue on runtime_ff's plot panel."""
        canvas = Image.open(str(output_path)).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        size, margin = 512, 48
        panel_left = canvas.width - size
        origin = np.zeros((1, 2), dtype=np.float32)
        raw_xy = np.asarray(raw_pred, dtype=np.float32)[:, :2]
        corrected_xy = np.asarray(corrected_pred, dtype=np.float32)[:, :2]
        gt_xy = np.asarray(gt, dtype=np.float32)[:, :2]
        gt_future = gt_xy[1:] if len(gt_xy) > 1 else gt_xy
        history_xy = np.asarray(history, dtype=np.float32)
        published_xy = np.asarray(published, dtype=np.float32)[:, :2]
        parts = [origin, raw_xy]
        if gt_future.size:
            parts.append(gt_future)
        if history_xy.size:
            parts.append(history_xy)
        if published_xy.size:
            parts.append(published_xy)
        if corrected_xy.size:
            parts.append(corrected_xy)
        all_xy = np.concatenate(parts, axis=0)
        if lateral_limit_m is None:
            x_extent = max(float(np.abs(all_xy[:, 0]).max()) + 1.0, 5.0)
        else:
            x_extent = max(float(lateral_limit_m), 1e-3)
        y_min = min(float(all_xy[:, 1].min()), 0.0) - 1.0
        y_max = max(float(all_xy[:, 1].max()), 0.0) + 1.0
        if y_max - y_min < 5.0:
            padding = (5.0 - (y_max - y_min)) / 2.0
            y_min, y_max = y_min - padding, y_max + padding
        scale_x = (size - 2 * margin) / (2.0 * x_extent)
        scale_y = (size - 2 * margin) / max(y_max - y_min, 1e-6)
        if lateral_limit_m is None:
            scale_x = scale_y = min(scale_x, scale_y)
        center_x = panel_left + size // 2
        origin_y = size - margin + y_min * scale_y

        def pixel(point):
            return (
                center_x - float(point[0]) * scale_x,
                origin_y - float(point[1]) * scale_y,
            )

        points = np.concatenate([origin, corrected_xy], axis=0)
        pixels = [pixel(point) for point in points]
        if len(pixels) > 1:
            draw.line(pixels, fill=(35, 105, 245), width=4)
        for px, py in pixels:
            draw.ellipse(
                (px - 3, py - 3, px + 3, py + 3), fill=(15, 65, 185)
            )
        draw.rectangle(
            (panel_left + margin - 2, 64, panel_left + margin + 190, 80),
            fill=(250, 250, 250),
        )
        draw.text(
            (panel_left + margin, 65),
            "VLM corrected blue",
            fill=(35, 85, 220),
        )
        canvas.save(str(output_path))

    def _write_plot(self, rec: dict) -> None:
        """
        Plot raw (red), VLM-corrected (blue), and published spline (orange).

        The L2 numbers and the with_gt comparison stay on the raw prediction:
        its six points are one sample_interval apart and so line up with the
        six driven waypoints, which the distance-spaced path does not.
        """
        future = rec.get("future", []) if rec.get("include_gt", False) else []
        if future:
            gt = np.vstack([np.zeros((1, 3)), np.asarray(future, dtype=np.float64)])
            xy_error = legacy._trajectory_xy_error(
                torch.from_numpy(rec["pred"]).float().unsqueeze(0),
                torch.from_numpy(gt).float().unsqueeze(0),
            )
            l2 = torch.linalg.norm(xy_error, dim=-1)[0].numpy()
        else:
            gt = np.zeros((0, 3), dtype=np.float32)
            l2 = np.zeros((0,), dtype=np.float32)

        corrected = rec.get("vlm_pred", rec["pred"])
        published, _ = self._published_path(corrected)
        out_dir = self._ensure_plot_dir()
        lateral_limit = 1.0 if not rec.get("include_gt", False) else None
        legacy.save_inference_plot(
            rgb_224=rec["rgb_224"],
            pred=rec["pred"],
            gt=gt,
            l2=l2,
            t_ref=rec["t_ref"],
            seq_idx=rec["seq_idx"],
            out_dir=out_dir,
            input_xy=rec["input_xy"],
            input_yaw=rec["input_yaw"],
            extra_panels=self._seg_panels(rec) + self._depth_panel(rec),
            lateral_limit_m=lateral_limit,
            smoothed_xy=published[:, :2],
        )
        self._overlay_vlm_trajectory(
            output_path=out_dir / f"{rec['seq_idx']:06d}_{int(rec['t_ref'])}.png",
            raw_pred=rec["pred"],
            corrected_pred=corrected,
            gt=gt,
            history=rec["input_xy"],
            published=published,
            lateral_limit_m=lateral_limit,
        )


def main():
    legacy.rospy.init_node("realtime_planner_node_ff", anonymous=False)
    node = RealtimePlannerNodeFF()
    legacy.rospy.on_shutdown(lambda: node._flush_pending(force=True))
    legacy.rospy.spin()


if __name__ == "__main__":
    main()
