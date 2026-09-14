#!/usr/bin/env python3
"""ROS realtime planner for Planning_ASAP_ff and Planning_ASAP_hybrid_ff.

The copied legacy node remains untouched. This entry point injects a new-model
runtime, fixes the semantic palette and calibration, and adds coherent VIO
versus fixed-speed ego input modes.

Every pose in this node comes from VIO (nav_msgs/Odometry on ~vio_topic,
published by mpcbitch/scripts/vio_pose_to_odom.py): the model's ego history,
the anchor of path_global / array_topic, and the GT plots. /odom is ignored, so
the planner and an MPC running with localization_source:=vio share one frame.
"""

from contextlib import nullcontext
from pathlib import Path
import os
import sys
import time

import numpy as np
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


def prepend_back_point(points: np.ndarray, times: np.ndarray, back_m: float):
    """
    Prepend one point `back_m` behind the path start, along the start direction.

    mpc_back_test tracks a point 0.5 m behind the pose (plus a short forward
    state projection). At low speed that point sits behind the path start, the
    nearest-point search clamps onto the start, and the longitudinal gap is
    reported as ~0.5 m of cross-track error. A collinear back point keeps the
    tracked point on a real segment without bending the path.

    `points` is (N, 3) of (x_left, y_front, yaw) starting at the vehicle, and
    `times` (N,) seconds. The direction is the first segment; if it has no
    length the vehicle heading (+y_front) is used. The back point's time is
    extrapolated at the first segment's rate.
    """
    if back_m <= 0.0 or len(points) == 0:
        return points, times
    direction = np.array([0.0, 1.0])
    time_back = times[0]
    if len(points) >= 2:
        segment = points[1, :2] - points[0, :2]
        length = float(np.linalg.norm(segment))
        if length > 1e-6:
            direction = segment / length
            time_back = times[0] - (times[1] - times[0]) * back_m / length
    back = np.empty((1, points.shape[1]), dtype=points.dtype)
    back[0, :2] = points[0, :2] - back_m * direction
    back[0, 2:] = points[0, 2:]
    return np.vstack([back, points]), np.concatenate([[time_back], times])

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
        # The inherited pipeline passes its last_odom pose here. That is already
        # the VIO snapshot for this image; the explicit history pose keeps the
        # buffer from ever storing a pose that was not selected for this frame.
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


class RealtimePlannerNodeFF(legacy.RealtimePlannerNode):
    def __init__(self):
        # Source every pose (model history, published path anchor, GT plots)
        # from VIO. The legacy constructor still subscribes /odom, but cb_odom
        # below discards it.
        self.vio_topic = legacy.rospy.get_param("~vio_topic", "/vio_odom")
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
        if requested_segmentation_backend not in {"segformer", "yolo26_sem"}:
            raise ValueError(
                "~segmentation_backend must be segformer (or legacy) or yolo26_sem"
            )

        # The inherited constructor loads the legacy backend and starts ROS
        # subscribers. Keep callbacks on it until YOLO is completely ready.
        self.segmentation_backend = "segformer"
        self.yolo_sem = None
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
        # Length of the collinear point prepended behind the path start so the
        # MPC's tracked point (0.5 m behind the pose) stays on the path at low
        # speed. Defaults to one point spacing; <= 0 disables it.
        self.path_back_extension_m = float(
            legacy.rospy.get_param(
                "~path_back_extension_m", self.path_point_spacing_m
            )
        )
        # (source trajectory, points, times) for the current cycle. The three
        # builders are called once each per cycle on the same trajectory, and
        # the smoothing must not run three times.
        self._published_cache = None

        # OpenVINS can briefly publish an all-zero or numerically tiny history
        # during initialization. A hybrid FORWARD coarse path extrapolated from
        # that history has no longitudinal direction, so visual residual noise
        # can dominate and make the prediction mostly lateral. In that specific
        # case, feed a small synthetic forward history and enforce the same
        # minimum progress on the output waypoints.
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

        yolo_sem_model = Path(
            legacy.rospy.get_param("~yolo_sem_model", DEFAULT_YOLO_SEM_MODEL)
        ).expanduser()
        if not yolo_sem_model.is_absolute():
            yolo_sem_model = _THIS_DIR / yolo_sem_model
        self.yolo_sem_model_path = str(yolo_sem_model.resolve())
        self.yolo_sem_imgsz = int(legacy.rospy.get_param("~yolo_sem_imgsz", 1024))
        if self.yolo_sem_imgsz <= 0:
            raise ValueError("~yolo_sem_imgsz must be positive")

        if requested_segmentation_backend == "yolo26_sem":
            self._busy = True
            try:
                self._load_yolo_semantic_backend()
                self.segmentation_backend = "yolo26_sem"
                # Do not let a three-frame model input mix SegFormer and YOLO
                # maps if images arrived while the new backend was loading.
                self.buffer = RealtimeSequenceBufferFF(self.use_depth)
                self.last_sample_time = None
                # The base constructor necessarily loaded SegFormer. Release it
                # after YOLO is ready so new mode does not retain both models.
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
        else:
            legacy.rospy.loginfo(
                f"[FF planner] segmentation backend: YOLO26 semantic | "
                f"model={self.yolo_sem_model_path} | imgsz={self.yolo_sem_imgsz}"
            )
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
        if self.path_back_extension_m > 0.0:
            legacy.rospy.loginfo(
                f"[FF planner] path back extension on: one point "
                f"{self.path_back_extension_m:.2f} m behind the start, along the "
                "start direction (all three published topics)"
            )

        if self.vio_small_motion_fallback:
            legacy.rospy.loginfo(
                "[FF planner] tiny-VIO fallback on: when every recent step is "
                f"<= {self.vio_small_translation_m:.3f} m and yaw <= "
                f"{self.vio_small_yaw_deg:.1f} deg, synthesize/enforce "
                f"{self.vio_fallback_forward_step_m:.2f} m forward per waypoint"
            )

        # Register this last so image callbacks cannot consume VIO and enter the
        # subclass pipeline before all FF-specific attributes are initialized.
        self.sub_vio = legacy.rospy.Subscriber(
            self.vio_topic, legacy.Odometry, self.cb_vio, queue_size=1
        )
        legacy.rospy.loginfo(
            f"[FF planner] subscribe VIO pose {self.vio_topic} (nav_msgs/Odometry) "
            "for ego history, path_global/array_topic anchor and plots; "
            f"{self.odom_topic} is ignored"
        )

    @property
    def _uses_fixed_ego(self) -> bool:
        return self.ego_input_mode == "fixed_speed"

    def cb_odom(self, msg) -> None:
        # The legacy constructor subscribes /odom with this callback. Ignore it:
        # last_odom is the per-image VIO snapshot taken in cb_image, and a LiDAR
        # pose overwriting it mid-inference would anchor the path in another frame.
        pass

    def cb_vio(self, msg) -> None:
        self.last_vio = msg

    def cb_image(self, msg) -> None:
        # This override is also the callback registered by the inherited
        # constructor. Do not advance the sampling clock until a VIO pose exists.
        if self.last_vio is None:
            legacy.rospy.loginfo_throttle(
                2.0, f"[FF planner] waiting for VIO pose {self.vio_topic}"
            )
            return
        # Snapshot the VIO pose at image arrival. The legacy process() reads
        # last_odom both before inference (history, GT) and after it
        # (path_global, array_topic); the prediction is relative to the vehicle
        # at image time, so all of those must use this one pose, not whatever
        # VIO reports once inference finishes. Never replace it while a previous
        # image is still being processed.
        if not self._busy:
            self.last_odom = self.last_vio
        super().cb_image(msg)

    def process(self, msg) -> None:
        # last_odom is the VIO snapshot selected in cb_image for this image.
        vio_pose = legacy.pose_matrix_from_odom(self.last_odom)
        self.buffer.set_next_history_pose(vio_pose)
        super().process(msg)

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
        """Return either backend as identical (224,224) PALETTE4 class IDs."""
        if self.segmentation_backend == "segformer":
            # This inherited call is intentionally unchanged: old preprocessing,
            # autocast, logit grouping, resize, and crop all remain byte-for-byte.
            return super().segment(rgb)

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
        With ~path_back_extension_m > 0 the first point is instead one collinear
        point behind the start (negative time), followed by the t0 start point.
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
        points, times = prepend_back_point(points, times, self.path_back_extension_m)
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
        The smoothed path in the VIO global frame, (N, 3) of (gx, gy, gyaw).
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
        """Same points as build_array_topic, as a Path in the VIO global frame."""
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

    def _seg_panels(self, rec: dict) -> list:
        seg_id = rec.get("seg_id_224")
        if not self.plot_seg or seg_id is None:
            return []
        return [("SEG model-input PALETTE4", legacy.colorize_cls4_rgb(seg_id))]

    def _write_plot(self, rec: dict) -> None:
        """
        The legacy plot with the published path drawn over the raw prediction.

        The L2 numbers and the with_gt comparison stay on the raw prediction:
        its six points are one sample_interval apart and so line up with the
        six driven waypoints, which the distance-spaced path does not.

        `rec['pred']` is a copy of the trajectory plan() returned, so re-running
        the smoothing on it reproduces the published path exactly. It costs one
        extra spline fit per saved plot rather than carrying the path through
        the with_gt queue.
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

        published, _ = self._published_path(rec["pred"])
        legacy.save_inference_plot(
            rgb_224=rec["rgb_224"],
            pred=rec["pred"],
            gt=gt,
            l2=l2,
            t_ref=rec["t_ref"],
            seq_idx=rec["seq_idx"],
            out_dir=self._ensure_plot_dir(),
            input_xy=rec["input_xy"],
            input_yaw=rec["input_yaw"],
            extra_panels=self._seg_panels(rec) + self._depth_panel(rec),
            lateral_limit_m=(1.0 if not rec.get("include_gt", False) else None),
            smoothed_xy=published[:, :2],
        )


def main():
    legacy.rospy.init_node("realtime_planner_node_ff", anonymous=False)
    node = RealtimePlannerNodeFF()
    legacy.rospy.on_shutdown(lambda: node._flush_pending(force=True))
    legacy.rospy.spin()


if __name__ == "__main__":
    main()
