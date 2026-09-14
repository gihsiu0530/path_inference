"""Shared TwinLiteNet segmentation backend for the FF planner nodes.

realtime_planner_node_ff.py and realtime_planner_node_ff_VIO_VLM.py are sibling
subclasses of the legacy node rather than one deriving from the other, so the
backend lives here instead of being copied into both.

The mixin deliberately defines no method the node classes already own —
``segment`` and ``cb_image`` stay with each node, which keeps their existing
backend dispatch untouched and makes the MRO uninteresting.
"""

import cv2
import numpy as np
import torch

import realtime_planner_node as legacy
import twinlite_da


# Every spelling that selects this backend.
TWINLITE_BACKEND_ALIASES = {
    "twinlitenet", "twinlite", "twinlite_da", "twinlite_net", "tln",
}
TWINLITE_BACKEND = "twinlitenet"


class TwinLiteBackendMixin:
    """Parameters, loading, camera-rate pre-pass, and the seg-id conversion."""

    def _twinlite_init_state(self) -> None:
        """Call before the inherited constructor starts ROS subscribers."""
        self.twinlite = None
        self._twinlite_stamp = None
        self._twinlite_last_prepass = None
        self._current_image_stamp = None

    def _twinlite_read_params(self) -> None:
        # The offline defaults live in TwinLiteNet/test_video_filtered.py and are
        # re-exported by twinlite_da so the two cannot drift apart.
        self.twinlite_weights = str(
            legacy.rospy.get_param(
                "~twinlite_weights", str(twinlite_da.DEFAULT_TWINLITE_WEIGHTS)
            )
        )
        self.twinlite_wood_filter = bool(
            legacy.rospy.get_param("~twinlite_wood_filter", True)
        )
        self.twinlite_vote_window = int(
            legacy.rospy.get_param(
                "~twinlite_vote_window", twinlite_da.DEFAULT_VOTE_WINDOW
            )
        )
        self.twinlite_vote_min = int(
            legacy.rospy.get_param("~twinlite_vote_min", twinlite_da.DEFAULT_VOTE_MIN)
        )
        self.twinlite_largest_component = bool(
            legacy.rospy.get_param("~twinlite_largest_component", True)
        )
        # The boardwalk texture test also fires on distant mown grass, which can
        # erase almost the whole drivable area on a turn. Below this kept
        # fraction the filter is treated as misfiring and the raw mask is used.
        self.twinlite_min_keep_ratio = float(
            legacy.rospy.get_param(
                "~twinlite_min_keep_ratio", twinlite_da.DEFAULT_MIN_KEEP_RATIO
            )
        )
        # The boardwalk vote needs consecutive camera frames, so TwinLiteNet runs
        # in cb_image at the camera rate rather than once per 0.5 s plan. This
        # caps that pre-pass; the planner's own cadence is far slower, so a
        # sampled frame is never the one skipped.
        self.twinlite_prepass_max_hz = float(
            legacy.rospy.get_param("~twinlite_prepass_max_hz", 15.0)
        )
        if self.twinlite_prepass_max_hz <= 0.0:
            raise ValueError("~twinlite_prepass_max_hz must be positive")
        if self.twinlite_prepass_max_hz < 1.0 / self.sample_interval:
            raise ValueError(
                f"~twinlite_prepass_max_hz={self.twinlite_prepass_max_hz} is slower "
                f"than the {1.0 / self.sample_interval:.1f} Hz planner cadence"
            )

    def _load_twinlite_backend(self) -> None:
        self.twinlite = twinlite_da.TwinLiteDrivableArea(
            weights=self.twinlite_weights,
            device=self.device,
            use_fp16=self.use_fp16,
            vote_window=self.twinlite_vote_window,
            vote_min=self.twinlite_vote_min,
            wood_filter=self.twinlite_wood_filter,
            largest_component=self.twinlite_largest_component,
            min_keep_ratio=self.twinlite_min_keep_ratio,
        )

    def _twinlite_log_startup(self) -> None:
        legacy.rospy.loginfo(
            f"[FF planner] segmentation backend: TwinLiteNet drivable area | "
            f"weights={self.twinlite.weights_path} | "
            f"wood_filter={self.twinlite_wood_filter} "
            f"(vote {self.twinlite_vote_min}/{self.twinlite_vote_window}) | "
            f"largest_component={self.twinlite_largest_component} | "
            f"min_keep_ratio={self.twinlite_min_keep_ratio:.2f} | "
            f"pre-pass <= {self.twinlite_prepass_max_hz:.1f} Hz"
        )
        legacy.rospy.loginfo(
            "[FF planner] TwinLiteNet is binary: drivable -> class 0 (road), "
            "everything else -> class 3 (static); classes 1/2 stay empty"
        )

    def _twinlite_reset(self) -> None:
        """Drop the vote history, e.g. after a rosbag clock restart."""
        if self.twinlite is None:
            return
        self.twinlite.reset()
        self._twinlite_stamp = None
        self._twinlite_last_prepass = None

    def _twinlite_prepass(self, msg, now: float, force: bool = False) -> None:
        """Feed the boardwalk vote at camera rate, ahead of the cadence gate.

        The offline script votes over a five-frame window at video rate; running
        TwinLiteNet only on sampled frames would stretch that window to 2.5 s and
        vote across a scene that has completely changed.

        ``force`` is for the frame actually being planned. The camera stream is
        not evenly spaced -- consecutive frames can arrive far closer together
        than the nominal rate -- so the throttle below can skip the very frame
        the cadence gate then selects, and segment() would have no mask for it.
        A forced call also ignores ``_busy``, since it comes from inside the
        plan it would otherwise be waiting on.
        """
        if self.segmentation_backend != TWINLITE_BACKEND:
            return
        if not force and self._busy:
            return
        # Never push the same frame twice: that would let one frame vote twice
        # in the boardwalk window.
        if self._twinlite_stamp is not None and abs(now - self._twinlite_stamp) <= 1e-9:
            return
        min_period = 1.0 / self.twinlite_prepass_max_hz
        if (
            not force
            and self._twinlite_last_prepass is not None
            and 0.0 <= now - self._twinlite_last_prepass < min_period
        ):
            return
        try:
            self.twinlite.push(legacy.rosimg_to_rgb_numpy(msg))
        except Exception:
            legacy.rospy.logerr_throttle(
                2.0,
                "[FF planner] TwinLiteNet pre-pass failed:\n"
                + __import__("traceback").format_exc(),
            )
            return
        self._twinlite_last_prepass = now
        self._twinlite_stamp = now

    def _twinlite_segment(self, rgb: np.ndarray) -> np.ndarray:
        """Turn the cached drivable-area mask into (224,224) PALETTE4 IDs."""
        if not self.twinlite.ready:
            raise RuntimeError("TwinLiteNet pre-pass produced no frame for this cycle")
        # Planning on a stale mask would silently steer by an old scene, so
        # require the pre-pass to have run on exactly this image.
        if (
            self._twinlite_stamp is None
            or self._current_image_stamp is None
            or abs(self._twinlite_stamp - self._current_image_stamp) > 1e-6
        ):
            raise RuntimeError(
                "TwinLiteNet mask does not belong to the frame being planned "
                f"(pre-pass stamp={self._twinlite_stamp}, "
                f"image stamp={self._current_image_stamp})"
            )

        mask = self.twinlite.current_mask()
        if self.twinlite.last_fallback_used:
            legacy.rospy.logwarn_throttle(
                2.0,
                "[FF planner] boardwalk filter kept only "
                f"{self.twinlite.last_filtered_area_px}/"
                f"{self.twinlite.last_unfiltered_area_px} drivable pixels "
                f"(< {self.twinlite_min_keep_ratio:.2f}); using the unfiltered "
                "TwinLiteNet mask for this frame",
            )

        # Binary drivable area, as configured: road (0) or static (3). The
        # FF planner's off-road cost reads only the road channel, so this is
        # the channel that matters; person/movable stay empty by design.
        cls4_small = np.where(mask, 0, 3).astype(np.uint8)
        # Undo the 640x360 working resolution before the keep-ratio centre crop,
        # so the result lines up with rgb_224 exactly like the YOLO path does.
        height, width = rgb.shape[:2]
        cls4_full = cv2.resize(
            cls4_small, (width, height), interpolation=cv2.INTER_NEAREST
        )
        cls4 = torch.from_numpy(cls4_full).unsqueeze(0).unsqueeze(0)
        return legacy.resize_keep_ratio_center_crop_uint8(cls4)[0, 0].numpy()
