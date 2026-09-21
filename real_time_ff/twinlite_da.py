"""TwinLiteNet drivable-area segmentation for the realtime FF planner.

This wraps TwinLiteNet's drivable-area head plus the texture-based boardwalk
removal that ``TwinLiteNet/test_video_filtered.py`` already validated offline.
The offline script emits the centre of a five-frame window, which costs two
frames of latency; the realtime wrapper votes causally over the frames seen so
far instead, so the mask for the current frame is available immediately.

The module has no ROS dependency, so it can be exercised from a plain Python
session.
"""

from collections import deque
from pathlib import Path
import importlib.util
import math
import sys

import cv2
import numpy as np
import torch


_THIS_DIR = Path(__file__).resolve().parent
TWINLITE_DIR = _THIS_DIR / "TwinLiteNet"
DEFAULT_TWINLITE_WEIGHTS = TWINLITE_DIR / "pretrained" / "best.pth"

# TwinLiteNet's fixed working resolution. wood_mask()'s SCALE ramp hard-codes
# the horizon at row 170 of these 360, so the filter only makes sense here.
TWINLITE_H, TWINLITE_W = 360, 640


def _load_module_from_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_twinlite_arch():
    """Import TwinLiteNet/model/TwinLite.py by path.

    A plain ``from model import TwinLite`` would resolve ``model`` to this
    directory's model/ folder, which holds .ckpt files and is not a package.
    TwinLite.py imports nothing but torch, so a file-path import is enough.
    """
    path = TWINLITE_DIR / "model" / "TwinLite.py"
    if not path.is_file():
        raise FileNotFoundError(f"TwinLiteNet architecture not found: {path}")
    return _load_module_from_file("twinlite_arch", path)


def _load_offline_filters():
    """Import the offline filter helpers instead of duplicating them.

    test_video_filtered.py does ``from model import TwinLite`` at module level,
    so TwinLiteNet/ has to lead sys.path while it executes. Both sys.path and
    any pre-existing sys.modules['model'] are restored afterwards, so the rest
    of the planner still sees its own import environment.
    """
    path = TWINLITE_DIR / "test_video_filtered.py"
    if not path.is_file():
        raise FileNotFoundError(f"TwinLiteNet filter script not found: {path}")
    saved_model_module = sys.modules.pop("model", None)
    sys.path.insert(0, str(TWINLITE_DIR))
    try:
        return _load_module_from_file("twinlite_filters", path)
    finally:
        try:
            sys.path.remove(str(TWINLITE_DIR))
        except ValueError:
            pass
        sys.modules.pop("model", None)
        if saved_model_module is not None:
            sys.modules["model"] = saved_model_module


_FILTERS = _load_offline_filters()

# Offline defaults, re-exported so callers can expose them as parameters.
DEFAULT_VOTE_WINDOW = int(_FILTERS.WIN)
DEFAULT_VOTE_MIN = int(_FILTERS.VOTE)

# Safety net for a wood_mask() misfire: below this fraction of the mask it
# would have produced without the boardwalk subtraction, the filter is treated
# as wrong rather than as a real rejection. Offline an over-aggressive frame
# only looked odd; here it would leave the planner's BEV with no road at all.
#
# Measured at 15 Hz over the whole of bkgd_right_raw.mp4, the boardwalk
# subtraction never takes the cleaned mask below 0.69 of the unsubtracted one,
# so 0.25 leaves every genuine rejection intact and only catches a real misfire.
DEFAULT_MIN_KEEP_RATIO = 0.25


class TwinLiteDrivableArea:
    """Per-frame TwinLiteNet drivable area with causal boardwalk voting."""

    def __init__(
        self,
        weights=DEFAULT_TWINLITE_WEIGHTS,
        device="cuda",
        use_fp16=True,
        vote_window=DEFAULT_VOTE_WINDOW,
        vote_min=DEFAULT_VOTE_MIN,
        wood_filter=True,
        largest_component=True,
        min_keep_ratio=DEFAULT_MIN_KEEP_RATIO,
    ):
        weights = Path(weights).expanduser()
        if not weights.is_absolute():
            weights = _THIS_DIR / weights
        if not weights.is_file():
            raise FileNotFoundError(
                f"TwinLiteNet checkpoint not found: {weights}. "
                "Set ~twinlite_weights to a valid .pth"
            )
        vote_window = int(vote_window)
        vote_min = int(vote_min)
        if vote_window < 1:
            raise ValueError("~twinlite_vote_window must be >= 1")
        if not 1 <= vote_min <= vote_window:
            raise ValueError("~twinlite_vote_min must be in [1, ~twinlite_vote_window]")
        min_keep_ratio = float(min_keep_ratio)
        if not 0.0 <= min_keep_ratio < 1.0:
            raise ValueError("~twinlite_min_keep_ratio must be in [0, 1)")

        self.weights_path = str(weights.resolve())
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16) and self.device.type == "cuda"
        self.vote_window = vote_window
        self.vote_min = vote_min
        self.wood_filter = bool(wood_filter)
        self.largest_component = bool(largest_component)
        self.min_keep_ratio = min_keep_ratio

        arch = _load_twinlite_arch()
        model = arch.TwinLiteNet()
        state_dict = torch.load(self.weights_path, map_location="cpu")
        # The released checkpoint was saved from a DataParallel wrapper. Strip
        # the prefix rather than re-wrapping, so a single GPU stays single.
        state_dict = {
            (key[len("module."):] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)
        self.model = model.to(self.device).eval()

        # (drivable area, wood mask) for the last `vote_window` pushed frames.
        self._history = deque(maxlen=vote_window)
        self.last_raw_area_px = 0
        self.last_filtered_area_px = 0
        self.last_unfiltered_area_px = 0
        self.last_fallback_used = False

    def reset(self) -> None:
        """Drop the voting history, e.g. after a rosbag clock restart."""
        self._history.clear()
        self.last_raw_area_px = 0
        self.last_filtered_area_px = 0
        self.last_unfiltered_area_px = 0
        self.last_fallback_used = False

    @property
    def ready(self) -> bool:
        return len(self._history) > 0

    @torch.inference_mode()
    def push(self, rgb_full: np.ndarray) -> None:
        """Run one full-resolution RGB frame through TwinLiteNet's DA head."""
        if rgb_full.ndim != 3 or rgb_full.shape[2] != 3:
            raise ValueError(f"Expected (H,W,3) RGB, got {rgb_full.shape}")

        # The offline scripts work on OpenCV BGR and feed the network
        # img[:, :, ::-1], i.e. RGB. Keep the network input identical and hand
        # wood_mask() the BGR frame it expects.
        rgb = cv2.resize(rgb_full, (TWINLITE_W, TWINLITE_H), interpolation=cv2.INTER_LINEAR)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])

        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        tensor = tensor.unsqueeze(0).to(self.device).float().div_(255.0)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
            outputs = self.model(tensor)
        _, da_predict = torch.max(outputs[0].float(), 1)
        drivable = (da_predict.byte().cpu().numpy()[0] * 255) > 100

        wood = (
            _FILTERS.wood_mask(bgr, drivable)
            if self.wood_filter
            else np.zeros_like(drivable, dtype=bool)
        )
        self._history.append((drivable, wood))

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        """The offline emit()'s clean-up: open, fill holes, keep the ego blob."""
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, _FILTERS.EK5
        ).astype(bool)
        mask = _FILTERS.fill_holes(mask)
        if self.largest_component:
            mask = _FILTERS.largest_bottom_component(mask)
        return mask

    def current_mask(self) -> np.ndarray:
        """Return the (360,640) bool drivable mask for the most recent frame."""
        if not self._history:
            raise RuntimeError("TwinLiteDrivableArea.current_mask() before any push()")

        drivable = self._history[-1][0]
        self.last_raw_area_px = int(drivable.sum())
        self.last_fallback_used = False

        if not self.wood_filter:
            mask = self._clean(drivable)
            self.last_filtered_area_px = int(mask.sum())
            self.last_unfiltered_area_px = self.last_filtered_area_px
            return mask

        votes = np.sum([entry[1] for entry in self._history], axis=0)
        # Scale the offline threshold to the frames actually seen, so the first
        # frames after a reset are filtered instead of unprotected.
        threshold = max(
            1, int(math.ceil(self.vote_min * len(self._history) / self.vote_window))
        )
        mask = self._clean(drivable & ~(votes >= threshold))
        self.last_filtered_area_px = int(mask.sum())
        self.last_unfiltered_area_px = self.last_filtered_area_px

        # A mask the boardwalk filter all but erased leaves the planner's BEV
        # with no road evidence, which is worse than an unfiltered one. Drop only
        # the wood subtraction — the morphology and component clean-up still
        # apply — and let the caller log it.
        #
        # The comparison has to be against the *cleaned* unfiltered mask: the
        # raw pixel count also drops when largest_bottom_component discards
        # disconnected blobs, and charging that to the wood filter would fire
        # this fallback on frames where TwinLiteNet itself produced a fragmented
        # drivable area. _clean() is only paid when the cheap bound below
        # (cleaned <= raw, always) cannot already clear the frame.
        if (
            self.last_raw_area_px > 0
            and self.last_filtered_area_px
            < self.min_keep_ratio * self.last_raw_area_px
        ):
            unfiltered = self._clean(drivable)
            self.last_unfiltered_area_px = int(unfiltered.sum())
            if self.last_filtered_area_px < self.min_keep_ratio * self.last_unfiltered_area_px:
                self.last_fallback_used = True
                mask = unfiltered
        return mask
