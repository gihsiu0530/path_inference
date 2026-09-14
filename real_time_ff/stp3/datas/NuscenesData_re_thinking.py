import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw

from stp3.datas.NuscenesData import locate_message
from stp3.utils.geometry import get_global_pose


COMMAND_TO_ONEHOT = {
    "LEFT": [1.0, 0.0, 0.0],
    "FORWARD": [0.0, 1.0, 0.0],
    "RIGHT": [0.0, 0.0, 1.0],
}


def _as_xyz(value, default=0.0):
    arr = np.asarray(value if value is not None else [default, default, default], dtype=np.float32)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), constant_values=default)
    return arr[:3]


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.zeros_like(values, dtype=np.float32)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


class ADMLPFeatureCache:
    """Generate AD-MLP style ego-state features from the ST-P3 dataset metadata."""

    def __init__(self, dataset, split_name: str, cfg):
        self.dataset = dataset
        self.cfg = cfg
        self.split_name = split_name
        self.feature_source = str(getattr(cfg, "ADMLP_FEATURE_SOURCE", "generated"))
        if self.feature_source not in {"generated", "official_pkl"}:
            raise ValueError(f"Unsupported ADMLP_FEATURE_SOURCE={self.feature_source!r}")
        self.official_pkl = str(getattr(cfg, "ADMLP_OFFICIAL_PKL", ""))
        self.official_data = None
        if self.feature_source == "official_pkl":
            if not self.official_pkl:
                raise ValueError("ADMLP_OFFICIAL_PKL must be set when ADMLP_FEATURE_SOURCE='official_pkl'")
            if not os.path.exists(self.official_pkl):
                raise FileNotFoundError(f"ADMLP_OFFICIAL_PKL not found: {self.official_pkl}")
            with open(self.official_pkl, "rb") as f:
                self.official_data = pickle.load(f)
            print(f"[ADMLP OFFICIAL] loaded {len(self.official_data)} token features from {self.official_pkl}")
        self.feature_mode = str(getattr(cfg, "ADMLP_FEATURE_MODE", "past4_command"))
        if self.feature_mode not in {"past4_command", "past5_no_command"}:
            raise ValueError(f"Unsupported ADMLP_FEATURE_MODE={self.feature_mode!r}")
        self.past_frames = int(getattr(cfg, "ADMLP_PAST_FRAMES", 4))
        self.sample_interval = float(getattr(dataset, "SAMPLE_INTERVAL", 0.5))
        self.traj_ema = bool(getattr(cfg, "ADMLP_TRAJ_EMA", False))
        legacy_can_ema = bool(getattr(cfg, "ADMLP_CAN_EMA", True))
        self.vel_ema = bool(getattr(cfg, "ADMLP_VEL_EMA", legacy_can_ema))
        self.acc_ema = bool(getattr(cfg, "ADMLP_ACC_EMA", legacy_can_ema))
        self.ema_alpha = float(getattr(cfg, "ADMLP_EMA_ALPHA", 0.2))
        self.ema_window = int(getattr(cfg, "ADMLP_EMA_WINDOW", 12))
        self.yaw_acc_clip = float(getattr(cfg, "ADMLP_YAW_ACC_CLIP", 2.0))
        self.zero_acc = bool(getattr(cfg, "ADMLP_ZERO_ACC", False))
        self.zero_yaw_acc = bool(getattr(cfg, "ADMLP_ZERO_YAW_ACC", False))

        cache_root = getattr(cfg, "ADMLP_CACHE_ROOT", None)
        if cache_root is None:
            cache_root = os.path.join(str(getattr(cfg.DATASET, "SAVE_DIR", "datas")), "re_thinking")
        os.makedirs(cache_root, exist_ok=True)
        version = str(getattr(cfg.DATASET, "VERSION", "unknown"))
        if self.feature_source == "official_pkl":
            official_name = os.path.splitext(os.path.basename(self.official_pkl))[0]
            cache_name = (
                f"admlp_re_thinking_{version}_{split_name}_official_{official_name}"
                f"_{self.feature_mode}_pf{self.past_frames}_nf{cfg.N_FUTURE_FRAMES}.pkl"
            )
        else:
            cache_name = (
                f"admlp_re_thinking_{version}_{split_name}_{self.feature_mode}_pf{self.past_frames}_nf{cfg.N_FUTURE_FRAMES}"
                f"_rawswap_tema{int(self.traj_ema)}_vema{int(self.vel_ema)}_aema{int(self.acc_ema)}"
                f"_a{self.ema_alpha:g}_w{self.ema_window}"
                f"_yac{self.yaw_acc_clip:g}_zacc{int(self.zero_acc)}_zyawacc{int(self.zero_yaw_acc)}.pkl"
            )
        self.cache_path = os.path.join(cache_root, cache_name)

        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self.records = pickle.load(f)
            print(f"[ADMLP CACHE] loaded {len(self.records)} records from {self.cache_path}")
        else:
            self.records = self._build_records()
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.records, f)
            print(f"[ADMLP CACHE] saved {len(self.records)} records to {self.cache_path}")

    def _official_feature(self, token: str) -> np.ndarray:
        if token not in self.official_data:
            raise KeyError(f"Token {token} not found in official AD-MLP pkl: {self.official_pkl}")
        token_data = self.official_data[token]
        values = []
        for key in sorted(token_data.keys()):
            if key == "gt":
                continue
            values.append(np.asarray(token_data[key], dtype=np.float32).reshape(-1))
        feature = np.concatenate(values, axis=0).astype(np.float32)
        if feature.shape[0] != 21:
            raise ValueError(f"Official feature for token {token} has shape {feature.shape}, expected 21")
        return feature

    def _relative_pose(self, rec_ref, rec_other) -> np.ndarray:
        ref_from_global = get_global_pose(rec_ref, self.dataset.nusc, inverse=True)
        global_from_other = get_global_pose(rec_other, self.dataset.nusc, inverse=False)
        other_from_ref = ref_from_global.dot(global_from_other)
        theta = quaternion_yaw(Quaternion(matrix=other_from_ref))
        origin = np.asarray(other_from_ref[:3, 3], dtype=np.float32)
        return np.asarray([origin[0], origin[1], theta], dtype=np.float32)

    def _past_trajectory(self, present_idx: int) -> np.ndarray:
        present_rec = self.dataset.ixes[present_idx]
        past = []
        start_idx = present_idx - self.past_frames
        for idx in range(start_idx, present_idx):
            if idx < 0:
                past.append(np.zeros(3, dtype=np.float32))
                continue
            rec = self.dataset.ixes[idx]
            if rec["scene_token"] != present_rec["scene_token"]:
                past.append(np.zeros(3, dtype=np.float32))
            else:
                past.append(self._relative_pose(present_rec, rec))
        past = np.stack(past, axis=0).astype(np.float32)
        if self.traj_ema:
            past = _ema(past, self.ema_alpha)
        return past

    def _motion_from_pose_pair(self, pose_data, prev_pose, pose_index: int) -> Tuple[np.ndarray, np.ndarray]:
        vel_raw = _as_xyz(pose_data.get("vel"))
        rot_rate = _as_xyz(pose_data.get("rotation_rate"))
        # CAN pose vel is closest to ST-P3 planning coordinates after swapping
        # xy: planning x is lateral, planning y is forward.
        vel = np.asarray([vel_raw[1], vel_raw[0], rot_rate[2]], dtype=np.float32)

        accel_raw = _as_xyz(pose_data.get("accel"))
        prev_rot_rate = _as_xyz(prev_pose.get("rotation_rate"))
        dt = max((float(pose_data["utime"]) - float(prev_pose["utime"])) * 1e-6, 1e-6)
        angular_acc = (rot_rate[2] - prev_rot_rate[2]) / dt if pose_index > 0 else 0.0
        if self.yaw_acc_clip > 0:
            angular_acc = float(np.clip(angular_acc, -self.yaw_acc_clip, self.yaw_acc_clip))
        # Empirically the best acceleration candidate is raw:swap_negx.
        acc = np.asarray([-accel_raw[1], accel_raw[0], angular_acc], dtype=np.float32)
        return vel, acc

    def _can_motion(self, rec) -> Tuple[np.ndarray, np.ndarray]:
        scene = self.dataset.nusc.get("scene", rec["scene_token"])
        try:
            pose_msgs = self.dataset.nusc_can.get_messages(scene["name"], "pose")
        except Exception:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        if len(pose_msgs) == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        pose_uts = [msg["utime"] for msg in pose_msgs]
        pose_index = locate_message(pose_uts, rec["timestamp"])
        pose_data = pose_msgs[pose_index]
        prev_pose = pose_msgs[max(0, pose_index - 1)]
        raw_vel, raw_acc = self._motion_from_pose_pair(pose_data, prev_pose, pose_index)

        if not self.vel_ema and not self.acc_ema:
            return raw_vel, raw_acc

        # ema_window <= 0 means causal EMA over all previous CAN pose messages
        # in the current scene, matching an offline scene-level smoothing setup.
        start = 0 if self.ema_window <= 0 else max(0, pose_index - self.ema_window + 1)
        vel_hist = []
        acc_hist = []
        for idx in range(start, pose_index + 1):
            cur = pose_msgs[idx]
            prev = pose_msgs[max(0, idx - 1)]
            vel_i, acc_i = self._motion_from_pose_pair(cur, prev, idx)
            vel_hist.append(vel_i)
            acc_hist.append(acc_i)
        # EMA is applied independently along the CAN time axis. Velocity and
        # acceleration are never stacked together before smoothing, and each
        # stream can be kept raw independently.
        vel = _ema(np.stack(vel_hist, axis=0), self.ema_alpha)[-1] if self.vel_ema else raw_vel
        acc = _ema(np.stack(acc_hist, axis=0), self.ema_alpha)[-1] if self.acc_ema else raw_acc
        return vel.astype(np.float32), acc.astype(np.float32)

    def _fallback_motion(self, past: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(past) < 2:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        vel = (past[-1] - past[-2]) / self.sample_interval
        if len(past) >= 3:
            prev_vel = (past[-2] - past[-3]) / self.sample_interval
            acc = (vel - prev_vel) / self.sample_interval
        else:
            acc = np.zeros(3, dtype=np.float32)
        return vel.astype(np.float32), acc.astype(np.float32)

    def _build_one(self, dataset_index: int) -> Dict[str, np.ndarray]:
        seq = self.dataset.indices[dataset_index]
        present_idx = int(seq[self.dataset.receptive_field - 1])
        present_rec = self.dataset.ixes[present_idx]

        gt, command = self.dataset.get_gt_trajectory(present_rec, present_idx)
        if self.feature_source == "official_pkl":
            feature = self._official_feature(present_rec["token"])
            return {
                "token": present_rec["token"],
                "input": feature,
                "gt": np.asarray(gt, dtype=np.float32),
                "command": command,
            }

        past = self._past_trajectory(present_idx)
        vel, acc = self._can_motion(present_rec)
        if not np.isfinite(vel).all() or not np.isfinite(acc).all():
            vel, acc = self._fallback_motion(past)
        if self.zero_acc:
            acc = np.zeros_like(acc, dtype=np.float32)
        elif self.zero_yaw_acc:
            acc = acc.copy()
            acc[2] = 0.0

        command_onehot = np.asarray(COMMAND_TO_ONEHOT[command], dtype=np.float32)
        if self.feature_mode == "past5_no_command":
            feature = np.concatenate([past.reshape(-1), vel, acc], axis=0).astype(np.float32)
        else:
            feature = np.concatenate([past.reshape(-1), vel, acc, command_onehot], axis=0).astype(np.float32)
        assert feature.shape[0] == 21, feature.shape
        return {
            "token": present_rec["token"],
            "input": feature,
            "gt": np.asarray(gt, dtype=np.float32),
            "command": command,
        }

    def _build_records(self) -> List[Dict[str, np.ndarray]]:
        records = []
        for i in range(len(self.dataset.indices)):
            records.append(self._build_one(i))
            if (i + 1) % 5000 == 0:
                print(f"[ADMLP CACHE] built {i + 1}/{len(self.dataset.indices)} {self.split_name}")
        return records

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return self.records[index]


class ADMLPFeatureDataset(torch.utils.data.Dataset):
    """Wrap the existing ST-P3 dataset and attach cached AD-MLP 21-D features."""

    def __init__(self, dataset, split_name: str, cfg):
        self.dataset = dataset
        self.cache = ADMLPFeatureCache(dataset, split_name, cfg)
        self.commands_per_index = dataset.commands_per_index[:len(dataset.indices)]
        self.indices = dataset.indices

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        data = self.dataset[index]
        rec = self.cache[index]
        data["token"] = rec["token"]
        data["admlp_input"] = torch.from_numpy(rec["input"]).float()
        data["admlp_gt"] = torch.from_numpy(rec["gt"]).float()
        return data
