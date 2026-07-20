import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from pyquaternion import Quaternion

from stp3.datas.NuscenesData import FuturePredictionDataset as BaseFuturePredictionDataset


class FuturePredictionDataset(BaseFuturePredictionDataset):
    """
    在原本 NuscenesData 的基礎上，額外提供行人軌跡預測輸入。

    回傳新增欄位：
    - ped_traj_preds: (M, T, 8) float32
    - ped_traj_mask: (M,) bool
    - ped_traj_valid_steps: (M, T) bool
    - ped_traj_has_data: () bool

    注意：
    - 為了相容 DataLoader 預設 collate，以上欄位每個 sample 都會回傳固定 shape。
    - 若目前 frame 沒有對到 CSV，則 ped_traj_mask 會全 False，模型端可直接忽略。
    """

    RAW_PED_COLUMNS = [
        'pred_bbox_cx',
        'pred_bbox_cy',
        'pred_bbox_w',
        'pred_bbox_h',
        'pred_goal_cx',
        'pred_goal_cy',
        'pred_goal_w',
        'pred_goal_h',
    ]

    PED_FEATURE_NAMES = [
        'bottom_x',
        'bottom_y',
        'bbox_h',
        'bbox_area',
        'delta_bottom_x',
        'delta_bottom_y',
        'goal_dx',
        'goal_dy',
        'ego_center_offset',
        'ego_corridor_score',
        'ttc_risk',
        'path_risk',
    ]

    def __init__(self, nusc, is_train, cfg):
        super().__init__(nusc, is_train, cfg)
        self.ped_pred_csv_path = getattr(
            cfg,
            'PED_TRAJ_PRED_CSV',
            '/home/cyc/self_dataset/0320_dataset_codex/CAM_FRONT_history_sequences_bitrap_predictions.csv',
        )
        self.ped_max_agents = int(getattr(cfg, 'PED_MAX_AGENTS', 64))
        self.ped_feat_dim = len(self.PED_FEATURE_NAMES)
        self.ped_n_future = int(getattr(cfg, 'PED_INPUT_FRAMES', 9))
        self.ped_source_future = int(getattr(cfg, 'PED_SOURCE_FRAMES', 45))
        self.image_w = float(getattr(cfg.IMAGE, 'ORIGINAL_WIDTH', 1600))
        self.image_h = float(getattr(cfg.IMAGE, 'ORIGINAL_HEIGHT', 900))
        self.ped_pred_index = self._load_ped_prediction_index(self.ped_pred_csv_path)
        print(
            f"[PedTraj] loaded {len(self.ped_pred_index)} keyframes from {self.ped_pred_csv_path} "
            f"for n_future={self.ped_n_future}, max_agents={self.ped_max_agents}"
        )

    def _build_geom_ped_features(self, traj_raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
        geom = np.zeros((traj_raw.shape[0], self.ped_feat_dim), dtype=np.float32)
        if not valid.any():
            return geom

        bbox_cx = traj_raw[:, 0]
        bbox_cy = traj_raw[:, 1]
        bbox_w = traj_raw[:, 2]
        bbox_h = traj_raw[:, 3]
        goal_cx = traj_raw[:, 4]
        goal_cy = traj_raw[:, 5]
        goal_h = traj_raw[:, 7]

        bottom_x = bbox_cx
        bottom_y = bbox_cy + 0.5 * bbox_h
        bbox_area = bbox_w * bbox_h
        goal_bottom_x = goal_cx
        goal_bottom_y = goal_cy + 0.5 * goal_h

        delta_bottom_x = np.zeros_like(bottom_x)
        delta_bottom_y = np.zeros_like(bottom_y)
        prev_t = None
        for t in range(traj_raw.shape[0]):
            if not valid[t]:
                continue
            if prev_t is not None:
                delta_bottom_x[t] = bottom_x[t] - bottom_x[prev_t]
                delta_bottom_y[t] = bottom_y[t] - bottom_y[prev_t]
            prev_t = t

        goal_dx = goal_bottom_x - bottom_x
        goal_dy = goal_bottom_y - bottom_y

        ego_center_offset = bottom_x - 0.5
        corridor_half_width = float(getattr(self.cfg, 'PED_EGO_CORRIDOR_HALF_WIDTH', 0.15))
        ego_corridor_score = np.clip(1.0 - np.abs(ego_center_offset) / max(corridor_half_width, 1e-3), 0.0, 1.0)

        forward_gap = np.clip(1.0 - bottom_y, 0.0, 1.0)
        approach_speed = np.clip(delta_bottom_y, 0.0, None)
        ttc = np.full_like(bottom_y, 10.0)
        moving = approach_speed > 1e-4
        ttc[moving] = forward_gap[moving] / np.maximum(approach_speed[moving], 1e-4)
        ttc_risk = np.clip(1.0 / (1.0 + ttc), 0.0, 1.0)

        goal_center_offset = goal_bottom_x - 0.5
        goal_corridor_score = np.clip(1.0 - np.abs(goal_center_offset) / max(corridor_half_width, 1e-3), 0.0, 1.0)
        future_corridor = np.maximum(ego_corridor_score, goal_corridor_score)
        cross_strength = np.clip(np.abs(delta_bottom_x) + 0.5 * np.abs(goal_dx), 0.0, 0.5) / 0.5
        close_score = np.clip(bottom_y, 0.0, 1.0)
        path_risk = np.clip(0.45 * future_corridor + 0.30 * ttc_risk + 0.15 * close_score + 0.10 * cross_strength, 0.0, 1.0)

        geom[:, 0] = bottom_x
        geom[:, 1] = bottom_y
        geom[:, 2] = bbox_h
        geom[:, 3] = bbox_area
        geom[:, 4] = delta_bottom_x
        geom[:, 5] = delta_bottom_y
        geom[:, 6] = goal_dx
        geom[:, 7] = goal_dy
        geom[:, 8] = ego_center_offset
        geom[:, 9] = ego_corridor_score
        geom[:, 10] = ttc_risk
        geom[:, 11] = path_risk
        geom[~valid] = 0.0
        return geom

    def _uniform_indices(self, n_src: int, n_dst: int) -> np.ndarray:
        if n_dst <= 1:
            return np.asarray([0], dtype=np.int64)
        return np.linspace(0, n_src - 1, num=n_dst, dtype=np.int64)

    def _downsample_ped_sequence(self, traj: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = self._uniform_indices(traj.shape[0], self.ped_n_future)
        return traj[idx].copy(), valid[idx].copy()

    def _ped_risk_score(self, traj: np.ndarray, valid: np.ndarray) -> float:
        valid_feats = traj[valid]
        if len(valid_feats) == 0:
            return 0.0

        bbox_h = valid_feats[:, 2]
        bbox_area = valid_feats[:, 3]
        ego_corridor_score = valid_feats[:, 9]
        ttc_risk = valid_feats[:, 10]
        path_risk = valid_feats[:, 11]

        size_score = np.clip(2.0 * bbox_h + 2.0 * bbox_area, 0.0, 1.0)
        step_score = 0.60 * path_risk + 0.25 * ego_corridor_score + 0.10 * ttc_risk + 0.05 * size_score
        return float(step_score.max())

    def _load_ped_prediction_index(self, csv_path: str) -> Dict[str, List[Tuple[float, int, np.ndarray, np.ndarray]]]:
        if not os.path.exists(csv_path):
            print(f"[PedTraj] CSV not found: {csv_path}. Dataset will return empty pedestrian trajectory inputs.")
            return {}

        grouped: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {}
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                try:
                    keyframe_name = row['keyframe_name'].strip()
                    person_idx = int(float(row['keyframe_person_index']))
                    pred_step = int(float(row['pred_step']))
                    if pred_step < 1:
                        continue
                    feat = np.asarray([float(row[c]) for c in self.RAW_PED_COLUMNS], dtype=np.float32)
                    feat[0] /= self.image_w
                    feat[2] /= self.image_w
                    feat[4] /= self.image_w
                    feat[6] /= self.image_w
                    feat[1] /= self.image_h
                    feat[3] /= self.image_h
                    feat[5] /= self.image_h
                    feat[7] /= self.image_h
                except Exception:
                    continue

                key_group = grouped.setdefault(keyframe_name, {})
                person_group = key_group.setdefault(person_idx, {})
                person_group[pred_step] = feat

        packed: Dict[str, List[Tuple[float, int, np.ndarray, np.ndarray]]] = {}
        for keyframe_name, person_map in grouped.items():
            persons = []
            for person_idx, step_map in person_map.items():
                traj_raw_full = np.zeros((self.ped_source_future, len(self.RAW_PED_COLUMNS)), dtype=np.float32)
                valid_full = np.zeros((self.ped_source_future,), dtype=np.bool_)
                for pred_step, feat in step_map.items():
                    if pred_step > self.ped_source_future:
                        continue
                    traj_raw_full[pred_step - 1] = feat
                    valid_full[pred_step - 1] = True

                if not valid_full.any():
                    continue

                traj_full = self._build_geom_ped_features(traj_raw_full, valid_full)
                idx = self._uniform_indices(traj_full.shape[0], self.ped_n_future)
                traj = traj_full[idx].copy()
                valid = valid_full[idx].copy()
                traj_raw_ds = traj_raw_full[idx].copy()
                if not valid.any():
                    continue

                risk_score = self._ped_risk_score(traj, valid)
                persons.append((risk_score, person_idx, traj, valid, traj_raw_ds))

            persons.sort(key=lambda x: (-x[0], x[1]))
            packed[keyframe_name] = persons

        return packed

    def _make_empty_ped_traj(self):
        ped_traj_preds = torch.zeros(self.ped_max_agents, self.ped_n_future, self.ped_feat_dim, dtype=torch.float32)
        ped_traj_mask = torch.zeros(self.ped_max_agents, dtype=torch.bool)
        ped_traj_valid_steps = torch.zeros(self.ped_max_agents, self.ped_n_future, dtype=torch.bool)
        ped_traj_person_ids = torch.full((self.ped_max_agents,), -1, dtype=torch.long)
        return ped_traj_preds, ped_traj_mask, ped_traj_valid_steps, ped_traj_person_ids

    def _get_present_keyframe_name(self, index: int) -> str:
        present_index = self.indices[index][self.receptive_field - 1]
        present_rec = self.ixes[present_index]
        cam = self.cfg.IMAGE.NAMES[0]
        cam_sample = self.nusc.get('sample_data', present_rec['data'][cam])
        return os.path.basename(cam_sample['filename'])

    def _build_ped_traj_tensors(self, keyframe_name: str):
        ped_traj_preds, ped_traj_mask, ped_traj_valid_steps, ped_traj_person_ids = self._make_empty_ped_traj()
        persons = self.ped_pred_index.get(keyframe_name)
        if not persons:
            return ped_traj_preds, ped_traj_mask, ped_traj_valid_steps, ped_traj_person_ids, False

        kept = persons[:self.ped_max_agents]
        for i, (_, person_idx, traj, valid, traj_raw_ds) in enumerate(kept):
            ped_traj_preds[i] = torch.from_numpy(traj)
            ped_traj_mask[i] = True
            ped_traj_valid_steps[i] = torch.from_numpy(valid)
            ped_traj_person_ids[i] = int(person_idx)

        return ped_traj_preds, ped_traj_mask, ped_traj_valid_steps, ped_traj_person_ids, True

    def __getitem__(self, index):
        data = super().__getitem__(index)
        keyframe_name = self._get_present_keyframe_name(index)
        ped_traj_preds, ped_traj_mask, ped_traj_valid_steps, ped_traj_person_ids, has_data = self._build_ped_traj_tensors(keyframe_name)

        data['ped_traj_preds'] = ped_traj_preds
        data['ped_traj_mask'] = ped_traj_mask
        data['ped_traj_valid_steps'] = ped_traj_valid_steps
        data['ped_traj_person_ids'] = ped_traj_person_ids
        data['ped_traj_has_data'] = torch.tensor(has_data, dtype=torch.bool)
        data['ped_traj_keyframe_name'] = keyframe_name
        return data
