import os

import numpy as np
import torch

from stp3.datas.NuscenesData_change_GPVL import FuturePredictionDataset as _GPVLFuturePredictionDataset


class FuturePredictionDataset(_GPVLFuturePredictionDataset):
    """VAD experiment dataset.

    Reuses the GPVL calibration/pseudo-BEV inputs without changing the legacy
    dataset paths. The extra teacher feature field from the GPVL variant is
    harmless for this trainer and keeps the sample-token mapping available.
    """

    def _teacher_bev_root(self):
        root = str(
            getattr(
                self.cfg,
                "BEVFORMER_TEACHER_BEV_ROOT",
                getattr(self.cfg, "VAD_TEACHER_BEV_ROOT", ""),
            )
            or ""
        )
        if root:
            return root
        return "/home/cyc/ST-P3_please/BEVFormer/BEVFormer/data/nuscenes/bevformer_bev_features"

    def _teacher_bev_candidates(self, sample_token):
        root = self._teacher_bev_root()
        return [
            os.path.join(root, self.mode, f"{sample_token}.npy"),
            os.path.join(root, f"{sample_token}.npy"),
        ]

    def _load_teacher_bev(self, sample_token):
        for path in self._teacher_bev_candidates(sample_token):
            if not os.path.exists(path):
                continue
            bev = np.load(path, allow_pickle=False)
            bev = np.asarray(bev, dtype=np.float32)
            if bev.ndim != 3:
                continue
            if bev.shape[-1] == int(getattr(self.cfg, "BEVFORMER_TEACHER_DIM", 256)):
                bev = np.transpose(bev, (2, 0, 1))
            return torch.from_numpy(np.ascontiguousarray(bev)), torch.tensor(1.0, dtype=torch.float32)

        teacher_dim = int(getattr(self.cfg, "BEVFORMER_TEACHER_DIM", 256))
        teacher_size = int(getattr(self.cfg, "BEVFORMER_TEACHER_SIZE", 50))
        return (
            torch.zeros(teacher_dim, teacher_size, teacher_size, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )

    def _teacher_risk_root(self):
        root = str(getattr(self.cfg, "VAD_TEACHER_RISK_ROOT", "") or "")
        if root:
            return root
        return "/home/cyc/ST-P3_please/BEVFormer/BEVFormer/data/nuscenes/bevformer_teacher_risk"

    def _teacher_risk_candidates(self, sample_token):
        root = self._teacher_risk_root()
        return [
            os.path.join(root, self.mode, f"{sample_token}.npy"),
            os.path.join(root, f"{sample_token}.npy"),
        ]

    def _load_teacher_risk(self, sample_token):
        for path in self._teacher_risk_candidates(sample_token):
            if not os.path.exists(path):
                continue
            heatmap = np.load(path, allow_pickle=False)
            heatmap = np.asarray(heatmap, dtype=np.float32)
            if heatmap.ndim == 2:
                heatmap = heatmap[None]
            if heatmap.ndim != 3:
                continue
            if heatmap.shape[-1] in (1, 2) and heatmap.shape[0] not in (1, 2):
                heatmap = np.transpose(heatmap, (2, 0, 1))
            if heatmap.shape[0] == 1:
                heatmap = np.concatenate([heatmap, heatmap], axis=0)
            if heatmap.shape[0] > 2:
                heatmap = heatmap[:2]
            heatmap = np.clip(heatmap, 0.0, 1.0)
            return torch.from_numpy(np.ascontiguousarray(heatmap)), torch.tensor(1.0, dtype=torch.float32)

        size = int(getattr(self.cfg, "VAD_TEACHER_RISK_SIZE", 100))
        return (
            torch.zeros(2, size, size, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        )

    def __getitem__(self, index):
        data = super().__getitem__(index)
        teacher_bev, teacher_bev_valid = self._load_teacher_bev(data["sample_token"])
        teacher_risk, teacher_risk_valid = self._load_teacher_risk(data["sample_token"])
        data["teacher_bev_feature"] = teacher_bev
        data["teacher_bev_valid"] = teacher_bev_valid
        data["teacher_risk_heatmap"] = teacher_risk
        data["teacher_risk_valid"] = teacher_risk_valid
        return data
