import os

import cv2
import numpy as np
import torch

from stp3.datas.NuscenesData import (
    FuturePredictionDataset as BaseFuturePredictionDataset,
    build_depth_path,
    build_seg2d_path,
)


class FuturePredictionDataset(BaseFuturePredictionDataset):
    """
    Dataset variant for codex_change experiments.

    It keeps the base NuscenesData behavior and adds seg_id_224_seq loaded from
    *_cls4_224.npy class-id maps. Other training scripts that import
    stp3.datas.NuscenesData are unaffected.
    """

    def __getitem__(self, index):
        data = super().__getitem__(index)

        T_rf = self.receptive_field
        ego_history_frames = int(getattr(self.cfg, "EGO_HISTORY_FRAMES", T_rf))
        cam = self.cfg.IMAGE.NAMES[0]
        seg_id_seq_list = []
        depth_seq_list = []
        depth_root = getattr(self.cfg.LIFT, "DEPTH_ROOT", "/home/cyc/dataset/nuscenes/trainval/depths")

        for i_idx in range(T_rf):
            idx_i = self.indices[index][i_idx]
            rec_i = self.ixes[idx_i]
            cam_sample_i = self.nusc.get('sample_data', rec_i['data'][cam])
            front_img_path_i = os.path.join(self.dataroot, cam_sample_i['filename'])

            seg2d_path_i = build_seg2d_path(front_img_path_i, self.cfg.SEG2D_ROOT)
            seg224_path_i = seg2d_path_i.replace("/seg2d/", "/seg_cl4_png/").replace(".npy", "_cls4_224.png")
            segid224_path_i = seg224_path_i.replace(".png", ".npy")

            assert os.path.exists(segid224_path_i), f"讀不到 seg_id224：{segid224_path_i}"
            seg_id_seq_list.append(np.load(segid224_path_i).astype(np.uint8))

            depth_path_i = build_depth_path(depth_root, cam_sample_i, cam)
            assert os.path.exists(depth_path_i), f"讀不到 depth224：{depth_path_i}"
            depth_seq_list.append(np.load(depth_path_i).astype(np.float32))

        data['seg_id_224_seq'] = torch.from_numpy(np.stack(seg_id_seq_list, axis=0)).to(torch.uint8)
        data['depth_224_seq'] = torch.from_numpy(np.stack(depth_seq_list, axis=0)).float()

        if ego_history_frames > T_rf:
            present_idx = int(self.indices[index][T_rf - 1])
            present_rec = self.ixes[present_idx]
            ego_list = []
            first_hist_idx = present_idx - (ego_history_frames - 1)
            for hist_i in range(ego_history_frames):
                idx0 = first_hist_idx + hist_i
                idx1 = idx0 + 1
                if hist_i == ego_history_frames - 1:
                    ego = torch.zeros(1, 6, dtype=torch.float32)
                elif idx0 < 0 or idx1 >= len(self.ixes):
                    ego = torch.zeros(1, 6, dtype=torch.float32)
                else:
                    rec0 = self.ixes[idx0]
                    rec1 = self.ixes[idx1]
                    if rec0['scene_token'] == present_rec['scene_token'] and rec1['scene_token'] == present_rec['scene_token']:
                        ego = self.get_egomotion_between(rec0, rec1)
                    else:
                        ego = torch.zeros(1, 6, dtype=torch.float32)
                ego_list.append(ego)
            data['ego_history_egomotion'] = torch.cat(ego_list, dim=0)
        return data
