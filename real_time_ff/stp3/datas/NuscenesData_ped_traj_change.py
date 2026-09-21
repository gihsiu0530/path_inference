import os

import numpy as np
import torch

from stp3.datas.NuscenesData import build_depth_path, build_seg2d_path
from stp3.datas.NuscenesData_ped_traj import FuturePredictionDataset as PedTrajDataset


class FuturePredictionDataset(PedTrajDataset):
    """
    Pedestrian-trajectory dataset variant with additional seg-id/depth inputs
    required by codex_pure / codex_pure_super_ft.
    """

    def __getitem__(self, index):
        data = super().__getitem__(index)

        cam = self.cfg.IMAGE.NAMES[0]
        seg_id_seq_list = []
        depth_seq_list = []
        depth_root = getattr(self.cfg.LIFT, "DEPTH_ROOT", "/home/cyc/dataset/nuscenes/trainval/depths")

        for i_idx in range(self.receptive_field):
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
        return data
