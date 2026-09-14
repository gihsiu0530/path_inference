import os

import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion

from stp3.datas.NuscenesData_change import FuturePredictionDataset as BaseFuturePredictionDataset


class FuturePredictionDataset(BaseFuturePredictionDataset):
    """GPVL variant that adds single-camera calibration for 224x224 pseudo-BEV lifting."""

    def _teacher_feature_candidates(self, sample_token):
        root = str(getattr(self.cfg, "GPVL_TEACHER_FEATURE_ROOT", "") or "")
        if not root:
            return []
        return [
            os.path.join(root, self.mode, f"{sample_token}.npy"),
            os.path.join(root, f"{sample_token}.npy"),
            os.path.join(root, "det_motion_map_features_base", self.mode, f"{sample_token}.npy"),
            os.path.join(root, "det_motion_map_features_tiny", self.mode, f"{sample_token}.npy"),
        ]

    def _load_teacher_scene_feature(self, sample_token):
        for path in self._teacher_feature_candidates(sample_token):
            if not os.path.exists(path):
                continue
            feature = np.load(path, allow_pickle=True)
            feature = np.asarray(feature, dtype=np.float32)
            if feature.ndim == 0:
                continue
            if feature.ndim > 1:
                feature = feature.reshape(-1, feature.shape[-1]).mean(axis=0)
            teacher_dim = int(getattr(self.cfg, "GPVL_TEACHER_FEATURE_DIM", 256))
            feature = torch.from_numpy(feature.astype(np.float32))
            if feature.numel() != teacher_dim:
                feature = F.adaptive_avg_pool1d(feature.view(1, 1, -1), teacher_dim).view(-1)
            return feature
        teacher_dim = int(getattr(self.cfg, "GPVL_TEACHER_FEATURE_DIM", 256))
        return torch.zeros(teacher_dim, dtype=torch.float32)

    def _get_224_calibration(self, rec, cam):
        lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
        yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
        lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_translation = np.array(lidar_pose['translation'])[:, None]
        lidar_to_world = np.vstack([
            np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
            np.array([0, 0, 0, 1]),
        ])

        camera_sample = self.nusc.get('sample_data', rec['data'][cam])
        car_egopose = self.nusc.get('ego_pose', camera_sample['ego_pose_token'])
        egopose_rotation = Quaternion(car_egopose['rotation']).inverse
        egopose_translation = -np.array(car_egopose['translation'])[:, None]
        world_to_car_egopose = np.vstack([
            np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
            np.array([0, 0, 0, 1]),
        ])

        sensor_sample = self.nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
        intrinsic = torch.tensor(sensor_sample['camera_intrinsic'], dtype=torch.float32)
        input_size = int(getattr(self.cfg, "CLIP_INPUT_SIZE", 224))
        sx = input_size / float(self.cfg.IMAGE.ORIGINAL_WIDTH)
        sy = input_size / float(self.cfg.IMAGE.ORIGINAL_HEIGHT)
        intrinsic = intrinsic.clone()
        intrinsic[0, 0] *= sx
        intrinsic[0, 2] *= sx
        intrinsic[1, 1] *= sy
        intrinsic[1, 2] *= sy

        sensor_rotation = Quaternion(sensor_sample['rotation'])
        sensor_translation = np.array(sensor_sample['translation'])[:, None]
        car_egopose_to_sensor = np.vstack([
            np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
            np.array([0, 0, 0, 1]),
        ])
        car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)
        lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
        sensor_to_lidar = torch.from_numpy(np.linalg.inv(lidar_to_sensor)).float()
        return intrinsic, sensor_to_lidar

    def __getitem__(self, index):
        data = super().__getitem__(index)

        cam = self.cfg.IMAGE.NAMES[0]
        intrinsics = []
        extrinsics = []
        for i_idx in range(self.receptive_field):
            idx_i = self.indices[index][i_idx]
            rec_i = self.ixes[idx_i]
            intrinsic, sensor_to_lidar = self._get_224_calibration(rec_i, cam)
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(sensor_to_lidar.unsqueeze(0).unsqueeze(0))

        data['intrinsics'] = torch.cat(intrinsics, dim=0)
        data['extrinsics'] = torch.cat(extrinsics, dim=0)

        present_idx = self.indices[index][self.receptive_field - 1]
        present_rec = self.ixes[present_idx]
        data['sample_token'] = present_rec['token']
        data['teacher_gpv_feature'] = self._load_teacher_scene_feature(present_rec['token'])
        try:
            data['hdmap'] = self.voxelize_hd_map(present_rec).squeeze(0).float()
        except Exception:
            data['hdmap'] = torch.empty(0)

        return data
