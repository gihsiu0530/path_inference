"""FF dataset additions: calibrated front camera and route polyline, without HD maps."""

import numpy as np
import torch
from pyquaternion import Quaternion

import stp3.datas.NuscenesData as nuscenes_data_base
from stp3.datas.NuscenesData_change import FuturePredictionDataset as BaseFuturePredictionDataset


class FuturePredictionDataset(BaseFuturePredictionDataset):
    """Add only the data required by codex_pure_ASAP_ff.

    The existing 224 inputs use a keep-aspect-ratio resize followed by a center
    crop.  Intrinsics below reproduce that operation exactly.
    """

    def __init__(self, nusc, is_train, cfg):
        # The inherited dataset constructor eagerly opens every nuScenes map,
        # even when __getitem__ never requests HD-map labels. Replace only that
        # constructor hook while FF is initialized, then restore the module.
        original_get_nusc_maps = nuscenes_data_base.get_nusc_maps
        nuscenes_data_base.get_nusc_maps = lambda _map_folder: {}
        try:
            super().__init__(nusc, is_train, cfg)
        finally:
            nuscenes_data_base.get_nusc_maps = original_get_nusc_maps
        self.nusc_maps = {}

    def _front_calibration_224(self, rec, target_size=224):
        cam_name = self.cfg.IMAGE.NAMES[0]
        camera_sample = self.nusc.get("sample_data", rec["data"][cam_name])
        calibrated = self.nusc.get("calibrated_sensor", camera_sample["calibrated_sensor_token"])
        intrinsic = np.asarray(calibrated["camera_intrinsic"], dtype=np.float32).copy()

        original_h = int(camera_sample.get("height", self.cfg.IMAGE.ORIGINAL_HEIGHT))
        original_w = int(camera_sample.get("width", self.cfg.IMAGE.ORIGINAL_WIDTH))
        scale = float(target_size) / float(min(original_h, original_w))
        resized_h = int(round(original_h * scale))
        resized_w = int(round(original_w * scale))
        crop_top = max(0, (resized_h - target_size) // 2)
        crop_left = max(0, (resized_w - target_size) // 2)
        intrinsic[0, :] *= scale
        intrinsic[1, :] *= scale
        intrinsic[0, 2] -= float(crop_left)
        intrinsic[1, 2] -= float(crop_top)

        lidar_sample = self.nusc.get("sample_data", rec["data"]["LIDAR_TOP"])
        lidar_pose = self.nusc.get("ego_pose", lidar_sample["ego_pose_token"])
        yaw = Quaternion(lidar_pose["rotation"]).yaw_pitch_roll[0]
        lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_to_world = np.eye(4, dtype=np.float64)
        lidar_to_world[:3, :3] = lidar_rotation.rotation_matrix
        lidar_to_world[:3, 3] = np.asarray(lidar_pose["translation"], dtype=np.float64)

        camera_pose = self.nusc.get("ego_pose", camera_sample["ego_pose_token"])
        pose_rotation_inv = Quaternion(camera_pose["rotation"]).inverse.rotation_matrix
        world_to_camera_pose = np.eye(4, dtype=np.float64)
        world_to_camera_pose[:3, :3] = pose_rotation_inv
        world_to_camera_pose[:3, 3] = -pose_rotation_inv @ np.asarray(camera_pose["translation"], dtype=np.float64)

        sensor_to_pose = np.eye(4, dtype=np.float64)
        sensor_to_pose[:3, :3] = Quaternion(calibrated["rotation"]).rotation_matrix
        sensor_to_pose[:3, 3] = np.asarray(calibrated["translation"], dtype=np.float64)
        pose_to_sensor = np.linalg.inv(sensor_to_pose)
        lidar_to_sensor = pose_to_sensor @ world_to_camera_pose @ lidar_to_world
        sensor_to_lidar = np.linalg.inv(lidar_to_sensor).astype(np.float32)
        return torch.from_numpy(intrinsic), torch.from_numpy(sensor_to_lidar)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        calibration_k = []
        calibration_t = []
        for sequence_index in self.indices[index][: self.receptive_field]:
            record = self.ixes[int(sequence_index)]
            intrinsic, extrinsic = self._front_calibration_224(record)
            calibration_k.append(intrinsic.unsqueeze(0))
            calibration_t.append(extrinsic.unsqueeze(0))
        data["intrinsics"] = torch.stack(calibration_k, dim=0)  # T,1,3,3
        data["extrinsics"] = torch.stack(calibration_t, dim=0)  # T,1,4,4

        # No nuScenes map query here.  The trainer interprets this empty tensor
        # as no external drivable mask, so FF uses image-projected road evidence
        # plus the cheap GT trajectory corridor below.
        data["hdmap"] = torch.empty(0, dtype=torch.float32)

        # nuScenes has no route planner target in the original dataset.  The
        # expert future is used as its training-time local route polyline.
        # Deployment should pass the navigation route in this same P,2 format.
        route = data["gt_trajectory"][..., :2].clone()
        data["route_polyline"] = route
        data["target_point"] = route
        return data
