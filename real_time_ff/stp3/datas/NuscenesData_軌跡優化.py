import os
from PIL import Image

import numpy as np
import cv2
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.utils import quaternion_yaw
from stp3.utils.tools import ( gen_dx_bx, get_nusc_maps)

from stp3.utils.geometry import (
    resize_and_crop_image,
    update_intrinsics,
    calculate_birds_eye_view_parameters,
    convert_egopose_to_matrix_numpy,
    pose_vec2mat,
    mat2pose_vec,
    invert_matrix_egopose_numpy,
    get_global_pose
)
from stp3.utils.instance import convert_instance_mask_to_center_and_offset_label
import stp3.utils.sampler as trajectory_sampler

import os

import math
from typing import List, Tuple


cv2.setNumThreads(1)





def build_seg2d_path(front_img_path: str, seg2d_root: str) -> str:
    """
    依照固定規則：SEG2D_ROOT/seq_id/frame_id.npy
    影像檔名：n008-...__CAM_FRONT__1526915245012465.jpg
    """
    stem = os.path.splitext(os.path.basename(front_img_path))[0]
    parts = stem.split("__")
    assert len(parts) >= 3, f"影像檔名不符預期（三段 __ 分隔）：{front_img_path}"
    seq_id  = parts[0]
    frame_id = parts[-1]
    seg_path = os.path.join(seg2d_root, seq_id, frame_id + ".npy")
    assert os.path.exists(seg_path), f"找不到 seg2d 檔案：{seg_path}"
    return seg_path


def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

class FuturePredictionDataset(torch.utils.data.Dataset):
    SAMPLE_INTERVAL = 0.5 #SECOND
    def __init__(self, nusc, is_train, cfg):
        self.nusc = nusc
        self.dataroot = self.nusc.dataroot
        self.nusc_exp = NuScenesExplorer(nusc)
        self.nusc_can = NuScenesCanBus(dataroot=self.dataroot)
        self.is_train = is_train
        self.cfg = cfg

        if self.is_train == 0:
            self.mode = 'train'
        elif self.is_train == 1:
            self.mode = 'val'
        elif self.is_train == 2:
            self.mode = 'test'
        else:
            raise NotImplementedError

        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()
        self.indices = self.get_indices()

        # Image resizing and cropping
        self.augmentation_parameters = self.get_resizing_and_cropping_parameters()

        # Normalising input images
        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Bird's-eye view parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )
        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

        # The number of sampled trajectories
        self.n_samples = self.cfg.PLANNING.SAMPLE_NUM

        # HD-map feature extractor
        self.nusc_maps = get_nusc_maps(self.cfg.DATASET.MAP_FOLDER)
        self.scene2map = {}
        for sce in self.nusc.scene:
            log = self.nusc.get('log', sce['log_token'])
            self.scene2map[sce['name']] = log['location']
        self.save_dir = cfg.DATASET.SAVE_DIR



    # ====== 便宜預篩 + 快速評分 + Traj-NMS 工具 ======

    def _xy_to_bev_idx(self, xy: np.ndarray) -> np.ndarray:
        """
        支援 xy 為 (2,) 或 (T,2)。
        回傳：
        - 若輸入 (2,) → (2,) 的 (row, col)
        - 若輸入 (T,2) → (T,2) 的 (row, col)
        """
        arr = np.asarray(xy)
        if arr.ndim == 1:
            # arr = [x, y]
            pts = (arr[:2] - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]
            pts = np.round(pts).astype(np.int32)
            # 交換軸到 (row=y_idx, col=x_idx)
            pts = pts[[1, 0]]
            return pts
        elif arr.ndim == 2:
            # arr = [[x,y], ...]
            pts = (arr - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]
            pts = np.round(pts).astype(np.int32)
            pts[:, [0, 1]] = pts[:, [1, 0]]
            return pts
        else:
            raise ValueError(f"_xy_to_bev_idx expects (2,) or (T,2), got shape={arr.shape}")


    def _traj_kinematics_ok(self, traj_xy: np.ndarray, dt: float,
                             acc_max: float, jerk_max: float, omega_max: float) -> bool:
        """
        traj_xy: (T,2) 降採樣後 (0.5s 間隔) 的 x,y
        """
        v = np.diff(traj_xy, axis=0) / dt                         # (T-1,2)
        a = np.diff(v, axis=0) / dt                               # (T-2,2)
        j = np.diff(a, axis=0) / dt                               # (T-3,2)
        # 角速度（由切向方向變化估計）
        heading = np.arctan2(v[:,1], v[:,0])
        # 或：heading = np.arctan2(v[:,1], v[:,0] + 1e-9)
        dpsi = np.diff(np.unwrap(heading))                         # (T-2,)
        omega = np.abs(dpsi / dt)

        if a.size > 0 and np.nanmax(np.linalg.norm(a, axis=1)) > acc_max:
            return False
        if j.size > 0 and np.nanmax(np.linalg.norm(j, axis=1)) > jerk_max:
            return False
        if omega.size > 0 and np.nanmax(omega) > omega_max:
            return False
        return True

    def _off_drivable_ratio(self, traj_xy: np.ndarray, seg_bev: torch.Tensor, ped_bev: torch.Tensor) -> float:
        """
        使用未來多幀 BEV 快速查表。
        seg_bev: (Tf, 1, H, W) 可行駛區(1)/背景(0)
        ped_bev: (Tf, 1, H, W) 行人(1)/非行人(0)
        回傳在「不可行駛或行人」上的命中比例。
        """
        Tf = seg_bev.shape[0]
        T  = traj_xy.shape[0]
        t_align = min(T, Tf)  # 與標註對齊的長度
        hits = 0
        total = 0
        H, W = seg_bev.shape[-2], seg_bev.shape[-1]
        for t in range(t_align):
            uv = self._xy_to_bev_idx(traj_xy[t])  # (2,) → (row,col)
            r, c = int(uv[0]), int(uv[1])
            if 0 <= r < H and 0 <= c < W:
                # 不可行駛 = seg==0； 或 命中行人 ped==1
                off = (seg_bev[t, 0, r, c].item() == 0) or (ped_bev[t, 0, r, c].item() == 1)
                hits += int(off)
                total += 1
        return (hits / max(1, total))

    def _fast_score(self, traj_xy: np.ndarray,
                    seg_bev: torch.Tensor, ped_bev: torch.Tensor,
                    lane_adherence: float = 0.0) -> float:
        """
        一個可解釋的快速分數（越大越好）：
        progress: 末端前進距離的投影 (x_end)
        lane_adherence: 由 HD-map/路徑先驗給的貼合度（此處留作參數）
        collision_proxy: 以 off_drivable_ratio 當作碰撞代理
        jerk_cost: 以平均 jerk 範數近似舒適度懲罰
        """
        dt = self.SAMPLE_INTERVAL
        # x_end = traj_xy[-1, 1] if False else traj_xy[-1, 0]  # 視你座標系定義，若 x 朝前就用 x
        y_end = traj_xy[-1, 1]

        # jerk 近似
        v = np.diff(traj_xy, axis=0)/dt
        a = np.diff(v, axis=0)/dt
        j = np.diff(a, axis=0)/dt
        jerk_cost = 0.0 if j.size == 0 else float(np.mean(np.linalg.norm(j, axis=1)))

        off_ratio = self._off_drivable_ratio(traj_xy, seg_bev, ped_bev)

        score = (
            + 1.0 * y_end
            + 0.6 * lane_adherence
            - 0.5 * off_ratio
            - 0.4 * jerk_cost
        )
        return float(score)

    def _traj_distance(self, A: np.ndarray, B: np.ndarray) -> float:
        """平均 L2 距離（你也可以換成 DTW/Fréchet）"""
        T = min(len(A), len(B))
        if T == 0: return 1e9
        return float(np.mean(np.linalg.norm(A[:T] - B[:T], axis=1)))

    def _traj_nms(self, trajs_xy: List[np.ndarray], scores: List[float], tau: float, k_target: int) -> List[int]:
        """回傳被保留的 index 清單"""
        order = np.argsort(scores)[::-1].tolist()
        keep = []
        for idx in order:
            if len(keep) >= k_target:
                break
            ok = True
            for j in keep:
                if self._traj_distance(trajs_xy[idx], trajs_xy[j]) < tau:
                    ok = False
                    break
            if ok:
                keep.append(idx)
        return keep

    def _sample_diverse_pool(self, v0: float, Kappa: float, T0: np.ndarray, N0: np.ndarray,
                             tt: np.ndarray, N_pool: int) -> np.ndarray:
        """
        多樣化超額取樣（回傳 fine 軌跡，shape: (N_pool, len(tt), D)）
        """
        v_scales   = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
        k_multip   = np.array([0.0, -0.5, +0.5, -1.0, +1.0, -1.5, +1.5])
        pairs = [(max(0.0, v0*vs), Kappa*(1.0 + km)) for vs in v_scales for km in k_multip]
        if len(pairs) < N_pool:
            # 不足時循環補齊
            reps = (N_pool + len(pairs) - 1)//len(pairs)
            pairs = (pairs * reps)[:N_pool]
        else:
            pairs = pairs[:N_pool]

        trajs = []
        for v_i, k_i in pairs:
            # fine = trajectory_sampler.sample(v_i, k_i, T0, N0, tt, 1)  # 每次 1 條
            # trajs.append(fine)
            M_each = 5  # 或 3/7 都可，≥3 即可避免類別為 0
            fine = trajectory_sampler.sample(v_i, k_i, T0, N0, tt, M_each)  # (M_each, T, 3)
            pick = np.random.randint(fine.shape[0])                         # 選 1 條代表這個 (v, κ)
            trajs.append(fine[pick:pick+1])                                  # 保持 shape: (1,T,3)

        return np.concatenate(trajs, axis=0)  # (N_pool, len(tt), D)


    def get_scenes(self):
        # filter by scene split
        split = {'v1.0-trainval': {0: 'train', 1: 'val', 2: 'test'},
                 'v1.0-mini': {0: 'mini_train', 1: 'mini_val'},}[
            self.nusc.version
        ][self.is_train]

        blacklist = [419] + self.nusc_can.can_blacklist  # # scene-0419 does not have vehicle monitor data
        blacklist = ['scene-' + str(scene_no).zfill(4) for scene_no in blacklist]

        scenes = create_splits_scenes()[split][:]
        for scene_no in blacklist:
            if scene_no in scenes:
                scenes.remove(scene_no)

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_indices(self):
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)

    def get_resizing_and_cropping_parameters(self):
        original_height, original_width = self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH
        final_height, final_width = self.cfg.IMAGE.FINAL_DIM

        resize_scale = self.cfg.IMAGE.RESIZE_SCALE
        resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = self.cfg.IMAGE.TOP_CROP
        crop_w = int(max(0, (resized_width - final_width) / 2))
        # Left, top, right, bottom crops.
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        if resized_width != final_width:
            print('Zero padding left and right parts of the image.')
        if crop_h + final_height != resized_height:
            print('Zero padding bottom part of the image.')

        return {'scale_width': resize_scale,
                'scale_height': resize_scale,
                'resize_dims': resize_dims,
                'crop': crop,
                }

    def get_input_data(self, rec):
        """
        Parameters
        ----------
            rec: nuscenes identifier for a given timestamp

        Returns
        -------
            images: torch.Tensor<float> (N, 3, H, W)
            intrinsics: torch.Tensor<float> (3, 3)
            extrinsics: torch.Tensor(N, 4, 4)
        """
        images = []
        intrinsics = []
        extrinsics = []
        depths = []
        cameras = self.cfg.IMAGE.NAMES

        # The extrinsics we want are from the camera sensor to "flat egopose" as defined
        # https://github.com/nutonomy/nuscenes-devkit/blob/9b492f76df22943daf1dc991358d3d606314af27/python-sdk/nuscenes/nuscenes.py#L279
        # which corresponds to the position of the lidar.
        # This is because the labels are generated by projecting the 3D bounding box in this lidar's reference frame.

        # From lidar egopose to world.
        lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
        yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
        lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_translation = np.array(lidar_pose['translation'])[:, None]
        lidar_to_world = np.vstack([
            np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
            np.array([0, 0, 0, 1])
        ])

        for cam in cameras:
            camera_sample = self.nusc.get('sample_data', rec['data'][cam])

            # Transformation from world to egopose
            car_egopose = self.nusc.get('ego_pose', camera_sample['ego_pose_token'])
            egopose_rotation = Quaternion(car_egopose['rotation']).inverse
            egopose_translation = -np.array(car_egopose['translation'])[:, None]
            world_to_car_egopose = np.vstack([
                np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
                np.array([0, 0, 0, 1])
            ])

            # From egopose to sensor
            sensor_sample = self.nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
            intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
            sensor_rotation = Quaternion(sensor_sample['rotation'])
            sensor_translation = np.array(sensor_sample['translation'])[:, None]
            car_egopose_to_sensor = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                np.array([0, 0, 0, 1])
            ])
            car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

            # Combine all the transformation.
            # From sensor to lidar.
            lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
            sensor_to_lidar = torch.from_numpy(np.linalg.inv(lidar_to_sensor)).float()

            # Load image
            image_filename = os.path.join(self.dataroot, camera_sample['filename'])
            img = Image.open(image_filename)
            # Resize and crop
            img = resize_and_crop_image(
                img, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
            )
            # Normalise image
            normalised_img = self.normalise_image(img)

            # Combine resize/cropping in the intrinsics
            top_crop = self.augmentation_parameters['crop'][1]
            left_crop = self.augmentation_parameters['crop'][0]
            intrinsic = update_intrinsics(
                intrinsic, top_crop, left_crop,
                scale_width=self.augmentation_parameters['scale_width'],
                scale_height=self.augmentation_parameters['scale_height']
            )

            # Get Depth
            # Depth data should under the dataroot path 
            if self.cfg.LIFT.GT_DEPTH:
                base_root = os.path.join(self.dataroot, 'depths') 
                filename = os.path.basename(camera_sample['filename']).split('.')[0] + '.npy'
                depth_file_name = os.path.join(base_root, cam, 'npy', filename)
                depth = torch.from_numpy(np.load(depth_file_name)).unsqueeze(0).unsqueeze(0)
                depth = F.interpolate(depth, scale_factor=self.cfg.IMAGE.RESIZE_SCALE, mode='bilinear')
                depth = depth.squeeze()
                crop = self.augmentation_parameters['crop']
                depth = depth[crop[1]:crop[3], crop[0]:crop[2]]
                depth = torch.round(depth)
                depths.append(depth.unsqueeze(0).unsqueeze(0))

            images.append(normalised_img.unsqueeze(0).unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(sensor_to_lidar.unsqueeze(0).unsqueeze(0))

        images, intrinsics, extrinsics = (torch.cat(images, dim=1),
                                          torch.cat(intrinsics, dim=1),
                                          torch.cat(extrinsics, dim=1)
                                          )
        if len(depths) > 0:
            depths = torch.cat(depths, dim=1)

        return images, intrinsics, extrinsics, depths

    def _get_top_lidar_pose(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot

    def get_depth_from_lidar(self, lidar_sample, cam_sample):
        points, coloring, im = self.nusc_exp.map_pointcloud_to_image(lidar_sample, cam_sample)
        tmp_cam = np.zeros((self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH))
        points = points.astype(np.int)
        tmp_cam[points[1, :], points[0,:]] = coloring
        tmp_cam = torch.from_numpy(tmp_cam).unsqueeze(0).unsqueeze(0)
        tmp_cam = F.interpolate(tmp_cam, scale_factor=self.cfg.IMAGE.RESIZE_SCALE, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        tmp_cam = tmp_cam.squeeze()
        crop = self.augmentation_parameters['crop']
        tmp_cam = tmp_cam[crop[1]:crop[3], crop[0]:crop[2]]
        tmp_cam = torch.round(tmp_cam)
        return tmp_cam


    def get_birds_eye_view_label(self, rec, instance_map, in_pred):
        translation, rotation = self._get_top_lidar_pose(rec)
        segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        pedestrian = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        # Background is ID 0
        instance = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))

        for annotation_token in rec['anns']:
            # Filter out all non vehicle instances
            annotation = self.nusc.get('sample_annotation', annotation_token)

            if self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1 and in_pred is False:
                continue
            if in_pred is True and annotation['instance_token'] not in instance_map:
                continue

            # NuScenes filter
            if 'vehicle' in annotation['category_name']:
                if annotation['instance_token'] not in instance_map:
                    instance_map[annotation['instance_token']] = len(instance_map) + 1
                instance_id = instance_map[annotation['instance_token']]
                poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
                cv2.fillPoly(instance, [poly_region], instance_id)
                cv2.fillPoly(segmentation, [poly_region], 1.0)
            elif 'human' in annotation['category_name']:
                if annotation['instance_token'] not in instance_map:
                    instance_map[annotation['instance_token']] = len(instance_map) + 1
                poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
                cv2.fillPoly(pedestrian, [poly_region], 1.0)


        return segmentation, instance, pedestrian, instance_map

    def _get_poly_region_in_image(self, instance_annotation, ego_translation, ego_rotation):
        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(ego_translation)
        box.rotate(ego_rotation)

        pts = box.bottom_corners()[:2].T
        pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]

        z = box.bottom_corners()[2, 0]
        return pts, z

    def get_label(self, rec, instance_map, in_pred):
        segmentation_np, instance_np, pedestrian_np, instance_map = \
            self.get_birds_eye_view_label(rec, instance_map, in_pred)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0)
        instance = torch.from_numpy(instance_np).long().unsqueeze(0)
        pedestrian = torch.from_numpy(pedestrian_np).long().unsqueeze(0).unsqueeze(0)

        return segmentation, instance, pedestrian, instance_map

    def get_future_egomotion(self, rec, index):
        rec_t0 = rec

        # Identity
        future_egomotion = np.eye(4, dtype=np.float32)

        if index < len(self.ixes) - 1:
            rec_t1 = self.ixes[index + 1]

            if rec_t0['scene_token'] == rec_t1['scene_token']:
                egopose_t0 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t0['data']['LIDAR_TOP'])['ego_pose_token']
                )
                egopose_t1 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t1['data']['LIDAR_TOP'])['ego_pose_token']
                )

                egopose_t0 = convert_egopose_to_matrix_numpy(egopose_t0)
                egopose_t1 = convert_egopose_to_matrix_numpy(egopose_t1)

                future_egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
                future_egomotion[3, :3] = 0.0
                future_egomotion[3, 3] = 1.0

        future_egomotion = torch.Tensor(future_egomotion).float()

        # Convert to 6DoF vector
        future_egomotion = mat2pose_vec(future_egomotion)
        return future_egomotion.unsqueeze(0)

    def get_trajectory_seed(self, rec=None, sample_indice=None):
        """
        回傳軌跡取樣所需的 seed 參數：v0, Kappa, T0, N0, tt
        （原本 get_trajectory_sampling 直接產生 N 條，現在改成回傳 seed）
        """
        if rec is None and sample_indice is None:
            raise ValueError("No valid input rec or token")
        if rec is None and sample_indice is not None:
            rec = self.ixes[sample_indice]

        ref_scene = self.nusc.get("scene", rec['scene_token'])
        pose_msgs = self.nusc_can.get_messages(ref_scene['name'],'pose')
        pose_uts = [msg['utime'] for msg in pose_msgs]
        steer_msgs = self.nusc_can.get_messages(ref_scene['name'], 'steeranglefeedback')
        steer_uts = [msg['utime'] for msg in steer_msgs]

        ref_utime = rec['timestamp']
        pose_index = locate_message(pose_uts, ref_utime)
        pose_data = pose_msgs[pose_index]
        steer_index = locate_message(steer_uts, ref_utime)
        steer_data = steer_msgs[steer_index]

        v0 = pose_data["vel"][0]  # m/s
        steering = steer_data["value"]

        location = self.scene2map[ref_scene['name']]
        flip_flag = True if location.startswith('singapore') else False
        if flip_flag:
            steering *= -1
        Kappa = 2 * steering / 2.588

        T0 = np.array([0.0, 1.0])
        N0 = np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])

        t_start = 0.0
        t_end = self.cfg.N_FUTURE_FRAMES * self.SAMPLE_INTERVAL
        t_interval = self.SAMPLE_INTERVAL / 10.0
        tt = np.arange(t_start, t_end + t_interval, t_interval)
        return v0, Kappa, T0, N0, tt


    def voxelize_hd_map(self, rec):
        dx, bx, _ = gen_dx_bx(self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND)
        stretch = [self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1]]
        dx, bx = dx[:2].numpy(), bx[:2].numpy()

        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1,0], rot[0,0]) # in radian
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        box_coords = (
            center[0],
            center[1],
            stretch[0]*2,
            stretch[1]*2
        ) # (x_center, y_center, width, height)
        canvas_size = (
                int(self.cfg.LIFT.X_BOUND[1] * 2 / self.cfg.LIFT.X_BOUND[2]),
                int(self.cfg.LIFT.Y_BOUND[1] * 2 / self.cfg.LIFT.Y_BOUND[2])
        )

        elements = self.cfg.SEMANTIC_SEG.HDMAP.ELEMENTS
        hd_features = self.nusc_maps[map_name].get_map_mask(box_coords, rot * 180 / np.pi , elements, canvas_size=canvas_size)
        #traffic = self.hd_traffic_light(map_name, center, stretch, dx, bx, canvas_size)
        #return torch.from_numpy(np.concatenate((hd_features, traffic), axis=0)[None]).float()
        hd_features = torch.from_numpy(hd_features[None]).float()
        hd_features = torch.transpose(hd_features,-2,-1) # (y,x) replace horizontal and vertical coordinates
        return hd_features

    def hd_traffic_light(self, map_name, center, stretch, dx, bx, canvas_size):

        roads = np.zeros(canvas_size)
        my_patch = (
            center[0] - stretch[0],
            center[1] - stretch[1],
            center[0] + stretch[0],
            center[1] + stretch[1],
        )
        tl_token = self.nusc_maps[map_name].get_records_in_patch(my_patch, ['traffic_light'], mode='intersect')['traffic_light']
        polys = []
        for token in tl_token:
            road_token =self.nusc_maps[map_name].get('traffic_light', token)['from_road_block_token']
            pt = self.nusc_maps[map_name].get('road_block', road_token)['polygon_token']
            polygon = self.nusc_maps[map_name].extract_polygon(pt)
            polys.append(np.array(polygon.exterior.xy).T)

        def get_rot(h):
            return torch.Tensor([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])
        # convert to local coordinates in place
        rot = get_rot(np.arctan2(center[3], center[2])).T
        for rowi in range(len(polys)):
            polys[rowi] -= center[:2]
            polys[rowi] = np.dot(polys[rowi], rot)

        for la in polys:
            pts = (la - bx) / dx
            pts = np.int32(np.around(pts))
            cv2.fillPoly(roads, [pts], 1)

        return roads[None]

    def get_gt_trajectory(self, rec, ref_index):
        n_output = self.cfg.N_FUTURE_FRAMES
        gt_trajectory = np.zeros((n_output+1, 3), np.float64)

        egopose_cur = get_global_pose(rec, self.nusc, inverse=True)

        for i in range(n_output+1):
            index = ref_index + i
            if index < len(self.ixes):
                rec_future = self.ixes[index]

                egopose_future = get_global_pose(rec_future, self.nusc, inverse=False)

                egopose_future = egopose_cur.dot(egopose_future)
                theta = quaternion_yaw(Quaternion(matrix=egopose_future))

                origin = np.array(egopose_future[:3, 3])

                gt_trajectory[i, :] = [origin[0], origin[1], theta]

        if gt_trajectory[-1][0] >= 2:
            command = 'RIGHT'
        elif gt_trajectory[-1][0] <= -2:
            command = 'LEFT'
        else:
            command = 'FORWARD'

        return gt_trajectory, command

    def get_routed_map(self, gt_points):
        dx, bx, _ = gen_dx_bx(self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND)
        dx, bx = dx[:2].numpy(), bx[:2].numpy()

        canvas_size = (
            int(self.cfg.LIFT.X_BOUND[1] * 2 / self.cfg.LIFT.X_BOUND[2]),
            int(self.cfg.LIFT.Y_BOUND[1] * 2 / self.cfg.LIFT.Y_BOUND[2])
        )

        roads = np.zeros(canvas_size)
        W = 1.85
        pts = np.array([
            [-4.084 / 2. + 0.5, W / 2.],
            [4.084 / 2. + 0.5, W / 2.],
            [4.084 / 2. + 0.5, -W / 2.],
            [-4.084 / 2. + 0.5, -W / 2.],
        ])
        pts = (pts - bx) / dx
        pts[:, [0, 1]] = pts[:, [1, 0]]

        pts = np.int32(np.around(pts))
        cv2.fillPoly(roads, [pts], 1)

        gt_points = gt_points[:-1].numpy()
        # 坐标原点在左上角
        target = pts.copy()
        target[:,0] = pts[:,0] + gt_points[0] / dx[0]
        target[:,1] = pts[:,1] - gt_points[1] / dx[1]
        target = np.int32(np.around(target))
        cv2.fillPoly(roads, [target], 1)
        return roads

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns
        -------
            data: dict with the following keys:
                image: torch.Tensor<float> (T, N, 3, H, W)
                    normalised cameras images with T the sequence length, and N the number of cameras.
                intrinsics: torch.Tensor<float> (T, N, 3, 3)
                    intrinsics containing resizing and cropping parameters.
                extrinsics: torch.Tensor<float> (T, N, 4, 4)
                    6 DoF pose from world coordinates to camera coordinates.
                segmentation: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                    (H_bev, W_bev) are the pixel dimensions in bird's-eye view.
                instance: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                centerness: torch.Tensor<float> (T, 1, H_bev, W_bev)
                offset: torch.Tensor<float> (T, 2, H_bev, W_bev)
                flow: torch.Tensor<float> (T, 2, H_bev, W_bev)
                future_egomotion: torch.Tensor<float> (T, 6)
                    6 DoF egomotion t -> t+1

        """
        data = {}
        keys = ['image', 'intrinsics', 'extrinsics', 'depths',
                'segmentation', 'instance', 'centerness', 'offset', 'flow', 'pedestrian',
                'future_egomotion', 'hdmap', 'gt_trajectory', 'indices',
                ]
        for key in keys:
            data[key] = []

        instance_map = {}
        # Loop over all the frames in the sequence.
        data['image']      = torch.empty(0)
        data['intrinsics'] = torch.empty(0)
        data['extrinsics'] = torch.empty(0)
        data['depths']     = torch.empty(0)   # 即使 GT_DEPTH=False 也直接給空張量即可
        data['hdmap']      = torch.empty(0)   # 如果你不打算用 hdmap

        seed_tuple = None   # ← 迴圈前
        for i, index_t in enumerate(self.indices[index]):
            if i >= self.receptive_field:
                in_pred = True
            else:
                in_pred = False
            rec = self.ixes[index_t]

            # if i < self.receptive_field:
            #     images, intrinsics, extrinsics, depths = self.get_input_data(rec)
            #     data['image'].append(images)
            #     data['intrinsics'].append(intrinsics)
            #     data['extrinsics'].append(extrinsics)
            #     data['depths'].append(depths)

            # if i < self.receptive_field:
            # images, intrinsics, extrinsics, depths = self.get_input_data(rec)
            # 在 for i ... 回圈結束之後、cat 之前，加上：
            


            segmentation, instance, pedestrian, instance_map = self.get_label(rec, instance_map, in_pred)

            future_egomotion = self.get_future_egomotion(rec, index_t)
            # hd_map_feature = self.voxelize_hd_map(rec)

            data['segmentation'].append(segmentation)
            data['instance'].append(instance)
            data['pedestrian'].append(pedestrian)
            data['future_egomotion'].append(future_egomotion)
            data['indices'].append(index_t)

            if i == self.cfg.TIME_RECEPTIVE_FIELD-1:
                gt_trajectory, command = self.get_gt_trajectory(rec, index_t)
                data['gt_trajectory'] = torch.from_numpy(gt_trajectory).float()
                data['command'] = command
                # 保留 present 的 rec 與 seed，用於之後批次產生候選
                seed_tuple = self.get_trajectory_seed(rec)

        # for key, value in data.items():
        #     if key in ['image', 'intrinsics', 'extrinsics', 'depths', 'segmentation', 'instance', 'future_egomotion', 'hdmap', 'pedestrian']:
        #         if key == 'depths' and self.cfg.LIFT.GT_DEPTH is False:
        #             continue
        #         data[key] = torch.cat(value, dim=0)

        # 只把這些會是 Tensor list 的做 cat
        for key in ['segmentation', 'instance', 'pedestrian', 'future_egomotion']:
            # 這四個在上面的 for i 回圈裡有 append(Tensor)
            data[key] = torch.cat(data[key], dim=0)

        # 其它 keys（image / intrinsics / extrinsics / depths / hdmap）
        # 已在上面直接設成空張量，不要再 cat

                # ====== 在這裡進行：超額取樣 → 預篩 → 快速評分 → Traj-NMS → 取 30 ======
        # 超額取樣池大小與參數（可從 cfg 取，無則用預設）
        N_target = int(self.n_samples)                       # 你的最終 N（例如 30）
        POOL_MULT = int(getattr(self.cfg.PLANNING, "POOL_MULT", 6))
        N_pool    = max(N_target, POOL_MULT * N_target)      # 例如 150
        TOPK_PRE  = int(getattr(self.cfg.PLANNING, "TOPK_PRE", 5*N_target))  # 例如 60
        TAU_NMS   = float(getattr(self.cfg.PLANNING, "TAU_NMS", 1.0))        # m

        # 運動學限制（保守起手）
        ACC_MAX   = float(getattr(self.cfg.PLANNING, "ACC_MAX", 5))
        JERK_MAX  = float(getattr(self.cfg.PLANNING, "JERK_MAX", 5))
        OMEGA_MAX = float(getattr(self.cfg.PLANNING, "OMEGA_MAX", 1.2))  # rad/s
        OFF_MAX   = float(getattr(self.cfg.PLANNING, "OFF_DRIVABLE_RATIO_MAX", 0.1))

        # 準備「未來幀」的 BEV（用於快速查表）
        rf = self.receptive_field
        seg_future = data['segmentation'][rf:]    # (Tf,1,H,W)
        ped_future = data['pedestrian'][rf:]      # (Tf,1,H,W)

        # 1) 超額取樣（fine: 0.05s 間隔）→ 2) 降採樣回 0.5s
        v0, Kappa, T0, N0, tt = seed_tuple
        traj_pool_fine = self._sample_diverse_pool(v0, Kappa, T0, N0, tt, N_pool=N_pool)
        traj_pool = traj_pool_fine[:, ::10, :2]   # (N_pool, T, 2) 只取 x,y

        # 3) 便宜預篩
        dt = self.SAMPLE_INTERVAL
        keep_xy   = []
        keep_idx  = []
        for i_tr, tr in enumerate(traj_pool):
            if not self._traj_kinematics_ok(tr, dt, ACC_MAX, JERK_MAX, OMEGA_MAX):
                continue
            off_ratio = self._off_drivable_ratio(tr, seg_future, ped_future)
            if off_ratio > OFF_MAX:
                continue
            keep_xy.append(tr)
            keep_idx.append(i_tr)

        if len(keep_xy) == 0:
            # 保底：不通過就放寬到僅運動學
            for i_tr, tr in enumerate(traj_pool):
                if self._traj_kinematics_ok(tr, dt, ACC_MAX, JERK_MAX, OMEGA_MAX):
                    keep_xy.append(tr)
                    keep_idx.append(i_tr)
            if len(keep_xy) == 0:
                # 還是不行就直接用原池
                keep_xy  = [traj_pool[i] for i in range(len(traj_pool))]
                keep_idx = list(range(len(traj_pool)))

        # 4) 快速評分
        scores = [ self._fast_score(tr, seg_future, ped_future) for tr in keep_xy ]
        order  = np.argsort(scores)[::-1]
        pre_sel_idx = order[:max(TOPK_PRE, N_target)]
        pre_trajs_xy = [ keep_xy[i] for i in pre_sel_idx ]
        pre_scores   = [ scores[i] for i in pre_sel_idx ]

        # 5) Traj-NMS（多樣性）
        keep_final_rel = self._traj_nms(pre_trajs_xy, pre_scores, TAU_NMS, N_target)
        final_xy = [ pre_trajs_xy[i] for i in keep_final_rel ]

        # 不足時，用距離最大化補齊
        # ====== 不足時：改用 k-center greedy + τ 退火 ======
        
        # 先把 pre 集合打包成 (traj, score)

        def _kcenter_fill(cands, selected, need, tau):
            """從 cands (list of (traj_xy, score)) 裡補到 selected，優先最大化與已選集合的最小距離。
            若距離 < tau，先跳過；補不到時逐步放寬 tau。"""
            added = 0
            tau_curr = tau
            tries = 0
            max_tries = 5  # 最多退火 5 次

            # 把已選集合整理成 list
            sel = list(selected)

            while added < need and tries <= max_tries:
                # 依「與已選集合的最小距離」排序（大到小）
                scored = []
                for tr, s in cands:
                    if any(np.allclose(tr, t) for t in sel):
                        continue
                    dmin = np.inf
                    for t in sel:
                        dmin = min(dmin, self._traj_distance(tr, t))
                        if dmin < tau_curr:  # 早停
                            break
                    scored.append((dmin, s, tr))

                # 先按 dmin 排，再以快速分數 s 當次序輔助
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

                progressed = False
                for dmin, s, tr in scored:
                    if dmin >= tau_curr:
                        sel.append(tr)
                        added += 1
                        progressed = True
                        if added >= need:
                            break

                if not progressed:
                    # 沒補到 → 放寬 τ（退火）
                    tau_curr *= 0.85
                    tries += 1

            return sel, added, tau_curr
        pre_pairs = list(zip(pre_trajs_xy, pre_scores))



        need = N_target - len(final_xy)
        if need > 0:
            sel_after, added_cnt, tau_final = _kcenter_fill(pre_pairs, final_xy, need, TAU_NMS)
            final_xy = sel_after[:]
            # print(f"[補齊] k-center 追加 {added_cnt} 條，τ→{tau_final:.2f}，最終 {len(final_xy)}")

        if len(final_xy) < N_target:
            # 先從 keep_xy (還沒進 pre 的) 擴充
            rest_keep = []
            used_pre_set = set([id(x) for x in pre_trajs_xy])
            for tr in keep_xy:
                if id(tr) not in used_pre_set:
                    rest_keep.append((tr, self._fast_score(tr, seg_future, ped_future)))

            need = N_target - len(final_xy)
            if len(rest_keep) > 0 and need > 0:
                final_xy, added_cnt2, tau_final2 = _kcenter_fill(rest_keep, final_xy, need, TAU_NMS*0.9)
                # print(f"[補齊-keep] 再補 {added_cnt2} 條，τ→{tau_final2:.2f}，最終 {len(final_xy)}")

        # 還不足最後再從原始 traj_pool 隨機/最遠挑
        if len(final_xy) < N_target:
            need = N_target - len(final_xy)
            pool_pairs = [(tr, 0.0) for tr in traj_pool]   # 沒有分數就當 0
            final_xy, added_cnt3, tau_final3 = _kcenter_fill(pool_pairs, final_xy, need, TAU_NMS*0.8)
            # print(f"[補齊-pool] 最終再補 {added_cnt3} 條，τ→{tau_final3:.2f}，最終 {len(final_xy)}")

        final_xy = np.stack(final_xy, axis=0)  # (N_target, T, 2) — 保證夠數量

            


        # 若後續模組期望 (N, T, 3)（含 theta），這裡可補一個 heading 估計
        if final_xy.shape[-1] == 2:
            # 估計 theta（以 v 的方向）
            theta_list = []
            for tr in final_xy:
                v = np.diff(tr, axis=0, prepend=tr[:1])
                theta = np.arctan2(v[:,1], v[:,0])
                theta_list.append(theta[..., None])
            theta_arr = np.stack(theta_list, axis=0)  # (N,T,1)
            final_traj = np.concatenate([final_xy, theta_arr], axis=-1)
        else:
            final_traj = final_xy

        data['sample_trajectory'] = torch.from_numpy(final_traj).float()



        data['target_point'] = torch.tensor([0., 0.])
        


        # === 取得 receptive_field 幀（過去 → 現在）的序列 224x224 影像 ===
        T_rf = self.receptive_field
        cam = self.cfg.IMAGE.NAMES[0]  # 例如 CAM_FRONT
        rgb_seq_list, seg_seq_list = [], []

        for i_idx in range(T_rf):
            idx_i = self.indices[index][i_idx]
            rec_i = self.ixes[idx_i]
            cam_sample_i = self.nusc.get('sample_data', rec_i['data'][cam])
            front_img_path_i = os.path.join(self.dataroot, cam_sample_i['filename'])

            # seg2d .npy 對應路徑（沿用你的規則）
            seg2d_root = self.cfg.SEG2D_ROOT
            seg2d_path_i = build_seg2d_path(front_img_path_i, seg2d_root)

            # 224 檔名規則（沿用你現在的）
            rgb224_path_i = front_img_path_i.replace(
                "/samples/CAM_FRONT/", "/samples_224*224/CAM_FRONT/"
            )
            seg224_path_i = seg2d_path_i.replace("/seg2d/", "/seg_cl4_png/").replace(".npy", "_cls4_224.png")

            # 讀圖 → RGB → uint8
            rgb224_i = cv2.cvtColor(cv2.imread(rgb224_path_i, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            seg224_i = cv2.cvtColor(cv2.imread(seg224_path_i, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            assert rgb224_i is not None, f"讀不到 rgb224：{rgb224_path_i}"
            assert seg224_i is not None, f"讀不到 seg224：{seg224_path_i}"

            rgb_seq_list.append(rgb224_i)
            seg_seq_list.append(seg224_i)

        # 堆成 (T_rf,H,W,3) → Tensor[uint8]
        rgb_224_seq = torch.from_numpy(np.stack(rgb_seq_list, axis=0)).to(torch.uint8)
        seg_224_seq = torch.from_numpy(np.stack(seg_seq_list, axis=0)).to(torch.uint8)

        data['rgb_224_seq'] = rgb_224_seq   # (T_rf,H,W,3) uint8
        data['seg_224_seq'] = seg_224_seq   # (T_rf,H,W,3) uint8


        # 原有：塞進 batch_meta（保留路徑，方便 debug）
        # data['batch_meta'] = {
        #     'front_img_path': front_img_path,
        #     'seg2d_path': seg2d_path,
        # }

        return data
