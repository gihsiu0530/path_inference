import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pyquaternion import Quaternion
import torch.utils.data
import torchvision

# 保留原本的工具引用，確保數學計算一致
from stp3.utils.geometry import mat2pose_vec
from nuscenes.eval.common.utils import quaternion_yaw

# 設定 OpenCV 執行緒數，避免 DataLoader 卡死
cv2.setNumThreads(1)

class CustomCampusDataset(torch.utils.data.Dataset):
    SAMPLE_INTERVAL = 0.5  # 固定 0.5秒
    SKIP_INITIAL_SECONDS = 12  # 0.5
    SKIP_FINAL_SECONDS = 0.0
    SEG_PALETTE = np.array([
        [0, 0, 0],
        [128, 64, 128],
        [220, 20, 60],
        [0, 142, 0],
    ], dtype=np.uint8)

    def __init__(self, cfg, is_train=False):
        """
        Args:
            cfg: 設定檔
            is_train: 用於區分模式，但在自製資料集中邏輯相同
        """
        self.cfg = cfg
        # 0603 resampled video folder:
        #   resample_index.csv maps target/img/seg/depth/odom/poseimu timestamps
        #   poseimu_zero.csv is used as model input egomotion
        #   odom.csv is kept as gt_trajectory ground truth
        default_dataroot = "/home/cyc/dataset/0603_go_to_bed/resample/video1"
        cfg_dataset = getattr(cfg, "DATASET", None)
        cfg_dataroot = getattr(cfg_dataset, "DATAROOT", None) if cfg_dataset is not None else None
        self.dataroot = cfg_dataroot or default_dataroot

        self.index_file = os.path.join(self.dataroot, "resample_index.csv")
        self.odom_file = os.path.join(self.dataroot, "odom.csv")
        self.poseimu_file = os.path.join(self.dataroot, "poseimu_zero.csv")
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"找不到 resample index CSV 檔案: {self.index_file}")
        if not os.path.exists(self.odom_file):
            raise FileNotFoundError(f"找不到 odom CSV 檔案: {self.odom_file}")
        if not os.path.exists(self.poseimu_file):
            raise FileNotFoundError(f"找不到 poseimu_zero CSV 檔案: {self.poseimu_file}")

        index_df = pd.read_csv(self.index_file)
        odom_df = pd.read_csv(self.odom_file)
        poseimu_df = pd.read_csv(self.poseimu_file)

        odom_df = odom_df.rename(columns={
            "timestep": "odom_timestep",
            "position_x": "odom_x",
            "position_y": "odom_y",
            "position_z": "odom_z",
            "orientation_x": "odom_qx",
            "orientation_y": "odom_qy",
            "orientation_z": "odom_qz",
            "orientation_w": "odom_qw",
        })
        poseimu_df = poseimu_df.rename(columns={
            "timestamp(ns)": "poseimu_timestep",
            "tx": "poseimu_x",
            "ty": "poseimu_y",
            "tz": "poseimu_z",
            "qx": "poseimu_qx",
            "qy": "poseimu_qy",
            "qz": "poseimu_qz",
            "qw": "poseimu_qw",
        })

        self.df = index_df.merge(
            odom_df,
            left_on="odom_timestep",
            right_on="odom_timestep",
            how="inner",
            validate="one_to_one",
        )
        self.df = self.df.merge(
            poseimu_df,
            left_on="poseimu_timestep",
            right_on="poseimu_timestep",
            how="inner",
            validate="one_to_one",
        )
        self.df = self.df.rename(columns={
            "target_timestep": "timestamp_us",
        })
        self.df["filename"] = self.df["img_timestep"].astype(str) + ".png"
        self.df = self.df.sort_values(by='timestamp_us').reset_index(drop=True)

        if len(self.df) > 0:
            first_ts = int(self.df["timestamp_us"].iloc[0])
            skip_ns = int(round(self.SKIP_INITIAL_SECONDS * 1e9))
            self.df = self.df[self.df["timestamp_us"] >= first_ts + skip_ns].reset_index(drop=True)

        if len(self.df) > 0:
            last_ts = int(self.df["timestamp_us"].iloc[-1])
            skip_ns = int(round(self.SKIP_FINAL_SECONDS * 1e9))
            if skip_ns > 0:
                self.df = self.df[self.df["timestamp_us"] <= last_ts - skip_ns].reset_index(drop=True)
        
        # 設定序列長度
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD
        self.n_future = cfg.N_FUTURE_FRAMES
        self.sequence_length = self.receptive_field + self.n_future
        
        # 建立索引 (Sliding Window)
        self.indices = []
        # 簡單檢查：若 CSV 資料量不足以構成一個序列，則不建立索引
        if len(self.df) >= self.sequence_length:
            for i in range(len(self.df) - self.sequence_length + 1):
                # 這裡假設你的 Bag 錄製是連續的。如果有多個 Bag 合併，需額外檢查時間跳變。
                self.indices.append(list(range(i, i + self.sequence_length)))
        else:
            print(f"[Warning] 資料量不足 ({len(self.df)} 幀)，無法建立長度為 {self.sequence_length} 的序列。")

        # 預先計算 BEV 參數 (雖然沒用到 HDMap，但為了保持格式一致性)
        # from stp3.utils.geometry import calculate_birds_eye_view_parameters
        # bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
        #     cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        # )
        # self.bev_resolution = bev_resolution.numpy()
        # self.bev_start_position = bev_start_position.numpy()
        # self.bev_dimension = bev_dimension.numpy()
        # self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

        # print(f"[CustomCampusDataset] 載入 {len(self.df)} 幀，產生 {len(self.indices)} 個序列。")

    def __len__(self):
        return len(self.indices)

    # def get_pose_matrix(self, row_idx):
    #     """
    #     從 CSV row 解析 Pose 並轉為 4x4 矩陣
    #     CSV columns: timestamp_us, filename, x, y, z, qx, qy, qz, qw
    #     """
    #     row = self.df.iloc[row_idx]
    #     # 注意: PyQuaternion 順序為 (w, x, y, z)
    #     q = Quaternion(row['qw'], row['qx'], row['qy'], row['qz'])
    #     t = np.array([row['x'], row['y'], row['z']])
        
    #     mat = np.eye(4)
    #     mat[:3, :3] = q.rotation_matrix
    #     mat[:3, 3] = t
    #     return mat
    
    def _pose_matrix_from_row(self, row, prefix):
        q = Quaternion(
            row[f'{prefix}_qw'],
            row[f'{prefix}_qx'],
            row[f'{prefix}_qy'],
            row[f'{prefix}_qz'],
        )
        t = np.array([
            row[f'{prefix}_x'],
            row[f'{prefix}_y'],
            row[f'{prefix}_z'],
        ])

        mat = np.eye(4)
        mat[:3, :3] = q.rotation_matrix
        mat[:3, 3] = t
        return mat

    def get_pose_matrix_input(self, row_idx):
        """
        給模型輸入 future_egomotion 用的 pose，來源是 poseimu_zero.csv。
        """
        row = self.df.iloc[row_idx]
        return self._pose_matrix_from_row(row, "poseimu")

    def get_pose_matrix(self, row_idx):
        return self.get_pose_matrix_input(row_idx)
    
    def get_pose_matrix_gt(self, row_idx):
        """
        給 gt_trajectory 用的 pose，來源是 odom.csv。
        """
        row = self.df.iloc[row_idx]
        return self._pose_matrix_from_row(row, "odom")


    # def get_future_egomotion(self, curr_idx, next_idx):
    #     """
    #     計算 t -> t+1 的相對運動 (6DoF)
    #     """
    #     pose_t0 = self.get_pose_matrix(curr_idx)
    #     pose_t1 = self.get_pose_matrix(next_idx)
        
    #     # Motion = T1^-1 @ T0 (將 t0 的點轉到 t1 座標系)
    #     future_egomotion = np.linalg.inv(pose_t1) @ pose_t0
        
    #     # 強制平面化 (假設車輛貼地行駛，忽略 Z 軸跳動與 Pitch/Roll，這對預測較穩定)
    #     future_egomotion[3, :3] = 0.0
    #     future_egomotion[3, 3] = 1.0
        
    #     # 轉成向量 (dx, dy, dz, roll, pitch, yaw)
    #     vec = mat2pose_vec(future_egomotion)
    #     return torch.from_numpy(vec).float().unsqueeze(0) # (1, 6)

    def get_future_egomotion(self, curr_idx, next_idx):
        """
        計算 t -> t+1 的相對運動 (6DoF)
        回傳 shape: (1, 6) 的 torch.FloatTensor
        """
        pose_t0 = self.get_pose_matrix_input(curr_idx)   # numpy (4,4)
        pose_t1 = self.get_pose_matrix_input(next_idx)   # numpy (4,4)
        
        # Motion = T1^-1 @ T0 (將 t0 的點轉到 t1 座標系)
        future_egomotion = np.linalg.inv(pose_t1) @ pose_t0

        # 強制平面化
        future_egomotion[3, :3] = 0.0
        future_egomotion[3, 3] = 1.0

        # 先轉成 torch.Tensor，再丟給 mat2pose_vec
        future_egomotion_t = torch.from_numpy(future_egomotion).float()  # (4,4) tensor

        vec = mat2pose_vec(future_egomotion_t)  # 這裡應該回傳 (6,) 或 (1,6) 的 tensor

        # 保證回傳 shape 是 (1, 6)
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)  # (1,6)

        return vec  # 已是 torch.FloatTensor，不要再 from_numpy


    # def get_gt_trajectory(self, current_idx, future_indices):
    #     """
    #     計算未來軌跡點在當前座標系下的位置 (x, y, yaw)
    #     """
    #     pose_curr_inv = np.linalg.inv(self.get_pose_matrix(current_idx))
        
    #     gt_traj = []
    #     for f_idx in future_indices:
    #         pose_future = self.get_pose_matrix(f_idx)
            
    #         # 座標轉換: T_curr^-1 @ T_future
    #         rel_pose = pose_curr_inv @ pose_future
            
    #         x, y = rel_pose[0, 3], rel_pose[1, 3]
    #         # 取得相對 Yaw
    #         yaw = quaternion_yaw(Quaternion(matrix=rel_pose))
    #         gt_traj.append([x, y, yaw])
            
    #     return np.array(gt_traj)
    
    # gpt說上面的沒加0,0,0
    # def get_gt_trajectory(self, current_idx, future_indices):
    #     pose_curr = self.get_pose_matrix(current_idx)
    #     pose_curr_inv = np.linalg.inv(pose_curr)

    #     traj = []
    #     # 先把當前點塞進去 (0,0,0)
    #     traj.append([0.0, 0.0, 0.0])

    #     for f_idx in future_indices:
    #         pose_future = self.get_pose_matrix(f_idx)
    #         rel_pose = pose_curr_inv @ pose_future

    #         x, y = rel_pose[0, 3], rel_pose[1, 3]
    #         yaw = quaternion_yaw(Quaternion(matrix=rel_pose))
    #         traj.append([x, y, yaw])

    #     return np.array(traj)   # (N_FUTURE + 1, 3)
    
    def get_gt_trajectory(self, current_idx, future_indices):
        pose_curr = self.get_pose_matrix_gt(current_idx)
        pose_curr_inv = np.linalg.inv(pose_curr)

        traj = []
        # 先把當前點塞進去 (0,0,0)
        traj.append([0.0, 0.0, 0.0])

        for f_idx in future_indices:
            pose_future = self.get_pose_matrix_gt(f_idx)
            rel_pose = pose_curr_inv @ pose_future

            # rel_pose 的平移是車體相對座標 (x_forward, y_left)。
            # 模型/plot 使用 (x_left, y_front)，需對調成 (rel_y, rel_x)。
            x_forward = rel_pose[0, 3]
            y_left = rel_pose[1, 3]
            x = y_left
            y = x_forward
            yaw = quaternion_yaw(Quaternion(matrix=rel_pose))
            traj.append([x, y, yaw])

        return np.array(traj)   # (N_FUTURE + 1, 3)

    def get_plot_history_trajectory(self, current_idx, history_seconds=3.0):
        """
        只供 park_L2.py 畫圖檢查用，不影響模型輸入。
        回傳過去 history_seconds 到 current 的固定長度軌跡，座標為 (x_left, y_front, yaw)。
        """
        history_steps = int(round(history_seconds / self.SAMPLE_INTERVAL))
        pose_curr_inv = np.linalg.inv(self.get_pose_matrix_input(current_idx))

        traj = []
        for idx in range(current_idx - history_steps, current_idx + 1):
            row_idx = max(0, idx)
            rel_pose = pose_curr_inv @ self.get_pose_matrix_input(row_idx)
            x_forward = rel_pose[0, 3]
            y_left = rel_pose[1, 3]
            yaw = quaternion_yaw(Quaternion(matrix=rel_pose))
            traj.append([y_left, x_forward, yaw])

        return np.array(traj, dtype=np.float32)


    def get_paths(self, row):
        """
        根據 resample_index.csv 中的 timestamp 組出 224x224 RGB、seg id npy 與 depth_infer npy。
        """
        rgb_name = f"{int(row['img_timestep'])}.png"
        seg_name = f"{int(row['seg_timestep'])}.npy"
        depth_name = f"{int(row['depth_timestep'])}.npy"
        rgb224_path = os.path.join(self.dataroot, "img", rgb_name)
        seg_path = os.path.join(self.dataroot, "seg", seg_name)
        depth_path = os.path.join(self.dataroot, "depth_infer", depth_name)

        return rgb224_path, seg_path, depth_path

    def load_seg_npy(self, seg_path):
        seg_id = np.load(seg_path)
        if seg_id.ndim == 3:
            if seg_id.shape[-1] == 1:
                seg_id = seg_id[..., 0]
            else:
                seg_id = np.argmax(seg_id, axis=-1)
        if seg_id.shape != (224, 224):
            seg_id = cv2.resize(seg_id.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
        seg_id = np.clip(seg_id.astype(np.uint8), 0, len(self.SEG_PALETTE) - 1)
        seg_rgb = self.SEG_PALETTE[seg_id]
        return seg_id, seg_rgb

    def load_depth_npy(self, depth_path):
        depth = np.load(depth_path).astype(np.float32)
        if depth.ndim == 3:
            depth = np.squeeze(depth)
        if depth.shape != (224, 224):
            depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)

    def __getitem__(self, index):
        seq_indices = self.indices[index]
        
        # 切分 Index: 過去(含現在) / 未來
        past_indices = seq_indices[:self.receptive_field]
        current_idx = past_indices[-1]
        future_indices = seq_indices[self.receptive_field:]
        
        # 初始化輸出字典 (格式與舊版完全一致)
        data = {}
        # 這些是 VLM 不會用到或我們沒有的，設為空張量
        data['image'] = torch.empty(0)
        data['intrinsics'] = torch.empty(0)
        data['extrinsics'] = torch.empty(0)
        data['depths'] = torch.empty(0)
        # data['segmentation'] = torch.empty(0)
        data['instance'] = torch.empty(0)
        # data['pedestrian'] = torch.empty(0)
        data['centerness'] = torch.empty(0)
        data['offset'] = torch.empty(0)
        data['flow'] = torch.empty(0)
        data['hdmap'] = torch.empty(0)
        data['sample_trajectory'] = torch.empty(0)

        # -----------------------------------------------------------------
        # 1. 讀取與處理序列影像 (RGB 224 & Seg 224)
        # -----------------------------------------------------------------
        rgb_seq_list = []
        seg_seq_list = []
        seg_id_seq_list = []
        depth_seq_list = []
        future_egomotion_list = []

        for i, idx in enumerate(past_indices):
            row = self.df.iloc[idx]
            
            # 取得路徑
            rgb224_path, seg224_path, depth224_path = self.get_paths(row)
            
            # 嚴格檢查：檔案不存在直接報錯
            if not os.path.exists(rgb224_path):
                raise FileNotFoundError(f"找不到 RGB 224 影像: {rgb224_path}")
            if not os.path.exists(seg224_path):
                raise FileNotFoundError(f"找不到 Segmentation npy: {seg224_path}")
            if not os.path.exists(depth224_path):
                raise FileNotFoundError(f"找不到 depth_infer npy: {depth224_path}")
            
            # 讀取 RGB
            bgr = cv2.imread(rgb224_path)
            if bgr is None: raise ValueError(f"OpenCV 無法讀取: {rgb224_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            seg_id, seg = self.load_seg_npy(seg224_path)
            depth = self.load_depth_npy(depth224_path)
            
            rgb_seq_list.append(rgb)
            seg_seq_list.append(seg)
            seg_id_seq_list.append(seg_id)
            depth_seq_list.append(depth)
            
            # 計算 Future Egomotion (每幀相對於下一幀的移動)
            if i < len(past_indices) - 1:
                next_idx_in_seq = past_indices[i+1]
                ego = self.get_future_egomotion(idx, next_idx_in_seq)
            else:
                # 最後一幀 (current)，計算相對於未來第一幀的移動
                if len(future_indices) > 0:
                    ego = self.get_future_egomotion(idx, future_indices[0])
                else:
                    # 如果沒有未來幀 (例如資料集結尾)，給 Identity
                    ego = torch.zeros(1, 6)
            
            future_egomotion_list.append(ego)

        # 堆疊 Tensor
        # shape: (T_rf, H, W, 3) -> uint8
        data['rgb_224_seq'] = torch.from_numpy(np.stack(rgb_seq_list)).to(torch.uint8)
        data['seg_224_seq'] = torch.from_numpy(np.stack(seg_seq_list)).to(torch.uint8)
        data['seg_id_224_seq'] = torch.from_numpy(np.stack(seg_id_seq_list)).long()
        data['depth_224_seq'] = torch.from_numpy(np.stack(depth_seq_list)).float()
        
        # shape: (T_rf, 6)
        data['future_egomotion'] = torch.cat(future_egomotion_list, dim=0)

        # -----------------------------------------------------------------
        # 2. 計算 Ground Truth 軌跡 (與指令)
        # -----------------------------------------------------------------
        gt_traj_np = self.get_gt_trajectory(current_idx, future_indices)
        data['gt_trajectory'] = torch.from_numpy(gt_traj_np).float()
        plot_history_np = self.get_plot_history_trajectory(current_idx, history_seconds=3.0)
        data['plot_input_xy_3s'] = torch.from_numpy(plot_history_np[:, :2]).float()
        data['plot_input_yaw_3s'] = torch.from_numpy(plot_history_np[:, 2]).float()

        # print("data['gt_trajectory']",data['gt_trajectory'])

        # 跟 nuScenes 一樣用最後一點的 x 來決定指令
        if gt_traj_np[-1, 0] >= 1.0:
            # data['command'] = 'RIGHT'
            data['command'] = 'LEFT'
        elif gt_traj_np[-1, 0] <= -1.0:
            # data['command'] = 'LEFT'
            data['command'] = 'RIGHT'
        else:
            data['command'] = 'FORWARD'

        data['target_point'] = torch.tensor([0., 0.])
        data['indices'] = torch.tensor(seq_indices)

        # -----------------------------------------------------------------
        # 3. ★ 新增：當前全域座標 + 影像編號，專門給 CSV / RViz 用
        # -----------------------------------------------------------------
        row = self.df.iloc[current_idx]

        # (a) curr_pose 保持為 odom/GT 座標，供 CSV / RViz 對齊 gt_trajectory。
        pose_curr = self.get_pose_matrix_gt(current_idx)      # 4x4
        x_curr = pose_curr[0, 3]
        y_curr = pose_curr[1, 3]
        yaw_curr = quaternion_yaw(Quaternion(matrix=pose_curr))

        # 給 eval() 寫 CSV 用
        data['curr_pose'] = torch.tensor([x_curr, y_curr, yaw_curr], dtype=torch.float32)
        pose_input_curr = self.get_pose_matrix_input(current_idx)
        data['curr_pose_input'] = torch.tensor([
            pose_input_curr[0, 3],
            pose_input_curr[1, 3],
            quaternion_yaw(Quaternion(matrix=pose_input_curr)),
        ], dtype=torch.float32)

        # (b) 影像編號：timestamp 或 filename 都可以，視你 ROS 播放怎麼找圖
        data['timestamp_us'] = torch.tensor(row['timestamp_us'], dtype=torch.long)
        # 字串讓 DataLoader collate 成 list[str] 就好
        data['filename'] = row['filename']

        T_rf = len(past_indices)  # 正常應該等於 self.receptive_field

        # 全 0，shape: (T_rf, 1, 1, 1)
        dummy_seg = torch.zeros(T_rf, 1, 1, 1, dtype=torch.long)
        dummy_ped = torch.zeros(T_rf, 1, 1, 1, dtype=torch.long)

        data['segmentation'] = dummy_seg
        data['pedestrian'] = dummy_ped

        return data

# 為了讓外部程式 import 不報錯，保留原本的 class 名稱
# 直接將 FuturePredictionDataset 指向我們的新 Class
FuturePredictionDataset = CustomCampusDataset
