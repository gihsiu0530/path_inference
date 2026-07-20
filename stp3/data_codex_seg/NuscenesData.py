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
        cfg_dataroot = getattr(getattr(cfg, "DATASET", cfg), "DATAROOT", None)
        default_dataroot = "/home/cyc/dataset/1222_obstacle"
        if cfg_dataroot and os.path.exists(os.path.join(cfg_dataroot, "meta", "campus_pose.csv")):
            self.dataroot = cfg_dataroot
        else:
            self.dataroot = default_dataroot
        
        # 讀取 Pose CSV
        self.meta_file = os.path.join(self.dataroot, "meta", "campus_pose.csv")
        if not os.path.exists(self.meta_file):
            raise FileNotFoundError(f"找不到 Pose CSV 檔案: {self.meta_file}")
        
        # 讀取並依照時間排序
        self.df = pd.read_csv(self.meta_file)
        self.df = self.df.sort_values(by='timestamp_us').reset_index(drop=True)
        
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
    
    def get_pose_matrix(self, row_idx):
        """
        從 CSV row 解析 Pose 並轉為「nuScenes 座標系」下的 4x4 矩陣
        CSV columns: timestamp_us, filename, x, y, z, qx, qy, qz, qw
        """
        row = self.df.iloc[row_idx]
        # PyQuaternion: (w, x, y, z)
        q = Quaternion(row['qw'], row['qx'], row['qy'], row['qz'])

        # ---- 1) 先組出 custom 座標系下的 4x4 ----
        R_c = q.rotation_matrix                 # 3x3
        t_c = np.array([row['x'], row['y'], row['z']])

        T_c = np.eye(4)
        T_c[:3, :3] = R_c
        T_c[:3, 3] = t_c

        # ---- 2) 定義 custom -> nuScenes 的固定旋轉 S ----
        S = np.array([[0., -1., 0.],
                    [1.,  0., 0.],
                    [0.,  0., 1.]], dtype=np.float64)

        S4 = np.eye(4)
        S4[:3, :3] = S

        # ---- 3) 用座標變換公式 T_n = S * T_c * S^T ----
        T_n = S4 @ T_c @ S4.T

        return T_n
    
    def get_pose_matrix_gt(self, row_idx):
        """
        給 gt_trajectory 用的 pose：
        - 原始 CSV: x 向左, y 向後
        - 目標 GT 座標: x 向左, y 向前  => 等同於把 y 軸取反

        回傳: 4x4 齊次變換矩陣，在「x 左、y 前、z 上」座標系下的 pose
        """
        row = self.df.iloc[row_idx]

        # 先組出「原始自錄座標系」(x 左, y 後) 下的 T_c
        q = Quaternion(row['qw'], row['qx'], row['qy'], row['qz'])  # (w,x,y,z)
        R_c = q.rotation_matrix
        t_c = np.array([row['x'], row['y'], row['z']])  # x 左, y 後, z 上

        T_c = np.eye(4)
        T_c[:3, :3] = R_c
        T_c[:3, 3] = t_c

        # 定義「自錄座標系 -> GT 座標系」的固定變換:
        # GT: x 左 (同原來), y 前 = -y_c, z 上
        # S = np.array([
        #     [1.,  0., 0.],   # x_gt =  x_c
        #     [0., -1., 0.],   # y_gt = -y_c  （把 y 軸反向）
        #     [0.,  0., 1.],   # z_gt =  z_c
        # ], dtype=np.float64)

        S = np.array([
            [-1.,  0., 0.],   # x_gt =  x_c
            [0., -1., 0.],   # y_gt = -y_c  （把 y 軸反向）
            [0.,  0., 1.],   # z_gt =  z_c
        ], dtype=np.float64)

        S4 = np.eye(4)
        S4[:3, :3] = S

        # 座標系變換公式：T_gt = S * T_c * S^T
        T_gt = S4 @ T_c @ S4.T

        return T_gt


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
        pose_t0 = self.get_pose_matrix(curr_idx)   # numpy (4,4)
        pose_t1 = self.get_pose_matrix(next_idx)   # numpy (4,4)
        
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

            x, y = rel_pose[0, 3], rel_pose[1, 3]
            yaw = quaternion_yaw(Quaternion(matrix=rel_pose))
            traj.append([x, y, yaw])

        return np.array(traj)   # (N_FUTURE + 1, 3)


    def get_paths(self, row):
        """
        根據 CSV 中的相對路徑，組出 224x224 RGB 與 Segmentation npy 的絕對路徑
        CSV filename: samples/CAM_FRONT/xxxx.jpg
        """
        raw_rel_path = row['filename']
        filename = os.path.basename(raw_rel_path)      # xxxx.jpg
        stem = os.path.splitext(filename)[0]           # xxxx

        parts = stem.split("__")
        seq_id = parts[0]           # n001-campus
        frame_id = parts[-1]        # 1765179718877364
            
        # 1. RGB 224 Path
        # 規則: samples/CAM_FRONT -> samples_224*224/CAM_FRONT
        rgb224_rel_dir = os.path.dirname(raw_rel_path).replace("samples", "samples_224*224")
        rgb224_path = os.path.join(self.dataroot, rgb224_rel_dir, filename)
        
        # 2. Segmentation class-id path
        # /dataroot/seg2d/<seq_id>/<frame_id>_cls4_224.npy
        seg_name = f"{frame_id}_cls4_224.npy"
        seg_path = os.path.join(self.dataroot, "seg2d", seq_id, seg_name)

        # 3. Precomputed depth path
        # /dataroot/depth_224*224/CAM_FRONT/<original_stem>.npy
        depth_name = f"{stem}.npy"
        depth_path = os.path.join(self.dataroot, "depth_224*224", "CAM_FRONT", depth_name)
        
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
        return np.load(depth_path).astype(np.float32)

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
                raise FileNotFoundError(f"找不到 Depth npy: {depth224_path}")
            
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

        # print("data['gt_trajectory']",data['gt_trajectory'])

        # 跟 nuScenes 一樣用最後一點的 x 來決定指令
        if gt_traj_np[-1, 0] >= 0.5:
            data['command'] = 'RIGHT'
            # data['command'] = 'LEFT'
        elif gt_traj_np[-1, 0] <= -0.5:
            data['command'] = 'LEFT'
            # data['command'] = 'RIGHT'
        else:
            data['command'] = 'FORWARD'

        data['command'] = 'FORWARD'
        data['target_point'] = torch.tensor([0., 0.])
        data['indices'] = torch.tensor(seq_indices)

        # -----------------------------------------------------------------
        # 3. ★ 新增：當前全域座標 + 影像編號，專門給 CSV / RViz 用
        # -----------------------------------------------------------------
        row = self.df.iloc[current_idx]

        # (a) 取當前 pose，轉成 (x, y, yaw)
        pose_curr = self.get_pose_matrix(current_idx)      # 4x4
        x_curr = pose_curr[0, 3]
        y_curr = pose_curr[1, 3]
        yaw_curr = quaternion_yaw(Quaternion(matrix=pose_curr))

        # 給 eval() 寫 CSV 用
        data['curr_pose'] = torch.tensor([x_curr, y_curr, yaw_curr], dtype=torch.float32)

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
