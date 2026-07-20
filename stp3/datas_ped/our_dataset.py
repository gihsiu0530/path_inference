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
from stp3.utils.tools import gen_dx_bx
from stp3.utils.geometry import mat2pose_vec
from nuscenes.eval.common.utils import quaternion_yaw

# 設定 OpenCV 執行緒數，避免 DataLoader 卡死
cv2.setNumThreads(1)

class CustomCampusDataset(torch.utils.data.Dataset):
    SAMPLE_INTERVAL = 0.5  # 固定 0.5秒

    def __init__(self, cfg, is_train=False):
        """
        Args:
            cfg: 設定檔
            is_train: 用於區分模式，但在自製資料集中邏輯相同
        """
        self.cfg = cfg
        # 強制指定資料根目錄
        self.dataroot = "/home/cyc/dataset/1208_school_stp3/output_dataset"
        
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

    def get_pose_matrix(self, row_idx):
        """
        從 CSV row 解析 Pose 並轉為 4x4 矩陣
        CSV columns: timestamp_us, filename, x, y, z, qx, qy, qz, qw
        """
        row = self.df.iloc[row_idx]
        # 注意: PyQuaternion 順序為 (w, x, y, z)
        q = Quaternion(row['qw'], row['qx'], row['qy'], row['qz'])
        t = np.array([row['x'], row['y'], row['z']])
        
        mat = np.eye(4)
        mat[:3, :3] = q.rotation_matrix
        mat[:3, 3] = t
        return mat
    
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


    def get_future_egomotion(self, curr_idx, next_idx):
        """
        計算 t -> t+1 的相對運動 (6DoF)
        """
        pose_t0 = self.get_pose_matrix(curr_idx)
        pose_t1 = self.get_pose_matrix(next_idx)
        
        # Motion = T1^-1 @ T0 (將 t0 的點轉到 t1 座標系)
        future_egomotion = np.linalg.inv(pose_t1) @ pose_t0
        
        # 強制平面化 (假設車輛貼地行駛，忽略 Z 軸跳動與 Pitch/Roll，這對預測較穩定)
        future_egomotion[3, :3] = 0.0
        future_egomotion[3, 3] = 1.0
        
        # 轉成向量 (dx, dy, dz, roll, pitch, yaw)
        vec = mat2pose_vec(future_egomotion)
        return torch.from_numpy(vec).float().unsqueeze(0) # (1, 6)

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
    def get_gt_trajectory(self, current_idx, future_indices):
        pose_curr = self.get_pose_matrix(current_idx)
        pose_curr_inv = np.linalg.inv(pose_curr)

        traj = []
        # 先把當前點塞進去 (0,0,0)
        traj.append([0.0, 0.0, 0.0])

        for f_idx in future_indices:
            pose_future = self.get_pose_matrix(f_idx)
            rel_pose = pose_curr_inv @ pose_future

            x, y = rel_pose[0, 3], rel_pose[1, 3]
            yaw = quaternion_yaw(Quaternion(matrix=rel_pose))
            traj.append([x, y, yaw])

        return np.array(traj)   # (N_FUTURE + 1, 3)


    def get_paths(self, row):
        """
        根據 CSV 中的相對路徑，組出 224x224 RGB 與 Segmentation 的絕對路徑
        CSV filename: samples/CAM_FRONT/xxxx.jpg
        """
        raw_rel_path = row['filename']
        filename = os.path.basename(raw_rel_path)      # xxxx.jpg
        stem = os.path.splitext(filename)[0]           # xxxx
        
        # 1. RGB 224 Path
        # 規則: samples/CAM_FRONT -> samples_224*224/CAM_FRONT
        rgb224_rel_dir = os.path.dirname(raw_rel_path).replace("samples", "samples_224*224")
        rgb224_path = os.path.join(self.dataroot, rgb224_rel_dir, filename)
        
        # 2. Segmentation Path
        # 規則: seg_cl4_png/xxxx_cls4_224.png
        # 假設 seg_cl4_png 直接在 dataroot 下
        seg_name = f"{stem}_cls4_224.png"
        seg_path = os.path.join(self.dataroot, "seg_cl4_png", seg_name)
        
        return rgb224_path, seg_path

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
        data['segmentation'] = torch.empty(0)
        data['instance'] = torch.empty(0)
        data['pedestrian'] = torch.empty(0)
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
        future_egomotion_list = []

        for i, idx in enumerate(past_indices):
            row = self.df.iloc[idx]
            
            # 取得路徑
            rgb224_path, seg224_path = self.get_paths(row)
            
            # 嚴格檢查：檔案不存在直接報錯
            if not os.path.exists(rgb224_path):
                raise FileNotFoundError(f"找不到 RGB 224 影像: {rgb224_path}")
            if not os.path.exists(seg224_path):
                raise FileNotFoundError(f"找不到 Segmentation 影像: {seg224_path}")
            
            # 讀取 RGB
            bgr = cv2.imread(rgb224_path)
            if bgr is None: raise ValueError(f"OpenCV 無法讀取: {rgb224_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # 讀取 Seg
            seg_bgr = cv2.imread(seg224_path)
            if seg_bgr is None: raise ValueError(f"OpenCV 無法讀取: {seg224_path}")
            seg = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
            
            rgb_seq_list.append(rgb)
            seg_seq_list.append(seg)
            
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
        
        # shape: (T_rf, 6)
        data['future_egomotion'] = torch.cat(future_egomotion_list, dim=0)

        # -----------------------------------------------------------------
        # 2. 計算 Ground Truth 軌跡 (與指令)
        # -----------------------------------------------------------------
        gt_traj_np = self.get_gt_trajectory(current_idx, future_indices)
        data['gt_trajectory'] = torch.from_numpy(gt_traj_np).float()

        # 跟 nuScenes 一樣用最後一點的 x 來決定指令
        if gt_traj_np[-1, 0] >= 2.0:
            data['command'] = 'RIGHT'
        elif gt_traj_np[-1, 0] <= -2.0:
            data['command'] = 'LEFT'
        else:
            data['command'] = 'FORWARD'
            
        print("command",data['command'])
        data['target_point'] = torch.tensor([0., 0.])
        data['indices'] = torch.tensor(seq_indices)

        return data

# 為了讓外部程式 import 不報錯，保留原本的 class 名稱
# 直接將 FuturePredictionDataset 指向我們的新 Class
FuturePredictionDataset = CustomCampusDataset