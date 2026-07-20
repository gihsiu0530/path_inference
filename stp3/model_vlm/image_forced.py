import os
import contextlib
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.cuda.amp import autocast
from scipy.interpolate import interp1d

try:
    import open_clip
    HAS_OPENCLIP = True
except Exception:
    HAS_OPENCLIP = False

from stp3.utils.tools import gen_dx_bx
from stp3.utils.geometry import calculate_birds_eye_view_parameters

# 1230

class BatchedCrossAttention(nn.Module):
    """與你現有 mult_planner 中相同介面的小型跨注意力，用於把文字/查詢與視覺 tokens 融合。"""
    def __init__(self, embed_dim: int, num_heads: int = 8, attn_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = (self.head_dim ** -0.5)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout and attn_dropout > 0 else nn.Identity()

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        # q: (B, Q, C)  kv: (B, S, C)
        B, Q, C = q.shape
        S = kv.shape[1]
        H = self.num_heads
        D = self.head_dim

        q_lin = self.q_proj(q).view(B, Q, H, D)
        k_lin = self.k_proj(kv).view(B, S, H, D)
        v_lin = self.v_proj(kv).view(B, S, H, D)

        # (B,H,Q,S)
        attn = torch.einsum('bqhd,bshd->bhqs', q_lin, k_lin) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum('bhqs,bshd->bqhd', attn, v_lin).contiguous().view(B, Q, C)
        return self.out_proj(out)


class VLM_Generative(nn.Module):
    """
    直接 **生成** 未來軌跡的規劃器：
    - 取多幀 RGB/Seg 的 CLIP 視覺 tokens（我們假設 caller 已提供 224x224 序列）
    - 加入每幀 ego-motion 嵌入
    - 文本（指令）→ text_feat，投影到視覺寬度
    - 用 cross-attn 把一組 learnable queries (每個時間步 1 個) 與視覺 tokens 融合
    - MLP 輸出 (dx,dy,dyaw) 或直接 (x,y,yaw)。此處實作 **直接 (x,y,0)**，z/yaw 先置 0
    """
    def __init__(self, clip_name: str, device: torch.device, n_future: int, input_size: int = 224,
                 prompts: Dict[str, str] = {
                     "LEFT":    "turn left, stay on drivable road, avoid obstacles, smooth path",
                     "FORWARD": "go straight, stay on drivable road, avoid obstacles, smooth path",
                     "RIGHT":   "turn right, stay on drivable road, avoid obstacles, smooth path",   
                 }):
        super().__init__()
        assert HAS_OPENCLIP, "需要 open_clip，請先安裝 open_clip_torch"
        self.device = device
        self.n_future = n_future
        self.input_size = input_size
        self.prompts = prompts

        self.state = "IDLE"
        self.stop_start_time = 0
        self.cooldown_start_time = 0
        self.STOP_DURATION = 3.0
        self.COOLDOWN_TIME = 8.0
        self.STOP_SPEED_THRES = 0.5 # 寬鬆一點

        self.clip, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained="openai")
        self.tokenizer = open_clip.get_tokenizer(clip_name)
        self.clip.to(device)

        # ===== 強制用影像：捷徑抑制超參 =====
        self.qdrop_text_p = float(getattr(self, "QDROP_TEXT_P", 0.0))   # 0~0.4  (v1: default off)
        self.qdrop_vis_p = float(getattr(self, "QDROP_VIS_P", 0.0))   # 0~0.4  (v1: default off)
        self.qdrop_timeq_p = float(getattr(self, "QDROP_TIMEQ_P", 0.0)) # 0~0.3 (v1: default off)
        self.coarse_drop_p = float(getattr(self, "COARSE_DROP_P", 0.4)) # 0~0.5 (v1: keep)


        # 凍結 CLIP（預設全凍結），可選擇性解凍視覺最後 K 層（v1 建議 K=1~2）
        for p in self.clip.parameters():
            p.requires_grad = False

        # 你可在外部（wrapper/cfg）設定 self.VIS_UNFREEZE_K 再呼叫 self.apply_visual_unfreeze()
        self.VIS_UNFREEZE_K = int(getattr(self, "VIS_UNFREEZE_K", 3))
        self.apply_visual_unfreeze(self.VIS_UNFREEZE_K)
        self.vis_width = int(getattr(self.clip.visual.transformer, "width", 768))

        # 取視覺 tokens 的 hook
        self._vis_tokens = None
        def _vis_hook(_, __, out):
            self._vis_tokens = out
        self.clip.visual.transformer.register_forward_hook(_vis_hook)

        # 文本寬度對齊到視覺寬度
        if hasattr(self.clip, "text_projection"):
            txt_dim = int(self.clip.text_projection.shape[-1])
        else:
            txt_dim = self.vis_width
        self.txt_to_vis = nn.Linear(txt_dim, self.vis_width, bias=False)

        # ego-motion 嵌入 (dx, dy, sin(dyaw), cos(dyaw))
        self.ego_mlp = nn.Sequential(
            nn.Linear(4, self.vis_width), nn.GELU(), nn.Linear(self.vis_width, self.vis_width)
        )

        # 跨注意力：T 個 learnable queries
        self.time_queries = nn.Parameter(torch.randn(self.n_future, self.vis_width) * 0.02)
        self.cross_attn = BatchedCrossAttention(self.vis_width, num_heads=8, attn_dropout=0.0)


        # ====== AR decoder: GRU-based ======
        # 時間位置編碼（讓每個 t 有可學的時間上下文）
        self.time_pe = nn.Parameter(torch.randn(self.n_future, self.vis_width) * 0.02)

        # 用於把全局語義（text + 視覺池化）投成 h0
        self.h0_proj = nn.Sequential(
            nn.Linear(self.vis_width * 2, self.vis_width),
            nn.GELU(),
            nn.Linear(self.vis_width, self.vis_width),
        )

        # 解碼器：每步吃進 [prev_xy(2) || fused_t(C)]，輸出新隱狀態
        # self.decoder_gru = nn.GRUCell(input_size=self.vis_width + 2, hidden_size=self.vis_width)


        # ====== coarse baseline（v1：不看影像，只用 ego 估計當前速度做 CV 外插） ======



        # 在 __init__ 裡加
        self.mix_gate = nn.Sequential(
            nn.Linear(self.vis_width * 2, self.vis_width),
            nn.GELU(),
            nn.Linear(self.vis_width, 1),
            nn.Sigmoid()
        )

        # （可選）讓一開始偏向 coarse：把最後一層 bias 初始化成負值
        nn.init.constant_(self.mix_gate[-2].bias, 0.0)  # sigmoid(0)=0.5

        # ====== AR decoder: GRU-based ======
        self.time_pe = nn.Parameter(torch.randn(self.n_future, self.vis_width) * 0.02)

        self.h0_proj = nn.Sequential(
            nn.Linear(self.vis_width * 2, self.vis_width),
            nn.GELU(),
            nn.Linear(self.vis_width, self.vis_width),
        )

        # ★ 新增：BEV 解碼器 (將 2D 的 CLIP 特徵轉成鳥瞰圖)
        # 預期輸入：(B, vis_width*2, H_f, W_f) -> 輸出：(B, vis_width, H_bev, W_bev)
        self.bev_decoder = nn.Sequential(
            nn.Conv2d(self.vis_width * 2, self.vis_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.vis_width),
            nn.GELU(),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False), # 放大以增加解析度
            nn.Conv2d(self.vis_width, self.vis_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.vis_width),
            nn.GELU(),
        )

        # ★ 修改：原本 input_size 是 vis_width * 2，現在加入 local_bev (大小 vis_width)，所以變 * 3
        self.decoder_gru = nn.GRUCell(input_size=self.vis_width * 3, hidden_size=self.vis_width)

        self.delta_head = nn.Sequential(
            nn.Linear(self.vis_width, self.vis_width),
            nn.GELU(),
            nn.Linear(self.vis_width, 2)
        )

        self.xy_embedder = nn.Sequential(
            nn.Linear(2, self.vis_width // 4),  
            nn.GELU(),
            nn.Linear(self.vis_width // 4, self.vis_width)
        )

        self.g_proj = nn.Sequential(nn.Linear(self.vis_width * 2, self.vis_width), nn.GELU(),
            nn.Linear(self.vis_width, self.vis_width), nn.Sigmoid())
        
        self.xy_post_ln = nn.LayerNorm(self.vis_width)
        # ★ 修改：同步把這裡的 LayerNorm 輸入維度改成 * 3
        self.dec_in_ln  = nn.LayerNorm(self.vis_width * 3)




        

        # ====== 一些可調參數（給不到 cfg 時也有預設）======
        self.use_delta = True  # 目前固定走 Δx,Δy
        # self.step_scale = 1.0  # Δ 的比例因子，資料單位不同時可調
        # self.max_step = 2.0    # 單步最大 Δ 限制（公尺），避免爆衝

        self.SCENARIOS = {
            "NORMAL": [
                "a clear road",
                "normal driving conditions",
                "safe to drive",
                "empty asphalt road"
            ],
            "BUMP": [
                # "a speed bump on the road",
                # "a road hump",
                # "speed breaker on the ground",
                "yellow and black speed bump"
            ],
            "OBSTACLE": [
                # "an unexpected obstacle blocking the road",
                "a soccer ball or box on the road",
                "a parked car in front of the road"
                # "a parked car on the right side of the road"
            ],
            "STOP": [
                "a red stop sign",
                "an octagon stop sign",
                "traffic sign saying stop",
                "red traffic light",           # 加入紅燈描述
                "traffic signal intersection", # 加入路口描述
                "stop signal"
            ],
            # ▼ 新增：右側障礙物 (需要往左閃)
            # "OBSTACLE_RIGHT": [
            #     "a parked car on the right side of the road",
            #     "a vehicle parked along the right street",
            #     "an obstacle on the right side"
            # ],
            # # ▼ 新增：左側障礙物 (需要往右閃)
            # "OBSTACLE_LEFT": [
            #     "a parked car on the left side of the road",
            #     "a vehicle parked along the left street",
            #     "an obstacle on the left side"
            # ]
        }
        
        # 預先計算 Scenario Embeddings (只做一次)
        self.scenario_feats = self._precompute_scenario_feats()


                # =========================================================
        # === 額外詳細列印：逐層顯示每個參數名稱與形狀、大小 ===
        # =========================================================
        def _millify(n):
            return f"{n/1e6:.2f}M" if n >= 1e6 else (f"{n/1e3:.1f}K" if n >= 1e3 else str(n))

        print("\n[CLIP Parameter Detail List]")
        total_params = 0
        trainable_params = 0
        for name, p in self.named_parameters():
            n = p.numel()
            total_params += n
            if p.requires_grad:
                trainable_params += n
            # tag = "T" if p.requires_grad else "F"
            # print(f"{tag} | {name:<80s} | shape={tuple(p.shape)!s:<25s} | {n:>10,d} params ({_millify(n)})")

        print("-" * 120)
        print(f"Total parameters    : {total_params:,} ({_millify(total_params)})")
        print(f"Trainable parameters: {trainable_params:,} ({_millify(trainable_params)})  "
            f"({100*trainable_params/max(1,total_params):.2f}%)\n")
        

    
    def apply_visual_unfreeze(self, k: int = 0):
        """
        選擇性解凍 CLIP visual 的最後 k 個 transformer blocks（以及 ln_pre/ln_post）。
        v1 建議：k=1~2（單卡 16G / batch=4 通常可負擔）。
        """
        k = int(max(0, k))
        # 先把 visual 全關（以免外部多次呼叫造成層數累積）
        if hasattr(self.clip, "visual"):
            for p in self.clip.visual.parameters():
                p.requires_grad = False

        if k > 0 and hasattr(self.clip, "visual") and hasattr(self.clip.visual, "transformer"):
            blocks = self.clip.visual.transformer.resblocks
            if len(blocks) > 0:
                for blk in blocks[-k:]:
                    for p in blk.parameters():
                        p.requires_grad = True
            # layernorm 通常也要解凍，讓分佈能對齊你的任務
            for n, p in self.clip.visual.named_parameters():
                if ("ln_post" in n) or ("ln_pre" in n):
                    p.requires_grad = True

        self._clip_vis_trainable = bool(hasattr(self.clip, "visual") and any(p.requires_grad for p in self.clip.visual.parameters()))

    def make_coarse_baseline(self, ego_seq: torch.Tensor, T: int = None) -> torch.Tensor:
        """
        v1 coarse baseline：不看影像，只用最近一個 ego-motion step 估計速度做 CV 外插。
        - ego_seq: (B,T_rf,4) 其中前兩維是 (ex,ey) = (dx,dy)/EGO_SCALE_M，對應最近歷史幀 -> 現在幀
        - 輸出 coarse_xy: (B,T,2)，座標系與你的 pred/gt 一致（第 0 維是側向、第 1 維是前向）。
        注意：baseline 故意簡化（側向=0），讓「轉彎/避障」必須靠影像+AR 來完成。
        """
        if T is None:
            T = self.n_future
        device = ego_seq.device
        B, T_rf, _ = ego_seq.shape

        ego_scale_m = float(getattr(self, "EGO_SCALE_M", 5.0))
        dt = float(getattr(self, "SAMPLE_DT", 0.5))

        # ego_seq 的最後一幀在 wrapper 裡被設為 0；用倒數第二幀近似最近一步位移
        if T_rf >= 2:
            dxy_norm = ego_seq[:, -2, 0:2]  # (B,2)  (dx,dy)/scale
        else:
            dxy_norm = ego_seq[:, -1, 0:2]

        dxy_m = dxy_norm * ego_scale_m          # (B,2) meters per dt
        speed = torch.norm(dxy_m, dim=-1)       # (B,)
        step_len = speed * dt                   # meters per future step (假設與資料 dt 一致)

        steps = torch.arange(1, T + 1, device=device, dtype=ego_seq.dtype).view(1, T)  # 1..T
        forward = step_len.view(B, 1) * steps  # (B,T)

        coarse_xy = torch.zeros(B, T, 2, device=device, dtype=ego_seq.dtype)
        coarse_xy[..., 1] = forward  # 前向
        return coarse_xy
    def _precompute_scenario_feats(self):
        feats_map = {}
        with torch.no_grad():
            for label, scenario_prompts in self.SCENARIOS.items():
                tokens = self.tokenizer(scenario_prompts).to(self.device)
                # 注意：這裡用 CLIP 原始的 encode_text，不要過 txt_to_vis
                feats = self.clip.encode_text(tokens)
                feats /= feats.norm(dim=-1, keepdim=True)
                # 取平均
                mean_feat = feats.mean(dim=0, keepdim=True)
                mean_feat /= mean_feat.norm(dim=-1, keepdim=True)
                feats_map[label] = mean_feat
        
        # 轉成 Tensor 矩陣: (N_classes, Dim)
        # 順序: [NORMAL, BUMP, OBSTACLE, STOP]
        self.scenario_labels = ["NORMAL", "BUMP", "OBSTACLE", "STOP"]
        # self.scenario_labels = ["NORMAL", "BUMP", "OBSTACLE", "OBSTACLE_RIGHT", "OBSTACLE_LEFT", "STOP"]
        feat_tensor = torch.cat([feats_map[l] for l in self.scenario_labels])
        return feat_tensor

    def detect_context(self, rgb_img_tensor):
        """
        對單張影像進行 Zero-shot 分類
        rgb_img_tensor: (1, 3, 224, 224) 已經 preprocess 過的 Tensor
        """
        with torch.no_grad():
            # encode image
            img_feat = self.clip.encode_image(rgb_img_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            
            # calculate similarity
            # scenario_feats shape: (4, C)
            similarity = (100.0 * img_feat @ self.scenario_feats.T).softmax(dim=-1)
            
            # get best class
            probs = similarity[0]
            best_idx = probs.argmax().item()
            best_label = self.scenario_labels[best_idx]
            
            return best_label, probs
        
    def update(self, scenario, current_speed):
        now = time.time()
        
        if self.state == "IDLE":
            if scenario == "STOP":
                self.state = "APPROACHING"
                return 0.0
            return None

        elif self.state == "APPROACHING":
            if current_speed < self.STOP_SPEED_THRES:
                self.state = "WAITING"
                self.stop_start_time = now
            if scenario != "STOP": # 誤判恢復
                self.state = "IDLE"
                return None
            return 0.0

        elif self.state == "WAITING":
            if (now - self.stop_start_time) < self.STOP_DURATION:
                return 0.0
            else:
                self.state = "COOLDOWN"
                self.cooldown_start_time = now
                return None

        elif self.state == "COOLDOWN":
            if scenario in ["OBSTACLE"]: # 遇到障礙物還是要停
                return 0.0
            if (now - self.cooldown_start_time) > self.COOLDOWN_TIME:
                self.state = "IDLE"
            elif scenario == "NORMAL": # 提早解除
                 self.state = "IDLE"
            return None
            
        return None
    

    @staticmethod  # <--- 請務必加上這行
    def resample_path_numpy(raw_traj, target_speed, dt=0.5):
        """
        raw_traj: (T, 2) numpy
        target_speed: float (m/s)
        """
        if target_speed <= 0.01: # 幾乎靜止
            return np.tile(raw_traj[0:1], (raw_traj.shape[0], 1))

        # 計算累積距離
        dists = np.linalg.norm(raw_traj[1:] - raw_traj[:-1], axis=1)
        cum_dist = np.cumsum(np.concatenate(([0], dists)))
        total_len = cum_dist[-1]
        
        if total_len < 1e-3: return raw_traj # 原地不動
        
        # 擬合曲線 (s -> x,y)
        # 注意: 如果點太少 interp1d 可能會報錯，需檢查 T
        f = interp1d(cum_dist, raw_traj, axis=0, kind='linear', fill_value="extrapolate")
        
        # 新的距離點
        T = raw_traj.shape[0]
        new_s = np.arange(T) * target_speed * dt
        new_s = np.clip(new_s, 0, total_len) # 不超過終點
        
        new_traj = f(new_s)
        return new_traj

    @staticmethod
    def amplify_lateral_swerve_numpy(raw_traj, scale=20.0):
        """
        平滑放大橫向位移 (x 軸)
        raw_traj: (T, 2) numpy array
        scale: 最終要放大的倍率 (預設 2.0 倍)
        """
        T = raw_traj.shape[0]
        new_traj = raw_traj.copy()
        
        # 建立 0 到 1 的時間進度條
        t_steps = np.linspace(0, 1, T)
        
        # 使用 Smoothstep 公式 (3t^2 - 2t^3)
        # 這樣起點和終點的變化率為 0，確保過渡非常平滑
        smooth_curve = 3 * (t_steps ** 2) - 2 * (t_steps ** 3)
        
        # 建立放大倍率陣列：從 1.0 漸變到 scale (例如 2.0)
        multiplier = 1.0 + smooth_curve * (scale - 1.0)
        
        # 只放大橫向 (x 軸) 位移，保持縱向 (y 軸) 前進節奏不變
        new_traj[:, 0] = raw_traj[:, 0] * multiplier
        
        return new_traj
        
        
        
    def generate_autoregressive(
            self,
            rgb_seq: np.ndarray,
            seg_seq: np.ndarray,
            ego_seq: torch.Tensor,
            commands: List[str],
            *,
            gt_trajs: torch.Tensor = None,               
            teacher_forcing_ratio: float = 0.0,
            bx: torch.Tensor = None,  # ★ 新增
            dx: torch.Tensor = None,  # ★ 新增
            bev_dim: List[int] = None # ★ 新增
        ) -> torch.Tensor:
        
        device = ego_seq.device
        B = ego_seq.shape[0]
        T = self.n_future

        # ★ 修改：接收兩個回傳值
        # ★ 修改：接收兩個回傳值
        vis_tokens, bev_input = self.build_vis_tokens(rgb_seq, seg_seq, ego_seq)   
        
        # ★ 新增：將 2D 視覺特徵解碼為 BEV Map (強制轉回 float32 以匹配卷積層權重)
        bev_map = self.bev_decoder(bev_input.to(torch.float32))  # (B, C, H_bev, W_bev)

        text_vis   = self.encode_text_vis(commands)                    # (B,C)

        # ===== Query Dropout：抑制 text/time 捷徑，逼模型用 vis_tokens =====
        text_vis_use = text_vis
        time_q_use   = self.time_queries.unsqueeze(0).expand(B, -1, -1)

        if self.training:
            p_vis = float(getattr(self, "qdrop_vis_p", 0.0))
            m = (torch.rand(vis_tokens.shape[:2], device=device) > p_vis).unsqueeze(-1).to(vis_tokens.dtype)
            vis_tokens = vis_tokens * m

        if self.training:
            # (A) Drop text_vis：部分 batch 直接把文字條件關掉
            p_txt = float(getattr(self, "qdrop_text_p", 0.0))
            if p_txt > 0:
                keep = (torch.rand(B, 1, device=device) > p_txt).to(text_vis.dtype)   # (B,1)
                text_vis_use = text_vis * keep

            # (B) Drop time_queries：部分 batch 把 time query 關掉（更強，但先小一點）
            p_tq = float(getattr(self, "qdrop_timeq_p", 0.0))
            if p_tq > 0:
                keep = (torch.rand(B, 1, 1, device=device) > p_tq).to(time_q_use.dtype)  # (B,1,1)
                time_q_use = time_q_use * keep

        # queries = time_q + text + time_pe
        queries = time_q_use + text_vis_use.unsqueeze(1) + self.time_pe.unsqueeze(0)    # (B,T,C)
        fused = self.cross_attn(queries, vis_tokens)                                    # (B,T,C)


        # 3) 初始隱狀態 h0：concat 全局 pooled 視覺語義 + text
        global_ctx = fused.mean(dim=1)                                          # (B,C)
        h = self.h0_proj(torch.cat([global_ctx, text_vis], dim=-1))            # (B,C)

        # === 並行粗軌跡（B,T,2） ===
        coarse_xy = self.make_coarse_baseline(ego_seq, T=T)   # (B,T,2)  (ego-only)


        # ===== Coarse scheduled drop：部分 batch 把 coarse 關掉，逼 AR 用視覺+GRU =====
        # if self.training:
        #     p_cd = float(getattr(self, "coarse_drop_p", 0.0))
        #     if p_cd > 0:
        #         keep = (torch.rand(B, 1, 1, device=device) > p_cd).to(coarse_xy.dtype)
        #         t0 = T // 2

        #         coarse_xy = coarse_xy * keep
        #         # coarse_xy[:, t0:, :] = coarse_xy[:, t0:, :] * keep   # 只 drop 後半段

        # ===== Coarse scheduled drop：只存 mask，不直接把 coarse 乘 0 =====
        coarse_keep_t = None
        if self.training:
            p_cd = float(self.coarse_drop_p)  # 你是 0.2
            if p_cd > 0:
                coarse_keep_t = (torch.rand(B, T, 1, device=device) > p_cd).to(coarse_xy.dtype)  # (B,T,1)
                coarse_keep_t[:, 0, :] = 1.0   # 永遠保留 coarse 第0點（AR 起點）





        self.last_coarse_xy = coarse_xy                # 之後在 planning 做輔助 loss 用
        self.last_coarse_keep_t = coarse_keep_t.detach() if coarse_keep_t is not None else None


        # === 殘差式 AR 解碼 ===
        traj_xy = []
        gate_stats = []   # 會存每個 t 的 gate 統計（B,）
        mix_gate_stats = []

        prev_xy = coarse_xy[:, 0, :]                   # 用粗軌跡第一點當起點（更穩）
        res_acc = torch.zeros(B, 2, device=device)     # 殘差累積器 r_0 = 0
        if gt_trajs is not None and gt_trajs.size(-1) >= 2:
            gt_xy = gt_trajs[..., :2]

        # AR 狀態
        ar_xy = coarse_xy[:, 0, :].clone()

        for t in range(T):
            ctx_t = fused[:, t, :]   # (B,C)

            embedded_xy = self.xy_post_ln(self.xy_embedder(prev_xy))
            g_ctx = self.g_proj(torch.cat([embedded_xy, ctx_t], dim=-1))
            ctx_gated = g_ctx * ctx_t + (1 - g_ctx) * embedded_xy
            
            # ==========================================================
            # ★ 新增：Local BEV 空間採樣 (上帝視角)
            # ==========================================================
            if bx is not None and dx is not None and bev_dim is not None:
                W_occ, H_occ = bev_dim[1], bev_dim[0]
                
                # 將當前實體座標 prev_xy (m) 轉為 BEV 網格索引
                yy = (prev_xy[..., 1] - bx[0]) / dx[0]
                xx = (prev_xy[..., 0] - bx[1]) / dx[1]
                
                # 正規化到 [-1, 1] 以符合 F.grid_sample 的格式
                y_norm = (yy / (H_occ - 1)) * 2 - 1
                x_norm = (xx / (W_occ - 1)) * 2 - 1
                
                # grid shape: (B, 1, 1, 2)
                grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1).unsqueeze(1)
                
                # 從 bev_map 中挖出當前位置的特徵：(B, C, 1, 1) -> (B, C)
                local_bev = F.grid_sample(
                    bev_map, grid, mode='bilinear', padding_mode='zeros', align_corners=True
                ).squeeze(-1).squeeze(-1)
            else:
                local_bev = torch.zeros(B, self.vis_width, device=device)

            # ★ 修改：把 local_bev 串接進去，送進 LayerNorm 與 GRU
            dec_in = torch.cat([embedded_xy, ctx_gated, local_bev], dim=-1)
            dec_in = self.dec_in_ln(dec_in)

            gate_stats.append(g_ctx.mean(dim=-1))


            h = self.decoder_gru(dec_in, h)
            delta = self.delta_head(h)
            # v1：避免 double-scale（單步上限由 max_step 控制，單位: meters）
            delta = torch.tanh(delta) * self.max_step

            # AR 推進（獨立）
            ar_xy = ar_xy + delta


            # mix gate（AR vs coarse）
            g_t = self.mix_gate(torch.cat([h, ctx_t], dim=-1))  # (B,1)
            # ---- 若 coarse 在此時間步被 drop，就強制走 AR（g_t = 1）----
            if self.training and coarse_keep_t is not None:
                drop_mask = (coarse_keep_t[:, t, :] < 0.5)  # (B,1) bool; True 表示 coarse 被 drop
                g_t = torch.where(drop_mask, torch.ones_like(g_t), g_t)
            # v1：不要 clamp 掉 g_t，否則 coarse_drop 會失效


            coarse_t = coarse_xy[:, t, :]
            final_xy_t = g_t * ar_xy + (1 - g_t) * coarse_t


            traj_xy.append(final_xy_t.unsqueeze(1))
            mix_gate_stats.append(g_t.squeeze(-1))

            # teacher forcing：用 GT 當下一步輸入（只影響 decoder 的輸入，不直接覆蓋輸出）
            prev_xy_next = final_xy_t
            if self.training and (gt_trajs is not None) and (teacher_forcing_ratio > 0.0) and (t < T - 1):
                use_tf = (torch.rand(B, 1, device=device) < teacher_forcing_ratio).to(final_xy_t.dtype)  # (B,1)
                gt_in = gt_xy[:, t, :]  # (B,2)
                prev_xy_next = use_tf * gt_in + (1.0 - use_tf) * final_xy_t
                # 讓 AR 的內部位置也跟上（更穩）
                ar_xy = use_tf * gt_in + (1.0 - use_tf) * ar_xy

            prev_xy = prev_xy_next

        xy = torch.cat(traj_xy, dim=1)
        self.last_mix_gate = torch.stack(mix_gate_stats, dim=1).detach()
                                         # (B,T,2)
        z  = torch.zeros(B, T, 1, device=device)

        # gate_stats: list[T] of (B,2) -> (B,T,2)
        self.last_gate_stats = torch.stack(gate_stats, dim=1).detach()


        return torch.cat([xy, z], dim=-1)                                       # (B,T,3)


    # --------- 前處理+編碼 ---------
    # def _preprocess_clip_tensor(self, imgs_np: np.ndarray) -> torch.Tensor:
    #     # imgs_np: (B,H,W,3) uint8
    #     x = torch.from_numpy(imgs_np).permute(0,3,1,2).float() / 255.0
    #     x = x.to(self.device, non_blocking=True)
    #     if x.shape[-1] != self.input_size:
    #         x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
    #     mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device)[:, None, None]
    #     std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device)[:, None, None]
    #     return (x - mean) / std

    def _preprocess_clip_tensor(self, imgs: torch.Tensor) -> torch.Tensor:
        # imgs: (B, H, W, 3) or (B, 3, H, W)
        if imgs.dim() == 4 and imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 3, 1, 2)     # → (B,3,H,W)
        x = imgs

        # ★ 這裡才可設 channels_last（現在是 4D NCHW 了）
        x = x.contiguous(memory_format=torch.channels_last)
        

        # if x.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
        # print("x")
        # x = x.to(torch.float16) / 255.0
        x = x.to(torch.float32) / 255.0

        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                            mode="bilinear", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device, dtype=x.dtype)[:, None, None]
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device, dtype=x.dtype)[:, None, None]
        return (x - mean) / std



    # def _encode_tokens(self, imgs_t: torch.Tensor) -> torch.Tensor:
    #     self._vis_tokens = None
    #     _ = self.clip.encode_image(imgs_t)  # 觸發 hook
    #     tok = self._vis_tokens
    #     assert tok is not None, "CLIP 視覺 tokens 取得失敗（hook 未觸發）"
    #     return tok / (tok.norm(dim=-1, keepdim=True) + 1e-6)

    def _encode_tokens(self, imgs_t: torch.Tensor) -> torch.Tensor:
        self._vis_tokens = None
        # 如果你解凍了 CLIP visual，就必須允許梯度通過 encode_image
        grad_ctx = contextlib.nullcontext() if getattr(self, "_clip_vis_trainable", False) else torch.no_grad()
        with grad_ctx:
            with autocast(dtype=torch.float16):
                _ = self.clip.encode_image(imgs_t)   # 觸發 hook（FP16）
        tok = self._vis_tokens
        assert tok is not None, "CLIP 視覺 tokens 取得失敗（hook 未觸發）"
        tok = tok.to(torch.float16)
        return tok / (tok.norm(dim=-1, keepdim=True) + 1e-6)

    # --------- 視覺序列 + ego 序列 → 單一大的 token set ---------
    def build_vis_tokens(self, rgb_seq, seg_seq, ego_seq: torch.Tensor) -> torch.Tensor:
        """
        將 (B,T_rf,...) 的 RGB/SEG 與 ego_seq 轉成單一 token set: (B, S_all, C)
        v1：使用向量化版本（一次 encode_image），速度更快、也更穩定。
        """
        # rgb/seg: (B,T,H,W,3) or (B,T,3,H,W) torch tensor on GPU
        if not torch.is_tensor(rgb_seq):
            rgb_seq = torch.from_numpy(rgb_seq)
        if not torch.is_tensor(seg_seq):
            seg_seq = torch.from_numpy(seg_seq)

        device = ego_seq.device
        rgb_seq = rgb_seq.to(device, non_blocking=True)
        seg_seq = seg_seq.to(device, non_blocking=True)

        B, T_rf = rgb_seq.shape[:2]

        # 標準化到 (B,T,3,H,W)
        if rgb_seq.dim() == 5 and rgb_seq.shape[-1] == 3:
            rgb_seq = rgb_seq.permute(0, 1, 4, 2, 3).contiguous()
            seg_seq = seg_seq.permute(0, 1, 4, 2, 3).contiguous()

        rgb_bt = rgb_seq.reshape(B * T_rf, *rgb_seq.shape[2:])  # (B*T,3,H,W)
        seg_bt = seg_seq.reshape(B * T_rf, *seg_seq.shape[2:])  # (B*T,3,H,W)

        rgb_bt = self._preprocess_clip_tensor(rgb_bt)
        seg_bt = self._preprocess_clip_tensor(seg_bt)

        imgs_bt2 = torch.cat([rgb_bt, seg_bt], dim=0)  # (2*B*T,3,224,224)

        tok_all = self._encode_tokens(imgs_bt2)        # (2*B*T, S, C)

        # ... (前方的 _encode_tokens 邏輯保持不變) ...

        # ★ 新增：解析空間維度 (ViT 的 token 數量 S = 1(CLS) + H_f * W_f)
        S = tok_all.shape[1]
        S_spatial = S - 1
        H_f = int(S_spatial ** 0.5)
        W_f = H_f

        # 1. 處理 1D 序列特徵 (給原本的 Cross Attention 用)
        tok_rgb_1d = tok_all[:B * T_rf].reshape(B, T_rf, S, -1)
        tok_seg_1d = tok_all[B * T_rf:].reshape(B, T_rf, S, -1)
        tok_1d = torch.cat([tok_rgb_1d, tok_seg_1d], dim=2)     # (B, T_rf, S*2, C)
        
        ego_embed = self.ego_mlp(ego_seq.to(device))   # (B, T_rf, C)
        tok_1d = tok_1d + ego_embed.unsqueeze(2)       # (B, T_rf, S*2, C)
        tok_1d_flat = tok_1d.reshape(B, T_rf * S * 2, -1)

        # 2. ★ 處理 2D 空間特徵 (給 BEV Decoder 用)
        # 取出空間 tokens (去除 CLS)，形狀為 (2*B*T_rf, H_f*W_f, C)
        spatial_tok = tok_all[:, 1:, :].view(2, B, T_rf, H_f, W_f, -1)
        
        # 我們只取「最新一幀 (T_rf - 1)」的空間特徵來建構當前的 BEV
        rgb_spatial = spatial_tok[0, :, -1, ...] # (B, H_f, W_f, C)
        seg_spatial = spatial_tok[1, :, -1, ...] # (B, H_f, W_f, C)
        
        # Concat 並 Permute 成 CNN 預期的 (B, C, H, W)
        bev_input = torch.cat([rgb_spatial, seg_spatial], dim=-1) # (B, H_f, W_f, 2C)
        bev_input = bev_input.permute(0, 3, 1, 2).contiguous()    # (B, 2C, H_f, W_f)

        return tok_1d_flat, bev_input


    def encode_text_vis(self, commands: List[str]) -> torch.Tensor:
        texts = [self.prompts[c] for c in commands]
        tok = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                text_feat = self.clip.encode_text(tok)
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
        # return self.txt_to_vis(text_feat.to(torch.float16))
        return self.txt_to_vis(text_feat.to(self.txt_to_vis.weight.dtype))

    
    def generate(self, rgb_seq, seg_seq, ego_seq, commands, gt_trajs: torch.Tensor = None, bx=None, dx=None, bev_dim=None):
        # Teacher Forcing 比例：訓練期可 >0（建議前期 0.7 → 後期 0 的 curriculum；先用固定值也可）
        tf_ratio = float(getattr(self, "AR_TF_RATIO", 0.0))
        if not self.training:
            tf_ratio = 0.0

        return self.generate_autoregressive(
            rgb_seq, seg_seq, ego_seq, commands,
            gt_trajs=gt_trajs,
            teacher_forcing_ratio=tf_ratio,
            bx=bx, dx=dx, bev_dim=bev_dim  # ★ 往下傳遞
        )



class VLM_STP3_Gen(nn.Module):
    """
    直接生成軌跡的 STP3 最小相容 wrapper：
      - forward()：僅負責把多幀序列與 ego 編碼所需的資料暫存
      - planning()：輸出生成軌跡，並計算 L2 + 碰撞率損失
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD
        self.n_future = cfg.N_FUTURE_FRAMES

        self.input_size = int(getattr(cfg, "CLIP_INPUT_SIZE", 224))
        clip_name = getattr(cfg, "CLIP_MODEL", "ViT-B-32")
        print("clip_name : ",clip_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vlm = VLM_Generative(clip_name, device, n_future=self.n_future, input_size=self.input_size)

        # 用於把 (x,y) 對映到 BEV 索引，和 metrics.PlanningMetric 一致
        dx, bx, _ = gen_dx_bx(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
        dx, bx = dx[:2], bx[:2]
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        _, _, bev_dim = calculate_birds_eye_view_parameters(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
        self.bev_dim = bev_dim.numpy().tolist()  # [H, W]

        # 暫存序列資料
        self._last_rgb_seq = None  # (B,T_rf,H,W,3) np.uint8
        self._last_seg_seq = None  # (B,T_rf,H,W,3) np.uint8
        self._last_ego_seq = None  # (B,T_rf,4) torch

        # 偽 encoder 輸出以符合原介面
        self.encoder_out_channels = 64
        self.fake_cam_front = nn.Parameter(torch.zeros(1, self.encoder_out_channels, 60, 28), requires_grad=False)

        # ====== 建議加入的超參，之後可移到 cfg ======
        self.vlm.AR_TF_RATIO = float(getattr(cfg, "AR_TF_RATIO", 0.5))  # 訓練期 teacher forcing 比例
        # self.vlm.step_scale  = float(getattr(cfg, "AR_STEP_SCALE", 2.0))  # 原本1
        self.vlm.max_step    = float(getattr(cfg, "AR_MAX_STEP", 8.0))

        print("model : image_force")
        print("dx:", self.dx, "bx:", self.bx)

        # training step counter（用於低頻率啟用 counterfactual loss，避免單卡 16G 爆）
        self._train_step = 0

        # 把 wrapper 的尺度資訊丟給 VLM_Generative（coarse baseline 需要用到）
        self.vlm.EGO_SCALE_M = float(getattr(cfg, "EGO_SCALE_M", 5.0))
        self.vlm.SAMPLE_DT = float(getattr(cfg, "SAMPLE_DT", 0.5))

        # v1 建議：解凍 CLIP visual 最後 1~2 層（可在 cfg 設定 VIS_UNFREEZE_K）
        self.vlm.VIS_UNFREEZE_K = int(getattr(cfg, "VIS_UNFREEZE_K", 1))
        self.vlm.apply_visual_unfreeze(self.vlm.VIS_UNFREEZE_K)

    def inflate_occupancy_1cell(self, occupancy: torch.Tensor) -> torch.Tensor:
        """
        硬膨脹 1 格（dx=0.5m）
        occupancy: (B,T,H,W) {0,1}
        """
        B, T, H, W = occupancy.shape
        occ = occupancy.float().view(B*T, 1, H, W)

        # 3x3 max-pool = 膨脹 1 cell
        occ_inf = F.max_pool2d(occ, kernel_size=3, stride=1, padding=1)
        occ_inf = (occ_inf > 0.5).float().view(B, T, H, W)
        return occ_inf
    

    def make_soft_cost_map(self, occupancy: torch.Tensor, blur_iters: int = 2) -> torch.Tensor:
        """
        將 0/1 occupancy 轉成 soft cost map
        blur_iters 控制影響範圍（dx=0.5m 時建議 2~3）
        """
        B, T, H, W = occupancy.shape
        cost = occupancy.float().view(B*T, 1, H, W)

        # 多次 avg_pool，近似距離衰減
        for _ in range(blur_iters):
            cost = F.avg_pool2d(cost, kernel_size=3, stride=1, padding=1)

        # 壓到 [0,1]，避免數值爆炸
        cost = torch.clamp(cost, 0.0, 1.0)

        return cost.view(B, T, H, W)


    
    def collision_loss_soft(self, trajs_xy: torch.Tensor, occupancy: torch.Tensor, return_per_t: bool = False) -> torch.Tensor:
        """
        最終版 collision loss：
        - 膨脹 1 格（0.5m）
        - soft 距離懲罰
        - grid_sample 可微
        """
        B, T, _ = trajs_xy.shape
        H, W = occupancy.shape[-2:]

        # 1) 硬安全邊界
        occ_inf = self.inflate_occupancy_1cell(occupancy)

        # 2) 軟距離 cost map
        cost_map = self.make_soft_cost_map(occ_inf, blur_iters=2)  # 可調 2~3

        # 3) 連續座標 -> 連續格座標
        yy = (trajs_xy[..., 1] - self.bx[0]) / self.dx[0]  # (B,T)
        xx = (trajs_xy[..., 0] - self.bx[1]) / self.dx[1]  # (B,T)

        # 4) normalize to [-1,1]（對齊你原 indexing）
        x_norm = (xx / (W - 1)) * 2 - 1
        y_norm = (yy / (H - 1)) * 2 - 1

        grid = torch.stack([x_norm, y_norm], dim=-1).view(B*T, 1, 1, 2)

        # 5) grid_sample（可微！）
        cost = cost_map.view(B*T, 1, H, W)
        sampled = F.grid_sample(
            cost,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )

        collision_cost = sampled.view(B, T)

        # 6) 回傳
        if return_per_t:
            return collision_cost  # (B,T)

        # 聚合（先用 mean，之後可試 max）
        # return collision_cost.mean()
        return (collision_cost ** 3).mean()


    

    # --------- 來自 mult_planner 的 forward 風格（多幀輸入） ---------
    def forward(self, image, intrinsics, extrinsics, future_egomotion, *, rgb_224_seq, seg_224_seq):
        B, T_rf = rgb_224_seq.shape[:2]
        device = future_egomotion.device

        # self._last_rgb_seq = rgb_224_seq.cpu().numpy()
        # self._last_seg_seq = seg_224_seq.cpu().numpy()

        # 直接在 GPU 留著，後續就不需要 from_numpy + to(device) 的往返
        # self._last_rgb_seq = rgb_224_seq.to(device, non_blocking=True).to(torch.float16)
        # self._last_seg_seq = seg_224_seq.to(device, non_blocking=True).to(torch.float16)

        self._last_rgb_seq = rgb_224_seq.to(device, non_blocking=True)
        self._last_seg_seq = seg_224_seq.to(device, non_blocking=True)


        # 構造每幀 → 現在幀的 ego 4D 輸入 (dx,dy,sin(dyaw),cos(dyaw))
        from stp3.utils.geometry import pose_vec2mat, mat2pose_vec
        fego = future_egomotion[:, :self.receptive_field, :]  # (B,T_rf,6)
        ego_seq_embed = []
        for t in range(T_rf):
            if t == T_rf - 1:
                dx = torch.zeros(B, 1, device=device); dy = torch.zeros(B, 1, device=device); dyaw = torch.zeros(B, 1, device=device)
            else:
                mats = [pose_vec2mat(fego[:, k, :]) for k in range(t, T_rf - 1)]
                M = mats[0]
                for m in mats[1:]:
                    M = torch.bmm(M, m)
                pose = mat2pose_vec(M)
                dx = pose[:, 0:1]; dy = pose[:, 1:2]; dyaw = pose[:, 5:6]
                dyaw = (dyaw + torch.pi) % (2*torch.pi) - torch.pi
            s_m = getattr(self.cfg, "EGO_SCALE_M", 5.0)
            ex, ey = dx / s_m, dy / s_m
            sy, cy = torch.sin(dyaw), torch.cos(dyaw)
            ego_seq_embed.append(torch.cat([ex, ey, sy, cy], dim=1))
        self._last_ego_seq = torch.stack(ego_seq_embed, dim=1).detach()  # (B,T_rf,4)

        return {}, self._last_rgb_seq  # 與原 trainer 介面保持一致

    # --------- 小工具：把連續 (x,y) 取樣到 occupancy 上，得到碰撞率 ---------
    def occupancy_collision_rate(self, trajs_xy: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        """
        trajs_xy: (B,T,2) 連續座標（以公尺，與 cfg.LIFT 邊界一致）
        occupancy: (B,T,H,W) 0/1 張量（trainer 已經準備好）
        回傳：碰撞比例（標量張量）
        """
        device = trajs_xy.device
        B, T, _ = trajs_xy.shape
        H, W = occupancy.shape[-2:]

        # 連續 → 整數索引
        # 注意 metrics 中對應： yi ↔ YBound 索引, xi ↔ XBound 索引
        yy = ((trajs_xy[..., 1] - self.bx[0]) / self.dx[0]).long()  # (B,T)
        xx = ((trajs_xy[..., 0] - self.bx[1]) / self.dx[1]).long()  # (B,T)
        yy = torch.clamp(yy, 0, H-1)
        xx = torch.clamp(xx, 0, W-1)
        ti = torch.arange(T, device=device).view(1, T).expand(B, T)
        bi = torch.arange(B, device=device).view(B, 1).expand(B, T)

        hit = occupancy[bi, ti, yy, xx].float()  # (B,T)
        coll_rate = hit.mean()  # 平均碰撞率
        return coll_rate
    
    def planning(self, *, bev_rgbs, trajs, gt_trajs, commands, target_points, occupancy=None):
        """
        整合了 VLM 感知 (Plan A) 與 生成式規劃 (Plan B) 的最終版 Planning
        """
        assert self._last_rgb_seq is not None and self._last_seg_seq is not None and self._last_ego_seq is not None, \
            "缺序列影像或 ego 嵌入：請先呼叫 forward()"

        device = gt_trajs.device
        



        # === C. 生成原始軌跡 (Plan B) ===
        pred = self.vlm.generate(
            self._last_rgb_seq,
            self._last_seg_seq,
            self._last_ego_seq.to(device),
            commands, 
            gt_trajs=gt_trajs,
            bx=self.bx,          # ★ 傳入 X,Y 偏移
            dx=self.dx,          # ★ 傳入解析度
            bev_dim=self.bev_dim # ★ 傳入長寬
        ).to(device)


        # === D. 幾何重採樣 (Plan A) ===
        # 1. 先複製一份作為「最終軌跡」，確保變數存在且維度正確 (B, T, 3)
        
        with torch.no_grad():
            final_traj = pred.clone()
            # === A. 初始化變數 ===
            is_inference = (not self.training)
            final_commands = list(commands) 
            vlm_target_speed = None 

            # === B. VLM 感知與決策 (僅在推論時執行) ===
            if is_inference:
                # 1. 提取當前幀
                # print("5555555555555")
                curr_img_raw = self._last_rgb_seq[0, -1, ...] 
                if curr_img_raw.dim() == 3:
                    curr_img_input = curr_img_raw.unsqueeze(0)
                else:
                    curr_img_input = curr_img_raw
                curr_img_tensor = self.vlm._preprocess_clip_tensor(curr_img_input)

                # 2. VLM 感知
                scenario, probs = self.vlm.detect_context(curr_img_tensor)
                
                # 3. 估算當前速度
                ego_scale = getattr(self.cfg, "EGO_SCALE_M", 5.0)
                last_ego = self._last_ego_seq[0, -1, :2] * ego_scale
                current_speed = torch.norm(last_ego).item() / 0.5
                
                vlm_target_speed = None 
                vlm_amplify_scale = 2.0  # ▼ 新增：預設橫向位移不放大 (1.0倍)

                # 4. 決策邏輯
                if scenario == "BUMP":
                    final_commands = ["SLOW"] * len(commands)
                    vlm_target_speed = 1.5 
                elif scenario in ["OBSTACLE", "OBSTACLE_RIGHT", "OBSTACLE_LEFT"]: # 涵蓋所有障礙物情境
                    final_commands = ["SLOW"] * len(commands)
                    # vlm_target_speed = 1.0 
                    vlm_amplify_scale = 2.0  # ▼ 新增：偵測到障礙物，將橫向位移放大 2 倍

                stop_speed_limit = self.vlm.update(scenario, current_speed)
                if stop_speed_limit is not None:
                    final_commands = ["STOP"] * len(commands)
                    vlm_target_speed = stop_speed_limit

                # ▼ 修改判斷式，加入 vlm_amplify_scale > 1.0 的觸發條件
                if is_inference and (vlm_target_speed is not None or vlm_amplify_scale > 1.0):
                    # 取出 (x, y) 轉為 Numpy
                    raw_traj_np = pred[0, :, :2].detach().cpu().numpy()
                    
                    # 步驟 A：先平滑放大橫向位移
                    if vlm_amplify_scale > 1.0:
                        raw_traj_np = self.vlm.amplify_lateral_swerve_numpy(raw_traj_np, scale=vlm_amplify_scale)
                    
                    # 步驟 B：再執行速度重採樣 (如有需要減速)
                    if vlm_target_speed is not None:
                        raw_traj_np = self.vlm.resample_path_numpy(raw_traj_np, vlm_target_speed, dt=0.5)
                    
                    # 寫回 Tensor
                    resampled_tensor = torch.from_numpy(raw_traj_np).float().to(device)
                    final_traj[0, :, :2] = resampled_tensor


        # === E. 計算 Loss (使用原始 pred) ===
        # Loss 計算保持不變，針對 pred (模型輸出) 進行監督
        err = ((pred[..., :2] - gt_trajs[..., :2])**2).sum(dim=-1).sqrt()
        T_len = err.shape[1]
        w = torch.linspace(1.3, 1.0, T_len, device=err.device)
        l2 = (err * w).mean()

        pred_xy = pred[..., :2]
        gt_xy   = gt_trajs[..., :2]
        pred_d  = torch.cat([pred_xy[:, :1] - 0, pred_xy[:, 1:] - pred_xy[:, :-1]], dim=1)
        gt_d    = torch.cat([gt_xy[:, :1]  - 0, gt_xy[:, 1:]  - gt_xy[:, :-1]],  dim=1)
        wv = torch.linspace(1.4, 1.0, T_len, device=pred_d.device)
        vel_l2 = ((pred_d - gt_d).pow(2).sum(-1) * wv).mean()

        coll_rate = torch.tensor(0.0, device=device)
        risk_t = None
        if occupancy is not None:
            risk_t = self.collision_loss_soft(pred[..., :2], occupancy, return_per_t=True)  # (B,T)
            coll_rate = (risk_t ** 3).mean()

        vel = pred[..., :2] - torch.cat([pred[:, :1, :2], pred[:, :-1, :2]], dim=1)
        smooth = (vel[:, 1:] - vel[:, :-1]).pow(2).sum(-1).mean()

        
        coarse_xy = getattr(self.vlm, "last_coarse_xy", None)
        coarse_l2 = torch.tensor(0.0, device=device)
        if coarse_xy is not None:
            err_c = ((coarse_xy - gt_trajs[..., :2])**2).sum(dim=-1).sqrt()
            T_c = err_c.shape[1]
            w_c = torch.linspace(1.6, 1.0, T_c, device=device)
            coarse_l2 = (err_c * w_c).mean()


        # === 方向一致性 Loss：讓預測的最後一點符合 command 幾何條件 ===
        # pred_last_x: (B,)
        pred_last_x = pred[..., 0][:, -1]

        # 將 batch 裡的 command 字串轉成 mask
        cmd_right   = torch.tensor([c == 'RIGHT'   for c in commands], device=device)
        cmd_left    = torch.tensor([c == 'LEFT'    for c in commands], device=device)
        cmd_forward = torch.tensor([c == 'FORWARD' for c in commands], device=device)

        # 閾值：跟你標 command 的 2m 一樣，或略微放寬一點
        margin_turn = getattr(self.cfg, 'DIR_MARGIN_TURN', 1.8)      # 右轉/左轉最小側向位移
        margin_fwd  = getattr(self.cfg, 'DIR_MARGIN_FORWARD',2.0)   # 直走允許的最大側向偏移

        dir_loss = torch.tensor(0.0, device=device)

        # 1. 恢復 RIGHT 和 LEFT 的強制性（這很重要）
        if cmd_right.any():
            x_r = pred_last_x[cmd_right]
            loss_r = F.relu(margin_turn - x_r).mean()
            dir_loss = dir_loss + loss_r

        if cmd_left.any():
            x_l = pred_last_x[cmd_left]
            loss_l = F.relu(x_l + margin_turn).mean()
            dir_loss = dir_loss + loss_l

        # 2. FORWARD 動態放寬：如果有高風險，允許較大的側向偏移來避障
        if cmd_forward.any():
            x_f = pred_last_x[cmd_forward].abs()
            
            if occupancy is not None and risk_t is not None:
                # 取得每個 FORWARD 樣本在未來的最大風險值 (B_fwd,)
                risk_max = risk_t[cmd_forward].max(dim=-1)[0].detach()
                # 風險越高，容忍的 margin 越大（例如基礎 2.0，有危險時最高放寬到 4.0）
                dynamic_margin = margin_fwd + 2.0 * risk_max 
                loss_f = F.relu(x_f - dynamic_margin).mean()
            else:
                loss_f = F.relu(x_f - margin_fwd).mean()
                
            dir_loss = dir_loss + loss_f

        # === 方向成功率指標（不進梯度，只用來監控） ===
        # success[i] = 該 sample 是否符合它自己的 command 幾何條件 (0/1)
        success = torch.zeros_like(pred_last_x, dtype=torch.float32, device=device)

        if cmd_right.any():
            x_r = pred_last_x[cmd_right]
            success[cmd_right] = (x_r >= margin_turn).float()

        if cmd_left.any():
            x_l = pred_last_x[cmd_left]
            success[cmd_left] = (x_l <= -margin_turn).float()

        if cmd_forward.any():
            x_f = pred_last_x[cmd_forward].abs()
            success[cmd_forward] = (x_f <= margin_fwd).float()

        # 只看轉彎樣本 (LEFT/RIGHT) 的成功率：避免被大量 FORWARD 稀釋
        turn_mask = cmd_right | cmd_left
        if turn_mask.any():
            dir_acc_turn = success[turn_mask].mean()
        else:
            dir_acc_turn = torch.tensor(float("nan"), device=device)

        # （可選）整體成功率：包含 FORWARD
        dir_acc_all = success.mean()


        if (is_inference) or (occupancy is None):
            hard_col = torch.tensor(0.0, device=device)
        else:
            with torch.no_grad():
                hard_col = self.occupancy_collision_rate(pred[..., :2], occupancy)

        lam_l2 = getattr(self.cfg, 'LOSS_L2_W', 8.0)     # l2
        lam_col = getattr(self.cfg, 'LOSS_COL_W', 30.0)   # 碰撞率
        lam_smo = getattr(self.cfg, 'LOSS_SMO_W', 0.1)   # 平滑
        lam_vel = getattr(self.cfg, 'LOSS_VEL_W', 0.6)   # 速度（間隔）
        lam_dir = getattr(self.cfg, 'LOSS_DIR_W', 8.0)   # 方向一致性 loss 權重
        lam_coarse = getattr(self.cfg, 'LOSS_COARSE_W', 0.0)  # 粗軌跡 head 的 L2

        # print("lam_col",lam_col)

        # loss = lam_l2 * l2 + lam_col * coll_rate + lam_smo * smooth + lam_vel * vel_l2 + lam_coarse * coarse_l2
        loss = (
            lam_l2 * l2 +
            lam_col * coll_rate +
            lam_smo * smooth +
            lam_vel * vel_l2 +
            lam_coarse * coarse_l2 +
            lam_dir * dir_loss
        )

        # =========================
        # v1: 強制模型「必須用影像」的額外 loss（不動資料）
        # =========================
        if self.training:
            self._train_step += 1

        # 反事實影像敏感 loss：同一筆資料用「黑影像」跑一次，並要求高風險步差異變大
        lam_vis = float(getattr(self.cfg, "LOSS_VIS_SENS_W", 0.5))
        vis_every = int(getattr(self.cfg, "VIS_SENS_EVERY", 4))          # 每 N step 做一次，省 VRAM
        risk_thres = float(getattr(self.cfg, "VIS_RISK_THRES", 0.15))    # risk_t > thres 才強迫看影像
        vis_margin = float(getattr(self.cfg, "VIS_SENS_MARGIN", 0.6))    # meters，差異至少要 > margin

        vis_sens = torch.tensor(0.0, device=device)
        risk_mask = None
        if (risk_t is not None):
            risk_w = torch.clamp(risk_t.detach(), 0.0, 1.0)  # (B,T)
            risk_mask = (risk_w > risk_thres).float()

            if self.training and lam_vis > 0 and (self._train_step % max(1, vis_every) == 0) and (risk_mask.sum() > 0):
                # 保存正常 forward 的 gate 結果（避免被黑影像 forward 覆蓋）
                saved_mix = getattr(self.vlm, "last_mix_gate", None)
                saved_keep = getattr(self.vlm, "last_coarse_keep_t", None)

                rgb_blk = torch.zeros_like(self._last_rgb_seq)
                seg_blk = torch.zeros_like(self._last_seg_seq)

                with torch.no_grad():
                    pred_blk = self.vlm.generate_autoregressive(
                        rgb_blk, seg_blk, self._last_ego_seq.to(device), commands,
                        gt_trajs=None,                    # 不給 GT
                        teacher_forcing_ratio=0.0         # 強制不用 TF
                    ).to(device)


                # restore
                if saved_mix is not None:
                    self.vlm.last_mix_gate = saved_mix
                if saved_keep is not None:
                    self.vlm.last_coarse_keep_t = saved_keep

                diff = torch.norm(pred[..., :2] - pred_blk[..., :2], dim=-1)  # (B,T)
                # hinge：diff 至少要大於 margin（只在高風險步）
                vis_sens = (F.relu(vis_margin - diff) * risk_mask * risk_w).sum() / (risk_mask.sum() + 1e-6)
                loss = loss + lam_vis * vis_sens

        # gate regularization：高風險時，鼓勵 mix_gate -> 1（更信 AR/影像）
        lam_gate_risk = float(getattr(self.cfg, "LOSS_GATE_RISK_W", 0.3))
        lam_gate_drop = float(getattr(self.cfg, "LOSS_GATE_DROP_W", 0.5))

        mix_gate = getattr(self.vlm, "last_mix_gate", None)          # (B,T)
        keep_t   = getattr(self.vlm, "last_coarse_keep_t", None)     # (B,T,1)

        gate_risk_loss = torch.tensor(0.0, device=device)
        gate_drop_loss = torch.tensor(0.0, device=device)

        if (mix_gate is not None) and (risk_mask is not None) and (lam_gate_risk > 0) and (risk_mask.sum() > 0):
            gate_risk_loss = (((1.0 - mix_gate) * risk_mask)).sum() / (risk_mask.sum() + 1e-6)
            loss = loss + lam_gate_risk * gate_risk_loss

        if self.training and (mix_gate is not None) and (keep_t is not None) and (lam_gate_drop > 0):
            # print("455456546")
            drop = (keep_t.squeeze(-1) < 0.5).float()   # (B,T)
            if drop.sum() > 0:
                gate_drop_loss = (((1.0 - mix_gate) * drop)).sum() / (drop.sum() + 1e-6)
                loss = loss + lam_gate_drop * gate_drop_loss

        loss_dict = {
            "l2": l2,
            "vel_l2": vel_l2,
            "smooth": smooth,
            "collision": coll_rate,
            "coarse_l2": coarse_l2,
            "dir_loss": dir_loss,
            "acc_turn": dir_acc_turn,   # ★ 新增：只看 LEFT/RIGHT 的成功率
            "acc_all": dir_acc_all,     # ★ 新增：包含 FORWARD 在內的成功率
            "col_with_lam": lam_col * coll_rate,
            "hard_collision": hard_col,
            "vis_sens": vis_sens.detach() if torch.is_tensor(vis_sens) else torch.tensor(0.0, device=device),
            "gate_risk_loss": gate_risk_loss.detach() if torch.is_tensor(gate_risk_loss) else torch.tensor(0.0, device=device),
            "gate_drop_loss": gate_drop_loss.detach() if torch.is_tensor(gate_drop_loss) else torch.tensor(0.0, device=device),
        }

        # ===== Gate penalty：coarse_drop 的時間步希望 mix_gate -> 1 =====
        lam_gate = float(getattr(self.cfg, "LOSS_GATE_W", 1.0))  # 你可在 cfg 設，先用 1.0

        mix_gate = getattr(self.vlm, "last_mix_gate", None)          # (B,T)
        keep_t   = getattr(self.vlm, "last_coarse_keep_t", None)     # (B,T,1)

        # if self.training and mix_gate is not None and keep_t is not None:
        #     drop = (keep_t.squeeze(-1) < 0.5).float()   # (B,T)
        #     gate_pen = ((1.0 - mix_gate) * drop).mean()
        #     loss = loss + lam_gate * gate_pen
        #     loss_dict["gate_pen"] = gate_pen.detach()


        mix_gate = getattr(self.vlm, "last_mix_gate", None)
        if mix_gate is not None:
            loss_dict["mix_gate_mean_over_time"] = mix_gate.mean()
            loss_dict["mix_gate_mean_last5"] = mix_gate[:, -5:].mean() if mix_gate.size(1) >= 5 else mix_gate.mean()

        ctx_gate = getattr(self.vlm, "last_gate_stats", None)
        if ctx_gate is not None:
            loss_dict["ctx_gate_mean_over_time"] = ctx_gate.mean()
            loss_dict["ctx_gate_mean_last5"] = ctx_gate[:, -5:].mean() if ctx_gate.size(1) >= 5 else ctx_gate.mean()


        # 回傳：Loss, 原始生成(pred), 最終安全軌跡(final_traj), 其他...
        # 這樣你就同時有「模型原本想走的」跟「被 VLM 修正後安全的」兩條線
        # return loss, pred, final_traj, torch.tensor(0.0, device=device), loss_dict
        # (將最後一行 return 修改成這樣)
        return loss, pred, final_traj, scenario, loss_dict