import os
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
        self.qdrop_text_p = float(getattr(self, "QDROP_TEXT_P", 0.2))   # 0~0.4
        self.qdrop_vis_p = float(getattr(self, "QDROP_VIS_P", 0.2))   # 0~0.4
        self.qdrop_timeq_p = float(getattr(self, "QDROP_TIMEQ_P", 0.1)) # 0~0.3
        self.coarse_drop_p = float(getattr(self, "COARSE_DROP_P", 0.2)) # 0~0.5


        # 凍結 CLIP，僅解凍視覺最後 K 層 & LN
        for p in self.clip.parameters():
            p.requires_grad = False
        # K_UNFREEZE = 2
        # if hasattr(self.clip, "visual") and hasattr(self.clip.visual, "transformer"):
        #     blocks = self.clip.visual.transformer.resblocks
        #     for blk in blocks[-K_UNFREEZE:]:
        #         for p in blk.parameters():
        #             p.requires_grad = True
        #     for n, p in self.clip.visual.named_parameters():
        #         if ("ln_post" in n) or ("ln_pre" in n):
        #             p.requires_grad = True

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

        self.decoder_gru = nn.GRUCell(input_size=self.vis_width * 2, hidden_size=self.vis_width)

        # 以「增量」方式輸出 Δx, Δy，比直接絕對座標更穩
        self.delta_head = nn.Sequential(
            nn.Linear(self.vis_width, self.vis_width),
            nn.GELU(),
            nn.Linear(self.vis_width, 2)
        )

        self.xy_embedder = nn.Sequential(
            nn.Linear(2, self.vis_width // 4),  # 2 for (x,y)
            nn.GELU(),
            nn.Linear(self.vis_width // 4, self.vis_width)
        )

        # ====== 並行「粗軌跡」頭：給 AR 做殘差修正的基底 ======
        self.traj_coarse = nn.Sequential(
            nn.Linear(self.vis_width, self.vis_width),
            nn.GELU(),
            nn.Linear(self.vis_width, 2)
        )

        self.g_proj = nn.Sequential(nn.Linear(self.vis_width * 2, self.vis_width), nn.GELU(),
            nn.Linear(self.vis_width, self.vis_width), nn.Sigmoid())
        
        self.xy_post_ln = nn.LayerNorm(self.vis_width)
        self.dec_in_ln  = nn.LayerNorm(self.vis_width * 2)

        # 在 __init__ 裡加
        self.mix_gate = nn.Sequential(
            nn.Linear(self.vis_width * 2, self.vis_width),
            nn.GELU(),
            nn.Linear(self.vis_width, 1),
            nn.Sigmoid()
        )

        # （可選）讓一開始偏向 coarse：把最後一層 bias 初始化成負值
        nn.init.constant_(self.mix_gate[-2].bias, -2.0)  # sigmoid(-2)≈0.12，初期 mostly coarse



        

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
                "a speed bump on the road",
                "a road hump",
                "speed breaker on the ground",
                "yellow and black speed bump"
            ],
            "OBSTACLE": [
                "an unexpected obstacle blocking the road",
                "debris or trash on the road",
                "a soccer ball or box on the road",
                "random object on the street"
            ],
            "STOP": [
                "a red stop sign",
                "an octagon stop sign",
                "traffic sign saying stop",
                "red traffic light",           # 加入紅燈描述
                "traffic signal intersection", # 加入路口描述
                "stop signal"
            ]
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
        
        
        
    def generate_autoregressive(
        self,
        rgb_seq: np.ndarray,
        seg_seq: np.ndarray,
        ego_seq: torch.Tensor,
        commands: List[str],
        *,
        gt_trajs: torch.Tensor = None,               # (B,T,3) or (B,T,2)
        teacher_forcing_ratio: float = 0.0,          # 訓練時 > 0；驗證/測試為 0
    ) -> torch.Tensor:
        """
        回傳: (B,T,3) ; 第三維 z 先放 0
        """
        device = ego_seq.device
        B = ego_seq.shape[0]
        T = self.n_future

        vis_tokens = self.build_vis_tokens(rgb_seq, seg_seq, ego_seq)   # (B,S,C)
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
        coarse_xy = self.traj_coarse(fused)            # (B,T,2)


        # ===== Coarse scheduled drop：部分 batch 把 coarse 關掉，逼 AR 用視覺+GRU =====
        # if self.training:
        #     p_cd = float(getattr(self, "coarse_drop_p", 0.0))
        #     if p_cd > 0:
        #         keep = (torch.rand(B, 1, 1, device=device) > p_cd).to(coarse_xy.dtype)
        #         t0 = T // 2

        #         coarse_xy = coarse_xy * keep
        #         # coarse_xy[:, t0:, :] = coarse_xy[:, t0:, :] * keep   # 只 drop 後半段

        if self.training:
            p_cd = self.coarse_drop_p
            if p_cd > 0:
                keep_t = (torch.rand(B, T, 1, device=device) > p_cd).to(coarse_xy.dtype)  # (B,T,1)
                keep_t[:, 0, :] = 1.0   # 永遠保留 coarse 的第0點，當AR起點
                coarse_xy = coarse_xy * keep_t




        self.last_coarse_xy = coarse_xy                # 之後在 planning 做輔助 loss 用

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
            dec_in = torch.cat([embedded_xy, ctx_gated], dim=-1)
            dec_in = self.dec_in_ln(dec_in)

            gate_stats.append(g_ctx.mean(dim=-1))  # (B,)  每步存一個值


            h = self.decoder_gru(dec_in, h)
            delta = self.delta_head(h)
            delta = torch.tanh(delta / max(1e-6, self.step_scale)) * self.max_step

            # AR 推進（獨立）
            if t > 0:
                ar_xy = ar_xy + delta * self.step_scale

            coarse_t = coarse_xy[:, t, :]

            # mix gate（AR vs coarse）
            g_t = self.mix_gate(torch.cat([h, ctx_t], dim=-1))  # (B,1)
            final_xy_t = g_t * ar_xy + (1 - g_t) * coarse_t

            traj_xy.append(final_xy_t.unsqueeze(1))
            mix_gate_stats.append(g_t.squeeze(-1))

            prev_xy = final_xy_t

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
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                _ = self.clip.encode_image(imgs_t)   # 觸發 hook（FP16）
        tok = self._vis_tokens
        assert tok is not None, "CLIP 視覺 tokens 取得失敗（hook 未觸發）"
        return tok.to(torch.float16) / (tok.norm(dim=-1, keepdim=True) + 1e-6)

    # --------- 視覺序列 + ego 序列 → 單一大的 token set ---------
    def build_vis_tokens(self, rgb_seq: np.ndarray, seg_seq: np.ndarray, ego_seq: torch.Tensor) -> torch.Tensor:
        B, T_rf = rgb_seq.shape[:2]
        all_tokens = []
        for t in range(T_rf):
            rgb_t = self._preprocess_clip_tensor(rgb_seq[:, t])
            seg_t = self._preprocess_clip_tensor(seg_seq[:, t])
            tok_rgb = self._encode_tokens(rgb_t)  # (B,S,C)
            tok_seg = self._encode_tokens(seg_t)
            tok = torch.cat([tok_rgb, tok_seg], dim=1)  # (B,S1+S2,C)
            ego_embed = self.ego_mlp(ego_seq[:, t].to(self.device))  # (B,C)
            tok = tok + ego_embed.unsqueeze(1)
            all_tokens.append(tok)
        vis_tokens = torch.cat(all_tokens, dim=1)  # (B, T_rf*(S1+S2), C)
        return vis_tokens

    # def build_vis_tokens(self, rgb_seq: torch.Tensor, seg_seq: torch.Tensor, ego_seq: torch.Tensor) -> torch.Tensor:
    #     # rgb/seg: (B, T, H, W, 3) 或 (B, T, 3, H, W)，且已在 GPU
    #     B, T_rf = rgb_seq.shape[:2]
    #     # 標準化到 (B,T,3,H,W)
    #     if rgb_seq.dim() == 5 and rgb_seq.shape[-1] == 3:
    #         rgb_seq = rgb_seq.permute(0,1,4,2,3)
    #         seg_seq = seg_seq.permute(0,1,4,2,3)
    #     # 展平到 (B*T,3,H,W)
    #     rgb_bt = rgb_seq.reshape(B*T_rf, *rgb_seq.shape[2:])
    #     seg_bt = seg_seq.reshape(B*T_rf, *seg_seq.shape[2:])

    #     rgb_bt = self._preprocess_clip_tensor(rgb_bt)               # (B*T,3,224,224)
    #     seg_bt = self._preprocess_clip_tensor(seg_bt)

    #     # 兩種模態合併一次丟進 CLIP
    #     imgs_bt2 = torch.cat([rgb_bt, seg_bt], dim=0)               # (2*B*T,3,224,224)

    #     # 取 tokens（一次前傳，省掉 Python 迴圈 & 多次 H2D）
    #     self._vis_tokens = None
    #     with torch.no_grad():
    #         with autocast(dtype=torch.float16):
    #             _ = self.clip.encode_image(imgs_bt2)
    #     tok_all = self._vis_tokens.to(torch.float16)                 # (2*B*T, S, C)
    #     tok_all = tok_all / (tok_all.norm(dim=-1, keepdim=True) + 1e-6)

    #     # 還原兩個模態
    #     S = tok_all.shape[1]
    #     tok_rgb = tok_all[:B*T_rf].reshape(B, T_rf, S, -1)
    #     tok_seg = tok_all[B*T_rf:].reshape(B, T_rf, S, -1)
    #     tok = torch.cat([tok_rgb, tok_seg], dim=2)                   # (B,T_rf,S1+S2,C)

    #     # ego 嵌入（每幀一個 C，broadcast 到 S 維）
    #     ego_embed = self.ego_mlp(ego_seq.to(self.device))            # (B,T_rf,C)
    #     tok = tok + ego_embed.unsqueeze(2)                           # (B,T_rf,S*,C)

    #     # 展平成 (B, T_rf*(S1+S2), C)
    #     B, T_rf, S_all, C = tok.shape
    #     return tok.reshape(B, T_rf*S_all, C)


    # --------- 文本條件 ---------
    # def encode_text_vis(self, commands: List[str]) -> torch.Tensor:
    #     texts = [self.prompts[c] for c in commands]
    #     tok = self.tokenizer(texts).to(self.device)
    #     text_feat = self.clip.encode_text(tok)
    #     text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
    #     return self.txt_to_vis(text_feat)  # (B,C)

    def encode_text_vis(self, commands: List[str]) -> torch.Tensor:
        texts = [self.prompts[c] for c in commands]
        tok = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                text_feat = self.clip.encode_text(tok)
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
        # return self.txt_to_vis(text_feat.to(torch.float16))
        return self.txt_to_vis(text_feat.to(self.txt_to_vis.weight.dtype))

    # --------- 生成軌跡 ---------
    # def generate(self, rgb_seq: np.ndarray, seg_seq: np.ndarray, ego_seq: torch.Tensor, commands: List[str]) -> torch.Tensor:
    #     B = ego_seq.shape[0]
    #     vis_tokens = self.build_vis_tokens(rgb_seq, seg_seq, ego_seq)      # (B,S,C)
    #     text_vis  = self.encode_text_vis(commands)                         # (B,C)

    #     # 將 text 作為額外查詢偏置加入所有 time queries
    #     queries = self.time_queries.unsqueeze(0).expand(B, -1, -1) + text_vis.unsqueeze(1)  # (B,T,C)
    #     fused = self.cross_attn(queries, vis_tokens)  # (B,T,C)
    #     xy = self.traj_head(fused)                    # (B,T,2)
    #     z = torch.zeros_like(xy[..., :1])             # 先不預測 z/yaw
    #     traj = torch.cat([xy, z], dim=-1)             # (B,T,3)
    #     return traj
    
    def generate(self, rgb_seq, seg_seq, ego_seq, commands, gt_trajs: torch.Tensor = None):
        # Teacher Forcing 比例（可以先固定，後面再做調度）
        # 若 cfg 有設定就讀，否則用預設值
        tf_ratio = float(getattr(self, "AR_TF_RATIO", getattr(self, "tf_ratio", 1.0)))
        if not self.training:
            tf_ratio = 0.0
        
        tf_ratio = 0.0
        if tf_ratio > 0.0:
            raise RuntimeError("tf_ratio supposed to be 0 !")
        # print("tf_ratio",tf_ratio)

        return self.generate_autoregressive(
            rgb_seq, seg_seq, ego_seq, commands,
            gt_trajs=gt_trajs,
            teacher_forcing_ratio=tf_ratio
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
        self.vlm.step_scale  = float(getattr(cfg, "AR_STEP_SCALE", 2.0))  # 原本1
        self.vlm.max_step    = float(getattr(cfg, "AR_MAX_STEP", 8.0))

        print("model : vlm_come_on")
        print("dx:", self.dx, "bx:", self.bx)

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


    
    def collision_loss_soft(self, trajs_xy: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
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

        # 6) 聚合（先用 mean，之後可試 max）
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
        # pred: (B, T, 3)
        pred = self.vlm.generate(
            self._last_rgb_seq,
            self._last_seg_seq,
            self._last_ego_seq.to(device),
            commands, 
            gt_trajs=gt_trajs
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
                
                # 4. 決策邏輯
                if scenario == "BUMP":
                    final_commands = ["SLOW"] * len(commands)
                    vlm_target_speed = 1.5 
                elif scenario == "OBSTACLE":
                    final_commands = ["SLOW"] * len(commands)
                    vlm_target_speed = 1.0 

                stop_speed_limit = self.vlm.update(scenario, current_speed)
                if stop_speed_limit is not None:
                    final_commands = ["STOP"] * len(commands)
                    vlm_target_speed = stop_speed_limit

                if is_inference and vlm_target_speed is not None:
                    # 2. 取出 (x, y) 轉為 Numpy
                    raw_traj_np = pred[0, :, :2].detach().cpu().numpy()
                    
                    # 3. 執行重採樣 -> (T, 2)
                    resampled_np = self.vlm.resample_path_numpy(raw_traj_np, vlm_target_speed, dt=0.5)
                    
                    # 4. 寫回 Tensor
                    resampled_tensor = torch.from_numpy(resampled_np).float().to(device)
                    
                    # 5. 覆蓋 final_traj 的 (x,y) 部分 (保留 z=0)
                    # 注意：這裡只覆蓋 batch 0，因為 inference 通常 B=1
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
        if occupancy is not None:
            coll_rate = self.collision_loss_soft(pred[..., :2], occupancy)

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

        # RIGHT：希望 pred_last_x >= margin_turn
        if cmd_right.any():
            x_r = pred_last_x[cmd_right]
            loss_r = F.relu(margin_turn - x_r).mean()
            dir_loss = dir_loss + loss_r

        # LEFT：希望 pred_last_x <= -margin_turn
        if cmd_left.any():
            x_l = pred_last_x[cmd_left]
            loss_l = F.relu(x_l + margin_turn).mean()
            dir_loss = dir_loss + loss_l

        # FORWARD：希望 |pred_last_x| 不要太大
        if cmd_forward.any():
            x_f = pred_last_x[cmd_forward].abs()
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


        if is_inference:
            hard_col = 0     
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
        }

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
        return loss, pred, final_traj, torch.tensor(0.0, device=device), loss_dict

    # def planning(self, *, bev_rgbs, trajs, gt_trajs, commands, target_points, occupancy=None):
    #     """忽略候選 `trajs`，直接生成軌跡並計算損失。
    #     回傳 (loss_total, final_traj, zero, zero) 以相容 trainer 現有記錄欄位。
    #     """
    #     assert self._last_rgb_seq is not None and self._last_seg_seq is not None and self._last_ego_seq is not None
    #     "缺序列影像或 ego 嵌入：請先呼叫 forward()"

    #     device = gt_trajs.device
    #     # 生成
    #     pred = self.vlm.generate(
    #         self._last_rgb_seq,
    #         self._last_seg_seq,
    #         self._last_ego_seq.to(device),
    #         commands,
    #         gt_trajs=gt_trajs
    #     ).to(device)  # (B,T,3)


    #     # with torch.no_grad():
    #     #     d0 = gt_trajs[..., :2][:, 0].norm(dim=-1).mean().item()
    #     #     print(f"[DEBUG] mean ||gt first step|| = {d0:.3f} m")



    #     # Loss 1: L2（ADE）
    #     # l2 = ((pred[..., :2] - gt_trajs[..., :2])**2).sum(dim=-1).sqrt().mean()

    #     # 新：近端加權 L2（前端 1.3 → 後端 1.0）
    #     err = ((pred[..., :2] - gt_trajs[..., :2])**2).sum(dim=-1).sqrt()  # (B,T)
    #     T = err.shape[1]
    #     w = torch.linspace(1.3, 1.0, T, device=err.device)                 # 可改 1.4→1.0 做 A/B
    #     l2 = (err * w).mean()

    #     # 速度監督（讓 AR 更快學步態）
    #     pred_xy = pred[..., :2]
    #     gt_xy   = gt_trajs[..., :2]
    #     pred_d  = torch.cat([pred_xy[:, :1] - 0, pred_xy[:, 1:] - pred_xy[:, :-1]], dim=1)  # (B,T,2)
    #     gt_d    = torch.cat([gt_xy[:, :1]  - 0, gt_xy[:, 1:]  - gt_xy[:, :-1]],  dim=1)
    #     # vel_l2  = (pred_d - gt_d).pow(2).sum(-1).mean()

    #     # 取代原 vel_l2 的均值
    #     T = pred_d.size(1)
    #     wv = torch.linspace(1.4, 1.0, T, device=pred_d.device)
    #     vel_l2 = ((pred_d - gt_d).pow(2).sum(-1) * wv).mean()




    #     # Loss 2: 碰撞率
    #     coll_rate = torch.tensor(0.0, device=device)
    #     if occupancy is not None:
    #         coll_rate = self.occupancy_collision_rate(pred[..., :2], occupancy)

    #     # 平滑正則：Δv 正則，避免抖動
    #     vel = pred[..., :2] - torch.cat([pred[:, :1, :2], pred[:, :-1, :2]], dim=1)
    #     smooth = (vel[:, 1:] - vel[:, :-1]).pow(2).sum(-1).mean()

    #     lam_l2 = getattr(self.cfg, 'LOSS_L2_W', 8.0)
    #     lam_col = getattr(self.cfg, 'LOSS_COL_W', 1.5)
    #     lam_smo = getattr(self.cfg, 'LOSS_SMO_W', 0.4)
    #     lam_vel = getattr(self.cfg, 'LOSS_VEL_W', 0.6)   # 新增：速度監督
    #     lam_coarse = getattr(self.cfg, 'LOSS_COARSE_W', 0.2)   # 小權重就好


    #     # --- 粗軌跡輔助 loss（讓並行頭先學一條可用的基線） ---
    #     coarse_xy = getattr(self.vlm, "last_coarse_xy", None)
    #     coarse_l2 = torch.tensor(0.0, device=device)
    #     if coarse_xy is not None:
    #         # 和主 L2 一樣用近端加權，保持一致
    #         err_c = ((coarse_xy - gt_trajs[..., :2])**2).sum(dim=-1).sqrt()   # (B,T)
    #         T_c = err_c.shape[1]
    #         w_c = torch.linspace(1.6, 1.0, T_c, device=device)                # 比主 L2 稍強調近端，收斂更快
    #         coarse_l2 = (err_c * w_c).mean()



    #     # lam_col = 0
    #     # lam_smo = 0
    #     loss = lam_l2 * l2 + lam_col * coll_rate + lam_smo * smooth + lam_vel * vel_l2 + lam_coarse * coarse_l2

    #     return loss, pred, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
