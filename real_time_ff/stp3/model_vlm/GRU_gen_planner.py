import os
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast


try:
    import open_clip
    HAS_OPENCLIP = True
except Exception:
    HAS_OPENCLIP = False

from stp3.utils.tools import gen_dx_bx
from stp3.utils.geometry import calculate_birds_eye_view_parameters


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

        self.clip, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained="openai")
        self.tokenizer = open_clip.get_tokenizer(clip_name)
        self.clip.to(device)

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


        

        # ====== 一些可調參數（給不到 cfg 時也有預設）======
        self.use_delta = True  # 目前固定走 Δx,Δy
        # self.step_scale = 1.0  # Δ 的比例因子，資料單位不同時可調
        # self.max_step = 2.0    # 單步最大 Δ 限制（公尺），避免爆衝


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

        # 1) 取 visu tokens & text 條件
        vis_tokens = self.build_vis_tokens(rgb_seq, seg_seq, ego_seq)           # (B,S,C)
        text_vis  = self.encode_text_vis(commands)                              # (B,C)

        # 2) 並行抽取每步語義特徵（僅作為條件，不直接出座標）
        #    queries = time_queries + text + time_pe
        queries = self.time_queries.unsqueeze(0).expand(B, -1, -1) + text_vis.unsqueeze(1) + self.time_pe.unsqueeze(0)
        fused = self.cross_attn(queries, vis_tokens)                            # (B,T,C)

        # 3) 初始隱狀態 h0：concat 全局 pooled 視覺語義 + text
        global_ctx = fused.mean(dim=1)                                          # (B,C)
        h = self.h0_proj(torch.cat([global_ctx, text_vis], dim=-1))            # (B,C)

        # === 並行粗軌跡（B,T,2） ===
        coarse_xy = self.traj_coarse(fused)            # (B,T,2)
        self.last_coarse_xy = coarse_xy                # 之後在 planning 做輔助 loss 用

        # === 殘差式 AR 解碼 ===
        traj_xy = []
        prev_xy = coarse_xy[:, 0, :]                   # 用粗軌跡第一點當起點（更穩）
        res_acc = torch.zeros(B, 2, device=device)     # 殘差累積器 r_0 = 0
        if gt_trajs is not None and gt_trajs.size(-1) >= 2:
            gt_xy = gt_trajs[..., :2]

        for t in range(T):
            ctx_t = fused[:, t, :]                     # (B,C)
            embedded_xy = self.xy_post_ln(self.xy_embedder(prev_xy))            
            # dec_in = torch.cat([embedded_xy, ctx_t], dim=-1)  # (B, 2C)

            g = self.g_proj(torch.cat([embedded_xy, ctx_t], dim=-1))  # (B,C) ∈ (0,1)
            ctx_gated = g * ctx_t + (1 - g) * embedded_xy            # 有選擇地信任誰
            dec_in = torch.cat([embedded_xy, ctx_gated], dim=-1)     # (B,2C)
            dec_in = self.dec_in_ln(dec_in)


            h = self.decoder_gru(dec_in, h)
            delta = self.delta_head(h)                 # (B,2) → 「殘差增量」Δr_t
            delta = torch.tanh(delta / max(1e-6, self.step_scale)) * self.max_step

            # 殘差累加 + 加到粗軌跡上： pred_t = coarse_t + (r_{t-1} + Δr_t)
            res_acc = res_acc + delta * self.step_scale
            pred_xy_t = coarse_xy[:, t, :] + res_acc   # (B,2)

            traj_xy.append(pred_xy_t.unsqueeze(1))     # ★ 輸出永遠用 model 的 pred，不被 TF 覆蓋

            # Teacher Forcing 僅影響「下一步的輸入 prev_xy」，並同步修正 res_acc 使其與 prev_xy 一致
            if (self.training and (gt_trajs is not None) and (teacher_forcing_ratio > 0.0)):
                use_tf = (torch.rand(B, device=device) < teacher_forcing_ratio).unsqueeze(-1)  # (B,1)
                next_prev = torch.where(use_tf, gt_xy[:, t, :], pred_xy_t)                     # (B,2)
                prev_xy = next_prev
                # 讓殘差與 chosen prev_xy 對齊： r_t = prev_xy - coarse_t
                res_acc = prev_xy - coarse_xy[:, t, :]
            else:
                prev_xy = pred_xy_t
                # res_acc 已在上面更新，這裡不用動


        xy = torch.cat(traj_xy, dim=1)                                          # (B,T,2)
        z  = torch.zeros(B, T, 1, device=device)
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



    def _encode_tokens(self, imgs_t: torch.Tensor) -> torch.Tensor:
        self._vis_tokens = None
        _ = self.clip.encode_image(imgs_t)  # 觸發 hook
        tok = self._vis_tokens
        assert tok is not None, "CLIP 視覺 tokens 取得失敗（hook 未觸發）"
        return tok / (tok.norm(dim=-1, keepdim=True) + 1e-6)

    # def _encode_tokens(self, imgs_t: torch.Tensor) -> torch.Tensor:
    #     self._vis_tokens = None
    #     with torch.no_grad():
    #         with autocast(dtype=torch.float16):
    #             _ = self.clip.encode_image(imgs_t)   # 觸發 hook（FP16）
    #     tok = self._vis_tokens
    #     assert tok is not None, "CLIP 視覺 tokens 取得失敗（hook 未觸發）"
    #     return tok.to(torch.float16) / (tok.norm(dim=-1, keepdim=True) + 1e-6)

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
    def encode_text_vis(self, commands: List[str]) -> torch.Tensor:
        texts = [self.prompts[c] for c in commands]
        tok = self.tokenizer(texts).to(self.device)
        text_feat = self.clip.encode_text(tok)
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
        return self.txt_to_vis(text_feat)  # (B,C)

    # def encode_text_vis(self, commands: List[str]) -> torch.Tensor:
    #     texts = [self.prompts[c] for c in commands]
    #     tok = self.tokenizer(texts).to(self.device)
    #     with torch.no_grad():
    #         with autocast(dtype=torch.float16):
    #             text_feat = self.clip.encode_text(tok)
    #     text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
    #     return self.txt_to_vis(text_feat.to(torch.float16))

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
        # tf_ratio = 0.0
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
        self.vlm.AR_TF_RATIO = float(getattr(cfg, "AR_TF_RATIO", 1.0))  # 訓練期 teacher forcing 比例
        self.vlm.step_scale  = float(getattr(cfg, "AR_STEP_SCALE", 1.0))
        self.vlm.max_step    = float(getattr(cfg, "AR_MAX_STEP", 5.0))

        print("model : gen_planner")
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
        """忽略候選 `trajs`，直接生成軌跡並計算損失。
        回傳 (loss_total, final_traj, zero, zero) 以相容 trainer 現有記錄欄位。
        """
        assert self._last_rgb_seq is not None and self._last_seg_seq is not None and self._last_ego_seq is not None
        "缺序列影像或 ego 嵌入：請先呼叫 forward()"

        device = gt_trajs.device
        # 生成
        pred = self.vlm.generate(
            self._last_rgb_seq,
            self._last_seg_seq,
            self._last_ego_seq.to(device),
            commands,
            gt_trajs=gt_trajs
        ).to(device)  # (B,T,3)


        # with torch.no_grad():
        #     d0 = gt_trajs[..., :2][:, 0].norm(dim=-1).mean().item()
        #     print(f"[DEBUG] mean ||gt first step|| = {d0:.3f} m")



        # Loss 1: L2（ADE）
        # l2 = ((pred[..., :2] - gt_trajs[..., :2])**2).sum(dim=-1).sqrt().mean()

        # 新：近端加權 L2（前端 1.3 → 後端 1.0）
        err = ((pred[..., :2] - gt_trajs[..., :2])**2).sum(dim=-1).sqrt()  # (B,T)
        T = err.shape[1]
        w = torch.linspace(1.3, 1.0, T, device=err.device)                 # 可改 1.4→1.0 做 A/B
        l2 = (err * w).mean()

        # 速度監督（讓 AR 更快學步態）
        pred_xy = pred[..., :2]
        gt_xy   = gt_trajs[..., :2]
        pred_d  = torch.cat([pred_xy[:, :1] - 0, pred_xy[:, 1:] - pred_xy[:, :-1]], dim=1)  # (B,T,2)
        gt_d    = torch.cat([gt_xy[:, :1]  - 0, gt_xy[:, 1:]  - gt_xy[:, :-1]],  dim=1)
        # vel_l2  = (pred_d - gt_d).pow(2).sum(-1).mean()

        # 取代原 vel_l2 的均值
        T = pred_d.size(1)
        wv = torch.linspace(1.4, 1.0, T, device=pred_d.device)
        vel_l2 = ((pred_d - gt_d).pow(2).sum(-1) * wv).mean()




        # Loss 2: 碰撞率
        coll_rate = torch.tensor(0.0, device=device)
        if occupancy is not None:
            coll_rate = self.occupancy_collision_rate(pred[..., :2], occupancy)

        # 平滑正則：Δv 正則，避免抖動
        vel = pred[..., :2] - torch.cat([pred[:, :1, :2], pred[:, :-1, :2]], dim=1)
        smooth = (vel[:, 1:] - vel[:, :-1]).pow(2).sum(-1).mean()

        lam_l2 = getattr(self.cfg, 'LOSS_L2_W', 8.0)
        lam_col = getattr(self.cfg, 'LOSS_COL_W', 1.5)
        lam_smo = getattr(self.cfg, 'LOSS_SMO_W', 0.4)
        lam_vel = getattr(self.cfg, 'LOSS_VEL_W', 0.6)   # 新增：速度監督
        lam_coarse = getattr(self.cfg, 'LOSS_COARSE_W', 0.2)   # 小權重就好


        # --- 粗軌跡輔助 loss（讓並行頭先學一條可用的基線） ---
        coarse_xy = getattr(self.vlm, "last_coarse_xy", None)
        coarse_l2 = torch.tensor(0.0, device=device)
        if coarse_xy is not None:
            # 和主 L2 一樣用近端加權，保持一致
            err_c = ((coarse_xy - gt_trajs[..., :2])**2).sum(dim=-1).sqrt()   # (B,T)
            T_c = err_c.shape[1]
            w_c = torch.linspace(1.6, 1.0, T_c, device=device)                # 比主 L2 稍強調近端，收斂更快
            coarse_l2 = (err_c * w_c).mean()



        # lam_col = 0
        # lam_smo = 0
        loss = lam_l2 * l2 + lam_col * coll_rate + lam_smo * smooth + lam_vel * vel_l2 + lam_coarse * coarse_l2

        return loss, pred, pred, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
