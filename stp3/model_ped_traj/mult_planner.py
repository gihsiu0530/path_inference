# stp3/models/seg_clip_panner.py
import os
from typing import List, Tuple, Dict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import open_clip
    HAS_OPENCLIP = True
except Exception:
    HAS_OPENCLIP = False


# ---------- 小工具：把 argmax 後的語意圖上色（可按你習慣改 palette） ----------
# 19 類 → 4 類的群組（可依需求微調）
"""
"id2label": {
    "0": "road",
    "1": "sidewalk",
    "2": "building",
    "3": "wall",
    "4": "fence",
    "5": "pole",
    "6": "traffic light",
    "7": "traffic sign",
    "8": "vegetation",
    "9": "terrain",
    "10": "sky",
    "11": "person",
    "12": "rider",
    "13": "car",
    "14": "truck",
    "15": "bus",
    "16": "train",
    "17": "motorcycle",
    "18": "bicycle"
  }
"""
GROUPS = {
    "road":   [0],                                # road, sidewalk
    "person": [11],                                  # person
    "movable": [12, 13, 14, 15, 16, 17, 18],         # rider, car, truck, bus, train, motorcycle, bicycle
    "static": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],          # building, wall, fence, pole, tl, ts, vegetation, terrain, sky
}

# 4 類配色（RGB）
FOUR_PALETTE = {
    0: (128, 64, 128),   # road（紫）
    1: (220, 20, 60),    # person（紅）
    2: (0, 0, 142),      # movable（深藍）
    3: (70, 70, 70),     # static（灰）
}

def center_crop_to_square_np(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    if H == W: return img
    if H < W:
        off = (W - H) // 2
        return img[:, off:off+H]
    else:
        off = (H - W) // 2
        return img[off:off+W, :]

def resize_to_input_np(img: np.ndarray, size: int, is_label: bool) -> np.ndarray:
    # is_label=True → 最近鄰（避免顏色混染），False → 雙線性
    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    # 若你希望嚴格對齊 CLIP 的預處理，可先 center crop 成正方形再縮
    sq = center_crop_to_square_np(img)
    return cv2.resize(sq, (size, size), interpolation=interp)


def colorize_4class(cls4: np.ndarray) -> np.ndarray:
    """ cls4: (H,W) with values in {0,1,2,3} """
    H, W = cls4.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for k, rgb in FOUR_PALETTE.items():
        out[cls4 == k] = np.array(rgb, dtype=np.uint8)
    return out

def colorize_class_map(class_map: np.ndarray) -> np.ndarray:
    """
    class_map: (H, W) int
    return: (H, W, 3) uint8 (RGB)
    """
    # 你可以改成 Cityscapes 配色；這裡放一組簡單 palette
    palette = {
        0:  (0, 0, 0),        # road → 黑（你也可改成(0,0,255)表示藍色）
        11: (220, 20, 60),    # person（紅）
        12: (255, 20, 147),   # rider（粉）
        13: (0, 0, 142),      # car（深藍）
        14: (0, 0, 70),       # truck
        15: (0, 60, 100),     # bus
        16: (0, 80, 100),     # train
        17: (0, 0, 230),      # motorcycle
        18: (119, 11, 32),    # bicycle
    }
    H, W = class_map.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for k, rgb in palette.items():
        out[class_map == k] = np.array(rgb, dtype=np.uint8)
    return out



class BatchedCrossAttention(nn.Module):
    """
    跨注意力（Q=軌跡；K,V=視覺 tokens），不複製 K,V：
      Q  : (B, N, 1, C)
      KV : (B, S, C)
      out: (B, N, 1, C)
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, attn_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必須可整除 num_heads"
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout and attn_dropout > 0 else nn.Identity()

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        """
        q  : (B, N, 1, C)
        kv : (B, S, C)
        """
        B, N, Q, C = q.shape
        S = kv.shape[1]
        H, D = self.num_heads, self.head_dim

        # 1) 線性投影
        q_lin = self.q_proj(q).view(B, N, Q, H, D)            # (B,N,1,H,D)
        k_lin = self.k_proj(kv).view(B, S, H, D)              # (B,S,H,D)
        v_lin = self.v_proj(kv).view(B, S, H, D)              # (B,S,H,D)

        # 2) 注意力分數：每個 batch b 內，N 條 query 共用同一份 K,V
        #    scores: (B, N, H, Q, S)
        scores = torch.einsum('bnqhd,bshd->bnhqs', q_lin, k_lin) * self.scale

        # 3) softmax over S
        attn = scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 4) 權重和：輸出 (B, N, Q, H, D)
        out = torch.einsum('bnhqs,bshd->bnqhd', attn, v_lin)

        # 5) 合併 heads → (B, N, Q, C)，再 out_proj
        out = out.contiguous().view(B, N, Q, C)
        out = self.out_proj(out)
        return out  # (B,N,1,C)


# ===================== 規劃器（CLIP + Cross-Attn） =====================
class VLMPlanning(nn.Module):
    """
    影像域版本：
      - 同時 encode RGB 與 Seg 著色圖 → 取兩者 patch tokens → concat
      - 軌跡 (x,y) → GRU → 向量當 Query
      - Cross-Attn 後接 MLP 出分數（每條軌跡一個 score）
    """
    def __init__(
        self,
        clip_name: str,
        device: torch.device,
        prompts: Dict[str, str] = {
            "LEFT":    "turn left, stay on drivable road, avoid obstacles, smooth path",
            "FORWARD": "go straight, stay on drivable road, avoid obstacles, smooth path",
            "RIGHT":   "turn right, stay on drivable road, avoid obstacles, smooth path",
        },
    ):
        super().__init__()
        self.prompts = prompts
        self.device = device

        assert HAS_OPENCLIP, "需要 open_clip，請先安裝 open_clip_torch"

        print("VLM model : ", clip_name)
        self.model, _, _ = open_clip.create_model_and_transforms(clip_name, pretrained="openai")
        self.tokenizer = open_clip.get_tokenizer(clip_name)
        self.model.to(device)

        # 全凍結，僅開視覺最後 K 層與 LN
        for p in self.model.parameters():
            p.requires_grad = False
        K_UNFREEZE = 2
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "transformer"):
            blocks = self.model.visual.transformer.resblocks
            for blk in blocks[-K_UNFREEZE:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for n, p in self.model.visual.named_parameters():
                if "ln_post" in n or "ln_pre" in n:
                    p.requires_grad = True

        self.model.train()           # 視覺塔打開訓練
        if hasattr(self.model, "transformer"):
            self.model.transformer.eval()  # 文字塔不訓練

        # 視覺寬度（例如 ViT-B/32 是 768）
        self.vis_width = int(getattr(self.model.visual.transformer, "width", 768))

        # ---- 取視覺 tokens 的 forward hook ----
        self._vis_tokens = None
        def _vis_hook(_, __, out):
            self._vis_tokens = out
        self.model.visual.transformer.register_forward_hook(_vis_hook)

        # 文本到視覺寬度的投影
        if hasattr(self.model, "text_projection"):
            txt_dim = int(self.model.text_projection.shape[-1])  # e.g., 512
        else:
            # 後備（通常不會到）
            txt_dim = self.vis_width
        self.txt_to_vis = nn.Linear(txt_dim, self.vis_width, bias=False)

        # 逐軌跡編碼（T×2 → C）
        self.traj_enc = nn.GRU(input_size=2, hidden_size=self.vis_width, num_layers=1,
                               batch_first=True, bidirectional=False)


        # Cross-Attn（批次版；不複製 K,V）
        self.cross_attn = BatchedCrossAttention(embed_dim=self.vis_width, num_heads=8, attn_dropout=0.0)



        # ★ 新增：egomotion 嵌入器（輸入 4 維：dx, dy, sin(dyaw), cos(dyaw)）
        self.ego_mlp = nn.Sequential(
            nn.Linear(4, self.vis_width),
            nn.GELU(),
            nn.Linear(self.vis_width, self.vis_width)
        )


        # 打分頭
        self.score_mlp = nn.Sequential(
            nn.Linear(self.vis_width, self.vis_width),
            nn.GELU(),
            nn.Linear(self.vis_width, 1)
        )

        print("[DBG] CLIP device:", next(self.model.parameters()).device)

        # 只印前幾次視覺尺寸
        self._first_vis_print = 0

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
            tag = "T" if p.requires_grad else "F"
            print(f"{tag} | {name:<80s} | shape={tuple(p.shape)!s:<25s} | {n:>10,d} params ({_millify(n)})")

        print("-" * 120)
        print(f"Total parameters    : {total_params:,} ({_millify(total_params)})")
        print(f"Trainable parameters: {trainable_params:,} ({_millify(trainable_params)})  "
            f"({100*trainable_params/max(1,total_params):.2f}%)\n")

    # ---------- 張量前處理（避免 PIL；維持 CLIP 的 mean/std） ----------
    def _preprocess_clip_tensor(self, imgs_np: np.ndarray, resize_to: int) -> torch.Tensor:
        # imgs_np: (B,H,W,3) uint8
        if imgs_np.shape[1] == resize_to and imgs_np.shape[2] == resize_to:
            x = torch.from_numpy(imgs_np).permute(0,3,1,2).float() / 255.0
            x = x.to(self.device, non_blocking=True)
        else:
            x = torch.from_numpy(imgs_np).permute(0,3,1,2).float() / 255.0
            x = x.to(self.device, non_blocking=True)
            x = F.interpolate(x, size=(resize_to, resize_to), mode="bilinear", align_corners=False)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device)[:, None, None]
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device)[:, None, None]
        return (x - mean) / std



    # ---------- 取 CLIP 視覺 tokens ----------
    def _encode_tokens(self, imgs_t: torch.Tensor) -> torch.Tensor:
        """
        imgs_t: (B,3,H,W) 已正規化
        return: (B, S, C) 視覺 tokens，L2 norm
        """
        self._vis_tokens = None
        _ = self.model.encode_image(imgs_t)   # 觸發 hook
        tok = self._vis_tokens
        assert tok is not None, "CLIP 視覺 tokens 取得失敗（hook 未觸發）"
        return tok / (tok.norm(dim=-1, keepdim=True) + 1e-6)

    # ---------- 打分（同時吃 RGB 與 Seg） ----------
    def score_batch(
        self,
        rgb_seq: np.ndarray,     # (B,T_rf,H,W,3) uint8
        seg_seq: np.ndarray,     # (B,T_rf,H,W,3) uint8
        trajs: torch.Tensor,     # (B,N,T,3)
        command: List[str],      # 長度 B
        ego_seq: torch.Tensor,   # (B,T_rf,4) 已做縮放+sin/cos 的 ego 嵌入輸入
    ) -> torch.Tensor:
        device = self.device
        B, N, T, _ = trajs.shape
        T_rf = rgb_seq.shape[1]

        # (1) 文本條件
        texts = [self.prompts[c] for c in command]
        tok = self.tokenizer(texts).to(device)
        text_feat = self.model.encode_text(tok)              # (B, D_txt)
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
        text_vis  = self.txt_to_vis(text_feat)               # (B, C)

        # (2) 逐幀取 tokens 並加上對應的 ego 嵌入
        all_tokens = []
        for t in range(T_rf):
            rgb_t = self._preprocess_clip_tensor(rgb_seq[:, t], resize_to=self.input_size)
            seg_t = self._preprocess_clip_tensor(seg_seq[:, t], resize_to=self.input_size)
            rgb_tok = self._encode_tokens(rgb_t)             # (B,S1,C)
            seg_tok = self._encode_tokens(seg_t)             # (B,S2,C)
            vis_tok = torch.cat([rgb_tok, seg_tok], dim=1)   # (B,S1+S2,C)

            # ego 嵌入（B,C）→ broadcast 到 (B,S,C) 後相加
            ego_embed_t = self.ego_mlp(ego_seq[:, t].to(device))  # (B,C)
            vis_tok = vis_tok + ego_embed_t.unsqueeze(1)          # (B,S,C)

            all_tokens.append(vis_tok)

        vis_tokens = torch.cat(all_tokens, dim=1)            # (B, T_rf*(S1+S2), C)

        # (3) 軌跡 → GRU → Query  （保留一份 K,V；把 Q reshape 成 (B,N,1,C)）
        traj_xy = trajs[..., :2].contiguous()                 # (B,N,T,2)
        q_in = traj_xy.view(B * N, T, 2).to(device)
        _, h_n = self.traj_enc(q_in)                          # (1,B*N,C)
        q = h_n.transpose(0, 1).view(B, N, 1, self.vis_width) # (B,N,1,C)

        # (4) Cross-Attn（批次版）：K,V 只保留 (B,S_all,C) 一份
        attn_out = self.cross_attn(q, vis_tokens)             # (B,N,1,C)
        h = attn_out.squeeze(2)                                # (B,N,C)

        # (5) 文本條件 + 打分（逐候選）
        tv = text_vis.unsqueeze(1).expand(B, N, -1)           # (B,N,C)
        h = h + tv                                            # (B,N,C)
        scores = self.score_mlp(h.reshape(B * N, self.vis_width)).view(B, N)
              # (B,N)
        return scores


# ===================== 主模型（保持與原 STP3 介面相容） =====================
class VLM_STP3(nn.Module):
    """
    影像域 Seg+RGB 版本：
      - forward()：讀取 batch_meta 的 front_img_path / seg2d_path → 產生 RGB 與 Seg 著色圖
      - 不計算 BEV；輸出仍組出最小相容的字典（segmentation/hdmap/costvolume）
      - planning()：簽名不變，但改用 forward() 存下來的影像陣列做 CLIP 打分
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD
        self.n_future = cfg.N_FUTURE_FRAMES

        # 前視相機名稱（用來對 batch 的影像路徑、內外參等）
        self.front_cam_name = cfg.IMAGE.NAMES[0] if len(cfg.IMAGE.NAMES) > 0 else "CAM_FRONT"

        # 分割 logits 根目錄（.npy）
        self.seg2d_root = getattr(cfg, "SEG2D_ROOT", None)
        assert self.seg2d_root is not None, "請在 config 設定 SEG2D_ROOT 指向 seg2d .npy 根目錄"

        # VLM 規劃器（CLIP）
        self.input_size = int(getattr(cfg, "CLIP_INPUT_SIZE", 224))  # ← 之後想改就改這裡或在 YAML 設定

        clip_name = getattr(cfg, "CLIP_MODEL", "ViT-B-32")
        self.vlm = VLMPlanning(clip_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.vlm.input_size = self.input_size   # ★ 關鍵修復：讓 VLMPlanning 擁有 input_size

        # 為配合 trainer 的介面：假 encoder 輸出（cam_front / costvolume）
        self.encoder_out_channels = 64
        self.fake_cam_front = nn.Parameter(torch.zeros(1, self.encoder_out_channels, 60, 28), requires_grad=False)

        # 存放 forward 最近一次的影像（給 planning 用）
        self._last_rgb_imgs = None
        self._last_seg_imgs = None

        self._last_rgb_seq = None
        self._last_seg_seq = None
        self._last_ego_seq_embed = None

        # 首次 debug 列印次數限制
        self._first_build_print = 0
        self._first_build_limit = 10



    def forward(self, image, intrinsics, extrinsics, future_egomotion, *,
                rgb_224_seq, seg_224_seq):
        """
        rgb_224_seq / seg_224_seq: (B, T_rf, H, W, 3) uint8
        future_egomotion: (B, T_all, 6) ，我們只用到前 T_rf-1 個相鄰變換組合成「每幀→現在幀」。
        """
        B, T_rf = rgb_224_seq.shape[:2]
        device = future_egomotion.device

        # 1) 存影像序列（轉 numpy，照你原本 score_batch 需求）
        self._last_rgb_seq = rgb_224_seq.cpu().numpy()
        self._last_seg_seq = seg_224_seq.cpu().numpy()

        # 2) 從 future_egomotion 組「每幀→現在幀」的增量（只取 2D 與 yaw）
        #    utils 中已有 pose_vec2mat / mat2pose_vec
        from stp3.utils.geometry import pose_vec2mat, mat2pose_vec

        # 只取 receptive_field 段
        T_all = future_egomotion.shape[1]
        assert T_all >= self.receptive_field, "future_egomotion 時長不足以覆蓋 receptive_field"
        fego = future_egomotion[:, :self.receptive_field, :]   # (B, T_rf, 6)

        ego_seq_embed = []
        for t in range(T_rf):
            # 目標：把 t → (T_rf-1) 的連乘變換求出
            if t == T_rf - 1:
                # 現在幀到自己：零位移零旋轉
                dx = torch.zeros(B, 1, device=device)
                dy = torch.zeros(B, 1, device=device)
                dyaw = torch.zeros(B, 1, device=device)
            else:
                mats = []
                for k in range(t, T_rf - 1):
                    mats.append(pose_vec2mat(fego[:, k, :]))   # (B,4,4)
                M = mats[0]
                for m in mats[1:]:
                    M = torch.bmm(M, m)                        # (B,4,4)
                pose = mat2pose_vec(M)                          # (B,6) [tx,ty,tz,rx,ry,rz]
                dx = pose[:, 0:1]                               # (B,1)
                dy = pose[:, 1:1+1]
                dyaw = pose[:, 5:6]                             # yaw（假設 rz）
                # wrap 到 [-pi,pi]
                dyaw = (dyaw + torch.pi) % (2*torch.pi) - torch.pi

            # 縮放（公尺）＋ sin/cos
            s_m = getattr(self.cfg, "EGO_SCALE_M", 5.0)
            ex = dx / s_m
            ey = dy / s_m
            sy = torch.sin(dyaw)
            cy = torch.cos(dyaw)
            ego_seq_embed.append(torch.cat([ex, ey, sy, cy], dim=1))  # (B,4)

        ego_seq_embed = torch.stack(ego_seq_embed, dim=1)   # (B, T_rf, 4)
        self._last_ego_seq_embed = ego_seq_embed.detach()

        return {}, self._last_rgb_seq




    def planning(self, *, bev_rgbs, trajs, gt_trajs, commands, target_points):
        device = trajs.device
        B, N, T, _ = trajs.shape
        assert (self._last_rgb_seq is not None) and (self._last_seg_seq is not None) \
            and (self._last_ego_seq_embed is not None), \
            "缺序列影像或 ego 嵌入：請先呼叫 forward()"

        # 用序列打分（逐幀 tokens + 每幀 ego 嵌入）
        scores = self.vlm.score_batch(
            self._last_rgb_seq,     # (B,T_rf,H,W,3) uint8 (numpy)
            self._last_seg_seq,     # (B,T_rf,H,W,3) uint8 (numpy)
            trajs,                  # (B,N,T,3)
            commands,               # list[str]
            self._last_ego_seq_embed.to(device),  # (B,T_rf,4)
            )
        
        # (2) 取最大分數那條
        idx = scores.argmax(dim=1)   # (B,)
        b_idx = torch.arange(B, device=device)
        final_traj = trajs[b_idx, idx]  # (B,T,3)

        # (3) ranking hinge（用 GT 最近的候選當正例）
        with torch.no_grad():
            d = ((trajs[...,:2] - gt_trajs[:,None,:,:2])**2).sum(-1).mean(-1)  # (B,N)
            pos = d.argmin(dim=1)  # (B,)
        # margin = 0.2
        # pos_score = scores[b_idx, pos]
        # neg_mask = torch.ones_like(scores, dtype=torch.bool); neg_mask[b_idx, pos] = False
        # neg_score = scores[neg_mask].view(B, N-1).max(dim=1).values
        # loss = F.relu(margin - pos_score + neg_score).mean()


        # === 改善版：Top-k / Soft 負例匯聚，平穩很多 ===
        margin = 0.2
        pos_score = scores[b_idx, pos]                          # (B,)

        # 去掉正例，取得所有負例分數
        neg_mask = torch.ones_like(scores, dtype=torch.bool)
        neg_mask[b_idx, pos] = False
        negs = scores[neg_mask].view(B, N - 1)                  # (B, N-1)

        # ---- 方案 A：Top-k 平均（建議先用；N=30 → k=8 是不錯的起點）----
        # k = min(8, negs.size(1))                                # 你也可以改成 6/10 做敏感度測試
        # topk_neg = torch.topk(negs, k=k, dim=1).values.mean(dim=1)   # (B,)

        # loss = loss_rank

        # ---- 方案 B（建議）：Soft 匯聚（LogSumExp）----
        tau = 0.7   # 0.5~1.0 之間調；越小越像 max，越大越像平均
        topk_neg = tau * torch.logsumexp(negs / tau, dim=1)            # (B,)

        # 平滑 hinge（可用 relu；softplus 更平滑）
        loss_rank = F.softplus(margin - pos_score + topk_neg).mean()
        # loss_rank = F.relu(margin - pos_score + topk_neg).mean()


        # === 指標對齊：期望成本（softmax(scores/τ) 加權） ===   期望降低L2
        with torch.no_grad():
            # ADE 當成本（每條候選的平均 L2）
            ade = ((trajs[..., :2] - gt_trajs[:, None, :, :2]) ** 2).sum(-1).sqrt().mean(dim=-1)  # (B, N)

            # 可選：標準化，避免尺度影響 λ
            mu = ade.mean(dim=1, keepdim=True)
            sd = ade.std(dim=1, keepdim=True) + 1e-6
            cost = (ade - mu) / sd                                 # (B, N)

        # 用分數的 softmax 當權重，最小化期望成本
        tau_cost = 1.0                                             # 0.7~1.5 可調
        w = torch.softmax(scores / tau_cost, dim=1)                # (B, N)
        loss_cost = (w * cost).sum(dim=1).mean()                   # scalar

        lam = 0.1                                                  # 0.05~0.2 可調
        loss = loss_rank + lam * loss_cost



        


        return loss, final_traj, loss_rank, loss_cost
