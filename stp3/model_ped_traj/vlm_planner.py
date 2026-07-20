# stp3/models/seg_clip_panner.py
import os
from typing import List, Tuple, Dict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Cross-Attn（Q=traj；K,V=concat 後的視覺 tokens）
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.vis_width, num_heads=8, batch_first=True)

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
        rgb_imgs: np.ndarray,          # (B,H,W,3) uint8
        seg_imgs: np.ndarray,          # (B,H,W,3) uint8（語意著色圖）
        trajs: torch.Tensor,           # (B,N,T,3)
        command: List[str],            # 長度 B
    ) -> torch.Tensor:
        device = self.device
        B, N, T, _ = trajs.shape

        if self._first_vis_print < 10:
            print(f"[SegCLIP] B={B}  RGB={rgb_imgs.shape[1]}x{rgb_imgs.shape[2]}  SEG={seg_imgs.shape[1]}x{seg_imgs.shape[2]}")
            self._first_vis_print += 1

        # (1) 文本條件
        texts = [self.prompts[c] for c in command]
        tok = self.tokenizer(texts).to(device)
        text_feat = self.model.encode_text(tok)  # (B, D_txt)
        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
        text_vis = self.txt_to_vis(text_feat)    # (B, C)

        # (2) 兩路影像 → tokens
        rgb_t = self._preprocess_clip_tensor(rgb_imgs, resize_to=self.input_size)
        rgb_tokens = self._encode_tokens(rgb_t)         # (B,S1,C)

        seg_t = self._preprocess_clip_tensor(seg_imgs, resize_to=self.input_size)
        seg_tokens = self._encode_tokens(seg_t)         # (B,S2,C)

        vis_tokens = torch.cat([rgb_tokens, seg_tokens], dim=1)  # (B,S1+S2,C)

        # (3) 軌跡 → GRU → Q
        traj_xy = trajs[..., :2].contiguous()                  # (B,N,T,2)
        q_in = traj_xy.view(B * N, T, 2).to(device)
        _, h_n = self.traj_enc(q_in)                           # h_n: (1,B*N,C)
        q = h_n.transpose(0, 1)                                # (B*N,1,C)

        # (4) Cross-Attn
        k = vis_tokens.repeat_interleave(N, dim=0)             # (B*N,S,C)
        v = k
        attn_out, _ = self.cross_attn(q, k, v)                 # (B*N,1,C)
        h = attn_out.squeeze(1)                                # (B*N,C)

        # (5) 文本條件 + 打分
        tv = text_vis.repeat_interleave(N, dim=0)              # (B*N,C)
        h = h + tv
        scores = self.score_mlp(h).view(B, N)                  # (B,N)
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

        # 首次 debug 列印次數限制
        self._first_build_print = 0
        self._first_build_limit = 10



    def forward(self, image, intrinsics, extrinsics, future_egomotion, *, rgb_224, seg_224):
        """
        我們現在直接接收由 DataLoader 準備好的 rgb_224 和 seg_224 張量。
        """
        # 刪除所有基於 batch_meta 讀檔的舊邏輯
        # ----------------------------------------------------
        # B = image.size(0)
        # assert batch_meta is not None and "front_img_path" in batch_meta, "..."
        # paths: List[str] = batch_meta["front_img_path"]
        # seg_paths: List[str] = batch_meta.get("seg2d_path", [None]*len(paths))
        # rgb_imgs, seg_imgs = [], []
        # for b in range(B):
        #     rgb = cv2.imread(...)
        #     ...
        # rgb_imgs = np.stack(rgb_imgs)
        # seg_imgs = np.stack(seg_imgs)
        # ----------------------------------------------------

        # VLMPlanning 需要 numpy array，所以我們轉換一下
        # DataLoader 傳來的應該是 (B, H, W, 3) 的 torch.Tensor
        rgb_imgs_np = rgb_224.cpu().numpy()
        seg_imgs_np = seg_224.cpu().numpy()
        
        # 存給 planning 用
        self._last_rgb_imgs = rgb_imgs_np
        self._last_seg_imgs = seg_imgs_np

        # 建立最小相容輸出
        output = {}

        # 注意回傳的第二個值
        return output, rgb_imgs_np

    def planning(self, *, bev_rgbs: np.ndarray, trajs: torch.Tensor, gt_trajs: torch.Tensor,
                 commands: List[str], target_points: torch.Tensor):
        """
        與舊介面相容（保留 bev_rgbs 參數），但實際上用 forward() 存好的 _last_rgb_imgs/_last_seg_imgs。
        回傳: (loss, final_traj)
        """
        device = trajs.device
        B, N, T, _ = trajs.shape

        assert self._last_rgb_imgs is not None and self._last_seg_imgs is not None, \
            "找不到影像快取：請確定先呼叫 forward() 再呼叫 planning()"

        # (1) 用影像域 Seg+RGB 做 CLIP 打分
        scores = self.vlm.score_batch(self._last_rgb_imgs, self._last_seg_imgs, trajs, commands)  # (B,N)

        # (2) 取最大分數那條
        idx = scores.argmax(dim=1)   # (B,)
        b_idx = torch.arange(B, device=device)
        final_traj = trajs[b_idx, idx]  # (B,T,3)

        # (3) ranking hinge（用 GT 最近的候選當正例）
        with torch.no_grad():
            d = ((trajs[...,:2] - gt_trajs[:,None,:,:2])**2).sum(-1).mean(-1)  # (B,N)
            pos = d.argmin(dim=1)  # (B,)
        margin = 0.2
        pos_score = scores[b_idx, pos]
        neg_mask = torch.ones_like(scores, dtype=torch.bool); neg_mask[b_idx, pos] = False
        neg_score = scores[neg_mask].view(B, N-1).max(dim=1).values
        loss = F.relu(margin - pos_score + neg_score).mean()

        return loss, final_traj
