from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from stp3.utils.geometry import calculate_birds_eye_view_parameters
from stp3.utils.tools import gen_dx_bx


class SeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.skip = None
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x if self.skip is None else self.skip(x)
        return self.act(self.block(x) + skip)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.skip = None
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x if self.skip is None else self.skip(x)
        return self.act(self.block(x) + skip)


class AttentionPool2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(channels, max(16, channels // 4), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(16, channels // 4), 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        attn = self.score(x).view(b, 1, h * w).softmax(dim=-1)
        feat = x.view(b, c, h * w)
        return torch.bmm(feat, attn.transpose(1, 2)).squeeze(-1)


class MultiScaleStem(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResidualConvBlock(64, 64),
        )
        self.stage2 = nn.Sequential(
            ResidualConvBlock(64, 128, stride=2),
            ResidualConvBlock(128, 128),
        )
        self.stage3 = nn.Sequential(
            ResidualConvBlock(128, 256, stride=2),
            ResidualConvBlock(256, 256),
        )
        self.stage4 = nn.Sequential(
            ResidualConvBlock(256, hidden_dim, stride=2),
            ResidualConvBlock(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.stage1(x)
        x = self.stage2(x)
        mid = self.stage3(x)   # 28x28 for 224 input.
        deep = self.stage4(mid)  # 14x14 for 224 input.
        return mid, deep


class WaypointDecoderLayer(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)

    def forward(self, query: torch.Tensor, spatial_tokens: torch.Tensor) -> torch.Tensor:
        q = query + self.self_attn(self.norm1(query), self.norm1(query), self.norm1(query), need_weights=False)[0]
        q = q + self.cross_attn(self.norm2(q), spatial_tokens, spatial_tokens, need_weights=False)[0]
        q = q + self.ffn(self.norm3(q))
        return q


class FastTrajectoryPlanner(nn.Module):
    COMMAND_TO_ID = {"LEFT": 0, "FORWARD": 1, "RIGHT": 2}

    def __init__(self, cfg, n_future: int, input_size: int = 224):
        super().__init__()
        self.cfg = cfg
        self.n_future = int(n_future)
        self.input_size = int(input_size)
        self.hidden_dim = int(getattr(cfg, "FAST_HIDDEN_DIM", 512))
        self.EGO_SCALE_M = float(getattr(cfg, "EGO_SCALE_M", 5.0))
        self.SAMPLE_DT = float(getattr(cfg, "SAMPLE_DT", 0.5))
        self.AR_TF_RATIO = 0.0
        c = self.hidden_dim

        self.rgb_stem = self._make_stem(3, c)
        self.seg_rgb_stem = self._make_stem(3, c)
        self.seg_id_embedding = nn.Embedding(int(getattr(cfg, "SEG_NUM_CLASSES", 4)), 16)
        self.seg_id_stem = self._make_stem(16, c)
        self.depth_stem = self._make_stem(1, c)

        self.mid_fusion = nn.Sequential(
            nn.Conv2d(256 * 4, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            ResidualConvBlock(c, c),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(c * 4, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            ResidualConvBlock(c, c),
            ResidualConvBlock(c, c),
            ResidualConvBlock(c, c),
        )
        self.pseudo_bev_grid_size = float(getattr(cfg, "PSEUDO_BEV_GRID_SIZE", 1.0))
        self.pseudo_bev_token_size = int(getattr(cfg, "PSEUDO_BEV_TOKEN_SIZE", 40))
        self.pseudo_bev_channels = int(getattr(cfg, "PSEUDO_BEV_CHANNELS", 128))
        x_bound = list(cfg.LIFT.X_BOUND)
        y_bound = list(cfg.LIFT.Y_BOUND)
        self.bev_x_min = float(x_bound[0])
        self.bev_x_max = float(x_bound[1])
        self.bev_y_min = float(y_bound[0])
        self.bev_y_max = float(y_bound[1])
        self.pseudo_bev_h = max(1, int(round((self.bev_x_max - self.bev_x_min) / self.pseudo_bev_grid_size)))
        self.pseudo_bev_w = max(1, int(round((self.bev_y_max - self.bev_y_min) / self.pseudo_bev_grid_size)))
        self.depth_min_m = float(getattr(cfg, "PSEUDO_BEV_DEPTH_MIN", 1.0))
        self.depth_max_m = float(getattr(cfg, "PSEUDO_BEV_DEPTH_MAX", 80.0))
        self.relative_depth_scale = float(getattr(cfg, "PSEUDO_BEV_REL_DEPTH_SCALE", 50.0))
        self.bev_input_proj = nn.Sequential(
            nn.Conv2d(c, self.pseudo_bev_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.pseudo_bev_channels),
            nn.GELU(),
        )
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(self.pseudo_bev_channels, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            ResidualConvBlock(c, c),
            ResidualConvBlock(c, c),
        )
        self.bev_obstacle_head = nn.Conv2d(c, 1, kernel_size=1)
        self.teacher_risk_head = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, c // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.GELU(),
            nn.Conv2d(c // 2, 1, kernel_size=1),
        )
        self.bev_teacher_dim = int(getattr(cfg, "BEVFORMER_TEACHER_DIM", 256))
        self.bev_teacher_adapter = nn.Sequential(
            nn.Conv2d(c, self.bev_teacher_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.bev_teacher_dim),
            nn.GELU(),
            nn.Conv2d(self.bev_teacher_dim, self.bev_teacher_dim, kernel_size=1),
        )
        self.bev_teacher_fusion_proj = nn.Sequential(
            nn.Conv2d(self.bev_teacher_dim, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=1),
        )
        self.bev_teacher_gate = nn.Sequential(
            nn.Linear(c * 3 + 1, c),
            nn.LayerNorm(c),
            nn.GELU(),
            nn.Linear(c, 1),
        )
        nn.init.constant_(self.bev_teacher_gate[-1].bias, float(getattr(cfg, "BEV_TEACHER_GATE_INIT_BIAS", -4.0)))
        self.bev_teacher_context_ln = nn.LayerNorm(c)
        self.bev_teacher_gate_min = float(getattr(cfg, "BEV_TEACHER_GATE_MIN", 0.0))
        self.bev_context_scale = float(getattr(cfg, "BEV_CONTEXT_SCALE", 0.1))
        self.bev_residual_scale = float(getattr(cfg, "BEV_RESIDUAL_SCALE", 0.5))
        self.bev_token_fusion_enabled = bool(getattr(cfg, "BEV_TOKEN_FUSION_ENABLED", False))
        self.bev_context_fusion_enabled = bool(getattr(cfg, "BEV_CONTEXT_FUSION_ENABLED", False))
        self.bev_ego_risk_context_enabled = bool(getattr(cfg, "BEV_EGO_RISK_CONTEXT_ENABLED", False))
        self.bev_residual_head = nn.Sequential(
            nn.Linear(c, c),
            nn.LayerNorm(c),
            nn.GELU(),
            nn.Linear(c, c // 2),
            nn.GELU(),
            nn.Linear(c // 2, self.n_future * 2),
        )
        nn.init.zeros_(self.bev_residual_head[-1].weight)
        nn.init.zeros_(self.bev_residual_head[-1].bias)
        self.spatial_pool = AttentionPool2d(c)
        self.temporal_gru = nn.GRU(c, c, batch_first=True)

        receptive_field = int(getattr(cfg, "TIME_RECEPTIVE_FIELD", 3))
        self.ego_mlp = nn.Sequential(
            nn.Linear(receptive_field * 4, c),
            nn.LayerNorm(c),
            nn.GELU(),
            nn.Linear(c, c),
            nn.GELU(),
        )
        self.command_embed = nn.Embedding(3, c)
        self.command_film = nn.Sequential(
            nn.Linear(c, c * 2),
            nn.GELU(),
            nn.Linear(c * 2, c * 2),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(c * 3, c),
            nn.LayerNorm(c),
            nn.GELU(),
            nn.Linear(c, c),
            nn.GELU(),
        )
        self.time_queries = nn.Parameter(torch.randn(self.n_future, c) * 0.02)
        self.query_context = nn.Sequential(
            nn.Linear(c * 3, c),
            nn.LayerNorm(c),
            nn.GELU(),
            nn.Linear(c, c),
        )
        self.spatial_pos_mlp = nn.Sequential(
            nn.Linear(2, c),
            nn.GELU(),
            nn.Linear(c, c),
        )
        self.spatial_token_ln = nn.LayerNorm(c)
        self.frame_embed = nn.Parameter(torch.randn(receptive_field, c) * 0.02)
        self.scale_embed = nn.Parameter(torch.randn(2, c) * 0.02)
        decoder_depth = int(getattr(cfg, "FAST_DECODER_LAYERS", 4))
        decoder_heads = int(getattr(cfg, "FAST_DECODER_HEADS", 8))
        self.bev_scale_embed = nn.Parameter(torch.randn(1, c) * 0.02)
        self.ego_query = nn.Parameter(torch.randn(1, c) * 0.02)
        self.ego_cross_attn = nn.MultiheadAttention(c, decoder_heads, batch_first=True)
        self.ego_ffn = nn.Sequential(
            nn.Linear(c, c * 4),
            nn.GELU(),
            nn.Dropout(float(getattr(cfg, "FAST_DROPOUT", 0.0))),
            nn.Linear(c * 4, c),
        )
        self.ego_norm1 = nn.LayerNorm(c)
        self.ego_norm2 = nn.LayerNorm(c)
        self.decoder_layers = nn.ModuleList([
            WaypointDecoderLayer(c, num_heads=decoder_heads, dropout=float(getattr(cfg, "FAST_DROPOUT", 0.0)))
            for _ in range(decoder_depth)
        ])
        self.waypoint_gru = nn.GRU(c, c, batch_first=True)
        self.waypoint_refine_ln = nn.LayerNorm(c)
        self.traj_abs_head = nn.Sequential(
            nn.Linear(c, c),
            nn.GELU(),
            nn.Linear(c, c // 2),
            nn.GELU(),
            nn.Linear(c // 2, 2),
        )
        self.traj_delta_head = nn.Sequential(
            nn.Linear(c, c),
            nn.GELU(),
            nn.Linear(c, c // 2),
            nn.GELU(),
            nn.Linear(c // 2, 2),
        )
        self.endpoint_head = nn.Sequential(
            nn.Linear(c, c),
            nn.GELU(),
            nn.Linear(c, c // 2),
            nn.GELU(),
            nn.Linear(c // 2, 2),
        )
        self.residual_scale = float(getattr(cfg, "FAST_RESIDUAL_SCALE", 30.0))
        self.delta_scale = float(getattr(cfg, "FAST_DELTA_SCALE", 4.0))
        self.last_coarse_xy = None
        self.last_endpoint_xy = None
        self.last_bev_feature = None
        self.last_obstacle_logits = None
        self.last_teacher_risk_logits = None
        self.last_bev_teacher_gate_mean = None
        self.last_bev_visible_mask = None
        self.last_bev_visible_ratio = None
        self.last_bev_residual_norm = None
        self.last_bev_branch_active = None

    @staticmethod
    def _make_stem(in_channels: int, hidden_dim: int) -> MultiScaleStem:
        return MultiScaleStem(in_channels, hidden_dim)

    def _rgb_like(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = x.float()
        if x.max().detach() > 2.0:
            x = x / 255.0
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return x

    def _seg_id(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == 1:
            x = x[:, 0]
        x = x.long().clamp(min=0, max=self.seg_id_embedding.num_embeddings - 1)
        emb = self.seg_id_embedding(x).permute(0, 3, 1, 2).contiguous()
        if emb.shape[-2:] != (self.input_size, self.input_size):
            emb = F.interpolate(emb, size=(self.input_size, self.input_size), mode="nearest")
        return emb

    def _depth(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] != 1:
            x = x.unsqueeze(1)
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=80.0, neginf=0.0)
        x = x.clamp(0.0, 80.0) / 80.0
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return x

    def _command_ids(self, commands: List[str], device: torch.device) -> torch.Tensor:
        ids = []
        for command in commands:
            if command not in self.COMMAND_TO_ID:
                raise ValueError(f"Unsupported command {command!r}; expected LEFT, FORWARD, or RIGHT.")
            ids.append(self.COMMAND_TO_ID[command])
        return torch.tensor(ids, device=device, dtype=torch.long)

    def _film(self, x: torch.Tensor, cmd_ctx: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.command_film(cmd_ctx).chunk(2, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma).view(cmd_ctx.shape[0], -1, 1, 1)
        beta = 0.1 * torch.tanh(beta).view(cmd_ctx.shape[0], -1, 1, 1)
        return x * gamma + beta

    def _check_finite(self, stage: str, **tensors):
        if not bool(getattr(self.cfg, "NAN_DEBUG", False)):
            return
        bad = []
        for name, tensor in tensors.items():
            if tensor is None or not torch.is_tensor(tensor):
                continue
            finite = torch.isfinite(tensor)
            if bool(finite.all().item()):
                continue
            detached = tensor.detach()
            finite_values = detached[finite]
            if finite_values.numel() > 0:
                min_v = float(finite_values.min().item())
                max_v = float(finite_values.max().item())
            else:
                min_v = float("nan")
                max_v = float("nan")
            bad.append(
                f"{name}: shape={tuple(tensor.shape)} "
                f"nonfinite_ratio={float((~finite).float().mean().item()):.6f} "
                f"finite_min={min_v:.6g} finite_max={max_v:.6g}"
            )
        if bad:
            msg = "[NaN debug] " + stage + "\n  " + "\n  ".join(bad)
            print(msg)
            if bool(getattr(self.cfg, "NAN_DEBUG_RAISE", True)):
                raise FloatingPointError(msg)

    def _tokens_from_feature(self, feat_seq: torch.Tensor, scale_index: int) -> torch.Tensor:
        b, t_rf, c, h, w = feat_seq.shape
        tokens = feat_seq.permute(0, 1, 3, 4, 2).reshape(b, t_rf, h * w, c)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=feat_seq.device, dtype=feat_seq.dtype),
            torch.linspace(-1.0, 1.0, w, device=feat_seq.device, dtype=feat_seq.dtype),
            indexing="ij",
        )
        pos = torch.stack([xx, yy], dim=-1).view(1, 1, h * w, 2)
        pos_embed = self.spatial_pos_mlp(pos)
        frame_embed = self.frame_embed[:t_rf].view(1, t_rf, 1, c)
        scale_embed = self.scale_embed[scale_index].view(1, 1, 1, c)
        tokens = self.spatial_token_ln(tokens + pos_embed + frame_embed + scale_embed)
        return tokens.reshape(b, t_rf * h * w, c).contiguous()

    def _tokens_from_bev(self, bev: torch.Tensor) -> torch.Tensor:
        if self.pseudo_bev_token_size > 0 and bev.shape[-1] != self.pseudo_bev_token_size:
            bev = F.adaptive_avg_pool2d(bev, (self.pseudo_bev_token_size, self.pseudo_bev_token_size))
        b, c, h, w = bev.shape
        tokens = bev.permute(0, 2, 3, 1).reshape(b, h * w, c)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=bev.device, dtype=bev.dtype),
            torch.linspace(-1.0, 1.0, w, device=bev.device, dtype=bev.dtype),
            indexing="ij",
        )
        pos = torch.stack([xx, yy], dim=-1).view(1, h * w, 2)
        return self.spatial_token_ln(tokens + self.spatial_pos_mlp(pos) + self.bev_scale_embed.view(1, 1, c))

    def _metric_depth(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 4 and depth.shape[1] != 1:
            depth = depth[:, :1]
        depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=self.depth_max_m, neginf=0.0)
        if float(depth.detach().amax()) <= 2.0:
            depth = depth * self.relative_depth_scale
        return depth.clamp(self.depth_min_m, self.depth_max_m)

    def _select_first_camera_calib(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            return x[:, :, 0]
        return x

    def _project_depth_to_bev_count(self, depth_bt: torch.Tensor, intrinsics: torch.Tensor,
                                    extrinsics: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        bt = depth_bt.shape[0]
        device = depth_bt.device
        dtype = intrinsics.dtype
        depth_grid = F.interpolate(depth_bt, size=(out_h, out_w), mode="bilinear", align_corners=False)

        ys = (torch.arange(out_h, device=device, dtype=dtype) + 0.5) * (self.input_size / float(out_h)) - 0.5
        xs = (torch.arange(out_w, device=device, dtype=dtype) + 0.5) * (self.input_size / float(out_w)) - 0.5
        vv, uu = torch.meshgrid(ys, xs, indexing="ij")
        uu = uu.reshape(1, -1).expand(bt, -1)
        vv = vv.reshape(1, -1).expand(bt, -1)
        z = depth_grid.reshape(bt, -1)

        fx = intrinsics[:, 0, 0].unsqueeze(1).clamp_min(1e-4)
        fy = intrinsics[:, 1, 1].unsqueeze(1).clamp_min(1e-4)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)
        x_cam = (uu - cx) / fx * z
        y_cam = (vv - cy) / fy * z
        pts_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=-1)
        pts_ego = torch.bmm(pts_cam, extrinsics.transpose(1, 2))[..., :3]

        forward = pts_ego[..., 0]
        side = pts_ego[..., 1]
        valid = (
            (z > self.depth_min_m)
            & (forward >= self.bev_x_min) & (forward < self.bev_x_max)
            & (side >= self.bev_y_min) & (side < self.bev_y_max)
        )
        iy = ((forward - self.bev_x_min) / self.pseudo_bev_grid_size).long().clamp(0, self.pseudo_bev_h - 1)
        ix = ((side - self.bev_y_min) / self.pseudo_bev_grid_size).long().clamp(0, self.pseudo_bev_w - 1)
        linear = iy * self.pseudo_bev_w + ix
        count = depth_grid.new_zeros(bt, 1, self.pseudo_bev_h * self.pseudo_bev_w)
        count.scatter_add_(2, linear.unsqueeze(1), valid.unsqueeze(1).to(depth_grid.dtype))
        return count

    def _pseudo_bev_from_feature(self, feat_seq: torch.Tensor, depth_seq: torch.Tensor,
                                 intrinsics: torch.Tensor, extrinsics: torch.Tensor):
        b, t_rf, c, h, w = feat_seq.shape
        bt = b * t_rf
        device = feat_seq.device
        dtype = feat_seq.dtype

        intrinsics = self._select_first_camera_calib(intrinsics).to(device=device, dtype=dtype).reshape(bt, 3, 3)
        extrinsics = self._select_first_camera_calib(extrinsics).to(device=device, dtype=dtype).reshape(bt, 4, 4)
        depth_full = depth_seq.reshape(bt, *depth_seq.shape[2:]).to(device, non_blocking=True)
        depth_full = self._metric_depth(depth_full)
        depth_bt = F.interpolate(depth_full, size=(h, w), mode="bilinear", align_corners=False)

        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * (self.input_size / float(h)) - 0.5
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * (self.input_size / float(w)) - 0.5
        vv, uu = torch.meshgrid(ys, xs, indexing="ij")
        uu = uu.reshape(1, -1).expand(bt, -1)
        vv = vv.reshape(1, -1).expand(bt, -1)
        z = depth_bt.reshape(bt, -1)

        fx = intrinsics[:, 0, 0].unsqueeze(1).clamp_min(1e-4)
        fy = intrinsics[:, 1, 1].unsqueeze(1).clamp_min(1e-4)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)
        x_cam = (uu - cx) / fx * z
        y_cam = (vv - cy) / fy * z
        pts_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=-1)
        pts_ego = torch.bmm(pts_cam, extrinsics.transpose(1, 2))[..., :3]

        forward = pts_ego[..., 0]
        side = pts_ego[..., 1]
        valid = (
            (z > self.depth_min_m)
            & (forward >= self.bev_x_min) & (forward < self.bev_x_max)
            & (side >= self.bev_y_min) & (side < self.bev_y_max)
        )
        iy = ((forward - self.bev_x_min) / self.pseudo_bev_grid_size).long().clamp(0, self.pseudo_bev_h - 1)
        ix = ((side - self.bev_y_min) / self.pseudo_bev_grid_size).long().clamp(0, self.pseudo_bev_w - 1)
        linear = iy * self.pseudo_bev_w + ix

        feat_bt = self.bev_input_proj(feat_seq.reshape(bt, c, h, w))
        feat_flat = feat_bt.reshape(bt, self.pseudo_bev_channels, h * w) * valid.unsqueeze(1).to(feat_bt.dtype)
        linear_expand = linear.unsqueeze(1).expand(-1, self.pseudo_bev_channels, -1)
        bev_flat = feat_bt.new_zeros(bt, self.pseudo_bev_channels, self.pseudo_bev_h * self.pseudo_bev_w)
        count = feat_bt.new_zeros(bt, 1, self.pseudo_bev_h * self.pseudo_bev_w)
        bev_flat.scatter_add_(2, linear_expand, feat_flat)
        count.scatter_add_(2, linear.unsqueeze(1), valid.unsqueeze(1).to(feat_bt.dtype))
        bev_flat = bev_flat / count.clamp_min(1.0)
        visible_size = int(getattr(self.cfg, "PSEUDO_BEV_VISIBLE_SIZE", 112))
        visible_size = max(h, min(self.input_size, visible_size))
        dense_count = self._project_depth_to_bev_count(depth_full, intrinsics, extrinsics, visible_size, visible_size)
        visible = (dense_count > 0).to(feat_bt.dtype)
        bev = bev_flat.view(bt, self.pseudo_bev_channels, self.pseudo_bev_h, self.pseudo_bev_w)
        visible = visible.view(bt, 1, self.pseudo_bev_h, self.pseudo_bev_w)
        return bev, visible

    def _ego_attend_risk_tokens(self, seed: torch.Tensor, risk_tokens: torch.Tensor) -> torch.Tensor:
        ego = self.ego_query.view(1, 1, -1).expand(seed.shape[0], 1, -1) + seed.unsqueeze(1)
        ego = ego + self.ego_cross_attn(self.ego_norm1(ego), risk_tokens, risk_tokens, need_weights=False)[0]
        ego = ego + self.ego_ffn(self.ego_norm2(ego))
        return ego.squeeze(1)

    def make_coarse_baseline(self, ego_seq: torch.Tensor) -> torch.Tensor:
        b, t_rf, _ = ego_seq.shape
        if t_rf >= 2:
            dxy_m = ego_seq[:, -2, 0:2] * self.EGO_SCALE_M
        else:
            dxy_m = ego_seq[:, -1, 0:2] * self.EGO_SCALE_M
        step_len = torch.norm(dxy_m, dim=-1).clamp(max=20.0)
        steps = torch.arange(1, self.n_future + 1, device=ego_seq.device, dtype=ego_seq.dtype).view(1, self.n_future)
        coarse = torch.zeros(b, self.n_future, 2, device=ego_seq.device, dtype=ego_seq.dtype)
        coarse[..., 1] = step_len.view(b, 1) * steps
        return coarse

    def forward(self, rgb_seq: torch.Tensor, seg_rgb_seq: torch.Tensor, seg_id_seq: torch.Tensor,
                depth_seq: torch.Tensor, ego_seq: torch.Tensor, commands: List[str],
                intrinsics: torch.Tensor = None, extrinsics: torch.Tensor = None) -> torch.Tensor:
        if seg_id_seq is None:
            raise ValueError("codex_pure requires seg_id_224_seq from NuscenesData_change.py.")
        if depth_seq is None:
            raise ValueError("codex_pure requires depth_224_seq from NuscenesData_change.py.")

        device = ego_seq.device
        b, t_rf = rgb_seq.shape[:2]
        rgb_seq = rgb_seq.to(device, non_blocking=True)
        seg_rgb_seq = seg_rgb_seq.to(device, non_blocking=True)
        seg_id_seq = seg_id_seq.to(device, non_blocking=True)
        depth_seq = depth_seq.to(device, non_blocking=True)

        rgb_bt = rgb_seq.reshape(b * t_rf, *rgb_seq.shape[2:])
        seg_rgb_bt = seg_rgb_seq.reshape(b * t_rf, *seg_rgb_seq.shape[2:])
        seg_id_bt = seg_id_seq.reshape(b * t_rf, *seg_id_seq.shape[2:])
        depth_bt = depth_seq.reshape(b * t_rf, *depth_seq.shape[2:])

        rgb_mid, rgb_deep = self.rgb_stem(self._rgb_like(rgb_bt))
        seg_rgb_mid, seg_rgb_deep = self.seg_rgb_stem(self._rgb_like(seg_rgb_bt))
        seg_id_mid, seg_id_deep = self.seg_id_stem(self._seg_id(seg_id_bt))
        depth_mid, depth_deep = self.depth_stem(self._depth(depth_bt))
        self._check_finite(
            "after_stems",
            rgb_mid=rgb_mid,
            rgb_deep=rgb_deep,
            seg_rgb_mid=seg_rgb_mid,
            seg_rgb_deep=seg_rgb_deep,
            seg_id_mid=seg_id_mid,
            seg_id_deep=seg_id_deep,
            depth_mid=depth_mid,
            depth_deep=depth_deep,
        )

        cmd_ctx = self.command_embed(self._command_ids(commands, device))
        fused_mid = self.mid_fusion(torch.cat([rgb_mid, seg_rgb_mid, seg_id_mid, depth_mid], dim=1))
        fused = self.fusion(torch.cat([rgb_deep, seg_rgb_deep, seg_id_deep, depth_deep], dim=1))
        fused_mid = self._film(fused_mid, cmd_ctx.repeat_interleave(t_rf, dim=0))
        fused = self._film(fused, cmd_ctx.repeat_interleave(t_rf, dim=0))
        self._check_finite("after_fusion", fused_mid=fused_mid, fused=fused)

        _, c, h_deep, w_deep = fused.shape
        fused_mid_seq = fused_mid.view(b, t_rf, c, *fused_mid.shape[-2:])
        fused_seq = fused.view(b, t_rf, c, h_deep, w_deep)
        frame_feat = self.spatial_pool(fused).view(b, t_rf, self.hidden_dim)
        _, h = self.temporal_gru(frame_feat)
        visual_ctx = h[-1]
        image_tokens = torch.cat([
            self._tokens_from_feature(fused_mid_seq, scale_index=0),
            self._tokens_from_feature(fused_seq, scale_index=1),
        ], dim=1)

        risk_tokens = None
        teacher_ctx_delta = None
        bev_plan_ctx = None
        self.last_bev_branch_active = torch.tensor(0.0, device=device)
        if intrinsics is not None and extrinsics is not None and intrinsics.numel() > 0 and extrinsics.numel() > 0:
            self.last_bev_branch_active = torch.tensor(1.0, device=device)
            freeze_all_bev = bool(getattr(self.cfg, "FREEZE_ALL_BEV", False))
            bev_source = fused_mid_seq.detach() if freeze_all_bev else fused_mid_seq
            bev_low, visible = self._pseudo_bev_from_feature(bev_source, depth_seq, intrinsics, extrinsics)
            bev_low = bev_low.view(b, t_rf, self.pseudo_bev_channels, self.pseudo_bev_h, self.pseudo_bev_w)
            visible = visible.view(b, t_rf, 1, self.pseudo_bev_h, self.pseudo_bev_w)
            current_bev = self.bev_encoder(bev_low[:, -1])
            self._check_finite("after_bev_encoder", bev_low=bev_low, current_bev=current_bev, visible=visible)
            if freeze_all_bev:
                current_bev = current_bev.detach()
            current_visible = visible[:, -1]
            self.last_bev_feature = current_bev
            if bool(getattr(self.cfg, "BEV_TEACHER_GATE_ENABLED", True)):
                teacher_like = self.bev_teacher_adapter(current_bev)
                teacher_residual = self.bev_teacher_fusion_proj(teacher_like)
                bev_ctx = current_bev.flatten(2).mean(dim=-1)
                teacher_ctx = teacher_residual.flatten(2).mean(dim=-1)
                visible_ctx = current_visible.to(dtype=current_bev.dtype).flatten(2).mean(dim=-1)
                gate_input = torch.cat([visual_ctx, bev_ctx, teacher_ctx, visible_ctx], dim=-1)
                self._check_finite(
                    "before_bev_gate",
                    visual_ctx=visual_ctx,
                    bev_ctx=bev_ctx,
                    teacher_ctx=teacher_ctx,
                    visible_ctx=visible_ctx,
                    gate_input=gate_input,
                )
                gate_logits = self.bev_teacher_gate(gate_input)
                self._check_finite("after_bev_gate", gate_logits=gate_logits)
                raw_gate = torch.sigmoid(gate_logits)
                gate_min = min(max(self.bev_teacher_gate_min, 0.0), 1.0)
                teacher_gate = gate_min + (1.0 - gate_min) * raw_gate
                bev_plan_ctx = teacher_gate * self.bev_teacher_context_ln(teacher_ctx)
                self._check_finite("after_bev_plan_ctx", teacher_gate=teacher_gate, bev_plan_ctx=bev_plan_ctx)
                if self.bev_context_fusion_enabled:
                    teacher_ctx_delta = self.bev_context_scale * bev_plan_ctx
                self.last_bev_teacher_gate_mean = teacher_gate.detach().float().mean()
            else:
                teacher_ctx_delta = None
                bev_plan_ctx = self.bev_teacher_context_ln(current_bev.flatten(2).mean(dim=-1))
                self._check_finite("after_bev_plan_ctx_no_gate", bev_plan_ctx=bev_plan_ctx)
                self.last_bev_teacher_gate_mean = torch.tensor(0.0, device=device, dtype=current_bev.dtype)
            self.last_obstacle_logits = self.bev_obstacle_head(current_bev)
            self.last_teacher_risk_logits = self.teacher_risk_head(current_bev)
            self._check_finite(
                "after_bev_heads",
                obstacle_logits=self.last_obstacle_logits,
                teacher_risk_logits=self.last_teacher_risk_logits,
            )
            self.last_bev_visible_mask = current_visible
            self.last_bev_visible_ratio = current_visible.detach().float().mean()
            risk_tokens = self._tokens_from_bev(current_bev)
            if self.bev_token_fusion_enabled:
                spatial_tokens = torch.cat([image_tokens, risk_tokens], dim=1)
            else:
                spatial_tokens = image_tokens
        else:
            self.last_bev_feature = None
            self.last_obstacle_logits = None
            self.last_teacher_risk_logits = None
            self.last_bev_teacher_gate_mean = None
            self.last_bev_visible_mask = None
            self.last_bev_visible_ratio = None
            self.last_bev_residual_norm = None
            self.last_bev_branch_active = torch.tensor(0.0, device=device)
            spatial_tokens = image_tokens

        ego_ctx = self.ego_mlp(ego_seq.reshape(b, -1).to(device))
        ctx = self.context_mlp(torch.cat([visual_ctx, ego_ctx, cmd_ctx], dim=-1))

        query_ctx = self.query_context(torch.cat([visual_ctx, ego_ctx, cmd_ctx], dim=-1))
        if teacher_ctx_delta is not None:
            ctx = ctx + teacher_ctx_delta
            query_ctx = query_ctx + teacher_ctx_delta
        if risk_tokens is not None and self.bev_ego_risk_context_enabled:
            ego_risk_ctx = self._ego_attend_risk_tokens(query_ctx, risk_tokens)
            ctx = ctx + ego_risk_ctx
            query_ctx = query_ctx + ego_risk_ctx
        queries = self.time_queries.unsqueeze(0).expand(b, -1, -1) + query_ctx.unsqueeze(1)
        queries = queries + ctx.unsqueeze(1)
        self._check_finite("before_decoder", ctx=ctx, query_ctx=query_ctx, spatial_tokens=spatial_tokens, queries=queries)
        for layer in self.decoder_layers:
            queries = layer(queries, spatial_tokens)
            self._check_finite("decoder_layer", queries=queries)
        refined, _ = self.waypoint_gru(queries)
        queries = self.waypoint_refine_ln(queries + refined)
        self._check_finite("after_waypoint_gru", refined=refined, queries=queries)

        coarse = self.make_coarse_baseline(ego_seq)
        self.last_coarse_xy = coarse.detach()
        abs_residual = torch.tanh(self.traj_abs_head(queries)).view(b, self.n_future, 2) * self.residual_scale
        delta_residual = torch.tanh(self.traj_delta_head(queries)).view(b, self.n_future, 2) * self.delta_scale
        residual = abs_residual + torch.cumsum(delta_residual, dim=1)
        xy = coarse + residual
        self._check_finite(
            "after_traj_heads",
            abs_residual=abs_residual,
            delta_residual=delta_residual,
            residual=residual,
            coarse=coarse,
            xy=xy,
        )
        if bev_plan_ctx is not None and self.bev_residual_scale > 0.0:
            bev_residual = torch.tanh(self.bev_residual_head(bev_plan_ctx)).view(b, self.n_future, 2)
            bev_residual = bev_residual * self.bev_residual_scale
            self._check_finite("after_bev_residual", bev_plan_ctx=bev_plan_ctx, bev_residual=bev_residual)
            xy = xy + bev_residual
            self._check_finite("after_bev_residual_add", xy=xy)
            self.last_bev_residual_norm = bev_residual.detach().norm(dim=-1).mean()
        else:
            self.last_bev_residual_norm = torch.tensor(0.0, device=device, dtype=xy.dtype)
        endpoint_residual = torch.tanh(self.endpoint_head(ctx)) * self.residual_scale
        self.last_endpoint_xy = coarse[:, -1, :] + endpoint_residual
        self._check_finite("before_output", endpoint_residual=endpoint_residual, endpoint_xy=self.last_endpoint_xy, xy=xy)
        z = torch.zeros(b, self.n_future, 1, device=device, dtype=xy.dtype)
        return torch.cat([xy, z], dim=-1)


class VLM_STP3_Gen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD
        self.n_future = cfg.N_FUTURE_FRAMES
        self.input_size = int(getattr(cfg, "CLIP_INPUT_SIZE", 224))
        self.vlm = FastTrajectoryPlanner(cfg, self.n_future, self.input_size)

        dx, bx, _ = gen_dx_bx(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
        self.dx = nn.Parameter(dx[:2], requires_grad=False)
        self.bx = nn.Parameter(bx[:2], requires_grad=False)
        _, _, bev_dim = calculate_birds_eye_view_parameters(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
        self.bev_dim = bev_dim.numpy().tolist()

        self.encoder_out_channels = 64
        self.fake_cam_front = nn.Parameter(torch.zeros(1, self.encoder_out_channels, 60, 28), requires_grad=False)
        self._last_rgb_seq = None
        self._last_seg_seq = None
        self._last_seg_id_seq = None
        self._last_depth_seq = None
        self._last_ego_seq = None
        self._last_intrinsics = None
        self._last_extrinsics = None

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("model : codex_pure_VAD_risk")
        print(f"Total parameters: {total:,}  Trainable parameters: {trainable:,}")

    def forward(self, image, intrinsics, extrinsics, future_egomotion, *,
                rgb_224_seq, seg_224_seq, seg_id_224_seq=None, depth_224_seq=None,
                ego_history_egomotion=None):
        if seg_id_224_seq is None:
            raise ValueError("codex_pure requires batch['seg_id_224_seq']; use NuscenesData_change.py.")
        if depth_224_seq is None:
            raise ValueError("codex_pure requires batch['depth_224_seq']; use NuscenesData_change.py.")

        device = future_egomotion.device
        self._last_rgb_seq = rgb_224_seq.to(device, non_blocking=True)
        self._last_seg_seq = seg_224_seq.to(device, non_blocking=True)
        self._last_seg_id_seq = seg_id_224_seq.to(device, non_blocking=True)
        self._last_depth_seq = depth_224_seq.to(device, non_blocking=True)
        self._last_intrinsics = intrinsics.to(device, non_blocking=True) if torch.is_tensor(intrinsics) else None
        self._last_extrinsics = extrinsics.to(device, non_blocking=True) if torch.is_tensor(extrinsics) else None

        from stp3.utils.geometry import mat2pose_vec, pose_vec2mat

        b = self._last_rgb_seq.shape[0]
        fego = future_egomotion[:, :self.receptive_field, :]
        ego_seq_embed = []
        for t in range(self.receptive_field):
            if t == self.receptive_field - 1:
                dx = torch.zeros(b, 1, device=device)
                dy = torch.zeros(b, 1, device=device)
                dyaw = torch.zeros(b, 1, device=device)
            else:
                mats = [pose_vec2mat(fego[:, k, :]) for k in range(t, self.receptive_field - 1)]
                transform = mats[0]
                for mat in mats[1:]:
                    transform = torch.bmm(transform, mat)
                pose = mat2pose_vec(transform)
                dx = pose[:, 0:1]
                dy = pose[:, 1:2]
                dyaw = pose[:, 5:6]
                dyaw = (dyaw + torch.pi) % (2 * torch.pi) - torch.pi
            scale = float(getattr(self.cfg, "EGO_SCALE_M", 5.0))
            ego_seq_embed.append(torch.cat([dx / scale, dy / scale, torch.sin(dyaw), torch.cos(dyaw)], dim=1))
        self._last_ego_seq = torch.stack(ego_seq_embed, dim=1).detach()
        return {}, self._last_rgb_seq

    def occupancy_collision_rate(self, trajs_xy: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        device = trajs_xy.device
        b, t, _ = trajs_xy.shape
        h, w = occupancy.shape[-2:]
        yy = ((trajs_xy[..., 1] - self.bx[0]) / self.dx[0]).long().clamp(0, h - 1)
        xx = ((trajs_xy[..., 0] - self.bx[1]) / self.dx[1]).long().clamp(0, w - 1)
        ti = torch.arange(t, device=device).view(1, t).expand(b, t)
        bi = torch.arange(b, device=device).view(b, 1).expand(b, t)
        return occupancy[bi, ti, yy, xx].float().mean()

    def collision_loss_soft(self, trajs_xy: torch.Tensor, occupancy: torch.Tensor,
                            return_per_t: bool = False) -> torch.Tensor:
        b, t, _ = trajs_xy.shape
        occ = occupancy.float()
        h, w = occ.shape[-2:]
        occ = F.max_pool2d(occ.view(b * t, 1, h, w), kernel_size=5, stride=1, padding=2).view(b, t, h, w)
        for _ in range(2):
            occ = F.avg_pool2d(occ.view(b * t, 1, h, w), kernel_size=5, stride=1, padding=2).view(b, t, h, w)
        y = (trajs_xy[..., 1] - self.bx[0]) / self.dx[0]
        x = (trajs_xy[..., 0] - self.bx[1]) / self.dx[1]
        grid_x = x / max(w - 1, 1) * 2.0 - 1.0
        grid_y = y / max(h - 1, 1) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)
        samples = F.grid_sample(
            occ.view(b * t, 1, h, w),
            grid.view(b * t, 1, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(b, t)
        return samples if return_per_t else samples.mean()

    def box_collision_loss_soft(self, trajs_xy: torch.Tensor, occupancy: torch.Tensor,
                                return_per_t: bool = False) -> torch.Tensor:
        b, t, _ = trajs_xy.shape
        occ = occupancy.float()
        h, w = occ.shape[-2:]
        occ = F.max_pool2d(occ.view(b * t, 1, h, w), kernel_size=3, stride=1, padding=1).view(b, t, h, w)
        for _ in range(2):
            occ = F.avg_pool2d(occ.view(b * t, 1, h, w), kernel_size=5, stride=1, padding=2).view(b, t, h, w)

        trajs_metric = trajs_xy * torch.tensor([-1.0, 1.0], device=trajs_xy.device, dtype=trajs_xy.dtype)
        ego_w = float(getattr(self.cfg.EGO, "WIDTH", 1.85))
        ego_h = float(getattr(self.cfg.EGO, "HEIGHT", 4.084))
        nx = int(getattr(self.cfg, "BOX_SAMPLE_X", 5))
        ny = int(getattr(self.cfg, "BOX_SAMPLE_Y", 9))
        x_offsets = torch.linspace(-ego_w / 2.0, ego_w / 2.0, nx, device=trajs_xy.device, dtype=trajs_xy.dtype)
        y_offsets = torch.linspace(-ego_h / 2.0 + 0.5, ego_h / 2.0 + 0.5, ny, device=trajs_xy.device, dtype=trajs_xy.dtype)
        yy_off, xx_off = torch.meshgrid(y_offsets, x_offsets, indexing="ij")
        offsets = torch.stack([xx_off.reshape(-1), yy_off.reshape(-1)], dim=-1)

        pts = trajs_metric.unsqueeze(2) + offsets.view(1, 1, -1, 2)
        y = (pts[..., 1] - self.bx[0]) / self.dx[0]
        x = (pts[..., 0] - self.bx[1]) / self.dx[1]
        grid_x = x / max(w - 1, 1) * 2.0 - 1.0
        grid_y = y / max(h - 1, 1) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(b * t, -1, 1, 2)
        sampled = F.grid_sample(
            occ.view(b * t, 1, h, w),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(b, t, -1)
        per_t = sampled.amax(dim=-1)
        return per_t if return_per_t else per_t.mean()

    def _as_bev_mask(self, mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() == 4 and mask.shape[1] != 1:
            mask = mask[:, :1]
        mask = mask.float().to(device=logits.device)
        if mask.shape[-2:] != logits.shape[-2:]:
            mask = F.interpolate(mask, size=logits.shape[-2:], mode="nearest")
        return (mask > 0.5).to(dtype=logits.dtype)

    def _dilate_bev_mask(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size <= 1:
            return mask
        pad = kernel_size // 2
        return (F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad) > 0.5).to(mask.dtype)

    def _front_corridor_mask(self, logits: torch.Tensor) -> torch.Tensor:
        h, w = logits.shape[-2:]
        device = logits.device
        dtype = logits.dtype
        x_min = float(getattr(self.cfg, "VAD_CORRIDOR_X_MIN", getattr(self.cfg, "BEV_CORRIDOR_X_MIN", 0.0)))
        x_max = float(getattr(self.cfg, "VAD_CORRIDOR_X_MAX", getattr(self.cfg, "BEV_CORRIDOR_X_MAX", 45.0)))
        y_abs = float(getattr(self.cfg, "VAD_CORRIDOR_Y_ABS", getattr(self.cfg, "BEV_CORRIDOR_Y_ABS", 10.0)))
        forward = torch.linspace(
            self.vlm.bev_x_min + 0.5 * self.vlm.pseudo_bev_grid_size,
            self.vlm.bev_x_max - 0.5 * self.vlm.pseudo_bev_grid_size,
            h,
            device=device,
            dtype=dtype,
        ).view(1, 1, h, 1)
        side = torch.linspace(
            self.vlm.bev_y_min + 0.5 * self.vlm.pseudo_bev_grid_size,
            self.vlm.bev_y_max - 0.5 * self.vlm.pseudo_bev_grid_size,
            w,
            device=device,
            dtype=dtype,
        ).view(1, 1, 1, w)
        corridor = (forward >= x_min) & (forward <= x_max) & (side.abs() <= y_abs)
        return corridor.to(dtype=dtype).expand(logits.shape[0], 1, h, w)

    def _teacher_risk_corridor_mask(self, logits: torch.Tensor) -> torch.Tensor:
        h, w = logits.shape[-2:]
        device = logits.device
        dtype = logits.dtype
        x_min = float(getattr(self.cfg, "VAD_TEACHER_RISK_X_MIN", 0.0))
        x_max = float(getattr(self.cfg, "VAD_TEACHER_RISK_X_MAX", 35.0))
        y_abs = float(getattr(self.cfg, "VAD_TEACHER_RISK_Y_ABS", 6.0))
        forward = torch.linspace(
            self.vlm.bev_x_min + 0.5 * self.vlm.pseudo_bev_grid_size,
            self.vlm.bev_x_max - 0.5 * self.vlm.pseudo_bev_grid_size,
            h,
            device=device,
            dtype=dtype,
        ).view(1, 1, h, 1)
        side = torch.linspace(
            self.vlm.bev_y_min + 0.5 * self.vlm.pseudo_bev_grid_size,
            self.vlm.bev_y_max - 0.5 * self.vlm.pseudo_bev_grid_size,
            w,
            device=device,
            dtype=dtype,
        ).view(1, 1, 1, w)
        corridor = (forward >= x_min) & (forward <= x_max) & (side.abs() <= y_abs)
        return corridor.to(dtype=dtype).expand(logits.shape[0], 1, h, w)

    def _teacher_bev_corridor_mask(self, h: int, w: int, device, dtype) -> torch.Tensor:
        x_min = float(getattr(self.cfg, "BEV_TEACHER_CORRIDOR_X_MIN", getattr(self.cfg, "VAD_CORRIDOR_X_MIN", 0.0)))
        x_max = float(getattr(self.cfg, "BEV_TEACHER_CORRIDOR_X_MAX", getattr(self.cfg, "VAD_CORRIDOR_X_MAX", 45.0)))
        y_abs = float(getattr(self.cfg, "BEV_TEACHER_CORRIDOR_Y_ABS", getattr(self.cfg, "VAD_CORRIDOR_Y_ABS", 10.0)))
        teacher_x_min = float(getattr(self.cfg, "BEVFORMER_PC_X_MIN", -50.0))
        teacher_x_max = float(getattr(self.cfg, "BEVFORMER_PC_X_MAX", 50.0))
        teacher_y_min = float(getattr(self.cfg, "BEVFORMER_PC_Y_MIN", -50.0))
        teacher_y_max = float(getattr(self.cfg, "BEVFORMER_PC_Y_MAX", 50.0))
        forward = torch.linspace(
            teacher_x_min + 0.5 * (teacher_x_max - teacher_x_min) / h,
            teacher_x_max - 0.5 * (teacher_x_max - teacher_x_min) / h,
            h,
            device=device,
            dtype=dtype,
        ).view(1, 1, h, 1)
        side = torch.linspace(
            teacher_y_min + 0.5 * (teacher_y_max - teacher_y_min) / w,
            teacher_y_max - 0.5 * (teacher_y_max - teacher_y_min) / w,
            w,
            device=device,
            dtype=dtype,
        ).view(1, 1, 1, w)
        return ((forward >= x_min) & (forward <= x_max) & (side.abs() <= y_abs)).to(dtype)

    def _bev_teacher_distill_loss(self, teacher_bev: torch.Tensor,
                                  teacher_valid: torch.Tensor = None) -> torch.Tensor:
        student_bev = getattr(self.vlm, "last_bev_feature", None)
        visible_mask = getattr(self.vlm, "last_bev_visible_mask", None)
        self._last_bev_teacher_mask_ratio = None
        if student_bev is None or teacher_bev is None:
            device = teacher_bev.device if torch.is_tensor(teacher_bev) else next(self.parameters()).device
            self._last_bev_teacher_mask_ratio = torch.tensor(0.0, device=device)
            return torch.tensor(0.0, device=device)

        device = student_bev.device
        teacher_bev = teacher_bev.to(device=device, dtype=student_bev.dtype)
        if teacher_bev.dim() == 3:
            teacher_bev = teacher_bev.unsqueeze(0)
        if teacher_bev.shape[1] != self.vlm.bev_teacher_dim and teacher_bev.shape[-1] == self.vlm.bev_teacher_dim:
            teacher_bev = teacher_bev.permute(0, 3, 1, 2).contiguous()

        student = self.vlm.bev_teacher_adapter(student_bev)
        student = F.interpolate(student, size=teacher_bev.shape[-2:], mode="bilinear", align_corners=False)
        mask = self._teacher_bev_corridor_mask(
            teacher_bev.shape[-2], teacher_bev.shape[-1], device, student.dtype
        ).expand(teacher_bev.shape[0], 1, -1, -1)

        if visible_mask is not None:
            visible = F.interpolate(
                visible_mask.to(device=device, dtype=student.dtype),
                size=teacher_bev.shape[-2:],
                mode="nearest",
            )
            visible = (visible > 0).to(student.dtype)
            dilate_k = int(getattr(self.cfg, "BEV_TEACHER_VISIBLE_DILATE", 5))
            if dilate_k > 1:
                if dilate_k % 2 == 0:
                    dilate_k += 1
                visible = F.max_pool2d(visible, kernel_size=dilate_k, stride=1, padding=dilate_k // 2)
            mask = mask * (visible > 0).to(student.dtype)
        if teacher_valid is not None:
            valid = teacher_valid.to(device=device, dtype=student.dtype).view(-1, 1, 1, 1)
            mask = mask * valid
        self._last_bev_teacher_mask_ratio = mask.detach().float().mean()
        if mask.sum() <= 0:
            return student.sum() * 0.0

        denom = mask.sum().clamp_min(1.0)
        student_norm = F.normalize(student, dim=1)
        teacher_norm = F.normalize(teacher_bev, dim=1)
        cosine = ((1.0 - (student_norm * teacher_norm).sum(dim=1, keepdim=True)) * mask).sum() / denom
        l2 = (((student - teacher_bev) ** 2).mean(dim=1, keepdim=True) * mask).sum() / denom
        return cosine + float(getattr(self.cfg, "LOSS_BEV_TEACHER_L2_RATIO", 0.1)) * l2

    def _bev_obstacle_aux_loss(self, logits: torch.Tensor, target: torch.Tensor,
                               visible_mask: torch.Tensor = None) -> torch.Tensor:
        target = self._as_bev_mask(target, logits)
        target = self._dilate_bev_mask(target, int(getattr(self.cfg, "VAD_RISK_TARGET_DILATE", 3)))
        valid_mask = self._front_corridor_mask(logits)
        if visible_mask is not None:
            visible = self._as_bev_mask(visible_mask, logits)
            visible = self._dilate_bev_mask(visible, int(getattr(self.cfg, "VAD_RISK_VISIBLE_DILATE", 3)))
            valid_mask = valid_mask * visible
        if valid_mask.sum() <= 0:
            return logits.sum() * 0.0

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pos_mask = valid_mask * target
        neg_mask = valid_mask * (1.0 - target)
        pos_den = pos_mask.sum()
        neg_den = neg_mask.sum()
        if pos_den <= 0:
            return (bce * neg_mask).sum() / neg_den.clamp_min(1.0)
        if neg_den <= 0:
            return (bce * pos_mask).sum() / pos_den.clamp_min(1.0)
        pos_loss = (bce * pos_mask).sum() / pos_den.clamp_min(1.0)
        neg_loss = (bce * neg_mask).sum() / neg_den.clamp_min(1.0)
        return 0.5 * (pos_loss + neg_loss)

    def _teacher_risk_distill_loss(self, logits: torch.Tensor, teacher_heatmap: torch.Tensor,
                                   teacher_valid: torch.Tensor = None,
                                   visible_mask: torch.Tensor = None) -> torch.Tensor:
        self._last_teacher_risk_mask_ratio = None
        if logits is None or teacher_heatmap is None:
            device = logits.device if logits is not None else next(self.parameters()).device
            self._last_teacher_risk_mask_ratio = torch.tensor(0.0, device=device)
            return torch.tensor(0.0, device=device)

        device = logits.device
        dtype = logits.dtype
        teacher = teacher_heatmap.to(device=device, dtype=dtype)
        if teacher.dim() == 3:
            teacher = teacher.unsqueeze(0)
        if teacher.shape[1] not in (1, 2) and teacher.shape[-1] in (1, 2):
            teacher = teacher.permute(0, 3, 1, 2).contiguous()
        if teacher.shape[1] == 1:
            teacher = torch.cat([teacher, teacher], dim=1)
        if teacher.shape[1] > 2:
            teacher = teacher[:, :2]
        teacher = teacher.clamp(0.0, 1.0)
        if teacher.shape[-2:] != logits.shape[-2:]:
            teacher = F.interpolate(teacher, size=logits.shape[-2:], mode="bilinear", align_corners=False)

        target_obj = teacher[:, :1].clamp(0.0, 1.0)
        target_risk = teacher[:, 1:2].clamp(0.0, 1.0)
        valid_mask = self._teacher_risk_corridor_mask(logits)
        if bool(getattr(self.cfg, "VAD_TEACHER_RISK_USE_VISIBLE", True)) and visible_mask is not None:
            visible = self._as_bev_mask(visible_mask, logits)
            visible = self._dilate_bev_mask(visible, int(getattr(self.cfg, "VAD_TEACHER_RISK_VISIBLE_DILATE", 1)))
            valid_mask = valid_mask * visible
        if teacher_valid is not None:
            valid = teacher_valid.to(device=device, dtype=dtype).view(-1, 1, 1, 1)
            valid_mask = valid_mask * valid

        self._last_teacher_risk_mask_ratio = valid_mask.detach().float().mean()
        if valid_mask.sum() <= 0:
            return logits.sum() * 0.0

        pred_risk = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target_obj, reduction="none")
        mse = (pred_risk - target_risk).pow(2)
        bce_w = float(getattr(self.cfg, "LOSS_TEACHER_RISK_BCE_W", 1.0))
        mse_w = float(getattr(self.cfg, "LOSS_TEACHER_RISK_MSE_W", 0.5))

        pos_mask = valid_mask * (target_obj > 0.05).to(dtype)
        neg_mask = valid_mask * (target_obj <= 0.05).to(dtype)
        pos_den = pos_mask.sum()
        neg_den = neg_mask.sum()
        if pos_den > 0 and neg_den > 0:
            bce_loss = 0.5 * (
                (bce * pos_mask).sum() / pos_den.clamp_min(1.0)
                + (bce * neg_mask).sum() / neg_den.clamp_min(1.0)
            )
        else:
            bce_loss = (bce * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        mse_loss = (mse * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        return bce_w * bce_loss + mse_w * mse_loss

    def _pseudo_bev_box_risk_loss(self, trajs_xy: torch.Tensor, obstacle_logits: torch.Tensor,
                                  gt_trajs_xy: torch.Tensor = None, occupancy: torch.Tensor = None,
                                  return_per_t: bool = False) -> torch.Tensor:
        b, t, _ = trajs_xy.shape
        risk = torch.sigmoid(obstacle_logits)
        risk = risk * self._front_corridor_mask(obstacle_logits)
        risk = F.max_pool2d(risk, kernel_size=3, stride=1, padding=1)
        h, w = risk.shape[-2:]

        ego_w = float(getattr(self.cfg.EGO, "WIDTH", 1.85))
        ego_h = float(getattr(self.cfg.EGO, "HEIGHT", 4.084))
        nx = int(getattr(self.cfg, "VAD_RISK_BOX_SAMPLE_X", 5))
        ny = int(getattr(self.cfg, "VAD_RISK_BOX_SAMPLE_Y", 9))
        side_offsets = torch.linspace(-ego_w / 2.0, ego_w / 2.0, nx, device=trajs_xy.device, dtype=trajs_xy.dtype)
        fwd_offsets = torch.linspace(-ego_h / 2.0 + 0.5, ego_h / 2.0 + 0.5, ny, device=trajs_xy.device, dtype=trajs_xy.dtype)
        fwd_off, side_off = torch.meshgrid(fwd_offsets, side_offsets, indexing="ij")
        offsets = torch.stack([side_off.reshape(-1), fwd_off.reshape(-1)], dim=-1)

        trajs_metric = trajs_xy * torch.tensor([-1.0, 1.0], device=trajs_xy.device, dtype=trajs_xy.dtype)
        pts = trajs_metric.unsqueeze(2) + offsets.view(1, 1, -1, 2)
        side = pts[..., 0]
        forward = pts[..., 1]
        grid_x = (side - self.vlm.bev_y_min) / (self.vlm.bev_y_max - self.vlm.bev_y_min)
        grid_y = (forward - self.vlm.bev_x_min) / (self.vlm.bev_x_max - self.vlm.bev_x_min)
        grid = torch.stack([grid_x * 2.0 - 1.0, grid_y * 2.0 - 1.0], dim=-1).view(b, t * offsets.shape[0], 1, 2)
        sampled = F.grid_sample(risk, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        per_t = sampled.view(b, 1, t, offsets.shape[0]).squeeze(1).amax(dim=-1)

        if gt_trajs_xy is not None and occupancy is not None:
            with torch.no_grad():
                gt_box_risk_t = self.box_collision_loss_soft(gt_trajs_xy, occupancy, return_per_t=True)
                valid_mask = (gt_box_risk_t < float(getattr(self.cfg, "VAD_GT_SAFE_COL_THRESH", 0.2))).to(per_t.dtype)
            per_t = per_t * valid_mask

        return per_t if return_per_t else per_t.mean()

    def planning(self, *, bev_rgbs, trajs, gt_trajs, commands, target_points, occupancy=None, drivable_mask=None,
                 occupancy_aux=None, drivable_aux=None, teacher_scene_feature=None,
                 teacher_bev_feature=None, teacher_bev_valid=None,
                 teacher_risk_heatmap=None, teacher_risk_valid=None):
        assert self._last_rgb_seq is not None and self._last_seg_seq is not None
        assert self._last_seg_id_seq is not None and self._last_depth_seq is not None and self._last_ego_seq is not None

        device = gt_trajs.device
        pred = self.vlm(
            self._last_rgb_seq,
            self._last_seg_seq,
            self._last_seg_id_seq,
            self._last_depth_seq,
            self._last_ego_seq.to(device),
            commands,
            intrinsics=self._last_intrinsics,
            extrinsics=self._last_extrinsics,
        ).to(device)
        if bool(getattr(self.cfg, "NAN_DEBUG", False)) and not torch.isfinite(pred).all():
            finite = torch.isfinite(pred)
            msg = (
                "[NaN debug] planner prediction is non-finite "
                f"nonfinite_ratio={float((~finite).float().mean().item()):.6f}"
            )
            print(msg)
            if bool(getattr(self.cfg, "NAN_DEBUG_RAISE", True)):
                raise FloatingPointError(msg)
        final_traj = pred

        pred_xy = pred[..., :2]
        gt_xy = gt_trajs[..., :2]
        err = ((pred_xy - gt_xy) ** 2).sum(dim=-1).sqrt()
        t_len = err.shape[1]
        w = torch.linspace(1.3, 1.0, t_len, device=device)
        l2 = (err * w).mean()
        fde = ((pred_xy[:, -1] - gt_xy[:, -1]) ** 2).sum(dim=-1).sqrt().mean()

        pred_d = torch.cat([pred_xy[:, :1], pred_xy[:, 1:] - pred_xy[:, :-1]], dim=1)
        gt_d = torch.cat([gt_xy[:, :1], gt_xy[:, 1:] - gt_xy[:, :-1]], dim=1)
        vel_l2 = ((pred_d - gt_d).pow(2).sum(-1) * w).mean()

        vel = pred_xy - torch.cat([pred_xy[:, :1], pred_xy[:, :-1]], dim=1)
        smooth = (vel[:, 1:] - vel[:, :-1]).pow(2).sum(-1).mean()

        collision = torch.tensor(0.0, device=device)
        box_collision = torch.tensor(0.0, device=device)
        hard_collision = torch.tensor(0.0, device=device)
        vad_risk_plan = torch.tensor(0.0, device=device)
        if occupancy is not None:
            risk_t = self.collision_loss_soft(pred_xy, occupancy, return_per_t=True)
            box_risk_t = self.box_collision_loss_soft(pred_xy, occupancy, return_per_t=True)
            with torch.no_grad():
                gt_box_risk_t = self.box_collision_loss_soft(gt_xy, occupancy, return_per_t=True)
                valid_col_mask = (gt_box_risk_t < 0.2).to(box_risk_t.dtype)
            col_w = torch.linspace(1.0, 2.0, risk_t.shape[1], device=device, dtype=risk_t.dtype).view(1, -1)
            collision = ((risk_t ** 2) * col_w).mean()
            box_collision = (((box_risk_t ** 2) * col_w * valid_col_mask).sum()
                             / (valid_col_mask.sum() + 1e-6))
            if self.training:
                with torch.no_grad():
                    hard_collision = self.occupancy_collision_rate(pred_xy, occupancy)

        obstacle_logits = getattr(self.vlm, "last_obstacle_logits", None)
        teacher_risk_logits = getattr(self.vlm, "last_teacher_risk_logits", None)
        visible_mask = getattr(self.vlm, "last_bev_visible_mask", None)
        visible_ratio = getattr(self.vlm, "last_bev_visible_ratio", None)
        bev_teacher_gate_mean = getattr(self.vlm, "last_bev_teacher_gate_mean", None)
        bev_residual_norm = getattr(self.vlm, "last_bev_residual_norm", None)
        bev_branch_active = getattr(self.vlm, "last_bev_branch_active", None)
        if visible_ratio is None:
            visible_ratio = torch.tensor(0.0, device=device)
        else:
            visible_ratio = visible_ratio.to(device=device)
        if bev_teacher_gate_mean is None:
            bev_teacher_gate_mean = torch.tensor(0.0, device=device)
        else:
            bev_teacher_gate_mean = bev_teacher_gate_mean.to(device=device)
        if bev_residual_norm is None:
            bev_residual_norm = torch.tensor(0.0, device=device)
        else:
            bev_residual_norm = bev_residual_norm.to(device=device)
        if bev_branch_active is None:
            bev_branch_active = torch.tensor(0.0, device=device)
        else:
            bev_branch_active = bev_branch_active.to(device=device)
        if teacher_bev_valid is None:
            teacher_bev_valid_ratio = torch.tensor(0.0, device=device)
        else:
            teacher_bev_valid_ratio = teacher_bev_valid.to(device=device, dtype=torch.float32).mean()
        if teacher_risk_valid is None:
            teacher_risk_valid_ratio = torch.tensor(0.0, device=device)
        else:
            teacher_risk_valid_ratio = teacher_risk_valid.to(device=device, dtype=torch.float32).mean()
        bev_obstacle_aux = torch.tensor(0.0, device=device)
        bev_teacher_distill = torch.tensor(0.0, device=device)
        bev_teacher_mask_ratio = torch.tensor(0.0, device=device)
        teacher_risk_distill = torch.tensor(0.0, device=device)
        teacher_risk_mask_ratio = torch.tensor(0.0, device=device)
        if teacher_risk_logits is not None and teacher_risk_heatmap is not None:
            teacher_risk_distill = self._teacher_risk_distill_loss(
                teacher_risk_logits,
                teacher_risk_heatmap,
                teacher_risk_valid,
                visible_mask,
            )
            logged_risk_mask_ratio = getattr(self, "_last_teacher_risk_mask_ratio", None)
            if logged_risk_mask_ratio is not None:
                teacher_risk_mask_ratio = logged_risk_mask_ratio.to(device=device)

        if obstacle_logits is not None:
            if occupancy_aux is not None:
                bev_obstacle_aux = self._bev_obstacle_aux_loss(obstacle_logits, occupancy_aux, visible_mask)
            if bool(getattr(self.cfg, "VAD_RISK_PLAN_ENABLED", True)):
                vad_risk_t = self._pseudo_bev_box_risk_loss(
                    pred_xy, obstacle_logits, gt_trajs_xy=gt_xy, occupancy=occupancy, return_per_t=True
                )
                risk_w = torch.linspace(1.0, 2.0, vad_risk_t.shape[1], device=device, dtype=vad_risk_t.dtype).view(1, -1)
                vad_risk_plan = ((vad_risk_t ** 2) * risk_w).mean()
            if teacher_bev_feature is not None:
                bev_teacher_distill = self._bev_teacher_distill_loss(teacher_bev_feature, teacher_bev_valid)
                logged_mask_ratio = getattr(self, "_last_bev_teacher_mask_ratio", None)
                if logged_mask_ratio is not None:
                    bev_teacher_mask_ratio = logged_mask_ratio.to(device=device)

        pred_last_x = pred_xy[:, -1, 0]
        cmd_right = torch.tensor([c == "RIGHT" for c in commands], device=device)
        cmd_left = torch.tensor([c == "LEFT" for c in commands], device=device)
        cmd_forward = torch.tensor([c == "FORWARD" for c in commands], device=device)
        margin_turn = float(getattr(self.cfg, "DIR_MARGIN_TURN", 1.8))
        margin_fwd = float(getattr(self.cfg, "DIR_MARGIN_FORWARD", 2.0))
        dir_loss = torch.tensor(0.0, device=device)
        if cmd_right.any():
            dir_loss = dir_loss + F.relu(margin_turn - pred_last_x[cmd_right]).mean()
        if cmd_left.any():
            dir_loss = dir_loss + F.relu(pred_last_x[cmd_left] + margin_turn).mean()
        if cmd_forward.any():
            dir_loss = dir_loss + F.relu(pred_last_x[cmd_forward].abs() - margin_fwd).mean()

        success = torch.zeros_like(pred_last_x)
        if cmd_right.any():
            success[cmd_right] = (pred_last_x[cmd_right] >= margin_turn).float()
        if cmd_left.any():
            success[cmd_left] = (pred_last_x[cmd_left] <= -margin_turn).float()
        if cmd_forward.any():
            success[cmd_forward] = (pred_last_x[cmd_forward].abs() <= margin_fwd).float()
        turn_mask = cmd_right | cmd_left
        acc_turn_valid = turn_mask.any()
        acc_turn = success[turn_mask].mean() if acc_turn_valid else torch.tensor(0.0, device=device)
        acc_turn_valid = torch.tensor(float(acc_turn_valid), device=device)
        acc_all = success.mean()

        coarse_l2 = torch.tensor(0.0, device=device)
        coarse = getattr(self.vlm, "last_coarse_xy", None)
        if coarse is not None:
            coarse_l2 = ((((coarse.to(device) - gt_xy) ** 2).sum(-1).sqrt()) * w).mean()
        endpoint_aux_l2 = torch.tensor(0.0, device=device)
        endpoint_xy = getattr(self.vlm, "last_endpoint_xy", None)
        if endpoint_xy is not None:
            endpoint_aux_l2 = ((endpoint_xy.to(device) - gt_xy[:, -1]) ** 2).sum(dim=-1).sqrt().mean()

        lam_l2 = float(getattr(self.cfg, "LOSS_L2_W", 8.0))
        lam_col = float(getattr(self.cfg, "LOSS_COL_W", 30.0))
        lam_box_col = float(getattr(self.cfg, "LOSS_BOX_COL_W", 20.0))
        lam_smo = float(getattr(self.cfg, "LOSS_SMO_W", 0.1))
        lam_vel = float(getattr(self.cfg, "LOSS_VEL_W", 0.6))
        lam_dir = float(getattr(self.cfg, "LOSS_DIR_W", 8.0))
        lam_coarse = float(getattr(self.cfg, "LOSS_COARSE_W", 0.0))
        lam_fde = float(getattr(self.cfg, "LOSS_FDE_W", 2.0))
        lam_endpoint = float(getattr(self.cfg, "LOSS_ENDPOINT_AUX_W", 0.0))
        lam_bev_obstacle = float(getattr(self.cfg, "LOSS_VAD_BEV_OBSTACLE_W", getattr(self.cfg, "LOSS_BEV_OBSTACLE_W", 0.3)))
        lam_vad_risk_plan = float(getattr(self.cfg, "LOSS_VAD_RISK_PLAN_W", 3.0))
        lam_bev_teacher = float(getattr(self.cfg, "LOSS_BEV_TEACHER_W", 0.3))
        lam_teacher_risk = float(getattr(self.cfg, "LOSS_TEACHER_RISK_W", 1.0))
        normal_loss = (
            lam_l2 * l2
            + lam_col * collision
            + lam_box_col * box_collision
            + lam_vad_risk_plan * vad_risk_plan
            + lam_smo * smooth
            + lam_vel * vel_l2
            + lam_dir * dir_loss
            + lam_coarse * coarse_l2
            + lam_fde * fde
            + lam_endpoint * endpoint_aux_l2
            + lam_bev_obstacle * bev_obstacle_aux
            + lam_bev_teacher * bev_teacher_distill
            + lam_teacher_risk * teacher_risk_distill
        )
        if bool(getattr(self.cfg, "BEV_TEACHER_ONLY", False)):  
            teacher_only_loss = lam_bev_teacher * bev_teacher_distill + lam_teacher_risk * teacher_risk_distill
            has_teacher_signal = (
                bool((teacher_bev_valid_ratio > 0).detach().item())
                or bool((teacher_risk_valid_ratio > 0).detach().item())
            )
            loss = teacher_only_loss if has_teacher_signal else normal_loss
            if not has_teacher_signal:
                print("BEV_TEACHER_ONLY requested but no valid teacher signal; fallback to normal planning loss.")
        else:
            loss = normal_loss
        if not torch.isfinite(loss):
            components = {
                "l2": l2,
                "fde": fde,
                "vel_l2": vel_l2,
                "smooth": smooth,
                "collision": collision,
                "box_collision": box_collision,
                "vad_risk_plan": vad_risk_plan,
                "dir_loss": dir_loss,
                "coarse_l2": coarse_l2,
                "endpoint_aux_l2": endpoint_aux_l2,
                "bev_obstacle_aux": bev_obstacle_aux,
                "bev_teacher_distill": bev_teacher_distill,
                "teacher_risk_distill": teacher_risk_distill,
            }
            bad = []
            for key, value in components.items():
                if torch.is_tensor(value) and not torch.isfinite(value.detach().float()).all():
                    bad.append(key)
            msg = f"[NaN debug] planning loss is non-finite; bad components={bad}"
            print(msg)
            if bool(getattr(self.cfg, "NAN_DEBUG_RAISE", True)):
                raise FloatingPointError(msg)

        loss_dict = {
            "l2": l2,
            "fde": fde,
            "vel_l2": vel_l2,
            "smooth": smooth,
            "collision": collision,
            "box_collision": box_collision,
            "vad_risk_plan": vad_risk_plan,
            "hard_collision": hard_collision,
            "coarse_l2": coarse_l2,
            "endpoint_aux_l2": endpoint_aux_l2,
            "bev_obstacle_aux": bev_obstacle_aux,
            "bev_teacher_distill": bev_teacher_distill,
            "bev_teacher_valid_ratio": teacher_bev_valid_ratio,
            "bev_teacher_mask_ratio": bev_teacher_mask_ratio,
            "teacher_risk_distill": teacher_risk_distill,
            "teacher_risk_valid_ratio": teacher_risk_valid_ratio,
            "teacher_risk_mask_ratio": teacher_risk_mask_ratio,
            "bev_branch_active": bev_branch_active,
            "bev_teacher_gate_mean": bev_teacher_gate_mean,
            "bev_residual_norm": bev_residual_norm,
            "bev_visible_ratio": visible_ratio,
            "dir_loss": dir_loss,
            "acc_turn": acc_turn,
            "acc_turn_valid": acc_turn_valid,
            "acc_all": acc_all,
            "col_with_lam": lam_col * collision,
            "box_col_with_lam": lam_box_col * box_collision,
            "vad_risk_plan_with_lam": lam_vad_risk_plan * vad_risk_plan,
            "bev_obstacle_with_lam": lam_bev_obstacle * bev_obstacle_aux,
            "bev_teacher_with_lam": lam_bev_teacher * bev_teacher_distill,
            "teacher_risk_with_lam": lam_teacher_risk * teacher_risk_distill,
            "fde_with_lam": lam_fde * fde,
            "endpoint_aux_with_lam": lam_endpoint * endpoint_aux_l2,
        }
        return loss, pred, final_traj, "FAST", loss_dict
