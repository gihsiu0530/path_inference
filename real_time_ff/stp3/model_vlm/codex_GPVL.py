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
        self.pseudo_bev_token_size = int(getattr(cfg, "PSEUDO_BEV_TOKEN_SIZE", 50))
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
        self.bev_drivable_head = nn.Conv2d(c, 1, kernel_size=1)
        self.bev_obstacle_head = nn.Conv2d(c, 1, kernel_size=1)
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
        self.last_drivable_logits = None
        self.last_obstacle_logits = None
        self.last_bev_visible_mask = None
        self.last_bev_visible_ratio = None
        self.last_student_scene_context = None

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
        return self.spatial_token_ln(tokens + self.spatial_pos_mlp(pos))

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

    def _pseudo_bev_from_feature(self, feat_seq: torch.Tensor, depth_seq: torch.Tensor,
                                 intrinsics: torch.Tensor, extrinsics: torch.Tensor):
        b, t_rf, c, h, w = feat_seq.shape
        bt = b * t_rf
        device = feat_seq.device
        dtype = feat_seq.dtype

        depth_bt = depth_seq.reshape(bt, *depth_seq.shape[2:]).to(device, non_blocking=True)
        depth_bt = self._metric_depth(depth_bt)
        depth_bt = F.interpolate(depth_bt, size=(h, w), mode="bilinear", align_corners=False)

        intrinsics = self._select_first_camera_calib(intrinsics).to(device=device, dtype=dtype).reshape(bt, 3, 3)
        extrinsics = self._select_first_camera_calib(extrinsics).to(device=device, dtype=dtype).reshape(bt, 4, 4)

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
        ones = torch.ones_like(z)
        pts_cam = torch.stack([x_cam, y_cam, z, ones], dim=-1)
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
        visible = (count > 0).to(feat_bt.dtype)
        bev = bev_flat.view(bt, self.pseudo_bev_channels, self.pseudo_bev_h, self.pseudo_bev_w)
        visible = visible.view(bt, 1, self.pseudo_bev_h, self.pseudo_bev_w)
        return bev, visible

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
            raise ValueError("codex_GPVL requires seg_id_224_seq from NuscenesData_change_GPVL.py.")
        if depth_seq is None:
            raise ValueError("codex_GPVL requires depth_224_seq from NuscenesData_change_GPVL.py.")

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

        cmd_ctx = self.command_embed(self._command_ids(commands, device))
        fused_mid = self.mid_fusion(torch.cat([rgb_mid, seg_rgb_mid, seg_id_mid, depth_mid], dim=1))
        fused = self.fusion(torch.cat([rgb_deep, seg_rgb_deep, seg_id_deep, depth_deep], dim=1))
        fused_mid = self._film(fused_mid, cmd_ctx.repeat_interleave(t_rf, dim=0))
        fused = self._film(fused, cmd_ctx.repeat_interleave(t_rf, dim=0))

        _, c, h_deep, w_deep = fused.shape
        fused_mid_seq = fused_mid.view(b, t_rf, c, *fused_mid.shape[-2:])
        fused_seq = fused.view(b, t_rf, c, h_deep, w_deep)
        image_tokens = torch.cat([
            self._tokens_from_feature(fused_mid_seq, scale_index=0),
            self._tokens_from_feature(fused_seq, scale_index=1),
        ], dim=1)

        bev_ctx = None
        if intrinsics is not None and extrinsics is not None and intrinsics.numel() > 0 and extrinsics.numel() > 0:
            bev_low, visible = self._pseudo_bev_from_feature(fused_mid_seq, depth_seq, intrinsics, extrinsics)
            bev_low = bev_low.view(b, t_rf, self.pseudo_bev_channels, self.pseudo_bev_h, self.pseudo_bev_w)
            visible = visible.view(b, t_rf, 1, self.pseudo_bev_h, self.pseudo_bev_w)
            current_bev = self.bev_encoder(bev_low[:, -1])
            bev_ctx = current_bev.flatten(2).mean(dim=-1)
            current_visible = visible[:, -1]
            self.last_bev_feature = current_bev
            self.last_drivable_logits = self.bev_drivable_head(current_bev)
            self.last_obstacle_logits = self.bev_obstacle_head(current_bev)
            self.last_bev_visible_mask = current_visible
            self.last_bev_visible_ratio = current_visible.detach().float().mean()
            spatial_tokens = torch.cat([image_tokens, self._tokens_from_bev(current_bev)], dim=1)
            frame_feat = self.spatial_pool(fused).view(b, t_rf, self.hidden_dim)
        else:
            self.last_bev_feature = None
            self.last_drivable_logits = None
            self.last_obstacle_logits = None
            self.last_bev_visible_mask = None
            self.last_bev_visible_ratio = None
            spatial_tokens = image_tokens
            frame_feat = self.spatial_pool(fused).view(b, t_rf, self.hidden_dim)

        _, h = self.temporal_gru(frame_feat)
        visual_ctx = h[-1]
        self.last_student_scene_context = visual_ctx if bev_ctx is None else visual_ctx + bev_ctx
        ego_ctx = self.ego_mlp(ego_seq.reshape(b, -1).to(device))
        ctx = self.context_mlp(torch.cat([visual_ctx, ego_ctx, cmd_ctx], dim=-1))

        query_ctx = self.query_context(torch.cat([visual_ctx, ego_ctx, cmd_ctx], dim=-1))
        queries = self.time_queries.unsqueeze(0).expand(b, -1, -1) + query_ctx.unsqueeze(1)
        queries = queries + ctx.unsqueeze(1)
        for layer in self.decoder_layers:
            queries = layer(queries, spatial_tokens)
        refined, _ = self.waypoint_gru(queries)
        queries = self.waypoint_refine_ln(queries + refined)

        abs_residual = torch.tanh(self.traj_abs_head(queries)).view(b, self.n_future, 2) * self.residual_scale
        delta_residual = torch.tanh(self.traj_delta_head(queries)).view(b, self.n_future, 2) * self.delta_scale
        residual = abs_residual + torch.cumsum(delta_residual, dim=1)
        coarse = self.make_coarse_baseline(ego_seq)
        self.last_coarse_xy = coarse.detach()
        endpoint_residual = torch.tanh(self.endpoint_head(ctx)) * self.residual_scale
        self.last_endpoint_xy = coarse[:, -1, :] + endpoint_residual
        xy = coarse + residual
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
        self.teacher_scene_dim = int(getattr(cfg, "GPVL_TEACHER_FEATURE_DIM", 256))
        self.teacher_scene_proj = nn.Sequential(
            nn.Linear(self.vlm.hidden_dim, self.teacher_scene_dim),
            nn.LayerNorm(self.teacher_scene_dim),
            nn.GELU(),
            nn.Linear(self.teacher_scene_dim, self.teacher_scene_dim),
        )
        self.last_teacher_scene_valid_ratio = None

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
        print("model : codex_GPVL_pseudo_bev")
        print(f"Total parameters: {total:,}  Trainable parameters: {trainable:,}")

    def forward(self, image, intrinsics, extrinsics, future_egomotion, *,
                rgb_224_seq, seg_224_seq, seg_id_224_seq=None, depth_224_seq=None,
        ego_history_egomotion=None):
        if seg_id_224_seq is None:
            raise ValueError("codex_GPVL requires batch['seg_id_224_seq']; use NuscenesData_change_GPVL.py.")
        if depth_224_seq is None:
            raise ValueError("codex_GPVL requires batch['depth_224_seq']; use NuscenesData_change_GPVL.py.")

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
        x_min = float(getattr(self.cfg, "BEV_CORRIDOR_X_MIN", 0.0))
        x_max = float(getattr(self.cfg, "BEV_CORRIDOR_X_MAX", 45.0))
        y_abs = float(getattr(self.cfg, "BEV_CORRIDOR_Y_ABS", 10.0))
        forward = torch.linspace(
            self.bev_x_min + 0.5 * self.pseudo_bev_grid_size,
            self.bev_x_max - 0.5 * self.pseudo_bev_grid_size,
            h,
            device=device,
            dtype=dtype,
        ).view(1, 1, h, 1)
        side = torch.linspace(
            self.bev_y_min + 0.5 * self.pseudo_bev_grid_size,
            self.bev_y_max - 0.5 * self.pseudo_bev_grid_size,
            w,
            device=device,
            dtype=dtype,
        ).view(1, 1, 1, w)
        corridor = (forward >= x_min) & (forward <= x_max) & (side.abs() <= y_abs)
        return corridor.to(dtype=dtype).expand(logits.shape[0], 1, h, w)

    def _bev_aux_loss(self, logits: torch.Tensor, target: torch.Tensor,
                      visible_mask: torch.Tensor = None, corridor_mask: torch.Tensor = None,
                      balanced: bool = True) -> torch.Tensor:
        target = self._as_bev_mask(target, logits)
        target = self._dilate_bev_mask(target, int(getattr(self.cfg, "BEV_AUX_TARGET_DILATE", 3)))

        valid_mask = torch.ones_like(target)
        if visible_mask is not None:
            visible_mask = self._as_bev_mask(visible_mask, logits)
            visible_mask = self._dilate_bev_mask(visible_mask, int(getattr(self.cfg, "BEV_AUX_VISIBLE_DILATE", 3)))
            valid_mask = valid_mask * visible_mask
        if corridor_mask is not None:
            valid_mask = valid_mask * self._as_bev_mask(corridor_mask, logits)

        if valid_mask.sum() <= 0:
            return logits.sum() * 0.0

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if not balanced:
            return (bce * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

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

    def _teacher_scene_loss(self, teacher_feature: torch.Tensor) -> torch.Tensor:
        scene_ctx = getattr(self.vlm, "last_student_scene_context", None)
        if scene_ctx is None or teacher_feature is None:
            self.last_teacher_scene_valid_ratio = None
            return self.dx.sum() * 0.0
        teacher = teacher_feature.to(device=scene_ctx.device, dtype=scene_ctx.dtype)
        if teacher.numel() == 0:
            self.last_teacher_scene_valid_ratio = torch.tensor(0.0, device=scene_ctx.device)
            return scene_ctx.sum() * 0.0
        if teacher.dim() > 2:
            teacher = teacher.reshape(teacher.shape[0], -1, teacher.shape[-1]).mean(dim=1)
        elif teacher.dim() == 1:
            teacher = teacher.unsqueeze(0)
        if teacher.shape[-1] != self.teacher_scene_dim:
            teacher = F.adaptive_avg_pool1d(teacher.unsqueeze(1), self.teacher_scene_dim).squeeze(1)
        valid = torch.isfinite(teacher).all(dim=1) & (teacher.abs().sum(dim=1) > 0)
        self.last_teacher_scene_valid_ratio = valid.float().mean().detach()
        if not valid.any():
            return scene_ctx.sum() * 0.0
        student = self.teacher_scene_proj(scene_ctx)
        student = F.normalize(student[valid], dim=-1)
        teacher = F.normalize(teacher[valid].detach(), dim=-1)
        return (1.0 - (student * teacher).sum(dim=-1)).mean()

    def planning(self, *, bev_rgbs, trajs, gt_trajs, commands, target_points, occupancy=None, drivable_mask=None,
                 occupancy_aux=None, drivable_aux=None, teacher_scene_feature=None):
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
        acc_turn = success[turn_mask].mean() if turn_mask.any() else torch.tensor(float("nan"), device=device)
        acc_all = success.mean()

        coarse_l2 = torch.tensor(0.0, device=device)
        coarse = getattr(self.vlm, "last_coarse_xy", None)
        if coarse is not None:
            coarse_l2 = ((((coarse.to(device) - gt_xy) ** 2).sum(-1).sqrt()) * w).mean()
        endpoint_aux_l2 = torch.tensor(0.0, device=device)
        endpoint_xy = getattr(self.vlm, "last_endpoint_xy", None)
        if endpoint_xy is not None:
            endpoint_aux_l2 = ((endpoint_xy.to(device) - gt_xy[:, -1]) ** 2).sum(dim=-1).sqrt().mean()

        bev_obstacle_aux = torch.tensor(0.0, device=device)
        bev_drivable_aux = torch.tensor(0.0, device=device)
        teacher_scene_aux = torch.tensor(0.0, device=device)
        teacher_scene_valid_ratio = torch.tensor(0.0, device=device)
        obstacle_logits = getattr(self.vlm, "last_obstacle_logits", None)
        drivable_logits = getattr(self.vlm, "last_drivable_logits", None)
        visible_mask = getattr(self.vlm, "last_bev_visible_mask", None)
        visible_ratio = getattr(self.vlm, "last_bev_visible_ratio", None)
        if visible_ratio is None:
            visible_ratio = torch.tensor(0.0, device=device)
        else:
            visible_ratio = visible_ratio.to(device=device)
        if obstacle_logits is not None and occupancy_aux is not None:
            corridor_mask = None
            if bool(getattr(self.cfg, "BEV_AUX_USE_FRONT_CORRIDOR", True)):
                corridor_mask = self._front_corridor_mask(obstacle_logits)
            bev_obstacle_aux = self._bev_aux_loss(
                obstacle_logits, occupancy_aux, visible_mask=visible_mask,
                corridor_mask=corridor_mask, balanced=True
            )
        if drivable_logits is not None and drivable_aux is not None:
            bev_drivable_aux = self._bev_aux_loss(
                drivable_logits, drivable_aux, visible_mask=visible_mask,
                corridor_mask=None, balanced=False
            )
        if teacher_scene_feature is not None:
            teacher_scene_aux = self._teacher_scene_loss(teacher_scene_feature)
            valid_ratio = getattr(self, "last_teacher_scene_valid_ratio", None)
            if valid_ratio is not None:
                teacher_scene_valid_ratio = valid_ratio.to(device=device)

        lam_l2 = float(getattr(self.cfg, "LOSS_L2_W", 8.0))
        lam_col = float(getattr(self.cfg, "LOSS_COL_W", 30.0))
        lam_box_col = float(getattr(self.cfg, "LOSS_BOX_COL_W", 20.0))
        lam_smo = float(getattr(self.cfg, "LOSS_SMO_W", 0.1))
        lam_vel = float(getattr(self.cfg, "LOSS_VEL_W", 0.6))
        lam_dir = float(getattr(self.cfg, "LOSS_DIR_W", 8.0))
        lam_coarse = float(getattr(self.cfg, "LOSS_COARSE_W", 0.0))
        lam_fde = float(getattr(self.cfg, "LOSS_FDE_W", 2.0))
        lam_endpoint = float(getattr(self.cfg, "LOSS_ENDPOINT_AUX_W", 2.0))
        lam_bev_obstacle = float(getattr(self.cfg, "LOSS_BEV_OBSTACLE_W", 0.3))
        lam_bev_drivable = float(getattr(self.cfg, "LOSS_BEV_DRIVABLE_W", 0.0))
        lam_teacher_scene = float(getattr(self.cfg, "LOSS_TEACHER_SCENE_W", 0.0))
        if lam_bev_drivable <= 0.0:
            bev_drivable_aux = bev_drivable_aux * 0.0
        loss = (
            lam_l2 * l2
            + lam_col * collision
            + lam_box_col * box_collision
            + lam_smo * smooth
            + lam_vel * vel_l2
            + lam_dir * dir_loss
            + lam_coarse * coarse_l2
            + lam_fde * fde
            + lam_endpoint * endpoint_aux_l2
            + lam_bev_obstacle * bev_obstacle_aux
            + lam_bev_drivable * bev_drivable_aux
            + lam_teacher_scene * teacher_scene_aux
        )

        loss_dict = {
            "l2": l2,
            "fde": fde,
            "vel_l2": vel_l2,
            "smooth": smooth,
            "collision": collision,
            "box_collision": box_collision,
            "hard_collision": hard_collision,
            "coarse_l2": coarse_l2,
            "endpoint_aux_l2": endpoint_aux_l2,
            "bev_obstacle_aux": bev_obstacle_aux,
            "bev_drivable_aux": bev_drivable_aux,
            "teacher_scene_aux": teacher_scene_aux,
            "teacher_scene_valid_ratio": teacher_scene_valid_ratio,
            "bev_visible_ratio": visible_ratio,
            "dir_loss": dir_loss,
            "acc_turn": acc_turn,
            "acc_all": acc_all,
            "col_with_lam": lam_col * collision,
            "box_col_with_lam": lam_box_col * box_collision,
            "fde_with_lam": lam_fde * fde,
            "endpoint_aux_with_lam": lam_endpoint * endpoint_aux_l2,
            "bev_obstacle_with_lam": lam_bev_obstacle * bev_obstacle_aux,
            "bev_drivable_with_lam": lam_bev_drivable * bev_drivable_aux,
            "teacher_scene_with_lam": lam_teacher_scene * teacher_scene_aux,
        }
        return loss, pred, final_traj, "FAST", loss_dict
