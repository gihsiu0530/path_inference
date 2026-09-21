"""Coarse-conditioned Frenet residual planner.

This module intentionally lives beside, and does not modify, codex_pure_ASAP.py.
The public VLM_STP3_Gen interface remains compatible with the existing trainer.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from stp3.model_vlm.codex_pure_ASAP import (
    FastTrajectoryPlanner as BaseFastTrajectoryPlanner,
    VLM_STP3_Gen as BaseVLM_STP3_Gen,
)


class MetricBEVProjectorFF(nn.Module):
    """Project the current front view to an ego-frame metric BEV feature map.

    Planner coordinates are (lateral-right, forward).  nuScenes lidar/ego
    coordinates are (forward, lateral-left), hence lateral-right = -ego_y.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(getattr(cfg, "SEG_NUM_CLASSES", 4))
        self.stride = max(1, int(getattr(cfg, "FF_BEV_PROJECT_STRIDE", 2)))
        self.depth_mode = str(getattr(cfg, "FF_DEPTH_MODE", "relative_inverse"))
        self.depth_min = float(getattr(cfg, "FF_DEPTH_MIN_M", 1.0))
        self.depth_max = float(getattr(cfg, "FF_DEPTH_MAX_M", 60.0))
        self.resolution = float(getattr(cfg, "FF_BEV_RESOLUTION_M", 0.5))
        self.forward_min = float(getattr(cfg, "FF_BEV_FORWARD_MIN_M", -10.0))
        self.forward_max = float(getattr(cfg, "FF_BEV_FORWARD_MAX_M", 50.0))
        self.side_min = float(getattr(cfg, "FF_BEV_SIDE_MIN_M", -25.0))
        self.side_max = float(getattr(cfg, "FF_BEV_SIDE_MAX_M", 25.0))
        self.height = int(math.ceil((self.forward_max - self.forward_min) / self.resolution))
        self.width = int(math.ceil((self.side_max - self.side_min) / self.resolution))
        self.out_channels = int(getattr(cfg, "FF_BEV_CHANNELS", 64))
        raw_channels = self.num_classes + 3 + 2  # semantic one-hot, RGB, height, density/depth
        self.encoder = nn.Sequential(
            nn.Conv2d(raw_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, self.out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU(),
        )
        self.last_valid = False
        self.last_observed_road = None
        self.last_drivable_probability = None
        self.last_observed_mask = None

    @staticmethod
    def _last_front_matrix(value: Optional[torch.Tensor], matrix_size: int) -> Optional[torch.Tensor]:
        if value is None or not torch.is_tensor(value) or value.numel() == 0:
            return None
        if value.shape[-2:] != (matrix_size, matrix_size):
            return None
        if value.dim() == 5:       # B,T,N,M,M
            return value[:, -1, 0]
        if value.dim() == 4:       # B,T,M,M or B,N,M,M
            return value[:, -1]
        if value.dim() == 3:       # B,M,M
            return value
        return None

    @staticmethod
    def _last_rgb(rgb_seq: torch.Tensor) -> torch.Tensor:
        rgb = rgb_seq[:, -1]
        if rgb.dim() == 4 and rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2).contiguous()
        rgb = rgb.float()
        if rgb.detach().amax() > 2.0:
            rgb = rgb / 255.0
        return rgb

    @staticmethod
    def _last_seg(seg_seq: torch.Tensor) -> torch.Tensor:
        seg = seg_seq[:, -1]
        if seg.dim() == 4 and seg.shape[1] == 1:
            seg = seg[:, 0]
        return seg.long()

    @staticmethod
    def _last_depth(depth_seq: torch.Tensor) -> torch.Tensor:
        depth = depth_seq[:, -1]
        if depth.dim() == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        return torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)

    def _depth_meters(self, raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = raw > 0
        if self.depth_mode == "metric":
            meters = raw
        elif self.depth_mode == "relative_inverse":
            # Compatibility path for the current Depth-Anything-like files.
            # Larger values are treated as closer.  This is pseudo-metric and
            # should be replaced by FF_DEPTH_MODE=metric for real deployment.
            meters = torch.full_like(raw, self.depth_max)
            for batch_index in range(raw.shape[0]):
                values = raw[batch_index][valid[batch_index]]
                if values.numel() == 0:
                    continue
                lo = torch.quantile(values.float(), 0.02)
                hi = torch.quantile(values.float(), 0.98)
                closeness = ((raw[batch_index] - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
                meters[batch_index] = self.depth_max - closeness * (self.depth_max - self.depth_min)
        else:
            raise ValueError(f"Unsupported FF_DEPTH_MODE={self.depth_mode!r}; use metric or relative_inverse")
        valid = valid & (meters >= self.depth_min) & (meters <= self.depth_max)
        return meters.clamp(self.depth_min, self.depth_max), valid

    def metric_to_grid(self, points_xy: torch.Tensor) -> torch.Tensor:
        side = points_xy[..., 0]
        forward = points_xy[..., 1]
        grid_x = (side - self.side_min) / max(self.side_max - self.side_min, 1e-6) * 2.0 - 1.0
        grid_y = (forward - self.forward_min) / max(self.forward_max - self.forward_min, 1e-6) * 2.0 - 1.0
        return torch.stack([grid_x, grid_y], dim=-1)

    def forward(
        self,
        rgb_seq: torch.Tensor,
        seg_seq: torch.Tensor,
        depth_seq: torch.Tensor,
        intrinsics: Optional[torch.Tensor],
        extrinsics: Optional[torch.Tensor],
    ) -> torch.Tensor:
        rgb = self._last_rgb(rgb_seq)
        seg = self._last_seg(seg_seq)
        raw_depth = self._last_depth(depth_seq)
        k = self._last_front_matrix(intrinsics, 3)
        sensor_to_ego = self._last_front_matrix(extrinsics, 4)
        batch_size = rgb.shape[0]
        device = rgb.device
        dtype = rgb.dtype

        if k is None or sensor_to_ego is None:
            self.last_valid = False
            raw_channels = self.num_classes + 5
            empty = torch.zeros(batch_size, raw_channels, self.height, self.width, device=device, dtype=dtype)
            self.last_observed_road = torch.zeros(batch_size, self.height, self.width, device=device, dtype=dtype)
            self.last_drivable_probability = self.last_observed_road[:, None]
            self.last_observed_mask = self.last_observed_road[:, None]
            return self.encoder(empty)

        if seg.shape[-2:] != rgb.shape[-2:]:
            seg = F.interpolate(seg[:, None].float(), size=rgb.shape[-2:], mode="nearest")[:, 0].long()
        if raw_depth.shape[-2:] != rgb.shape[-2:]:
            raw_depth = F.interpolate(raw_depth[:, None], size=rgb.shape[-2:], mode="bilinear", align_corners=False)[:, 0]

        meters, valid_depth = self._depth_meters(raw_depth)
        h, w = meters.shape[-2:]
        rows = torch.arange(0, h, self.stride, device=device)
        cols = torch.arange(0, w, self.stride, device=device)
        vv, uu = torch.meshgrid(rows, cols, indexing="ij")
        pixels = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1).reshape(-1, 3).to(dtype=dtype)

        sampled_depth = meters[:, :: self.stride, :: self.stride].reshape(batch_size, -1)
        sampled_valid = valid_depth[:, :: self.stride, :: self.stride].reshape(batch_size, -1)
        sampled_seg = seg[:, :: self.stride, :: self.stride].reshape(batch_size, -1)
        sampled_rgb = rgb[:, :, :: self.stride, :: self.stride].permute(0, 2, 3, 1).reshape(batch_size, -1, 3)

        k_inv = torch.linalg.inv(k.to(device=device, dtype=dtype))
        rays = torch.einsum("bij,nj->bni", k_inv, pixels)
        camera_points = rays * sampled_depth.unsqueeze(-1)
        camera_h = torch.cat([camera_points, torch.ones_like(camera_points[..., :1])], dim=-1)
        ego_points = torch.einsum(
            "bij,bnj->bni", sensor_to_ego.to(device=device, dtype=dtype), camera_h
        )[..., :3]
        forward = ego_points[..., 0]
        side_right = -ego_points[..., 1]
        height = ego_points[..., 2]
        in_grid = (
            sampled_valid
            & (forward >= self.forward_min)
            & (forward < self.forward_max)
            & (side_right >= self.side_min)
            & (side_right < self.side_max)
        )

        one_hot = F.one_hot(
            sampled_seg.clamp(0, self.num_classes - 1), num_classes=self.num_classes
        ).to(dtype=dtype)
        height_feature = (height / 3.0).clamp(-2.0, 2.0).unsqueeze(-1)
        depth_feature = (sampled_depth / max(self.depth_max, 1e-6)).unsqueeze(-1)
        source = torch.cat([one_hot, sampled_rgb, height_feature, depth_feature], dim=-1)
        raw_channels = source.shape[-1]
        bev = torch.zeros(batch_size, raw_channels, self.height * self.width, device=device, dtype=dtype)
        counts = torch.zeros(batch_size, 1, self.height * self.width, device=device, dtype=dtype)

        for batch_index in range(batch_size):
            keep = in_grid[batch_index]
            if not keep.any():
                continue
            ix = ((side_right[batch_index, keep] - self.side_min) / self.resolution).long().clamp(0, self.width - 1)
            iy = ((forward[batch_index, keep] - self.forward_min) / self.resolution).long().clamp(0, self.height - 1)
            linear_index = iy * self.width + ix
            expanded_index = linear_index.unsqueeze(0).expand(raw_channels, -1)
            bev[batch_index].scatter_add_(1, expanded_index, source[batch_index, keep].transpose(0, 1))
            counts[batch_index].scatter_add_(1, linear_index.unsqueeze(0), torch.ones(1, linear_index.numel(), device=device, dtype=dtype))

        bev = bev / counts.clamp_min(1.0)
        bev = bev.view(batch_size, raw_channels, self.height, self.width)
        observed = (counts.view(batch_size, 1, self.height, self.width) > 0).to(dtype)
        road = bev[:, 0:1].clamp(0.0, 1.0)
        splat_radius = float(getattr(self.cfg, "FF_ROAD_SPLAT_RADIUS_M", 0.75))
        splat_cells = max(0, int(math.ceil(splat_radius / max(self.resolution, 1e-6))))
        if splat_cells > 0:
            kernel = splat_cells * 2 + 1
            road = F.max_pool2d(road, kernel_size=kernel, stride=1, padding=splat_cells)
            observed = F.max_pool2d(observed, kernel_size=kernel, stride=1, padding=splat_cells)
        road = F.avg_pool2d(road, kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)
        self.last_observed_road = road[:, 0].detach()
        self.last_drivable_probability = road
        self.last_observed_mask = observed
        self.last_valid = bool(in_grid.any().item())
        return self.encoder(bev)


class FastTrajectoryPlannerFF(BaseFastTrajectoryPlanner):
    """Residual planner conditioned point-wise on coarse path and metric BEV."""

    def __init__(self, cfg, n_future: int, input_size: int = 224):
        super().__init__(cfg, n_future, input_size)
        channels = self.hidden_dim
        self.metric_bev = MetricBEVProjectorFF(cfg)
        self.coarse_encoder = nn.Sequential(
            nn.Linear(10, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.route_encoder = nn.Sequential(
            nn.Linear(4, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        route_heads = int(getattr(cfg, "FF_ROUTE_HEADS", 8))
        self.route_attention = nn.MultiheadAttention(channels, route_heads, batch_first=True)
        patch_points = 5
        self.bev_patch_points = patch_points
        self.bev_patch_radius = float(getattr(cfg, "FF_BEV_PATCH_RADIUS_M", 1.0))
        self.bev_waypoint_encoder = nn.Sequential(
            nn.Linear(self.metric_bev.out_channels * patch_points, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.ff_query_norm = nn.LayerNorm(channels)
        self.lateral_head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 1),
        )
        nn.init.zeros_(self.lateral_head[-1].weight)
        nn.init.zeros_(self.lateral_head[-1].bias)
        alpha_init = min(max(float(getattr(cfg, "FF_ALPHA_INIT", 0.10)), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.constant_(self.confidence_head[-1].bias, math.log(alpha_init / (1.0 - alpha_init)))
        self.max_lateral = float(getattr(cfg, "FF_MAX_LATERAL_M", 2.0))
        self._planning_context = {}
        self.last_delta_n_raw = None
        self.last_alpha_logits = None
        self.last_alpha = None
        self.last_coarse_tangent = None
        self.last_coarse_normal = None
        self.last_metric_bev = None

        # The inherited unrestricted XY heads are retained only for checkpoint
        # compatibility and are deliberately frozen in the FF architecture.
        for module in (self.traj_abs_head, self.traj_delta_head, self.endpoint_head, self.residual_gate_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def set_planning_context(self, **kwargs):
        self._planning_context = kwargs

    @staticmethod
    def _coarse_geometry(coarse: torch.Tensor, sample_dt: float):
        delta = torch.cat([coarse[:, :1], coarse[:, 1:] - coarse[:, :-1]], dim=1)
        if coarse.shape[1] > 1:
            first_bad = delta[:, 0].norm(dim=-1, keepdim=True) < 1e-4
            delta[:, 0] = torch.where(first_bad, delta[:, 1], delta[:, 0])
        tangent = F.normalize(delta, dim=-1, eps=1e-6)
        normal_right = torch.stack([tangent[..., 1], -tangent[..., 0]], dim=-1)
        previous_tangent = torch.cat([tangent[:, :1], tangent[:, :-1]], dim=1)
        cross = previous_tangent[..., 0] * tangent[..., 1] - previous_tangent[..., 1] * tangent[..., 0]
        curvature = cross / delta.norm(dim=-1).clamp_min(0.25)
        speed = delta.norm(dim=-1) / max(sample_dt, 1e-6)
        return delta, tangent, normal_right, curvature, speed

    def _route_tokens(self, route: Optional[torch.Tensor], coarse: torch.Tensor) -> torch.Tensor:
        batch_size = coarse.shape[0]
        if route is None or not torch.is_tensor(route) or route.numel() == 0:
            route = coarse
        else:
            route = route.to(device=coarse.device, dtype=coarse.dtype)
            if route.dim() == 2 and route.shape == (batch_size, 2):
                fraction = torch.linspace(0.0, 1.0, self.n_future + 1, device=coarse.device, dtype=coarse.dtype)
                route = route[:, None, :] * fraction.view(1, -1, 1)
                zero_endpoint = route[:, -1].norm(dim=-1) < 1e-4
                route[zero_endpoint] = torch.cat([torch.zeros_like(coarse[zero_endpoint, :1]), coarse[zero_endpoint]], dim=1)
            elif route.dim() != 3 or route.shape[-1] < 2:
                route = coarse
            else:
                route = route[..., :2]
        route_delta = torch.cat([route[:, :1], route[:, 1:] - route[:, :-1]], dim=1)
        feature = torch.cat([route / 30.0, route_delta / 10.0], dim=-1)
        return self.route_encoder(feature)

    def _sample_bev(self, bev: torch.Tensor, coarse: torch.Tensor, tangent: torch.Tensor, normal: torch.Tensor):
        radius = self.bev_patch_radius
        center = coarse.unsqueeze(2)
        sample_points = torch.cat(
            [
                center,
                center + radius * tangent.unsqueeze(2),
                center - radius * tangent.unsqueeze(2),
                center + radius * normal.unsqueeze(2),
                center - radius * normal.unsqueeze(2),
            ],
            dim=2,
        )
        grid = self.metric_bev.metric_to_grid(sample_points)
        sampled = F.grid_sample(bev, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled.permute(0, 2, 3, 1).reshape(coarse.shape[0], coarse.shape[1], -1)
        return self.bev_waypoint_encoder(sampled)

    def forward(
        self,
        rgb_seq: torch.Tensor,
        seg_rgb_seq: torch.Tensor,
        seg_id_seq: torch.Tensor,
        depth_seq: torch.Tensor,
        ego_seq: torch.Tensor,
        commands,
        coarse_xy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if seg_id_seq is None or depth_seq is None:
            raise ValueError("FF planner requires seg_id_224_seq and depth_224_seq")
        device = ego_seq.device
        batch_size, time_steps = rgb_seq.shape[:2]
        rgb_seq = rgb_seq.to(device, non_blocking=True)
        seg_rgb_seq = seg_rgb_seq.to(device, non_blocking=True)
        seg_id_seq = seg_id_seq.to(device, non_blocking=True)
        depth_seq = depth_seq.to(device, non_blocking=True)

        rgb_bt = rgb_seq.reshape(batch_size * time_steps, *rgb_seq.shape[2:])
        seg_rgb_bt = seg_rgb_seq.reshape(batch_size * time_steps, *seg_rgb_seq.shape[2:])
        seg_id_bt = seg_id_seq.reshape(batch_size * time_steps, *seg_id_seq.shape[2:])
        depth_bt = depth_seq.reshape(batch_size * time_steps, *depth_seq.shape[2:])
        rgb_mid, rgb_deep = self.rgb_stem(self._rgb_like(rgb_bt))
        seg_rgb_mid, seg_rgb_deep = self.seg_rgb_stem(self._rgb_like(seg_rgb_bt))
        seg_id_mid, seg_id_deep = self.seg_id_stem(self._seg_id(seg_id_bt))
        depth_mid, depth_deep = self.depth_stem(self._depth(depth_bt))

        command_context = self.command_embed(self._command_ids(commands, device))
        fused_mid = self.mid_fusion(torch.cat([rgb_mid, seg_rgb_mid, seg_id_mid, depth_mid], dim=1))
        fused = self.fusion(torch.cat([rgb_deep, seg_rgb_deep, seg_id_deep, depth_deep], dim=1))
        fused_mid = self._film(fused_mid, command_context.repeat_interleave(time_steps, dim=0))
        fused = self._film(fused, command_context.repeat_interleave(time_steps, dim=0))
        _, channels, deep_h, deep_w = fused.shape
        fused_mid_seq = fused_mid.view(batch_size, time_steps, channels, *fused_mid.shape[-2:])
        fused_seq = fused.view(batch_size, time_steps, channels, deep_h, deep_w)
        spatial_tokens = torch.cat(
            [self._tokens_from_feature(fused_mid_seq, 0), self._tokens_from_feature(fused_seq, 1)], dim=1
        )
        frame_feature = self.spatial_pool(fused).view(batch_size, time_steps, self.hidden_dim)
        _, temporal_hidden = self.temporal_gru(frame_feature)
        visual_context = temporal_hidden[-1]
        ego_context = self.ego_mlp(ego_seq.reshape(batch_size, -1).to(device))
        context = self.context_mlp(torch.cat([visual_context, ego_context, command_context], dim=-1))

        if coarse_xy is None:
            coarse = self.make_coarse_baseline(ego_seq)
        else:
            coarse = coarse_xy.to(device=device, dtype=context.dtype)
        if coarse.shape != (batch_size, self.n_future, 2):
            raise ValueError(f"coarse_xy should be {(batch_size, self.n_future, 2)}, got {tuple(coarse.shape)}")
        delta, tangent, normal, curvature, speed = self._coarse_geometry(coarse, self.SAMPLE_DT)
        coarse_feature = torch.cat(
            [coarse / 30.0, delta / 10.0, tangent, normal, curvature.unsqueeze(-1), (speed / 15.0).unsqueeze(-1)],
            dim=-1,
        )
        coarse_token = self.coarse_encoder(coarse_feature)

        metric_bev = self.metric_bev(
            rgb_seq,
            seg_id_seq,
            depth_seq,
            self._planning_context.get("intrinsics"),
            self._planning_context.get("extrinsics"),
        )
        bev_token = self._sample_bev(metric_bev, coarse, tangent, normal)
        route_tokens = self._route_tokens(self._planning_context.get("route_polyline"), coarse)

        query_context = self.query_context(torch.cat([visual_context, ego_context, command_context], dim=-1))
        queries = self.time_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + query_context.unsqueeze(1) + context.unsqueeze(1) + coarse_token + bev_token
        queries = queries + self.route_attention(queries, route_tokens, route_tokens, need_weights=False)[0]
        queries = self.ff_query_norm(queries)
        for layer in self.decoder_layers:
            queries = layer(queries, spatial_tokens)
        refined, _ = self.waypoint_gru(queries)
        queries = self.waypoint_refine_ln(queries + refined)

        delta_n_raw = torch.tanh(self.lateral_head(queries)).squeeze(-1) * self.max_lateral
        alpha_logits = self.confidence_head(queries).squeeze(-1)
        alpha = torch.sigmoid(alpha_logits)
        residual_raw = delta_n_raw.unsqueeze(-1) * normal
        residual = alpha.unsqueeze(-1) * residual_raw
        final_xy = coarse + residual

        self.last_coarse_xy = coarse.detach()
        self.last_delta_n_raw = delta_n_raw
        self.last_alpha_logits = alpha_logits
        self.last_alpha = alpha
        self.last_coarse_tangent = tangent.detach()
        self.last_coarse_normal = normal.detach()
        self.last_residual_gate_raw = alpha.unsqueeze(-1).expand(-1, -1, 2)
        self.last_residual_gate = self.last_residual_gate_raw.detach()
        self.last_residual_xy_raw = residual_raw
        self.last_residual_xy_gated = residual
        self.last_endpoint_xy = final_xy[:, -1]
        self.last_metric_bev = metric_bev.detach()
        z = torch.zeros(batch_size, self.n_future, 1, device=device, dtype=final_xy.dtype)
        return torch.cat([final_xy, z], dim=-1)


class VLM_STP3_Gen(BaseVLM_STP3_Gen):
    """Existing ASAP wrapper with FF residual, route, BEV and safety losses."""

    def _load_admlp_baseline(self):
        # A full FF Lightning checkpoint already contains
        # model.admlp_baseline.*. Realtime deployment may therefore defer the
        # constructor's separate baseline-file load and immediately strict-load
        # the complete checkpoint, making deployment portable across machines.
        if bool(getattr(self.cfg, "FF_SKIP_ADMLP_BASELINE_INIT", False)):
            print("[FF] deferring AD-MLP initialization to the full FF checkpoint")
            return
        return super()._load_admlp_baseline()

    def __init__(self, cfg):
        super().__init__(cfg)
        self.vlm = FastTrajectoryPlannerFF(cfg, self.n_future, self.input_size)
        self._last_intrinsics_ff = None
        self._last_extrinsics_ff = None
        print("model override: codex_pure_fast_ASAP_ff")
        if str(getattr(cfg, "FF_DEPTH_MODE", "relative_inverse")) != "metric":
            print("[FF WARNING] FF_DEPTH_MODE is not metric; the no-HD-map camera BEV is pseudo-metric.")

    def forward(self, image, intrinsics, extrinsics, future_egomotion, **kwargs):
        self._last_intrinsics_ff = intrinsics if torch.is_tensor(intrinsics) and intrinsics.numel() > 0 else None
        self._last_extrinsics_ff = extrinsics if torch.is_tensor(extrinsics) and extrinsics.numel() > 0 else None
        return super().forward(image, intrinsics, extrinsics, future_egomotion, **kwargs)

    def _admlp_coarse(self, device, dtype, commands):
        coarse = super()._admlp_coarse(device, dtype, commands)
        if not self.training or not bool(getattr(self.cfg, "FF_COARSE_AUG_ENABLED", True)):
            return coarse
        probability = float(getattr(self.cfg, "FF_COARSE_AUG_PROB", 0.75))
        active = (torch.rand(coarse.shape[0], 1, 1, device=device) < probability).to(coarse.dtype)
        lateral_max = float(getattr(self.cfg, "FF_COARSE_AUG_LATERAL_M", 1.0))
        yaw_max = math.radians(float(getattr(self.cfg, "FF_COARSE_AUG_YAW_DEG", 5.0)))
        scale_max = float(getattr(self.cfg, "FF_COARSE_AUG_SPEED_RATIO", 0.15))
        point_noise = float(getattr(self.cfg, "FF_COARSE_AUG_POINT_M", 0.20))
        lateral = (torch.rand(coarse.shape[0], 1, device=device, dtype=dtype) * 2.0 - 1.0) * lateral_max
        yaw = (torch.rand(coarse.shape[0], 1, device=device, dtype=dtype) * 2.0 - 1.0) * yaw_max
        scale = 1.0 + (torch.rand(coarse.shape[0], 1, device=device, dtype=dtype) * 2.0 - 1.0) * scale_max
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        x, y = coarse[..., 0], coarse[..., 1]
        rotated = torch.stack([cos_yaw * x - sin_yaw * y, sin_yaw * x + cos_yaw * y], dim=-1)
        rotated = rotated * scale.unsqueeze(-1)
        _, tangent, normal, _, _ = FastTrajectoryPlannerFF._coarse_geometry(rotated, self.vlm.SAMPLE_DT)
        smooth_noise = torch.randn_like(x).cumsum(dim=1)
        smooth_noise = smooth_noise / torch.arange(1, x.shape[1] + 1, device=device, dtype=dtype).sqrt().view(1, -1)
        augmented = rotated + (lateral + point_noise * smooth_noise).unsqueeze(-1) * normal
        return coarse + active * (augmented - coarse)

    def _footprint_offroad_per_t(self, trajectory_xy: torch.Tensor, drivable_mask: Optional[torch.Tensor]):
        if drivable_mask is None:
            return trajectory_xy.sum(dim=-1) * 0.0
        batch_size, horizon, _ = trajectory_xy.shape
        drivable = drivable_mask.to(device=trajectory_xy.device, dtype=trajectory_xy.dtype)
        if drivable.dim() == 3:
            drivable = drivable[:, None].expand(-1, horizon, -1, -1)
        if drivable.shape[1] == 1:
            drivable = drivable.expand(-1, horizon, -1, -1)
        elif drivable.shape[1] != horizon:
            drivable = drivable[:, :1].expand(-1, horizon, -1, -1)
        offroad = 1.0 - drivable.clamp(0.0, 1.0)
        margin = float(getattr(self.cfg, "FF_SAFETY_MARGIN_M", 0.35))
        cells = int(math.ceil(margin / max(float(self.dx.min().item()), 1e-6)))
        if cells > 0:
            kernel = cells * 2 + 1
            offroad = F.max_pool2d(
                offroad.reshape(batch_size * horizon, 1, *offroad.shape[-2:]), kernel, 1, cells
            ).reshape_as(offroad)
        offroad = F.avg_pool2d(
            offroad.reshape(batch_size * horizon, 1, *offroad.shape[-2:]), 3, 1, 1
        ).reshape_as(offroad)

        delta = torch.cat([trajectory_xy[:, :1], trajectory_xy[:, 1:] - trajectory_xy[:, :-1]], dim=1)
        if horizon > 1:
            bad = delta[:, 0].norm(dim=-1, keepdim=True) < 1e-4
            delta[:, 0] = torch.where(bad, delta[:, 1], delta[:, 0])
        tangent = F.normalize(delta, dim=-1, eps=1e-6)
        normal = torch.stack([tangent[..., 1], -tangent[..., 0]], dim=-1)
        half_width = float(getattr(self.cfg.EGO, "WIDTH", 1.85)) / 2.0
        half_length = float(getattr(self.cfg.EGO, "HEIGHT", 4.084)) / 2.0
        lateral_offsets = torch.linspace(-half_width, half_width, 3, device=trajectory_xy.device, dtype=trajectory_xy.dtype)
        longitudinal_offsets = torch.linspace(-half_length + 0.5, half_length + 0.5, 5, device=trajectory_xy.device, dtype=trajectory_xy.dtype)
        long_grid, lat_grid = torch.meshgrid(longitudinal_offsets, lateral_offsets, indexing="ij")
        points = (
            trajectory_xy.unsqueeze(2)
            + lat_grid.reshape(1, 1, -1, 1) * normal.unsqueeze(2)
            + long_grid.reshape(1, 1, -1, 1) * tangent.unsqueeze(2)
        )
        # HD-map grid uses (forward, left); planner uses (right, forward).
        forward = points[..., 1]
        side_left = -points[..., 0]
        height, width = offroad.shape[-2:]
        grid_x = ((side_left - self.bx[1]) / self.dx[1]) / max(width - 1, 1) * 2.0 - 1.0
        grid_y = ((forward - self.bx[0]) / self.dx[0]) / max(height - 1, 1) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(batch_size * horizon, -1, 1, 2)
        sampled = F.grid_sample(
            offroad.reshape(batch_size * horizon, 1, height, width),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(batch_size, horizon, -1)
        outside = (grid_x.abs() > 1.0) | (grid_y.abs() > 1.0)
        sampled = torch.maximum(sampled, outside.to(sampled.dtype))
        return sampled.amax(dim=-1)

    def _footprint_points(self, trajectory_xy: torch.Tensor):
        horizon = trajectory_xy.shape[1]
        delta = torch.cat([trajectory_xy[:, :1], trajectory_xy[:, 1:] - trajectory_xy[:, :-1]], dim=1)
        if horizon > 1:
            bad = delta[:, 0].norm(dim=-1, keepdim=True) < 1e-4
            first = torch.where(bad, delta[:, 1], delta[:, 0])
            delta = torch.cat([first.unsqueeze(1), delta[:, 1:]], dim=1)
        tangent = F.normalize(delta, dim=-1, eps=1e-6)
        normal = torch.stack([tangent[..., 1], -tangent[..., 0]], dim=-1)
        half_width = float(getattr(self.cfg.EGO, "WIDTH", 1.85)) / 2.0
        half_length = float(getattr(self.cfg.EGO, "HEIGHT", 4.084)) / 2.0
        lateral_offsets = torch.linspace(-half_width, half_width, 3, device=trajectory_xy.device, dtype=trajectory_xy.dtype)
        longitudinal_offsets = torch.linspace(-half_length + 0.5, half_length + 0.5, 5, device=trajectory_xy.device, dtype=trajectory_xy.dtype)
        long_grid, lat_grid = torch.meshgrid(longitudinal_offsets, lateral_offsets, indexing="ij")
        return (
            trajectory_xy.unsqueeze(2)
            + lat_grid.reshape(1, 1, -1, 1) * normal.unsqueeze(2)
            + long_grid.reshape(1, 1, -1, 1) * tangent.unsqueeze(2)
        )

    def _image_road_cost_per_t(self, trajectory_xy: torch.Tensor):
        """Evaluate footprint risk using only image-projected road evidence."""
        road = self.vlm.metric_bev.last_drivable_probability
        observed = self.vlm.metric_bev.last_observed_mask
        if road is None or observed is None or not self.vlm.metric_bev.last_valid:
            ones = torch.ones(trajectory_xy.shape[:2], device=trajectory_xy.device, dtype=trajectory_xy.dtype)
            return ones, torch.zeros_like(ones)
        road = road.to(device=trajectory_xy.device, dtype=trajectory_xy.dtype)
        observed = observed.to(device=trajectory_xy.device, dtype=trajectory_xy.dtype)
        points = self._footprint_points(trajectory_xy)
        grid = self.vlm.metric_bev.metric_to_grid(points)
        sampled_road = F.grid_sample(
            road, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )[:, 0]
        sampled_observed = F.grid_sample(
            observed, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )[:, 0]
        outside = (grid[..., 0].abs() > 1.0) | (grid[..., 1].abs() > 1.0)
        sampled_observed = sampled_observed * (~outside).to(sampled_observed.dtype)
        unknown_cost = float(getattr(self.cfg, "FF_UNKNOWN_POINT_COST", 0.15))
        point_cost = sampled_observed * (1.0 - sampled_road) + unknown_cost * (1.0 - sampled_observed)
        return point_cost.amax(dim=-1), sampled_observed.mean(dim=-1)

    def _safety_cost_per_t(self, trajectory_xy, drivable_mask):
        if drivable_mask is not None:
            cost = self._footprint_offroad_per_t(trajectory_xy, drivable_mask)
            return cost, torch.ones_like(cost)
        return self._image_road_cost_per_t(trajectory_xy)

    def _safety_filter(self, drivable_mask, trajectory):
        if not bool(getattr(self.cfg, "FF_SAFETY_FILTER_ENABLED", True)):
            return trajectory, torch.ones(trajectory.shape[0], device=trajectory.device, dtype=trajectory.dtype)
        coarse = self.vlm.last_coarse_xy.to(device=trajectory.device, dtype=trajectory.dtype)
        applied = self.vlm.last_residual_xy_gated.to(device=trajectory.device, dtype=trajectory.dtype)
        if drivable_mask is None and not self.vlm.metric_bev.last_valid:
            safe = torch.cat([coarse, trajectory[..., 2:]], dim=-1)
            return safe, torch.zeros(trajectory.shape[0], device=trajectory.device, dtype=trajectory.dtype)
        scales = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], device=trajectory.device, dtype=trajectory.dtype)
        # Safety filtering may only reduce the learned per-waypoint alpha; it
        # must never amplify the residual beyond the model's proposed path.
        candidates = coarse[:, None] + scales.view(1, -1, 1, 1) * applied[:, None]
        risk_values = []
        minimum_coverage = float(getattr(self.cfg, "FF_SAFETY_MIN_COVERAGE", 0.35))
        unknown_residual_penalty = float(getattr(self.cfg, "FF_UNKNOWN_RESIDUAL_PENALTY", 0.10))
        for index in range(scales.numel()):
            candidate_cost, coverage = self._safety_cost_per_t(candidates[:, index], drivable_mask)
            risk = candidate_cost.mean(dim=1)
            if drivable_mask is None:
                low_coverage = (coverage.mean(dim=1) < minimum_coverage).to(risk.dtype)
                risk = risk + low_coverage * unknown_residual_penalty * scales[index]
            risk_values.append(risk)
        risk = torch.stack(risk_values, dim=1)
        best = risk.argmin(dim=1)
        batch_index = torch.arange(trajectory.shape[0], device=trajectory.device)
        safe_xy = candidates[batch_index, best]
        safe = torch.cat([safe_xy, trajectory[..., 2:]], dim=-1)
        return safe, scales[best]

    def planning(self, *, bev_rgbs, trajs, gt_trajs, commands, target_points, occupancy=None, drivable_mask=None):
        self.vlm.set_planning_context(
            route_polyline=target_points,
            intrinsics=self._last_intrinsics_ff,
            extrinsics=self._last_extrinsics_ff,
        )
        base_loss, prediction, final_traj, tag, loss_dict = super().planning(
            bev_rgbs=bev_rgbs,
            trajs=trajs,
            gt_trajs=gt_trajs,
            commands=commands,
            target_points=target_points,
            occupancy=occupancy,
            drivable_mask=drivable_mask,
        )
        pred_xy = final_traj[..., :2]
        gt_xy = gt_trajs[..., :2]
        coarse = self.vlm.last_coarse_xy.to(device=pred_xy.device, dtype=pred_xy.dtype)
        normal = self.vlm.last_coarse_normal.to(device=pred_xy.device, dtype=pred_xy.dtype)
        target_delta_n = ((gt_xy - coarse) * normal).sum(dim=-1)
        raw_delta_n = self.vlm.last_delta_n_raw
        alpha_logits = self.vlm.last_alpha_logits
        alpha = self.vlm.last_alpha
        lateral_loss = F.smooth_l1_loss(raw_delta_n, target_delta_n, beta=0.25)
        alpha_deadzone = float(getattr(self.cfg, "FF_ALPHA_TARGET_DEADZONE_M", 0.15))
        alpha_temperature = float(getattr(self.cfg, "FF_ALPHA_TARGET_TEMPERATURE_M", 0.08))
        alpha_target = torch.sigmoid(
            (target_delta_n.abs() - alpha_deadzone) / max(alpha_temperature, 1e-6)
        )
        # BCE on sigmoid probabilities is explicitly disallowed under AMP.
        # Keep alpha for trajectory construction, but supervise its raw logits.
        alpha_loss = F.binary_cross_entropy_with_logits(alpha_logits, alpha_target.detach())
        identity_weight = torch.exp(-target_delta_n.abs() / max(alpha_deadzone, 1e-3))
        identity_loss = (alpha * raw_delta_n.abs() * identity_weight).mean()
        final_offroad_t, final_coverage_t = self._safety_cost_per_t(pred_xy, drivable_mask)
        coarse_offroad_t, _ = self._safety_cost_per_t(coarse, drivable_mask)
        coarse_offroad_t = coarse_offroad_t.detach()
        offroad_loss = final_offroad_t.mean()
        improve_loss = F.relu(final_offroad_t - coarse_offroad_t).mean()

        route_loss = pred_xy.sum() * 0.0
        corridor_loss = pred_xy.sum() * 0.0
        if torch.is_tensor(target_points) and target_points.numel() > 0 and target_points.dim() == 3:
            route = target_points[..., :2].to(device=pred_xy.device, dtype=pred_xy.dtype)
            route_distance = torch.cdist(pred_xy, route).amin(dim=-1)
            route_loss = route_distance.mean()
            corridor_radius = float(getattr(self.cfg, "FF_GT_CORRIDOR_RADIUS_M", 1.0))
            corridor_loss = F.relu(route_distance - corridor_radius).square().mean()

        weighted_lateral = float(getattr(self.cfg, "LOSS_FF_LATERAL_W", 2.0)) * lateral_loss
        weighted_alpha = float(getattr(self.cfg, "LOSS_FF_ALPHA_W", 0.25)) * alpha_loss
        weighted_identity = float(getattr(self.cfg, "LOSS_FF_IDENTITY_W", 0.5)) * identity_loss
        weighted_offroad = float(getattr(self.cfg, "LOSS_FF_OFFROAD_W", 8.0)) * offroad_loss
        weighted_improve = float(getattr(self.cfg, "LOSS_FF_IMPROVE_W", 4.0)) * improve_loss
        weighted_route = float(getattr(self.cfg, "LOSS_FF_ROUTE_W", 0.2)) * route_loss
        weighted_corridor = float(getattr(self.cfg, "LOSS_FF_CORRIDOR_W", 2.0)) * corridor_loss
        loss = (
            base_loss
            + weighted_lateral
            + weighted_alpha
            + weighted_identity
            + weighted_offroad
            + weighted_improve
            + weighted_route
            + weighted_corridor
        )

        safety_scale = torch.ones(pred_xy.shape[0], device=pred_xy.device, dtype=pred_xy.dtype)
        if not self.training:
            final_traj, safety_scale = self._safety_filter(drivable_mask, final_traj)
            prediction = final_traj
        loss_dict.update(
            {
                "ff_lateral_loss": lateral_loss,
                "ff_alpha_loss": alpha_loss,
                "ff_identity_loss": identity_loss,
                "ff_offroad_loss": offroad_loss,
                "ff_offroad_improve_loss": improve_loss,
                "ff_route_loss": route_loss,
                "ff_corridor_loss": corridor_loss,
                "ff_road_coverage_mean": final_coverage_t.mean(),
                "ff_alpha_mean": alpha.mean(),
                "ff_alpha_target_mean": alpha_target.mean(),
                "ff_delta_n_abs_mean": raw_delta_n.abs().mean(),
                "ff_target_delta_n_abs_mean": target_delta_n.abs().mean(),
                "ff_safety_scale_mean": safety_scale.mean(),
                "ff_metric_bev_valid": torch.tensor(float(self.vlm.metric_bev.last_valid), device=pred_xy.device),
                "ff_lateral_with_lam": weighted_lateral,
                "ff_alpha_with_lam": weighted_alpha,
                "ff_identity_with_lam": weighted_identity,
                "ff_offroad_with_lam": weighted_offroad,
                "ff_improve_with_lam": weighted_improve,
                "ff_route_with_lam": weighted_route,
                "ff_corridor_with_lam": weighted_corridor,
            }
        )
        return loss, prediction, final_traj, "FAST_FF", loss_dict
