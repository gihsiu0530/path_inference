import math
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from stp3.model_0512_graduate.codex_re_thinking import ADMLPPlanner
from stp3.model_0512_graduate.codex_pure_vlm import SharpTurnLateralAccelerationLimiter
from stp3.utils.geometry import calculate_birds_eye_view_parameters
from stp3.utils.tools import gen_dx_bx


DEFAULT_ADMLP_BASELINE_CKPT = (
    "/home/systemlab/senpai/checkpoint/last.ckpt"
)

# ASAP 急轉軌跡約束總開關：
# True  = 驗證/推論時啟用 SharpTurnLateralAccelerationLimiter
# False = 完全停用，不修改模型輸出的軌跡
SHARP_TURN_BOOST_ENABLED = False


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
        gate_init = float(getattr(cfg, "FAST_RESIDUAL_GATE_INIT", 0.05))
        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.residual_gate_head = nn.Sequential(
            nn.Linear(c, c // 2),
            nn.GELU(),
            nn.Linear(c // 2, self.n_future * 2),
        )
        nn.init.zeros_(self.residual_gate_head[-1].weight)
        nn.init.constant_(self.residual_gate_head[-1].bias, gate_logit)
        self.residual_scale = float(getattr(cfg, "FAST_RESIDUAL_SCALE", 30.0))
        self.delta_scale = float(getattr(cfg, "FAST_DELTA_SCALE", 4.0))
        self.residual_gate_min = float(getattr(cfg, "FAST_RESIDUAL_GATE_MIN", 0.0))
        self.disable_residual_gate = bool(getattr(cfg, "FAST_DISABLE_RESIDUAL_GATE", False))
        self.residual_gate_floor = 0.0
        self.residual_gate_free = False
        self.last_coarse_xy = None
        self.last_endpoint_xy = None
        self.last_residual_gate = None
        self.last_residual_gate_raw = None
        self.last_residual_xy_raw = None
        self.last_residual_xy_gated = None

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
                coarse_xy: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        cmd_ctx = self.command_embed(self._command_ids(commands, device))
        fused_mid = self.mid_fusion(torch.cat([rgb_mid, seg_rgb_mid, seg_id_mid, depth_mid], dim=1))
        fused = self.fusion(torch.cat([rgb_deep, seg_rgb_deep, seg_id_deep, depth_deep], dim=1))
        fused_mid = self._film(fused_mid, cmd_ctx.repeat_interleave(t_rf, dim=0))
        fused = self._film(fused, cmd_ctx.repeat_interleave(t_rf, dim=0))

        _, c, h_deep, w_deep = fused.shape
        fused_mid_seq = fused_mid.view(b, t_rf, c, *fused_mid.shape[-2:])
        fused_seq = fused.view(b, t_rf, c, h_deep, w_deep)
        spatial_tokens = torch.cat([
            self._tokens_from_feature(fused_mid_seq, scale_index=0),
            self._tokens_from_feature(fused_seq, scale_index=1),
        ], dim=1)

        frame_feat = self.spatial_pool(fused).view(b, t_rf, self.hidden_dim)
        _, h = self.temporal_gru(frame_feat)
        visual_ctx = h[-1]
        ego_ctx = self.ego_mlp(ego_seq.reshape(b, -1).to(device))
        if bool(getattr(self.cfg, "RESIDUAL_CONTEXT_DROP_EGO", True)):
            residual_ego_ctx = torch.zeros_like(ego_ctx)
        else:
            residual_ego_ctx = ego_ctx
        ctx = self.context_mlp(torch.cat([visual_ctx, residual_ego_ctx, cmd_ctx], dim=-1))

        query_ctx = self.query_context(torch.cat([visual_ctx, residual_ego_ctx, cmd_ctx], dim=-1))
        queries = self.time_queries.unsqueeze(0).expand(b, -1, -1) + query_ctx.unsqueeze(1)
        queries = queries + ctx.unsqueeze(1)
        for layer in self.decoder_layers:
            queries = layer(queries, spatial_tokens)
        refined, _ = self.waypoint_gru(queries)
        queries = self.waypoint_refine_ln(queries + refined)

        abs_residual = torch.tanh(self.traj_abs_head(queries)).view(b, self.n_future, 2) * self.residual_scale
        delta_residual = torch.tanh(self.traj_delta_head(queries)).view(b, self.n_future, 2) * self.delta_scale
        residual = abs_residual + torch.cumsum(delta_residual, dim=1)
        if coarse_xy is None:
            coarse = self.make_coarse_baseline(ego_seq)
        else:
            coarse = coarse_xy.to(device=device, dtype=residual.dtype)
            if coarse.shape != residual.shape:
                raise ValueError(f"coarse_xy should be {tuple(residual.shape)}, got {tuple(coarse.shape)}")
        residual_gate_raw = torch.sigmoid(self.residual_gate_head(ctx)).view(b, self.n_future, 2)
        gate_floor = float(getattr(self, "residual_gate_floor", 0.0)) if self.training else 0.0
        gate_min = float(getattr(self, "residual_gate_min", 0.0))
        gate_floor = max(gate_floor, gate_min)
        gate_floor = min(max(gate_floor, 0.0), 1.0)
        residual_gate = gate_floor + (1.0 - gate_floor) * residual_gate_raw
        if bool(getattr(self, "disable_residual_gate", False)):
            residual_gate = torch.ones_like(residual_gate)
        elif self.training and bool(getattr(self, "residual_gate_free", False)):
            residual_gate = torch.ones_like(residual_gate)
        residual_raw = residual
        residual = residual_raw * residual_gate
        self.last_coarse_xy = coarse.detach()
        self.last_residual_gate = residual_gate.detach()
        self.last_residual_gate_raw = residual_gate_raw.detach()
        self.last_residual_xy_raw = residual_raw
        self.last_residual_xy_gated = residual
        endpoint_residual = torch.tanh(self.endpoint_head(ctx)) * self.residual_scale
        endpoint_residual = endpoint_residual * residual_gate[:, -1]
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
        self.admlp_feature_source = str(getattr(cfg, "ADMLP_FEATURE_SOURCE", "generated"))
        self.admlp_feature_mode = str(getattr(cfg, "ADMLP_FEATURE_MODE", "past4_command"))
        self.admlp_past_frames = int(getattr(cfg, "ADMLP_PAST_FRAMES", 4))
        print(f"ADMLP_FEATURE_SOURCE: {self.admlp_feature_source}")
        print(f"ADMLP_FEATURE_MODE: {self.admlp_feature_mode}")
        print(f"ADMLP_PAST_FRAMES: {self.admlp_past_frames}")
        if self.admlp_feature_source != "generated":
            raise ValueError(
                "CustomCampusDataset generates AD-MLP features at runtime; "
                f"ADMLP_FEATURE_SOURCE must be 'generated', got {self.admlp_feature_source!r}."
            )
        if self.admlp_feature_mode != "past4_command" or self.admlp_past_frames != 4:
            raise ValueError(
                "NuscenesData_0624_ASAP produces past4_command features with four past poses; "
                f"got mode={self.admlp_feature_mode!r}, past_frames={self.admlp_past_frames}."
            )
        self.vlm = FastTrajectoryPlanner(cfg, self.n_future, self.input_size)
        self.admlp_baseline = ADMLPPlanner(cfg, self.n_future)
        self._load_admlp_baseline()
        self.admlp_baseline.eval()
        for param in self.admlp_baseline.parameters():
            param.requires_grad_(False)

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
        self._last_admlp_ego_motion = None
        self._last_admlp_input = None
        self._last_admlp_xy = None
        self.sharp_turn_limiter = SharpTurnLateralAccelerationLimiter(cfg)
        self.sharp_turn_limiter.enabled = SHARP_TURN_BOOST_ENABLED
        self.last_sharp_turn_info = None

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("model : codex_pure_fast_ASAP_admlp_coarse")
        print(f"SHARP_TURN_BOOST_ENABLED: {self.sharp_turn_limiter.enabled}")
        print(
            "SHARP_TURN_LATACC_LIMIT_DURING_TRAIN: "
            f"{self.sharp_turn_limiter.apply_during_train}"
        )
        print(f"Total parameters: {total:,}  Trainable parameters: {trainable:,}")
        self._print_planning_loss_weights()

    def _print_planning_loss_weights(self):
        lam_l2 = float(getattr(self.cfg, "LOSS_L2_W", 8.0))
        lam_col = float(getattr(self.cfg, "LOSS_COL_W", 30.0))
        lam_box_col = float(getattr(self.cfg, "LOSS_BOX_COL_W", 20.0))
        lam_smo = float(getattr(self.cfg, "LOSS_SMO_W", 0.1))
        lam_vel = float(getattr(self.cfg, "LOSS_VEL_W", 0.6))
        lam_dir = float(getattr(self.cfg, "LOSS_DIR_W", 8.0))
        lam_coarse = 0.0
        lam_fde = float(getattr(self.cfg, "LOSS_FDE_W", 2.0))
        lam_endpoint = float(getattr(self.cfg, "LOSS_ENDPOINT_AUX_W", 2.0))
        lam_residual_aux = float(getattr(self.cfg, "LOSS_RESIDUAL_AUX_W", 1.0))
        lam_easy_residual_reg = float(getattr(self.cfg, "LOSS_EASY_RESIDUAL_REG_W", 0.0))
        disable_residual_gate = bool(getattr(self.cfg, "FAST_DISABLE_RESIDUAL_GATE", False))
        lam_gate_target = 0.0 if disable_residual_gate else float(getattr(self.cfg, "LOSS_GATE_TARGET_W", 0.0))
        hard_l2_reweight_w = float(getattr(self.cfg, "HARD_L2_REWEIGHT_W", 0.0))
        easy_l2_weight = float(getattr(self.cfg, "EASY_L2_WEIGHT", 1.0))
        coarse_collision_hard_scale = float(getattr(self.cfg, "COARSE_COLLISION_HARD_SCALE", 1.0))
        coarse_collision_box_reweight_w = float(getattr(self.cfg, "COARSE_COLLISION_BOX_REWEIGHT_W", 0.0))
        residual_aux_on_raw = bool(getattr(self.cfg, "RESIDUAL_AUX_ON_RAW", True))
        residual_aux_thresh = float(getattr(self.cfg, "RESIDUAL_AUX_THRESH", 0.5))
        residual_aux_soft_scale = float(getattr(self.cfg, "RESIDUAL_AUX_SOFT_SCALE", 1.0))
        residual_aux_beta = float(getattr(self.cfg, "RESIDUAL_AUX_BETA", 0.5))
        gate_target_thresh = float(getattr(self.cfg, "GATE_TARGET_THRESH", 0.3))
        gate_target_scale = float(getattr(self.cfg, "GATE_TARGET_SCALE", 1.0))
        gate_min = float(getattr(self.cfg, "FAST_RESIDUAL_GATE_MIN", 0.0))
        gate_warmup_start = float(getattr(self.cfg, "FAST_RESIDUAL_GATE_WARMUP_FLOOR_START", 0.10))
        gate_warmup_end = float(getattr(self.cfg, "FAST_RESIDUAL_GATE_WARMUP_FLOOR_END", 0.0))
        gate_warmup_epochs = float(getattr(self.cfg, "FAST_RESIDUAL_GATE_WARMUP_EPOCHS", 5.0))
        residual_context_drop_ego = bool(getattr(self.cfg, "RESIDUAL_CONTEXT_DROP_EGO", True))
        residual_gate_free_epochs = float(getattr(self.cfg, "RESIDUAL_GATE_FREE_EPOCHS", 0.0))
        print("[ASAP planning loss weights]")
        print(f"  lam_l2 = {lam_l2}")
        print(f"  lam_col = {lam_col}")
        print(f"  lam_box_col = {lam_box_col}")
        print(f"  lam_smo = {lam_smo}")
        print(f"  lam_vel = {lam_vel}")
        print(f"  lam_dir = {lam_dir}")
        print(f"  lam_coarse = {lam_coarse}")
        print(f"  lam_fde = {lam_fde}")
        print(f"  lam_endpoint = {lam_endpoint}")
        print(f"  lam_residual_aux = {lam_residual_aux}")
        print(f"  lam_easy_residual_reg = {lam_easy_residual_reg}")
        print(f"  lam_gate_target = {lam_gate_target}")
        print(f"  easy_l2_weight = {easy_l2_weight}")
        print(f"  hard_l2_reweight_w = {hard_l2_reweight_w}")
        print(f"  coarse_collision_hard_scale = {coarse_collision_hard_scale}")
        print(f"  coarse_collision_box_reweight_w = {coarse_collision_box_reweight_w}")
        print(f"  residual_aux_on_raw = {residual_aux_on_raw}")
        print(f"  residual_aux_thresh = {residual_aux_thresh}")
        print(f"  residual_aux_soft_scale = {residual_aux_soft_scale}")
        print(f"  residual_aux_beta = {residual_aux_beta}")
        print(f"  gate_target_thresh = {gate_target_thresh}")
        print(f"  gate_target_scale = {gate_target_scale}")
        print("[ASAP residual gate warmup]")
        print(f"  gate_min = {gate_min}")
        print(f"  gate_warmup_start = {gate_warmup_start}")
        print(f"  gate_warmup_end = {gate_warmup_end}")
        print(f"  gate_warmup_epochs = {gate_warmup_epochs}")
        print(f"  disable_residual_gate = {disable_residual_gate}")
        print(f"  residual_context_drop_ego = {residual_context_drop_ego}")
        print(f"  residual_gate_free_epochs = {residual_gate_free_epochs}")

    def _load_admlp_baseline(self):
        ckpt_path = str(getattr(self.cfg, "ADMLP_BASELINE_CKPT", DEFAULT_ADMLP_BASELINE_CKPT))
        if not ckpt_path:
            raise ValueError("ADMLP_BASELINE_CKPT is empty; frozen AD-MLP coarse baseline requires a checkpoint.")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"AD-MLP baseline checkpoint not found: {ckpt_path}")

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        target = self.admlp_baseline.state_dict()
        prefixes = (
            "model.planner.",
            "planner.",
            "module.model.planner.",
            "module.planner.",
            "admlp_baseline.",
            "",
        )
        matched = {}
        for key, value in state_dict.items():
            for prefix in prefixes:
                if not key.startswith(prefix):
                    continue
                stripped = key[len(prefix):]
                if stripped in target and target[stripped].shape == value.shape:
                    matched[stripped] = value
                break

        if not matched:
            sample = list(state_dict.keys())[:8]
            raise RuntimeError(
                "Could not find AD-MLP planner weights in checkpoint. "
                f"checkpoint={ckpt_path}, sample_keys={sample}"
            )

        missing, unexpected = self.admlp_baseline.load_state_dict(matched, strict=False)
        print(
            f"Loaded frozen AD-MLP baseline from {ckpt_path} "
            f"({len(matched)} tensors, missing={len(missing)}, unexpected={len(unexpected)})"
        )

    def _admlp_coarse(self, device: torch.device, dtype: torch.dtype, commands: List[str]) -> torch.Tensor:
        with torch.no_grad():
            if self._last_admlp_input is not None:
                state = self._last_admlp_input.to(device=device, dtype=torch.float32)
                expected_dim = self.admlp_baseline.plan_head[0].in_features
                if state.dim() != 2 or state.shape[1] != expected_dim:
                    raise ValueError(
                        f"admlp_input should be (B,{expected_dim}), got {tuple(state.shape)}"
                    )
                pred_all = self.admlp_baseline.forward_state(state)
            else:
                if self._last_admlp_ego_motion is None:
                    raise RuntimeError("Call forward before planning so AD-MLP input or ego history is cached.")
                ego_motion = self._last_admlp_ego_motion.to(device=device, dtype=torch.float32)
                pred_all = self.admlp_baseline(ego_motion, commands)
        coarse_xy = pred_all[:, 1:, :2].to(device=device, dtype=dtype)
        self._last_admlp_xy = coarse_xy.detach()
        return coarse_xy

    def forward(self, image, intrinsics, extrinsics, future_egomotion, *,
                rgb_224_seq, seg_224_seq, seg_id_224_seq=None, depth_224_seq=None,
                ego_history_egomotion=None, admlp_input=None, **kwargs):
        if seg_id_224_seq is None:
            raise ValueError("codex_pure requires batch['seg_id_224_seq']; use NuscenesData_change.py.")
        if depth_224_seq is None:
            # 沒有深度資料時以零深度佔位：模型的 fusion 層固定含 depth 通道，
            # 無法真正移除，因此補一組零深度（_depth() 會 resize 到 input_size）。
            b_, t_ = rgb_224_seq.shape[0], rgb_224_seq.shape[1]
            depth_224_seq = torch.zeros(b_, t_, 224, 224, dtype=torch.float32)

        device = future_egomotion.device
        self._last_rgb_seq = rgb_224_seq.to(device, non_blocking=True)
        self._last_seg_seq = seg_224_seq.to(device, non_blocking=True)
        self._last_seg_id_seq = seg_id_224_seq.to(device, non_blocking=True)
        self._last_depth_seq = depth_224_seq.to(device, non_blocking=True)
        self._last_admlp_input = admlp_input
        if admlp_input is not None:
            self._last_admlp_ego_motion = None
        elif ego_history_egomotion is not None:
            self._last_admlp_ego_motion = (
                ego_history_egomotion[:, :-1]
                if ego_history_egomotion.shape[1] > 1
                else ego_history_egomotion
            )
        else:
            hist_len = max(1, self.receptive_field - 1)
            self._last_admlp_ego_motion = future_egomotion[:, :hist_len]

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

    def _apply_sharp_turn_limiter(
        self,
        pred: torch.Tensor,
        commands: Optional[List[str]] = None,
    ) -> torch.Tensor:
        final_traj, info = self.sharp_turn_limiter.apply(
            pred,
            commands,
            training=self.training,
        )
        self.last_sharp_turn_info = info
        return final_traj

    def planning(self, *, bev_rgbs, trajs, gt_trajs, commands, target_points, occupancy=None, drivable_mask=None):
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
            coarse_xy=self._admlp_coarse(device, gt_trajs.dtype, commands),
        ).to(device)
        final_traj = self._apply_sharp_turn_limiter(pred, commands)

        pred_xy = pred[..., :2]
        gt_xy = gt_trajs[..., :2]
        err = ((pred_xy - gt_xy) ** 2).sum(dim=-1).sqrt()
        t_len = err.shape[1]
        w = torch.linspace(1.3, 1.0, t_len, device=device)
        l2 = (err * w).mean()
        hard_l2 = l2
        fde_per = ((pred_xy[:, -1] - gt_xy[:, -1]) ** 2).sum(dim=-1).sqrt()
        fde = fde_per.mean()
        loss_fde = fde

        pred_d = torch.cat([pred_xy[:, :1], pred_xy[:, 1:] - pred_xy[:, :-1]], dim=1)
        gt_d = torch.cat([gt_xy[:, :1], gt_xy[:, 1:] - gt_xy[:, :-1]], dim=1)
        vel_l2 = ((pred_d - gt_d).pow(2).sum(-1) * w).mean()

        vel = pred_xy - torch.cat([pred_xy[:, :1], pred_xy[:, :-1]], dim=1)
        smooth = (vel[:, 1:] - vel[:, :-1]).pow(2).sum(-1).mean()

        collision = torch.tensor(0.0, device=device)
        box_collision = torch.tensor(0.0, device=device)
        box_collision_base = torch.tensor(0.0, device=device)
        hard_collision = torch.tensor(0.0, device=device)
        valid_col_mask = None
        risk_t = None
        box_risk_t = None
        col_w = None
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
            box_collision_base = box_collision
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
        residual_aux_l1 = torch.tensor(0.0, device=device)
        residual_aux_hard_ratio = torch.tensor(0.0, device=device)
        residual_aux_weight_mean = torch.tensor(0.0, device=device)
        residual_need_weight_mean = torch.tensor(0.0, device=device)
        coarse_collision_hard_score_mean = torch.tensor(0.0, device=device)
        coarse_collision_hard_ratio = torch.tensor(0.0, device=device)
        coarse_box_collision_loss = torch.tensor(0.0, device=device)
        box_collision_hard_weight_mean = torch.tensor(1.0, device=device)
        final_box_collision_on_coarse_hard = torch.tensor(0.0, device=device)
        residual_norm_mean = torch.tensor(0.0, device=device)
        residual_raw_norm_mean = torch.tensor(0.0, device=device)
        residual_target_norm_mean = torch.tensor(0.0, device=device)
        easy_residual_reg = torch.tensor(0.0, device=device)
        hard_l2_weight_mean = torch.tensor(1.0, device=device)
        easy_residual_weight_mean = torch.tensor(0.0, device=device)
        gate_target_loss = torch.tensor(0.0, device=device)
        gate_target_mean = torch.tensor(0.0, device=device)
        coarse = getattr(self.vlm, "last_coarse_xy", None)
        if coarse is not None:
            coarse_xy = coarse.to(device=device, dtype=pred_xy.dtype)
            target_residual = gt_xy - coarse_xy
            pred_residual = pred_xy - coarse_xy
            raw_residual = getattr(self.vlm, "last_residual_xy_raw", None)
            if raw_residual is not None:
                raw_residual = raw_residual.to(device=device, dtype=pred_xy.dtype)
            target_residual_norm = torch.sqrt((target_residual ** 2).sum(dim=-1) + 1e-8)
            pred_residual_norm = torch.sqrt((pred_residual ** 2).sum(dim=-1) + 1e-8)
            coarse_l2 = (target_residual_norm * w).mean()
            residual_norm_mean = pred_residual_norm.mean()
            raw_residual_norm = None
            if raw_residual is not None:
                raw_residual_norm = torch.sqrt((raw_residual ** 2).sum(dim=-1) + 1e-8)
                residual_raw_norm_mean = raw_residual_norm.mean()
            residual_target_norm_mean = target_residual_norm.mean()

            residual_aux_thresh = float(getattr(self.cfg, "RESIDUAL_AUX_THRESH", 0.5))
            residual_aux_soft_scale = max(float(getattr(self.cfg, "RESIDUAL_AUX_SOFT_SCALE", 1.0)), 1e-6)
            use_raw_residual_aux = bool(getattr(self.cfg, "RESIDUAL_AUX_ON_RAW", True))
            pred_residual_for_aux = raw_residual if (use_raw_residual_aux and raw_residual is not None) else pred_residual
            hard_mask = target_residual_norm > residual_aux_thresh
            residual_aux_hard_ratio = hard_mask.float().mean()
            residual_need_score = ((target_residual_norm - residual_aux_thresh) / residual_aux_soft_scale).clamp(0.0, 1.0)
            residual_need_weight_mean = residual_need_score.mean()
            coarse_collision_score = torch.zeros_like(residual_need_score)
            if occupancy is not None:
                with torch.no_grad():
                    coarse_box_risk_t = self.box_collision_loss_soft(coarse_xy, occupancy, return_per_t=True)
                    if col_w is not None and valid_col_mask is not None:
                        coarse_box_collision_loss = (
                            ((coarse_box_risk_t.to(device=device, dtype=pred_xy.dtype) ** 2)
                             * col_w
                             * valid_col_mask).sum()
                            / (valid_col_mask.sum() + 1e-6)
                        )
                    coarse_collision_scale = float(getattr(self.cfg, "COARSE_COLLISION_HARD_SCALE", 1.0))
                    coarse_collision_score = (coarse_box_risk_t.to(device=device, dtype=pred_xy.dtype)
                                              * coarse_collision_scale).clamp(0.0, 1.0)
                    if valid_col_mask is not None:
                        coarse_collision_score = coarse_collision_score * valid_col_mask.to(
                            device=device,
                            dtype=pred_xy.dtype,
                        )
            coarse_collision_hard_score_mean = coarse_collision_score.mean()
            coarse_collision_hard_ratio = (coarse_collision_score > 0.05).float().mean()
            residual_aux_weight = torch.maximum(residual_need_score, coarse_collision_score)
            residual_aux_weight_mean = residual_aux_weight.mean()
            if box_risk_t is not None and col_w is not None and valid_col_mask is not None:
                box_reweight_w = float(getattr(self.cfg, "COARSE_COLLISION_BOX_REWEIGHT_W", 0.0))
                box_col_hard_weight = 1.0 + box_reweight_w * coarse_collision_score
                box_collision_hard_weight_mean = box_col_hard_weight.mean()
                box_collision = (
                    ((box_risk_t ** 2) * col_w * valid_col_mask * box_col_hard_weight).sum()
                    / (valid_col_mask.sum() + 1e-6)
                )
                coarse_hard_mask = (coarse_collision_score > 0.05).to(dtype=pred_xy.dtype) * valid_col_mask
                final_box_collision_on_coarse_hard = (
                    ((box_risk_t ** 2) * col_w * coarse_hard_mask).sum()
                    / (coarse_hard_mask.sum() + 1e-6)
                )
            residual_aux_per_dim = F.smooth_l1_loss(
                pred_residual_for_aux,
                target_residual,
                beta=float(getattr(self.cfg, "RESIDUAL_AUX_BETA", 0.5)),
                reduction="none",
            )
            residual_aux_per_t = residual_aux_per_dim.mean(dim=-1)
            residual_aux_l1 = (residual_aux_per_t * residual_aux_weight).sum() / (residual_aux_weight.sum() + 1e-6)

            hard_l2_reweight_w = float(getattr(self.cfg, "HARD_L2_REWEIGHT_W", 0.0))
            easy_l2_weight = max(float(getattr(self.cfg, "EASY_L2_WEIGHT", 1.0)), 0.0)
            hard_l2_weight = easy_l2_weight + hard_l2_reweight_w * residual_aux_weight
            hard_l2_weight_mean = hard_l2_weight.mean()
            hard_l2 = (err * w * hard_l2_weight).sum() / (hard_l2_weight.sum() + 1e-6)
            fde_weight = hard_l2_weight[:, -1]
            loss_fde = (fde_per * fde_weight).sum() / (fde_weight.sum() + 1e-6)

            residual_for_easy_reg = raw_residual_norm if raw_residual_norm is not None else pred_residual_norm
            easy_residual_weight = 1.0 - residual_aux_weight
            easy_residual_weight_mean = easy_residual_weight.mean()
            easy_residual_reg = (
                (residual_for_easy_reg * easy_residual_weight).sum()
                / (easy_residual_weight.sum() + 1e-6)
            )

            disable_residual_gate = bool(getattr(self.cfg, "FAST_DISABLE_RESIDUAL_GATE", False))
            raw_gate = getattr(self.vlm, "last_residual_gate_raw", None)
            if raw_gate is not None and not disable_residual_gate:
                gate_target_thresh = float(getattr(self.cfg, "GATE_TARGET_THRESH", 0.3))
                gate_target_scale = max(float(getattr(self.cfg, "GATE_TARGET_SCALE", 1.0)), 1e-6)
                gate_target = ((target_residual_norm - gate_target_thresh) / gate_target_scale).clamp(0.0, 1.0)
                gate_target_mean = gate_target.mean()
                raw_gate_t = raw_gate.to(device=device, dtype=pred_xy.dtype).mean(dim=-1)
                gate_target_loss = F.mse_loss(raw_gate_t, gate_target)
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
        lam_coarse = 0.0
        lam_fde = float(getattr(self.cfg, "LOSS_FDE_W", 2.0))
        lam_endpoint = float(getattr(self.cfg, "LOSS_ENDPOINT_AUX_W", 2.0))
        lam_residual_aux = float(getattr(self.cfg, "LOSS_RESIDUAL_AUX_W", 1.0))
        lam_easy_residual_reg = float(getattr(self.cfg, "LOSS_EASY_RESIDUAL_REG_W", 0.0))
        disable_residual_gate = bool(getattr(self.cfg, "FAST_DISABLE_RESIDUAL_GATE", False))
        lam_gate_target = 0.0 if disable_residual_gate else float(getattr(self.cfg, "LOSS_GATE_TARGET_W", 0.0))
        loss = (
            lam_l2 * hard_l2
            + lam_col * collision
            + lam_box_col * box_collision
            + lam_smo * smooth
            + lam_vel * vel_l2
            + lam_dir * dir_loss
            + lam_coarse * coarse_l2
            + lam_fde * loss_fde
            + lam_endpoint * endpoint_aux_l2
            + lam_residual_aux * residual_aux_l1
            + lam_easy_residual_reg * easy_residual_reg
            + lam_gate_target * gate_target_loss
        )

        loss_dict = {
            "l2": l2,
            "hard_l2": hard_l2,
            "hard_l2_weight_mean": hard_l2_weight_mean,
            "fde": fde,
            "loss_fde": loss_fde,
            "vel_l2": vel_l2,
            "smooth": smooth,
            "collision": collision,
            "box_collision": box_collision,
            "box_collision_base": box_collision_base,
            "coarse_box_collision_loss": coarse_box_collision_loss,
            "box_collision_hard_weight_mean": box_collision_hard_weight_mean,
            "final_box_collision_on_coarse_hard": final_box_collision_on_coarse_hard,
            "hard_collision": hard_collision,
            "coarse_l2": coarse_l2,
            "endpoint_aux_l2": endpoint_aux_l2,
            "residual_aux_l1": residual_aux_l1,
            "residual_aux_hard_ratio": residual_aux_hard_ratio,
            "residual_aux_weight_mean": residual_aux_weight_mean,
            "residual_need_weight_mean": residual_need_weight_mean,
            "coarse_collision_hard_score_mean": coarse_collision_hard_score_mean,
            "coarse_collision_hard_ratio": coarse_collision_hard_ratio,
            "easy_residual_reg": easy_residual_reg,
            "easy_residual_weight_mean": easy_residual_weight_mean,
            "residual_norm_mean": residual_norm_mean,
            "residual_raw_norm_mean": residual_raw_norm_mean,
            "residual_target_norm_mean": residual_target_norm_mean,
            "residual_aux_on_raw": torch.tensor(
                1.0 if bool(getattr(self.cfg, "RESIDUAL_AUX_ON_RAW", True)) else 0.0,
                device=device,
            ),
            "gate_target_loss": gate_target_loss,
            "gate_target_mean": gate_target_mean,
            "residual_gate_mean": (
                self.vlm.last_residual_gate.mean()
                if self.vlm.last_residual_gate is not None
                else torch.tensor(0.0, device=device)
            ),
            "residual_gate_raw_mean": (
                self.vlm.last_residual_gate_raw.mean()
                if self.vlm.last_residual_gate_raw is not None
                else torch.tensor(0.0, device=device)
            ),
            "residual_gate_floor": torch.tensor(
                float(getattr(self.vlm, "residual_gate_floor", 0.0)) if self.training else 0.0,
                device=device,
            ),
            "residual_gate_min": torch.tensor(
                float(getattr(self.vlm, "residual_gate_min", 0.0)),
                device=device,
            ),
            "residual_gate_free": torch.tensor(
                1.0 if bool(getattr(self.vlm, "residual_gate_free", False)) and self.training else 0.0,
                device=device,
            ),
            "disable_residual_gate": torch.tensor(
                1.0 if bool(getattr(self.cfg, "FAST_DISABLE_RESIDUAL_GATE", False)) else 0.0,
                device=device,
            ),
            "dir_loss": dir_loss,
            "acc_turn": acc_turn,
            "acc_all": acc_all,
            "col_with_lam": lam_col * collision,
            "box_col_with_lam": lam_box_col * box_collision,
            "coarse_with_lam": lam_coarse * coarse_l2,
            "fde_with_lam": lam_fde * loss_fde,
            "endpoint_aux_with_lam": lam_endpoint * endpoint_aux_l2,
            "residual_aux_with_lam": lam_residual_aux * residual_aux_l1,
            "easy_residual_reg_with_lam": lam_easy_residual_reg * easy_residual_reg,
            "gate_target_with_lam": lam_gate_target * gate_target_loss,
        }
        sharp_turn_info = self.last_sharp_turn_info
        if sharp_turn_info is not None:
            apply_mask = sharp_turn_info["apply_mask"].to(device)
            sharp_turn_mask = sharp_turn_info["sharp_turn_mask"].to(device)
            command_mask = sharp_turn_info["command_mask"].to(device)
            traj_mask = sharp_turn_info["traj_mask"].to(device)
            forward_limited = sharp_turn_info["forward_limited"].to(device)
            loss_dict["sharp_turn_applied_ratio"] = apply_mask.float().mean()
            loss_dict["sharp_turn_trigger_ratio"] = sharp_turn_mask.float().mean()
            loss_dict["sharp_turn_command_ratio"] = command_mask.float().mean()
            loss_dict["sharp_turn_traj_ratio"] = traj_mask.float().mean()
            loss_dict["sharp_turn_forward_limited_ratio"] = forward_limited.float().mean()
            loss_dict["sharp_turn_forward_ratio"] = sharp_turn_info["forward_ratio"].to(device).mean()
            loss_dict["sharp_turn_forward_m"] = sharp_turn_info["forward"].to(device).mean()
            loss_dict["sharp_turn_forward_after_m"] = sharp_turn_info["forward_after"].to(device).mean()
            loss_dict["sharp_turn_forward_cap_m"] = sharp_turn_info["forward_limit_m"].to(device).mean()
            loss_dict["sharp_turn_cap_ratio"] = sharp_turn_info["cap_ratio"].to(device).mean()
            loss_dict["sharp_turn_speed_ratio_min"] = sharp_turn_info["speed_ratio_min"].to(device).mean()
            loss_dict["sharp_turn_max_lat_acc_mps2"] = sharp_turn_info["max_lat_acc_mps2"].to(device).mean()
            loss_dict["sharp_turn_max_curvature"] = sharp_turn_info["max_curvature"].to(device).mean()
            loss_dict["sharp_turn_turn_radius_m"] = sharp_turn_info["turn_radius_m"].to(device).mean()
            loss_dict["sharp_turn_min_turn_radius_m"] = sharp_turn_info["min_turn_radius_m"].to(device).mean()
            loss_dict["sharp_turn_train_turn_radius_m"] = sharp_turn_info["train_turn_radius_m"].to(device).mean()
            loss_dict["sharp_turn_radius_ratio"] = sharp_turn_info["radius_ratio"].to(device).mean()
            loss_dict["sharp_turn_curve_alpha"] = sharp_turn_info["curve_alpha"].to(device).mean()
            loss_dict["sharp_turn_max_heading_deg"] = sharp_turn_info["max_heading_deg"].to(device).mean()
        return loss, pred, final_traj, "FAST", loss_dict
