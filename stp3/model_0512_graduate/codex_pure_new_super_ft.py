from typing import List, Optional

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


class PedTrajectoryEncoder(nn.Module):
    def __init__(self, channels: int, feat_dim: int, max_agents: int, frames: int, use_goal: bool = True):
        super().__init__()
        self.channels = channels
        self.feat_dim = feat_dim
        self.max_agents = max_agents
        self.frames = frames
        self.use_goal = use_goal
        self.step_proj = nn.Sequential(
            nn.Linear(feat_dim + 1, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.time_pe = nn.Parameter(torch.randn(frames, channels) * 0.02)
        self.agent_embed = nn.Embedding(max_agents, channels)

    def _prepare(self, ped_traj_preds, ped_traj_mask, ped_traj_valid_steps, device):
        if ped_traj_preds is None:
            return None, None
        if not torch.is_tensor(ped_traj_preds):
            ped_traj_preds = torch.as_tensor(ped_traj_preds)
        ped = ped_traj_preds.to(device=device, dtype=torch.float32, non_blocking=True)
        if ped.dim() == 3:
            ped = ped.unsqueeze(1)
        if ped.dim() != 4:
            raise ValueError(f"ped_traj_preds should be (B,M,T,F) or (B,T,F), got shape={tuple(ped.shape)}")

        b, m, t, f = ped.shape
        if m > self.max_agents:
            ped = ped[:, :self.max_agents]
            m = ped.shape[1]
        if f < self.feat_dim:
            pad = torch.zeros(b, m, t, self.feat_dim - f, device=device, dtype=ped.dtype)
            ped = torch.cat([ped, pad], dim=-1)
        elif f > self.feat_dim:
            ped = ped[..., :self.feat_dim]
        if t < self.frames:
            pad = torch.zeros(b, m, self.frames - t, self.feat_dim, device=device, dtype=ped.dtype)
            ped = torch.cat([ped, pad], dim=2)
        elif t > self.frames:
            ped = ped[:, :, :self.frames]

        finite_step = torch.isfinite(ped).all(dim=-1)
        finite_agent = finite_step.any(dim=-1)
        ped = torch.nan_to_num(ped, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.use_goal and ped.shape[-1] >= 8:
            ped = ped.clone()
            ped[..., 6:8] = 0.0

        if ped_traj_mask is None:
            agent_mask = finite_agent
        else:
            if not torch.is_tensor(ped_traj_mask):
                ped_traj_mask = torch.as_tensor(ped_traj_mask)
            agent_mask = ped_traj_mask.to(device=device, dtype=torch.bool, non_blocking=True)
            if agent_mask.dim() == 1:
                agent_mask = agent_mask.unsqueeze(0).expand(b, -1)
            agent_mask = agent_mask[:, :ped.shape[1]] & finite_agent

        if ped_traj_valid_steps is None:
            step_mask = finite_step
        else:
            if not torch.is_tensor(ped_traj_valid_steps):
                ped_traj_valid_steps = torch.as_tensor(ped_traj_valid_steps)
            step_mask = ped_traj_valid_steps.to(device=device, dtype=torch.bool, non_blocking=True)
            if step_mask.dim() == 2:
                step_mask = step_mask.unsqueeze(0).expand(b, -1, -1)
            step_mask = step_mask[:, :ped.shape[1], :ped.shape[2]] & finite_step
        step_mask = step_mask & agent_mask[:, :, None]
        return ped, step_mask

    def forward(self, ped_traj_preds, ped_traj_mask, ped_traj_valid_steps, device):
        ped, step_mask = self._prepare(ped_traj_preds, ped_traj_mask, ped_traj_valid_steps, device)
        if ped is None or not step_mask.any():
            return None, None, None, None

        b, m, t, _ = ped.shape
        valid_scalar = step_mask[:, :, :, None].to(ped.dtype)
        tokens = self.step_proj(torch.cat([ped, valid_scalar], dim=-1))
        agent_ids = torch.arange(m, device=device).clamp(max=self.max_agents - 1)
        tokens = tokens + self.time_pe[:t].view(1, 1, t, -1)
        tokens = tokens + self.agent_embed(agent_ids).view(1, m, 1, -1)
        tokens = tokens * valid_scalar

        denom = step_mask.sum(dim=(1, 2)).clamp(min=1).to(tokens.dtype).unsqueeze(-1)
        ped_mean = tokens.sum(dim=(1, 2)) / denom
        neg_inf = -1e4 if tokens.dtype == torch.float16 else -1e9
        ped_max = tokens.masked_fill(~step_mask[:, :, :, None], neg_inf).amax(dim=(1, 2))
        no_valid = ~step_mask.any(dim=(1, 2))
        ped_max = torch.where(no_valid.unsqueeze(1), torch.zeros_like(ped_max), ped_max)
        ctx = self.ctx_proj(torch.cat([ped_mean, ped_max], dim=-1))
        return tokens.reshape(b, m * t, -1), ~step_mask.reshape(b, m * t), ctx, step_mask.any(dim=(1, 2))


class PedQueryFusion(nn.Module):
    def __init__(self, channels: int, num_heads: int, dropout: float, gate_bias: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(channels)
        self.norm_ped = nn.LayerNorm(channels)
        self.gate = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.GELU(),
            nn.Linear(channels, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[2].bias, gate_bias)

    def forward(self, queries, ped_tokens, ped_key_padding_mask, ped_ctx, active_gate):
        attn_out = self.attn(
            self.norm_q(queries),
            self.norm_ped(ped_tokens),
            self.norm_ped(ped_tokens),
            key_padding_mask=ped_key_padding_mask,
            need_weights=False,
        )[0]
        ctx = ped_ctx.unsqueeze(1).expand(-1, queries.shape[1], -1)
        gate = self.gate(torch.cat([queries, attn_out, ctx], dim=-1)) * active_gate
        return queries + gate * (attn_out + ctx), gate.squeeze(-1)


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
        self.residual_scale = float(getattr(cfg, "FAST_RESIDUAL_SCALE", 30.0))
        self.delta_scale = float(getattr(cfg, "FAST_DELTA_SCALE", 4.0))
        self.ped_gate_min = float(getattr(cfg, "PED_GATE_MIN", 0.05))
        self.ped_gate_init_bias = float(getattr(cfg, "PED_GATE_INIT_BIAS", -1.5))
        self.ped_start_step = int(getattr(cfg, "PED_START_STEP", 1))
        self.ped_ramp_steepness = float(getattr(cfg, "PED_RAMP_STEEPNESS", 2.0))
        self.ped_ctx_scale = float(getattr(cfg, "PED_CTX_SCALE", 0.35))
        self.ped_bev_scale = float(getattr(cfg, "PED_BEV_SCALE", 0.8))
        self.ped_bev_gate_min = float(getattr(cfg, "PED_BEV_GATE_MIN", 0.02))
        self.ped_encoder = PedTrajectoryEncoder(
            channels=c,
            feat_dim=int(getattr(cfg, "PED_TRAJ_FEAT_DIM", 12)),
            max_agents=int(getattr(cfg, "PED_MAX_AGENTS", 64)),
            frames=int(getattr(cfg, "PED_INPUT_FRAMES", 9)),
            use_goal=bool(getattr(cfg, "PED_USE_GOAL", True)),
        )
        self.ped_query_fusion = PedQueryFusion(
            c,
            num_heads=decoder_heads,
            dropout=float(getattr(cfg, "FAST_DROPOUT", 0.0)),
            gate_bias=self.ped_gate_init_bias,
        )
        self.ped_bev_encoder = nn.Sequential(
            nn.Conv2d(1, c // 4, kernel_size=3, padding=1),
            nn.GroupNorm(8 if (c // 4) % 8 == 0 else 4, c // 4),
            nn.GELU(),
            nn.Conv2d(c // 4, c // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8 if (c // 2) % 8 == 0 else 4, c // 2),
            nn.GELU(),
        )
        self.ped_bev_align_head = nn.Sequential(
            nn.Conv2d(c // 2, c, kernel_size=3, padding=1),
            nn.GroupNorm(8 if c % 8 == 0 else 4, c),
            nn.GELU(),
        )
        self.ped_bev_fuse_gate = nn.Sequential(
            nn.Conv2d(c * 2, c // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.last_coarse_xy = None
        self.last_endpoint_xy = None
        self.last_ped_gate = None
        self.last_ped_bev_gate = None

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

    def _future_ped_active_gate(self, b: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        step_idx = torch.arange(self.n_future, device=device, dtype=dtype)
        center = float(min(max(self.ped_start_step, 0), max(self.n_future - 1, 0)))
        if self.n_future <= 1:
            active = torch.ones(self.n_future, device=device, dtype=dtype)
        else:
            active = torch.sigmoid((step_idx - center) / max(self.ped_ramp_steepness, 1e-3))
        return active.view(1, self.n_future, 1).expand(b, -1, -1)

    def _fuse_ped_bev(self, fused_mid, fused, ped_bev_map, b: int, t_rf: int, device: torch.device):
        if ped_bev_map is None:
            self.last_ped_bev_gate = torch.zeros(b, device=device)
            return fused_mid, fused
        if not torch.is_tensor(ped_bev_map):
            ped_bev_map = torch.as_tensor(ped_bev_map)
        ped_bev_map = ped_bev_map.to(device=device, dtype=torch.float32, non_blocking=True)
        if ped_bev_map.dim() == 3:
            ped_bev_map = ped_bev_map.unsqueeze(1)
        map_has_data = (ped_bev_map.flatten(1).amax(dim=1) > 0).to(fused.dtype)
        if not map_has_data.any():
            self.last_ped_bev_gate = torch.zeros(b, device=device)
            return fused_mid, fused

        ped_bev_bt = ped_bev_map.repeat_interleave(t_rf, dim=0)
        map_has_data_bt = map_has_data.repeat_interleave(t_rf).view(-1, 1, 1, 1)
        ped_bev_mid = F.interpolate(ped_bev_bt, size=fused_mid.shape[-2:], mode="bilinear", align_corners=False)
        ped_bev_deep = F.interpolate(ped_bev_bt, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        ped_mid_feat = self.ped_bev_align_head(self.ped_bev_encoder(ped_bev_mid)) * map_has_data_bt
        ped_deep_feat = self.ped_bev_align_head(self.ped_bev_encoder(ped_bev_deep)) * map_has_data_bt
        ped_mid_gate = self.ped_bev_fuse_gate(torch.cat([fused_mid, ped_mid_feat], dim=1))
        ped_deep_gate = self.ped_bev_fuse_gate(torch.cat([fused, ped_deep_feat], dim=1))
        ped_mid_gate = (self.ped_bev_gate_min + (1.0 - self.ped_bev_gate_min) * ped_mid_gate) * map_has_data_bt
        ped_deep_gate = (self.ped_bev_gate_min + (1.0 - self.ped_bev_gate_min) * ped_deep_gate) * map_has_data_bt
        fused_mid = fused_mid + self.ped_bev_scale * ped_mid_gate * ped_mid_feat
        fused = fused + self.ped_bev_scale * ped_deep_gate * ped_deep_feat
        ped_mid_gate_b = ped_mid_gate.view(b, t_rf, *ped_mid_gate.shape[1:]).mean(dim=(1, 2, 3, 4))
        ped_deep_gate_b = ped_deep_gate.view(b, t_rf, *ped_deep_gate.shape[1:]).mean(dim=(1, 2, 3, 4))
        self.last_ped_bev_gate = (0.5 * (ped_mid_gate_b + ped_deep_gate_b)).detach()
        return fused_mid, fused

    def forward(
        self,
        rgb_seq: torch.Tensor,
        seg_rgb_seq: torch.Tensor,
        seg_id_seq: torch.Tensor,
        depth_seq: torch.Tensor,
        ego_seq: torch.Tensor,
        commands: List[str],
        ped_traj_preds: Optional[torch.Tensor] = None,
        ped_traj_mask: Optional[torch.Tensor] = None,
        ped_traj_valid_steps: Optional[torch.Tensor] = None,
        ped_bev_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        fused_mid, fused = self._fuse_ped_bev(fused_mid, fused, ped_bev_map, b, t_rf, device)

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
        ctx = self.context_mlp(torch.cat([visual_ctx, ego_ctx, cmd_ctx], dim=-1))
        ped_tokens, ped_key_padding_mask, ped_ctx, ped_has_data = self.ped_encoder(
            ped_traj_preds, ped_traj_mask, ped_traj_valid_steps, device
        )
        if ped_ctx is not None:
            ped_ctx = ped_ctx * ped_has_data.to(ped_ctx.dtype).unsqueeze(-1)
            ctx = ctx + self.ped_ctx_scale * ped_ctx

        query_ctx = self.query_context(torch.cat([visual_ctx, ego_ctx, cmd_ctx], dim=-1))
        queries = self.time_queries.unsqueeze(0).expand(b, -1, -1) + query_ctx.unsqueeze(1)
        queries = queries + ctx.unsqueeze(1)
        if ped_tokens is not None:
            if ped_key_padding_mask.all(dim=1).any():
                ped_key_padding_mask = ped_key_padding_mask.clone()
                ped_key_padding_mask[ped_key_padding_mask.all(dim=1)] = False
            active_gate = self._future_ped_active_gate(b, device, queries.dtype)
            active_gate = self.ped_gate_min + (1.0 - self.ped_gate_min) * active_gate
            active_gate = active_gate * ped_has_data.to(queries.dtype).view(b, 1, 1)
            queries, ped_gate = self.ped_query_fusion(
                queries, ped_tokens.to(queries.dtype), ped_key_padding_mask, ped_ctx.to(queries.dtype), active_gate
            )
            self.last_ped_gate = ped_gate.detach()
        else:
            self.last_ped_gate = torch.zeros(b, self.n_future, device=device)
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
        self._last_ped_traj_preds = None
        self._last_ped_traj_mask = None
        self._last_ped_traj_valid_steps = None
        self._last_ped_bev_map = None

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("model : codex_pure_fast")
        print(f"Total parameters: {total:,}  Trainable parameters: {trainable:,}")

    def forward(self, image, intrinsics, extrinsics, future_egomotion, *,
                rgb_224_seq, seg_224_seq, seg_id_224_seq=None, depth_224_seq=None,
                ped_traj_preds=None, ped_traj_mask=None, ped_traj_valid_steps=None, ped_bev_map=None):
        if seg_id_224_seq is None:
            raise ValueError("codex_pure requires batch['seg_id_224_seq']; use NuscenesData_change.py.")
        if depth_224_seq is None:
            raise ValueError("codex_pure requires batch['depth_224_seq']; use NuscenesData_change.py.")

        device = future_egomotion.device
        self._last_rgb_seq = rgb_224_seq.to(device, non_blocking=True)
        self._last_seg_seq = seg_224_seq.to(device, non_blocking=True)
        self._last_seg_id_seq = seg_id_224_seq.to(device, non_blocking=True)
        self._last_depth_seq = depth_224_seq.to(device, non_blocking=True)
        self._last_ped_traj_preds = ped_traj_preds
        self._last_ped_traj_mask = ped_traj_mask
        self._last_ped_traj_valid_steps = ped_traj_valid_steps
        self._last_ped_bev_map = ped_bev_map.to(device, non_blocking=True) if torch.is_tensor(ped_bev_map) else ped_bev_map

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

    def dynamic_repulsion_loss(self, trajs_xy, ped_bev_points, ped_bev_valid_steps, return_per_t=False):
        if ped_bev_points is None or ped_bev_valid_steps is None:
            zero = torch.zeros(
                trajs_xy.shape[0], trajs_xy.shape[1], device=trajs_xy.device, dtype=trajs_xy.dtype
            )
            return zero if return_per_t else zero.mean()
        device = trajs_xy.device
        dtype = trajs_xy.dtype
        if not torch.is_tensor(ped_bev_points):
            ped_bev_points = torch.as_tensor(ped_bev_points)
        if not torch.is_tensor(ped_bev_valid_steps):
            ped_bev_valid_steps = torch.as_tensor(ped_bev_valid_steps)
        ped_bev_points = ped_bev_points.to(device=device, dtype=dtype, non_blocking=True)
        ped_bev_valid_steps = ped_bev_valid_steps.to(device=device, dtype=torch.bool, non_blocking=True)
        if ped_bev_points.dim() != 4 or ped_bev_valid_steps.dim() != 3:
            raise ValueError("ped_bev_points should be (B,M,T,2) and ped_bev_valid_steps should be (B,M,T)")
        t = trajs_xy.shape[1]
        ped_bev_points = ped_bev_points[:, :, :t]
        ped_bev_valid_steps = ped_bev_valid_steps[:, :, :t]
        dist = torch.norm(trajs_xy[:, None, :, :] - ped_bev_points, dim=-1)
        safe_dist = float(getattr(self.cfg, "LOSS_PED_REPULSE_SAFE_DIST", 6.0))
        penalty = F.relu(safe_dist - dist) * ped_bev_valid_steps.to(dtype)
        per_t = penalty.max(dim=1).values
        if return_per_t:
            return per_t
        valid_t = ped_bev_valid_steps.any(dim=1)
        if not valid_t.any():
            return per_t.sum() * 0.0
        return (per_t * valid_t.to(dtype)).sum() / valid_t.to(dtype).sum().clamp(min=1.0)

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

    def planning(
        self,
        *,
        bev_rgbs,
        trajs,
        gt_trajs,
        commands,
        target_points,
        occupancy=None,
        drivable_mask=None,
        ped_bev_points=None,
        ped_bev_valid_steps=None,
    ):
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
            ped_traj_preds=self._last_ped_traj_preds,
            ped_traj_mask=self._last_ped_traj_mask,
            ped_traj_valid_steps=self._last_ped_traj_valid_steps,
            ped_bev_map=self._last_ped_bev_map,
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

        ped_repulse = torch.tensor(0.0, device=device)
        if ped_bev_points is not None and ped_bev_valid_steps is not None:
            ped_repulse = self.dynamic_repulsion_loss(pred_xy, ped_bev_points, ped_bev_valid_steps)

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
        acc_turn = success[turn_mask].mean() if turn_mask.any() else torch.tensor(0.0, device=device)
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
        lam_endpoint = float(getattr(self.cfg, "LOSS_ENDPOINT_AUX_W", 2.0))
        lam_ped_repulse = float(getattr(self.cfg, "LOSS_PED_REPULSE_W", 1.0))
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
            + lam_ped_repulse * ped_repulse
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
            "ped_repulse": ped_repulse,
            "dir_loss": dir_loss,
            "acc_turn": acc_turn,
            "acc_all": acc_all,
            "col_with_lam": lam_col * collision,
            "box_col_with_lam": lam_box_col * box_collision,
            "fde_with_lam": lam_fde * fde,
            "endpoint_aux_with_lam": lam_endpoint * endpoint_aux_l2,
            "ped_repulse_with_lam": lam_ped_repulse * ped_repulse,
        }
        ped_gate = getattr(self.vlm, "last_ped_gate", None)
        if ped_gate is not None:
            loss_dict["ped_gate_mean_over_time"] = ped_gate.mean()
            loss_dict["ped_gate_mean_last5"] = ped_gate[:, -5:].mean() if ped_gate.size(1) >= 5 else ped_gate.mean()
        ped_bev_gate = getattr(self.vlm, "last_ped_bev_gate", None)
        if ped_bev_gate is not None:
            loss_dict["ped_bev_gate_mean"] = ped_bev_gate.mean()
        return loss, pred, final_traj, "FAST", loss_dict
