import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from typing import List, Optional

from stp3.model_ped_traj.image_forced import (
    VLM_Generative as BaseVLM_Generative,
    VLM_STP3_Gen as BaseVLM_STP3_Gen,
)
from stp3.utils.tools import gen_dx_bx
from stp3.utils.geometry import calculate_birds_eye_view_parameters, pose_vec2mat, mat2pose_vec


class StableBatchedCrossAttention(nn.Module):
    """Ped-traj local copy that keeps the attention kernel in float32 under AMP."""

    def __init__(self, embed_dim: int, num_heads: int = 8, attn_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout and attn_dropout > 0 else nn.Identity()

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        B, Q, C = q.shape
        S = kv.shape[1]
        H = self.num_heads
        D = self.head_dim
        out_dtype = q.dtype

        with autocast(enabled=False):
            q_f = q.float()
            kv_f = kv.float()

            q_w = self.q_proj.weight.float()
            q_b = self.q_proj.bias.float() if self.q_proj.bias is not None else None
            q_lin = F.linear(q_f, q_w, q_b).view(B, Q, H, D)

            k_w = self.k_proj.weight.float()
            k_b = self.k_proj.bias.float() if self.k_proj.bias is not None else None
            k_lin = F.linear(kv_f, k_w, k_b).view(B, S, H, D)

            v_w = self.v_proj.weight.float()
            v_b = self.v_proj.bias.float() if self.v_proj.bias is not None else None
            v_lin = F.linear(kv_f, v_w, v_b).view(B, S, H, D)

            attn = torch.einsum('bqhd,bshd->bhqs', q_lin, k_lin) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = torch.einsum('bhqs,bshd->bqhd', attn, v_lin).contiguous().view(B, Q, C)
            out = F.layer_norm(out, [C])

            o_w = self.out_proj.weight.float()
            o_b = self.out_proj.bias.float() if self.out_proj.bias is not None else None
            out = F.linear(out, o_w, o_b)
        return out.to(out_dtype)


class VLM_Generative(BaseVLM_Generative):
    """
    `image_forced.py` 的增強版生成器。
    額外支援 ped_traj_preds: (B, M, T, F)
    預設 F=8，對應目前 dataset 的幾何 ped 特徵。

    這個版本刻意保持弱介入：
    - 保留 ped token fusion
    - 不加入 ped gate regularization / contrast loss / lateral-only correction
    - 讓行為盡量接近原本效果較穩的版本
    """

    def __init__(self, clip_name: str, device: torch.device, n_future: int, input_size: int = 224, prompts=None):
        if prompts is None:
            super().__init__(clip_name, device, n_future=n_future, input_size=input_size)
        else:
            super().__init__(clip_name, device, n_future=n_future, input_size=input_size, prompts=prompts)
        self.cross_attn = StableBatchedCrossAttention(self.vis_width, num_heads=8, attn_dropout=0.0)
        self.ped_traj_feat_dim = int(getattr(self, 'PED_TRAJ_FEAT_DIM', 12))
        self.max_ped_agents = int(getattr(self, 'PED_MAX_AGENTS', 64))
        self.ped_step_proj = nn.Sequential(
            nn.Linear(self.ped_traj_feat_dim + 1, self.vis_width),
            nn.GELU(),
            nn.Linear(self.vis_width, self.vis_width),
        )
        self.ped_ctx_proj = nn.Sequential(
            nn.Linear(self.vis_width * 2, self.vis_width),
            nn.GELU(),
            nn.Linear(self.vis_width, self.vis_width),
        )
        self.ped_frames = int(getattr(self, 'PED_INPUT_FRAMES', 9))
        self.ped_time_pe = nn.Parameter(torch.randn(self.ped_frames, self.vis_width) * 0.02)
        self.ped_agent_embed = nn.Embedding(self.max_ped_agents, self.vis_width)
        self.ped_fuse_gate = nn.Sequential(
            nn.Linear(self.vis_width * 2, self.vis_width),
            nn.GELU(),
            nn.Linear(self.vis_width, 1),
            nn.Sigmoid(),
        )
        # Independent BEV alignment head for pedestrian geometry.
        # It does not share weights with the image BEV decoder; instead it learns
        # how to translate a sparse geometric ped-BEV map into the planner's
        # learned BEV feature space before fusion.
        def _gn(c: int):
            groups = 8
            while c % groups != 0 and groups > 1:
                groups //= 2
            return nn.GroupNorm(groups, c)

        self.ped_bev_encoder = nn.Sequential(
            nn.Conv2d(1, self.vis_width // 4, kernel_size=3, padding=1),
            _gn(self.vis_width // 4),
            nn.GELU(),
            nn.Conv2d(self.vis_width // 4, self.vis_width // 2, kernel_size=3, padding=1),
            _gn(self.vis_width // 2),
            nn.GELU(),
        )
        self.ped_bev_align_head = nn.Sequential(
            nn.Conv2d(self.vis_width // 2, self.vis_width, kernel_size=3, padding=1),
            _gn(self.vis_width),
            nn.GELU(),
            nn.Conv2d(self.vis_width, self.vis_width, kernel_size=3, padding=1),
            _gn(self.vis_width),
            nn.GELU(),
        )
        self.ped_bev_fuse_gate = nn.Sequential(
            nn.Conv2d(self.vis_width * 2, self.vis_width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.vis_width // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.ped_bev_scale = float(getattr(self, 'PED_BEV_SCALE', 1.0))
        self.ped_bev_gate_min = float(getattr(self, 'PED_BEV_GATE_MIN', 0.05))
        self.cached_ped_bev_map = None
        self.ped_gate_min = float(getattr(self, 'PED_GATE_MIN', 0.10))
        self.ped_gate_init_bias = float(getattr(self, 'PED_GATE_INIT_BIAS', 0.0))
        self.ped_start_step = int(getattr(self, 'PED_START_STEP', 1))
        self.ped_ramp_steepness = float(getattr(self, 'PED_RAMP_STEEPNESS', 2.0))
        self.ped_use_goal = bool(getattr(self, 'PED_USE_GOAL', True))
        nn.init.constant_(self.ped_fuse_gate[2].bias, self.ped_gate_init_bias)
        self.cached_ped_traj_preds = None
        self.cached_ped_traj_mask = None
        self.cached_ped_traj_valid_steps = None
        self.last_ped_gate = None
        self.last_ped_bev_gate = None

        


    def _assert_finite_ped_bev_tensor(self, name, tensor, ped_bev_map_ref=None):
        if not torch.is_tensor(tensor):
            return
        chk = tensor.detach().float()
        if torch.isfinite(chk).all():
            return

        if ped_bev_map_ref is not None and torch.is_tensor(ped_bev_map_ref):
            ref = ped_bev_map_ref.detach().float()
            finite_ref = ref[torch.isfinite(ref)]
            if finite_ref.numel() > 0:
                ref_min = float(finite_ref.min().item())
                ref_max = float(finite_ref.max().item())
            else:
                ref_min = float('nan')
                ref_max = float('nan')
            if ref.dim() >= 4:
                ped_bev_valid_ratio = float((ref.abs() > 0).float().mean().item())
                batch_size = int(ref.shape[0])
            else:
                ped_bev_valid_ratio = float('nan')
                batch_size = int(chk.shape[0]) if chk.dim() > 0 else -1
        else:
            ref_min = float('nan')
            ref_max = float('nan')
            ped_bev_valid_ratio = float('nan')
            batch_size = int(chk.shape[0]) if chk.dim() > 0 else -1

        raise RuntimeError(
            f'Non-finite ped-BEV tensor detected: name={name}, '
            f'batch_size={batch_size}, '
            f'ped_bev_map_min={ref_min}, ped_bev_map_max={ref_max}, '
            f'ped_bev_valid_ratio={ped_bev_valid_ratio}, '
            f'train_step={getattr(self, "_debug_train_step", -1)}'
        )

    def _assert_finite_runtime_tensor(self, name, tensor):
        if not torch.is_tensor(tensor):
            return
        chk = tensor.detach().float()
        if torch.isfinite(chk).all():
            return
        finite = chk[torch.isfinite(chk)]
        if finite.numel() > 0:
            tmin = float(finite.min().item())
            tmax = float(finite.max().item())
        else:
            tmin = float('nan')
            tmax = float('nan')
        raise RuntimeError(
            f'Non-finite runtime tensor detected: name={name}, '
            f'shape={tuple(chk.shape)}, min={tmin}, max={tmax}, '
            f'train_step={getattr(self, "_debug_train_step", -1)}'
        )

    def _prepare_ped_traj_inputs(
        self,
        ped_traj_preds: Optional[torch.Tensor],
        ped_traj_mask: Optional[torch.Tensor],
        ped_traj_valid_steps: Optional[torch.Tensor],
        device: torch.device,
    ):
        if ped_traj_preds is None:
            return None, None, None

        if not torch.is_tensor(ped_traj_preds):
            ped_traj_preds = torch.as_tensor(ped_traj_preds)
        ped = ped_traj_preds.to(device=device, dtype=torch.float32, non_blocking=True)

        if ped.dim() == 3:
            ped = ped.unsqueeze(1)
        if ped.dim() != 4:
            raise ValueError(f'ped_traj_preds 應為 (B,M,T,F) 或 (B,T,F)，收到 shape={tuple(ped.shape)}')

        B, M, T, Fdim = ped.shape
        if M > self.max_ped_agents:
            ped = ped[:, :self.max_ped_agents]
            M = ped.shape[1]

        if Fdim < self.ped_traj_feat_dim:
            pad = torch.zeros(B, M, T, self.ped_traj_feat_dim - Fdim, device=device, dtype=ped.dtype)
            ped = torch.cat([ped, pad], dim=-1)
        elif Fdim > self.ped_traj_feat_dim:
            ped = ped[..., :self.ped_traj_feat_dim]

        if T < self.ped_frames:
            pad = torch.zeros(B, M, self.ped_frames - T, ped.shape[-1], device=device, dtype=ped.dtype)
            ped = torch.cat([ped, pad], dim=2)
        elif T > self.ped_frames:
            ped = ped[:, :, :self.ped_frames]

        finite = torch.isfinite(ped).all(dim=-1).all(dim=-1)
        ped = torch.nan_to_num(ped, nan=0.0, posinf=0.0, neginf=0.0)

        if not self.ped_use_goal and ped.shape[-1] >= 8:
            ped = ped.clone()
            ped[..., 6:8] = 0.0

        if ped_traj_mask is None:
            ped_mask = finite
        else:
            if not torch.is_tensor(ped_traj_mask):
                ped_traj_mask = torch.as_tensor(ped_traj_mask)
            ped_mask = ped_traj_mask.to(device=device, dtype=torch.bool, non_blocking=True)
            if ped_mask.dim() == 1:
                ped_mask = ped_mask.unsqueeze(0).expand(B, -1)
            ped_mask = ped_mask[:, :ped.shape[1]] & finite

        if ped_traj_valid_steps is None:
            step_mask = torch.isfinite(ped).all(dim=-1)
        else:
            if not torch.is_tensor(ped_traj_valid_steps):
                ped_traj_valid_steps = torch.as_tensor(ped_traj_valid_steps)
            step_mask = ped_traj_valid_steps.to(device=device, dtype=torch.bool, non_blocking=True)
            if step_mask.dim() == 2:
                step_mask = step_mask.unsqueeze(0).expand(B, -1, -1)
            step_mask = step_mask[:, :ped.shape[1], :ped.shape[2]]
        step_mask = step_mask & ped_mask[:, :, None]

        return ped, ped_mask, step_mask

    def encode_ped_traj(self, ped_traj_preds=None, ped_traj_mask=None, ped_traj_valid_steps=None):
        ped, ped_mask, step_mask = self._prepare_ped_traj_inputs(ped_traj_preds, ped_traj_mask, ped_traj_valid_steps, self.device)
        if ped is None:
            return None, None

        B, M, T, _ = ped.shape
        if not step_mask.any():
            return None, None

        valid_scalar = step_mask[:, :, :, None].to(ped.dtype)
        ped_feat = torch.cat([ped, valid_scalar], dim=-1)
        ped_tokens = self.ped_step_proj(ped_feat)

        agent_ids = torch.arange(M, device=ped.device).clamp(max=self.max_ped_agents - 1)
        ped_tokens = ped_tokens + self.ped_time_pe[:T].view(1, 1, T, -1)
        ped_tokens = ped_tokens + self.ped_agent_embed(agent_ids).view(1, M, 1, -1)
        ped_tokens = ped_tokens * step_mask[:, :, :, None].to(ped_tokens.dtype)

        ped_tokens_flat = ped_tokens.reshape(B, M * T, -1)
        denom = step_mask.sum(dim=(1, 2)).clamp(min=1).to(ped_tokens.dtype).unsqueeze(-1)
        ped_mean = ped_tokens.sum(dim=(1, 2)) / denom

        neg_inf = -1e4 if ped_tokens.dtype == torch.float16 else -1e9
        ped_mask_exp = step_mask[:, :, :, None]
        ped_max = ped_tokens.masked_fill(~ped_mask_exp, neg_inf).amax(dim=(1, 2))
        no_valid = ~step_mask.any(dim=(1, 2))
        ped_max = torch.where(no_valid.unsqueeze(1), torch.zeros_like(ped_max), ped_max)

        ped_ctx = self.ped_ctx_proj(torch.cat([ped_mean, ped_max], dim=-1))
        ped_tokens_flat = F.layer_norm(ped_tokens_flat, [ped_tokens_flat.shape[-1]])
        ped_ctx = F.layer_norm(ped_ctx, [ped_ctx.shape[-1]])
        return ped_tokens_flat, ped_ctx

    def generate_autoregressive(
        self,
        rgb_seq: np.ndarray,
        seg_seq: np.ndarray,
        ego_seq: torch.Tensor,
        commands: List[str],
        *,
        gt_trajs: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.0,
        ped_traj_preds: torch.Tensor = None,
        ped_traj_mask: torch.Tensor = None,
        ped_traj_valid_steps: torch.Tensor = None,
        ped_bev_map: torch.Tensor = None,
        bx: torch.Tensor = None,
        dx: torch.Tensor = None,
        bev_dim: List[int] = None,
    ) -> torch.Tensor:
        device = ego_seq.device
        B = ego_seq.shape[0]
        T = self.n_future
        self._debug_train_step = int(getattr(self, '_debug_train_step', 0))

        vis_tokens, bev_input = self.build_vis_tokens(rgb_seq, seg_seq, ego_seq)
        self._assert_finite_runtime_tensor('bev_input', bev_input)
        self._assert_finite_runtime_tensor('vis_tokens_pre_ped', vis_tokens)
        bev_map = self.bev_decoder(bev_input.to(torch.float32))
        if ped_bev_map is not None:
            if not torch.is_tensor(ped_bev_map):
                ped_bev_map = torch.as_tensor(ped_bev_map)
            ped_bev_map = ped_bev_map.to(device=device, dtype=torch.float32, non_blocking=True)
            self._assert_finite_ped_bev_tensor('ped_bev_map', ped_bev_map, ped_bev_map)
            if ped_bev_map.dim() == 3:
                ped_bev_map = ped_bev_map.unsqueeze(1)
            ped_bev_map = F.interpolate(ped_bev_map, size=bev_map.shape[-2:], mode='bilinear', align_corners=False)
            ped_bev_feat = self.ped_bev_encoder(ped_bev_map)
            ped_bev_feat = self.ped_bev_align_head(ped_bev_feat)
            self._assert_finite_ped_bev_tensor('ped_bev_feat', ped_bev_feat, ped_bev_map)
            ped_bev_gate = self.ped_bev_fuse_gate(torch.cat([bev_map, ped_bev_feat], dim=1))
            ped_bev_gate = self.ped_bev_gate_min + (1.0 - self.ped_bev_gate_min) * ped_bev_gate
            self._assert_finite_ped_bev_tensor('ped_bev_gate', ped_bev_gate, ped_bev_map)
            bev_map = bev_map + self.ped_bev_scale * ped_bev_gate * ped_bev_feat
            self.last_ped_bev_gate = ped_bev_gate.mean(dim=(1, 2, 3)).detach()
        else:
            self.last_ped_bev_gate = torch.zeros(B, device=device)
        self._assert_finite_runtime_tensor('bev_map_after_ped_fusion', bev_map)
        text_vis = self.encode_text_vis(commands)
        self._assert_finite_runtime_tensor('text_vis', text_vis)
        ped_tokens, ped_ctx = self.encode_ped_traj(ped_traj_preds, ped_traj_mask, ped_traj_valid_steps)
        if ped_ctx is None:
            ped_ctx = torch.zeros_like(text_vis)
            self.last_ped_gate = torch.zeros(B, T, device=device)

        text_vis_use = text_vis
        time_q_use = self.time_queries.unsqueeze(0).expand(B, -1, -1)

        if self.training:
            p_vis = float(getattr(self, 'qdrop_vis_p', 0.0))
            m = (torch.rand(vis_tokens.shape[:2], device=device) > p_vis).unsqueeze(-1).to(vis_tokens.dtype)
            vis_tokens = vis_tokens * m

            p_txt = float(getattr(self, 'qdrop_text_p', 0.0))
            if p_txt > 0:
                keep = (torch.rand(B, 1, device=device) > p_txt).to(text_vis.dtype)
                text_vis_use = text_vis * keep

            p_tq = float(getattr(self, 'qdrop_timeq_p', 0.0))
            if p_tq > 0:
                keep = (torch.rand(B, 1, 1, device=device) > p_tq).to(time_q_use.dtype)
                time_q_use = time_q_use * keep

        self._assert_finite_runtime_tensor('time_q_use', time_q_use)
        queries = time_q_use + text_vis_use.unsqueeze(1) + self.time_pe.unsqueeze(0)
        self._assert_finite_runtime_tensor('queries', queries)
        self._assert_finite_runtime_tensor('vis_tokens', vis_tokens)
        self._assert_finite_runtime_tensor('cross_attn.q_proj.weight', self.cross_attn.q_proj.weight)
        self._assert_finite_runtime_tensor('cross_attn.k_proj.weight', self.cross_attn.k_proj.weight)
        self._assert_finite_runtime_tensor('cross_attn.v_proj.weight', self.cross_attn.v_proj.weight)
        self._assert_finite_runtime_tensor('cross_attn.out_proj.weight', self.cross_attn.out_proj.weight)
        fused_vis_only = self.cross_attn(queries, vis_tokens)
        self._assert_finite_runtime_tensor('fused_vis_only', fused_vis_only)
        fused = fused_vis_only

        if ped_tokens is not None:
            ped_queries = queries + ped_ctx.unsqueeze(1)
            self._assert_finite_runtime_tensor('ped_queries', ped_queries)
            self._assert_finite_runtime_tensor('ped_tokens', ped_tokens)
            ped_fused = self.cross_attn(ped_queries, ped_tokens)
            self._assert_finite_runtime_tensor('ped_fused', ped_fused)
            ped_gate_raw = self.ped_fuse_gate(torch.cat([fused_vis_only, ped_fused], dim=-1))
            ped_gate = self.ped_gate_min + (1.0 - self.ped_gate_min) * ped_gate_raw

            step_idx = torch.arange(T, device=device, dtype=fused_vis_only.dtype)
            center = float(min(max(self.ped_start_step, 0), max(T - 1, 0)))
            if T <= 1:
                ped_active = torch.ones(B, T, 1, device=device, dtype=fused_vis_only.dtype)
            else:
                steepness = max(self.ped_ramp_steepness, 1e-3)
                ped_active_1d = torch.sigmoid((step_idx - center) / steepness)
                ped_active = ped_active_1d.view(1, T, 1).expand(B, -1, -1)

            ped_gate = ped_gate * ped_active
            fused = fused_vis_only + ped_gate * ped_fused
            self.last_ped_gate = ped_gate.squeeze(-1).detach()
        else:
            self.last_ped_gate = torch.zeros(B, T, device=device)
        self._assert_finite_runtime_tensor('fused_after_ped', fused)

        global_ctx = fused.mean(dim=1)
        h = self.h0_proj(torch.cat([global_ctx, text_vis], dim=-1))
        self._assert_finite_runtime_tensor('decoder_h0', h)

        coarse_xy = self.make_coarse_baseline(ego_seq, T=T)

        coarse_keep_t = None
        if self.training:
            p_cd = float(self.coarse_drop_p)
            if p_cd > 0:
                coarse_keep_t = (torch.rand(B, T, 1, device=device) > p_cd).to(coarse_xy.dtype)
                coarse_keep_t[:, 0, :] = 1.0

        self.last_coarse_xy = coarse_xy
        self.last_coarse_keep_t = coarse_keep_t.detach() if coarse_keep_t is not None else None

        traj_xy = []
        gate_stats = []
        mix_gate_stats = []
        ar_traj = []
        prev_xy = coarse_xy[:, 0, :]

        if gt_trajs is not None and gt_trajs.size(-1) >= 2:
            gt_xy = gt_trajs[..., :2]
        else:
            gt_xy = None

        ar_xy = coarse_xy[:, 0, :].clone()

        for t in range(T):
            ctx_t = fused[:, t, :]
            embedded_xy = self.xy_post_ln(self.xy_embedder(prev_xy))
            g_ctx = self.g_proj(torch.cat([embedded_xy, ctx_t], dim=-1))
            ctx_gated = g_ctx * ctx_t + (1.0 - g_ctx) * embedded_xy

            if bx is not None and dx is not None and bev_dim is not None:
                W_occ, H_occ = bev_dim[1], bev_dim[0]
                yy = (prev_xy[..., 1] - bx[0]) / dx[0]
                xx = (prev_xy[..., 0] - bx[1]) / dx[1]
                y_norm = (yy / (H_occ - 1)) * 2 - 1
                x_norm = (xx / (W_occ - 1)) * 2 - 1
                grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1).unsqueeze(1)
                local_bev = F.grid_sample(
                    bev_map, grid, mode='bilinear', padding_mode='border', align_corners=True
                ).squeeze(-1).squeeze(-1)
            else:
                local_bev = torch.zeros(B, self.vis_width, device=device)

            dec_in = torch.cat([embedded_xy, ctx_gated, local_bev], dim=-1)
            dec_in = self.dec_in_ln(dec_in)
            gate_stats.append(g_ctx.mean(dim=-1))
            self._assert_finite_runtime_tensor(f'dec_in_t{t}', dec_in)

            h = self.decoder_gru(dec_in, h)
            self._assert_finite_runtime_tensor(f'decoder_h_t{t}', h)
            delta = self.delta_head(h)
            delta = torch.tanh(delta) * self.max_step
            self._assert_finite_runtime_tensor(f'delta_t{t}', delta)
            ar_xy = ar_xy + delta
            self._assert_finite_runtime_tensor(f'ar_xy_t{t}', ar_xy)
            ar_traj.append(ar_xy.unsqueeze(1))

            g_t = self.mix_gate(torch.cat([h, ctx_t], dim=-1))
            if self.training and coarse_keep_t is not None:
                drop_mask = coarse_keep_t[:, t, :] < 0.5
                g_t = torch.where(drop_mask, torch.ones_like(g_t), g_t)

            coarse_t = coarse_xy[:, t, :]
            final_xy_t = g_t * ar_xy + (1.0 - g_t) * coarse_t
            self._assert_finite_runtime_tensor(f'final_xy_t{t}', final_xy_t)

            traj_xy.append(final_xy_t.unsqueeze(1))
            mix_gate_stats.append(g_t.squeeze(-1))

            prev_xy_next = final_xy_t
            if self.training and gt_xy is not None and teacher_forcing_ratio > 0.0 and (t < T - 1):
                use_tf = (torch.rand(B, 1, device=device) < teacher_forcing_ratio).to(final_xy_t.dtype)
                gt_in = gt_xy[:, t, :]
                prev_xy_next = use_tf * gt_in + (1.0 - use_tf) * final_xy_t
                ar_xy = use_tf * gt_in + (1.0 - use_tf) * ar_xy

            prev_xy = prev_xy_next

        xy = torch.cat(traj_xy, dim=1)
        self.last_mix_gate = torch.stack(mix_gate_stats, dim=1).detach()
        self.last_gate_stats = torch.stack(gate_stats, dim=1).detach()
        self.last_ar_xy = torch.cat(ar_traj, dim=1).detach() if ar_traj else None

        z = torch.zeros(B, T, 1, device=device)
        return torch.cat([xy, z], dim=-1)

    def generate(
        self,
        rgb_seq,
        seg_seq,
        ego_seq,
        commands,
        gt_trajs: torch.Tensor = None,
        ped_traj_preds: torch.Tensor = None,
        ped_traj_mask: torch.Tensor = None,
        ped_traj_valid_steps: torch.Tensor = None,
        ped_bev_map: torch.Tensor = None,
        bx=None,
        dx=None,
        bev_dim=None,
    ):
        tf_ratio = float(getattr(self, 'AR_TF_RATIO', 0.0))
        if not self.training:
            tf_ratio = 0.0

        if ped_traj_preds is None:
            ped_traj_preds = self.cached_ped_traj_preds
        if ped_traj_mask is None:
            ped_traj_mask = self.cached_ped_traj_mask
        ped_traj_valid_steps = getattr(self, 'cached_ped_traj_valid_steps', None) if ped_traj_valid_steps is None else ped_traj_valid_steps
        if ped_bev_map is None:
            ped_bev_map = getattr(self, 'cached_ped_bev_map', None)

        return self.generate_autoregressive(
            rgb_seq,
            seg_seq,
            ego_seq,
            commands,
            gt_trajs=gt_trajs,
            teacher_forcing_ratio=tf_ratio,
            ped_traj_preds=ped_traj_preds,
            ped_traj_mask=ped_traj_mask,
            ped_traj_valid_steps=ped_traj_valid_steps,
            ped_bev_map=ped_bev_map,
            bx=bx,
            dx=dx,
            bev_dim=bev_dim,
        )


class VLM_STP3_Gen(BaseVLM_STP3_Gen):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD
        self.n_future = cfg.N_FUTURE_FRAMES

        self.input_size = int(getattr(cfg, 'CLIP_INPUT_SIZE', 224))
        clip_name = getattr(cfg, 'CLIP_MODEL', 'ViT-B-32')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vlm = VLM_Generative(clip_name, device, n_future=self.n_future, input_size=self.input_size)

        dx, bx, _ = gen_dx_bx(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
        dx, bx = dx[:2], bx[:2]
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        _, _, bev_dim = calculate_birds_eye_view_parameters(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
        self.bev_dim = bev_dim.numpy().tolist()

        self._last_rgb_seq = None
        self._last_seg_seq = None
        self._last_ego_seq = None
        self._last_ped_traj_preds = None
        self._last_ped_traj_mask = None
        self._last_ped_traj_valid_steps = None

        self.encoder_out_channels = 64
        self.fake_cam_front = nn.Parameter(torch.zeros(1, self.encoder_out_channels, 60, 28), requires_grad=False)

        self.vlm.AR_TF_RATIO = float(getattr(cfg, 'AR_TF_RATIO', 0.5))
        self.vlm.max_step = float(getattr(cfg, 'AR_MAX_STEP', 5.0))
        self._train_step = 0

        self.vlm.EGO_SCALE_M = float(getattr(cfg, 'EGO_SCALE_M', 5.0))
        self.vlm.SAMPLE_DT = float(getattr(cfg, 'SAMPLE_DT', 0.5))
        self.vlm.VIS_UNFREEZE_K = int(getattr(cfg, 'VIS_UNFREEZE_K', 1))
        self.vlm.apply_visual_unfreeze(self.vlm.VIS_UNFREEZE_K)

        print('model : image_force_ped_traj')
        print('dx:', self.dx, 'bx:', self.bx)
        print('ped_cfg:')
        print('  LOSS_PED_REPULSE_W =', float(getattr(self.cfg, 'LOSS_PED_REPULSE_W', 16.0)))
        print('  LOSS_PED_REPULSE_AR_ALPHA =', float(getattr(self.cfg, 'LOSS_PED_REPULSE_AR_ALPHA', 1.0)))
        print('  LOSS_PED_REPULSE_SAFE_DIST =', float(getattr(self.cfg, 'LOSS_PED_REPULSE_SAFE_DIST', 6.0)))
        print('  PED_REPULSE_WARMUP_STEPS =', int(getattr(self.cfg, 'PED_REPULSE_WARMUP_STEPS', 0)))
        print('  PED_REPULSE_RAMP_STEPS =', int(getattr(self.cfg, 'PED_REPULSE_RAMP_STEPS', 1000)))
        print('  PED_GATE_MIN =', float(getattr(self.vlm, 'ped_gate_min', 0.10)))
        print('  PED_GATE_INIT_BIAS =', float(getattr(self.vlm, 'ped_gate_init_bias', 0.0)))
        print('  PED_START_STEP =', int(getattr(self.vlm, 'ped_start_step', 1)))
        print('  PED_RAMP_STEEPNESS =', float(getattr(self.vlm, 'ped_ramp_steepness', 2.0)))
        print('  PED_BEV_GATE_MIN =', float(getattr(self.vlm, 'ped_bev_gate_min', 0.05)))
        print('  PED_BEV_LATERAL_LIMIT_M =', float(getattr(self.cfg, 'PED_BEV_LATERAL_LIMIT_M', 15.0)))
        print('  PED_BEV_FORWARD_MIN_M =', float(getattr(self.cfg, 'PED_BEV_FORWARD_MIN_M', 0.0)))
        print('  PED_BEV_FORWARD_MAX_M =', float(getattr(self.cfg, 'PED_BEV_FORWARD_MAX_M', 30.0)))

    def current_ped_repulse_weight(self):
        target_w = float(getattr(self.cfg, 'LOSS_PED_REPULSE_W', 16.0))
        warmup_steps = int(getattr(self.cfg, 'PED_REPULSE_WARMUP_STEPS', 0))
        ramp_steps = int(getattr(self.cfg, 'PED_REPULSE_RAMP_STEPS', 1000))
        if not self.training:
            return target_w

        step = int(self._train_step)
        if step < warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            return target_w
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(ramp_steps)))
        return target_w * progress

    def set_pedestrian_trajectory_predictions(self, ped_traj_preds=None, ped_traj_mask=None, ped_traj_valid_steps=None, ped_bev_map=None):
        self._last_ped_traj_preds = ped_traj_preds
        self._last_ped_traj_mask = ped_traj_mask
        self._last_ped_traj_valid_steps = ped_traj_valid_steps
        self._last_ped_bev_map = ped_bev_map
        self.vlm.cached_ped_traj_preds = ped_traj_preds
        self.vlm.cached_ped_traj_mask = ped_traj_mask
        self.vlm.cached_ped_traj_valid_steps = ped_traj_valid_steps
        self.vlm.cached_ped_bev_map = ped_bev_map

    def _build_box_sample_offsets(self, device, dtype):
        half_w = float(self.cfg.EGO.WIDTH) * 0.5
        half_h = float(self.cfg.EGO.HEIGHT) * 0.5
        lat = torch.tensor([-half_w, 0.0, half_w], device=device, dtype=dtype)
        lon = torch.tensor([-half_h, 0.0, half_h], device=device, dtype=dtype)
        yy, xx = torch.meshgrid(lon, lat, indexing='ij')
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def box_collision_loss_soft(self, trajs_xy, occupancy, return_per_t=False):
        """
        Soft surrogate for ego-box collision.
        Samples a small grid over the ego footprint and aggregates occupancy with a soft union.
        """
        device = trajs_xy.device
        dtype = trajs_xy.dtype
        occupancy = occupancy.to(device=device, dtype=torch.float32)
        B, T, _ = trajs_xy.shape
        H, W = occupancy.shape[-2:]

        offsets = self._build_box_sample_offsets(device, dtype)
        sample_points = trajs_xy.unsqueeze(2) + offsets.view(1, 1, -1, 2)

        xx = sample_points[..., 0]
        yy = sample_points[..., 1]

        x_norm = ((xx - self.bx[1]) / self.dx[1])
        y_norm = ((yy - self.bx[0]) / self.dx[0])
        x_norm = (x_norm / max(1, W - 1)) * 2.0 - 1.0
        y_norm = (y_norm / max(1, H - 1)) * 2.0 - 1.0

        grid = torch.stack([x_norm, y_norm], dim=-1).view(B * T, -1, 1, 2)
        occ = occupancy.view(B * T, 1, H, W)
        sampled = F.grid_sample(occ, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled = sampled.view(B, T, -1).clamp(0.0, 1.0)

        risk_t = 1.0 - torch.prod(1.0 - sampled, dim=-1)
        if return_per_t:
            return risk_t

        risk_pow = float(getattr(self.cfg, 'BOX_RISK_POW', 2.0))
        risk_mode = str(getattr(self.cfg, 'BOX_RISK_REDUCE', 'topk'))
        topk_frac = float(getattr(self.cfg, 'BOX_RISK_TOPK_FRAC', 0.34))

        r = risk_t.clamp(min=0.0) ** risk_pow
        if risk_mode == 'mean':
            return r.mean()
        if risk_mode == 'max':
            return r.max(dim=1).values.mean()
        if risk_mode == 'topk':
            k = max(1, int(round(T * topk_frac)))
            topk_vals, _ = torch.topk(r, k=k, dim=1, largest=True, sorted=False)
            return topk_vals.mean()
        raise ValueError(f'Unknown BOX_RISK_REDUCE={risk_mode}')

    def dynamic_repulsion_loss(self, trajs_xy, ped_bev_points, ped_bev_valid_steps, return_per_t=False):
        if ped_bev_points is None or ped_bev_valid_steps is None:
            if return_per_t:
                return torch.zeros(trajs_xy.shape[:2], device=trajs_xy.device, dtype=trajs_xy.dtype)
            return torch.zeros((), device=trajs_xy.device, dtype=trajs_xy.dtype)

        ped_bev_points = ped_bev_points.to(device=trajs_xy.device, dtype=trajs_xy.dtype)
        ped_bev_valid_steps = ped_bev_valid_steps.to(device=trajs_xy.device, dtype=torch.bool)
        if ped_bev_points.dim() != 4 or ped_bev_valid_steps.dim() != 3:
            raise ValueError('ped_bev_points should be (B,M,T,2) and ped_bev_valid_steps should be (B,M,T)')

        B, T, _ = trajs_xy.shape
        ped_bev_points = ped_bev_points[:, :, :T]
        ped_bev_valid_steps = ped_bev_valid_steps[:, :, :T]

        ego = trajs_xy.unsqueeze(1)
        dist = torch.norm(ego - ped_bev_points, dim=-1)
        safe_dist = float(getattr(self.cfg, 'LOSS_PED_REPULSE_SAFE_DIST', 8.0))
        penalty = F.relu(safe_dist - dist) ** 2
        penalty = penalty * ped_bev_valid_steps.to(penalty.dtype)

        has_valid_t = ped_bev_valid_steps.any(dim=1)
        penalty_t = penalty.max(dim=1).values
        penalty_t = torch.where(has_valid_t, penalty_t, torch.zeros_like(penalty_t))
        if return_per_t:
            return penalty_t

        valid_count = has_valid_t.sum().clamp(min=1).to(trajs_xy.dtype)
        return penalty_t.sum() / valid_count

    def ped_bev_debug_stats(self, trajs_xy, ped_bev_points, ped_bev_valid_steps):
        zero = torch.zeros((), device=trajs_xy.device, dtype=trajs_xy.dtype)
        stats = {
            'ped_bev_valid_ratio': zero,
            'ped_min_dist_on_valid': zero,
            'ped_bev_xy_abs_mean': zero,
            'ego_xy_abs_mean': trajs_xy.abs().mean().detach(),
        }
        if ped_bev_points is None or ped_bev_valid_steps is None:
            return stats

        ped_bev_points = ped_bev_points.to(device=trajs_xy.device, dtype=trajs_xy.dtype)
        ped_bev_valid_steps = ped_bev_valid_steps.to(device=trajs_xy.device, dtype=torch.bool)
        if ped_bev_points.dim() != 4 or ped_bev_valid_steps.dim() != 3:
            return stats

        T = trajs_xy.shape[1]
        ped_bev_points = ped_bev_points[:, :, :T]
        ped_bev_valid_steps = ped_bev_valid_steps[:, :, :T]

        stats['ped_bev_valid_ratio'] = ped_bev_valid_steps.float().mean().detach()
        if ped_bev_valid_steps.any():
            valid_points = ped_bev_points[ped_bev_valid_steps]
            stats['ped_bev_xy_abs_mean'] = valid_points.abs().mean().detach()

            ego = trajs_xy.unsqueeze(1)
            dist = torch.norm(ego - ped_bev_points, dim=-1)
            inf = torch.full_like(dist, float('inf'))
            dist_valid = torch.where(ped_bev_valid_steps, dist, inf)
            min_dist_t = dist_valid.amin(dim=1)
            has_valid_t = ped_bev_valid_steps.any(dim=1)
            if has_valid_t.any():
                min_dist_valid = min_dist_t[has_valid_t]
                if torch.isfinite(min_dist_valid).any():
                    stats['ped_min_dist_on_valid'] = min_dist_valid[torch.isfinite(min_dist_valid)].mean().detach()

        return stats

    def planning(self, *, bev_rgbs, trajs, gt_trajs, commands, target_points, occupancy=None, ped_bev_points=None, ped_bev_valid_steps=None):
        if self.training:
            self._train_step += 1
        self.vlm._debug_train_step = int(self._train_step)

        loss, pred, final_traj, resample_traj, loss_dict = super().planning(
            bev_rgbs=bev_rgbs,
            trajs=trajs,
            gt_trajs=gt_trajs,
            commands=commands,
            target_points=target_points,
            occupancy=occupancy,
        )

        if occupancy is not None:
            pred_xy = pred[..., :2]
            ar_xy = getattr(self.vlm, 'last_ar_xy', None)
            if ar_xy is None:
                ar_xy = pred_xy

            box_risk_pred_t = self.box_collision_loss_soft(pred_xy, occupancy, return_per_t=True)
            box_risk_ar_t = self.box_collision_loss_soft(ar_xy, occupancy, return_per_t=True)

            box_coll_pred = self.box_collision_loss_soft(pred_xy, occupancy, return_per_t=False)
            box_coll_ar = self.box_collision_loss_soft(ar_xy, occupancy, return_per_t=False)

            alpha_box_ar = float(getattr(self.cfg, 'LOSS_BOX_COL_AR_ALPHA', 1.0))
            lam_box = float(getattr(self.cfg, 'LOSS_BOX_COL_W', 10.0))
            box_coll = box_coll_pred + alpha_box_ar * box_coll_ar

            loss = loss + lam_box * box_coll
            loss_dict['box_collision'] = box_coll.detach()
            loss_dict['box_collision_pred'] = box_coll_pred.detach()
            loss_dict['box_collision_ar'] = box_coll_ar.detach()
            loss_dict['box_col_with_lam'] = (lam_box * box_coll).detach()
            loss_dict['box_risk_mean'] = box_risk_pred_t.mean().detach()
            loss_dict['box_risk_max'] = box_risk_pred_t.max(dim=1).values.mean().detach()
            if getattr(self.vlm, 'last_ped_bev_gate', None) is not None:
                loss_dict['ped_bev_gate_mean'] = self.vlm.last_ped_bev_gate.mean().detach()

        if ped_bev_points is not None and ped_bev_valid_steps is not None:
            pred_xy = pred[..., :2]
            ar_xy = getattr(self.vlm, 'last_ar_xy', None)
            if ar_xy is None:
                ar_xy = pred_xy

            repulse_pred_t = self.dynamic_repulsion_loss(
                pred_xy, ped_bev_points, ped_bev_valid_steps, return_per_t=True
            )
            repulse_pred = self.dynamic_repulsion_loss(
                pred_xy, ped_bev_points, ped_bev_valid_steps, return_per_t=False
            )
            repulse_ar = self.dynamic_repulsion_loss(
                ar_xy, ped_bev_points, ped_bev_valid_steps, return_per_t=False
            )

            alpha_repulse_ar = float(getattr(self.cfg, 'LOSS_PED_REPULSE_AR_ALPHA', 0.5))
            lam_repulse = self.current_ped_repulse_weight()
            repulse = repulse_pred + alpha_repulse_ar * repulse_ar

            loss = loss + lam_repulse * repulse
            loss_dict['ped_repulse'] = repulse.detach()
            loss_dict['ped_repulse_pred'] = repulse_pred.detach()
            loss_dict['ped_repulse_ar'] = repulse_ar.detach()
            loss_dict['ped_repulse_with_lam'] = (lam_repulse * repulse).detach()
            loss_dict['ped_repulse_w_current'] = torch.tensor(lam_repulse, device=pred_xy.device, dtype=pred_xy.dtype)
            loss_dict['ped_repulse_t_mean'] = repulse_pred_t.mean().detach()
            loss_dict['ped_repulse_t_max'] = repulse_pred_t.max(dim=1).values.mean().detach()

            debug_stats = self.ped_bev_debug_stats(pred_xy, ped_bev_points, ped_bev_valid_steps)
            for k, v in debug_stats.items():
                loss_dict[k] = v.detach()
        else:
            pred_xy = pred[..., :2]
            loss_dict['ped_bev_valid_ratio'] = torch.zeros((), device=pred_xy.device, dtype=pred_xy.dtype)
            loss_dict['ped_min_dist_on_valid'] = torch.zeros((), device=pred_xy.device, dtype=pred_xy.dtype)
            loss_dict['ped_bev_xy_abs_mean'] = torch.zeros((), device=pred_xy.device, dtype=pred_xy.dtype)
            loss_dict['ego_xy_abs_mean'] = pred_xy.abs().mean().detach()

        return loss, pred, final_traj, resample_traj, loss_dict

    def forward(
        self,
        image,
        intrinsics,
        extrinsics,
        future_egomotion,
        *,
        rgb_224_seq,
        seg_224_seq,
        ped_traj_preds=None,
        ped_traj_mask=None,
        ped_traj_valid_steps=None,
        ped_bev_map=None,
    ):
        B, T_rf = rgb_224_seq.shape[:2]
        device = future_egomotion.device

        self._last_rgb_seq = rgb_224_seq.to(device, non_blocking=True)
        self._last_seg_seq = seg_224_seq.to(device, non_blocking=True)
        self.set_pedestrian_trajectory_predictions(ped_traj_preds, ped_traj_mask, ped_traj_valid_steps, ped_bev_map)

        fego = future_egomotion[:, :self.receptive_field, :]
        ego_seq_embed = []
        for t in range(T_rf):
            if t == T_rf - 1:
                dx = torch.zeros(B, 1, device=device)
                dy = torch.zeros(B, 1, device=device)
                dyaw = torch.zeros(B, 1, device=device)
            else:
                mats = [pose_vec2mat(fego[:, k, :]) for k in range(t, T_rf - 1)]
                M = mats[0]
                for m in mats[1:]:
                    M = torch.bmm(M, m)
                pose = mat2pose_vec(M)
                dx = pose[:, 0:1]
                dy = pose[:, 1:2]
                dyaw = pose[:, 5:6]
                dyaw = (dyaw + torch.pi) % (2 * torch.pi) - torch.pi
            s_m = getattr(self.cfg, 'EGO_SCALE_M', 5.0)
            ex, ey = dx / s_m, dy / s_m
            sy, cy = torch.sin(dyaw), torch.cos(dyaw)
            ego_seq_embed.append(torch.cat([ex, ey, sy, cy], dim=1))
        self._last_ego_seq = torch.stack(ego_seq_embed, dim=1).detach()

        return {}, self._last_rgb_seq
