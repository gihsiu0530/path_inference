"""ASAP-FF with a direct ego-motion extrapolation coarse trajectory.

This additive variant does not instantiate or load AD-MLP.  The coarse path is
obtained by rolling the recent ego-frame translation and yaw increment forward
as a constant SE(2) motion, then the existing FF network predicts only bounded
coarse-normal corrections.
"""

import math

import torch
import torch.nn as nn

from stp3.model_vlm.codex_pure_ASAP_ff import (
    FastTrajectoryPlannerFF,
    VLM_STP3_Gen as BaseFFVLM_STP3_Gen,
)
from stp3.utils.geometry import calculate_birds_eye_view_parameters
from stp3.utils.tools import gen_dx_bx


class VLM_STP3_Gen(BaseFFVLM_STP3_Gen):
    """FF planner whose coarse path is constant-motion extrapolation."""

    def __init__(self, cfg):
        # Deliberately bypass BaseFFVLM_STP3_Gen.__init__, because its parent
        # constructs and loads the frozen AD-MLP planner.
        nn.Module.__init__(self)
        self.cfg = cfg
        self.receptive_field = int(cfg.TIME_RECEPTIVE_FIELD)
        self.n_future = int(cfg.N_FUTURE_FRAMES)
        self.input_size = int(getattr(cfg, "CLIP_INPUT_SIZE", 224))
        self.vlm = FastTrajectoryPlannerFF(cfg, self.n_future, self.input_size)

        dx, bx, _ = gen_dx_bx(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
        self.dx = nn.Parameter(dx[:2], requires_grad=False)
        self.bx = nn.Parameter(bx[:2], requires_grad=False)
        _, _, bev_dim = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )
        self.bev_dim = bev_dim.numpy().tolist()
        self.encoder_out_channels = 64
        self.fake_cam_front = nn.Parameter(
            torch.zeros(1, self.encoder_out_channels, 60, 28), requires_grad=False
        )

        self._last_rgb_seq = None
        self._last_seg_seq = None
        self._last_seg_id_seq = None
        self._last_depth_seq = None
        self._last_ego_seq = None
        self._last_admlp_ego_motion = None  # compatibility name used by inherited forward
        self._last_admlp_input = None
        self._last_admlp_xy = None
        self._last_intrinsics_ff = None
        self._last_extrinsics_ff = None
        self._last_extrapolation_motion_ff = None
        self.last_extrap_speed_mps = torch.tensor(0.0)
        self.last_extrap_yaw_rate_deg_s = torch.tensor(0.0)

        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        print("model override: codex_pure_fast_ASAP_extrap_ff (no AD-MLP)")
        print(f"Total parameters: {total:,}  Trainable parameters: {trainable:,}")
        self._print_planning_loss_weights()
        if str(getattr(cfg, "FF_DEPTH_MODE", "relative_inverse")) != "metric":
            print("[FF WARNING] FF_DEPTH_MODE is not metric; the no-HD-map camera BEV is pseudo-metric.")

    @staticmethod
    def _strip_current_zero_sentinel(motion):
        if motion is None or not torch.is_tensor(motion) or motion.dim() != 3:
            return motion
        if motion.shape[1] > 1 and bool((motion[:, -1].abs().amax(dim=-1) < 1e-7).all().item()):
            return motion[:, :-1]
        return motion

    def forward(self, image, intrinsics, extrinsics, future_egomotion, **kwargs):
        ego_history = kwargs.get("ego_history_egomotion")
        if torch.is_tensor(ego_history) and ego_history.numel() > 0:
            extrapolation_motion = self._strip_current_zero_sentinel(ego_history)
        else:
            extrapolation_motion = self._strip_current_zero_sentinel(
                future_egomotion[:, : self.receptive_field]
            )
        output = super().forward(
            image, intrinsics, extrinsics, future_egomotion, **kwargs
        )
        # ADMLPFeatureDataset is intentionally not used by the extrapolation
        # train/validation entry points. Restore the raw ego motion regardless
        # of any compatibility field passed by an external caller.
        self._last_extrapolation_motion_ff = extrapolation_motion
        self._last_admlp_ego_motion = extrapolation_motion
        self._last_admlp_input = None
        return output

    def _constant_motion_coarse(self, device, dtype):
        motion = self._last_extrapolation_motion_ff
        if motion is None or not torch.is_tensor(motion) or motion.numel() == 0:
            raise RuntimeError(
                "Direct extrapolation requires ego_history_egomotion or future_egomotion."
            )
        motion = motion.to(device=device, dtype=dtype)[..., :6]
        history_steps = max(1, int(getattr(self.cfg, "FF_EXTRAP_HISTORY_STEPS", 4)))
        motion = motion[:, -history_steps:]
        body_steps = motion[..., :2]
        yaw_steps = motion[..., 5]
        if bool(getattr(self.cfg, "FF_EXTRAP_INVERT_DATASET_EGOMOTION", True)):
            # NuscenesData stores inv(pose_t1) @ pose_t0: the previous frame
            # expressed in the new frame. Invert each SE(2) increment to obtain
            # the forward vehicle motion t0 -> t1 before extrapolation.
            cosine, sine = torch.cos(yaw_steps), torch.sin(yaw_steps)
            stored_x, stored_y = body_steps[..., 0], body_steps[..., 1]
            forward_x = -cosine * stored_x - sine * stored_y
            forward_y = sine * stored_x - cosine * stored_y
            body_steps = torch.stack([forward_x, forward_y], dim=-1)
            yaw_steps = -yaw_steps
        decay = min(max(float(getattr(self.cfg, "FF_EXTRAP_RECENCY_DECAY", 0.7)), 1e-3), 1.0)
        age = torch.arange(motion.shape[1] - 1, -1, -1, device=device, dtype=dtype)
        weights = decay ** age
        weights = weights / weights.sum().clamp_min(1e-6)
        body_step = (body_steps * weights.view(1, -1, 1)).sum(dim=1)
        yaw_step = (yaw_steps * weights.view(1, -1)).sum(dim=1)

        dt = float(getattr(self.cfg, "SAMPLE_INTERVAL", 0.5))
        max_speed = float(getattr(self.cfg, "FF_EXTRAP_MAX_SPEED_MPS", 12.0))
        step_norm = body_step.norm(dim=-1, keepdim=True)
        maximum_step = max_speed * max(dt, 1e-6)
        body_step = body_step * (maximum_step / step_norm.clamp_min(1e-6)).clamp(max=1.0)
        max_yaw_rate = math.radians(float(getattr(self.cfg, "FF_EXTRAP_MAX_YAW_RATE_DEG_S", 60.0)))
        yaw_step = yaw_step.clamp(
            min=-max_yaw_rate * dt, max=max_yaw_rate * dt
        )

        forward = torch.zeros(motion.shape[0], device=device, dtype=dtype)
        left = torch.zeros_like(forward)
        heading = torch.zeros_like(forward)
        points = []
        for _ in range(self.n_future):
            cosine, sine = torch.cos(heading), torch.sin(heading)
            delta_forward = cosine * body_step[:, 0] - sine * body_step[:, 1]
            delta_left = sine * body_step[:, 0] + cosine * body_step[:, 1]
            forward = forward + delta_forward
            left = left + delta_left
            heading = heading + yaw_step
            # Planner coordinates are (lateral-right, forward).
            points.append(torch.stack([-left, forward], dim=-1))
        coarse = torch.stack(points, dim=1)
        self.last_extrap_speed_mps = body_step.norm(dim=-1) / max(dt, 1e-6)
        self.last_extrap_yaw_rate_deg_s = torch.rad2deg(yaw_step / max(dt, 1e-6))
        return coarse

    def _augment_extrapolated_coarse(self, coarse, device, dtype):
        if not self.training or not bool(getattr(self.cfg, "FF_COARSE_AUG_ENABLED", True)):
            return coarse
        probability = float(getattr(self.cfg, "FF_COARSE_AUG_PROB", 0.75))
        active = (torch.rand(coarse.shape[0], 1, 1, device=device) < probability).to(dtype)
        lateral_max = float(getattr(self.cfg, "FF_COARSE_AUG_LATERAL_M", 1.0))
        yaw_max = math.radians(float(getattr(self.cfg, "FF_COARSE_AUG_YAW_DEG", 5.0)))
        scale_max = float(getattr(self.cfg, "FF_COARSE_AUG_SPEED_RATIO", 0.15))
        point_noise = float(getattr(self.cfg, "FF_COARSE_AUG_POINT_M", 0.20))
        lateral = (torch.rand(coarse.shape[0], 1, device=device, dtype=dtype) * 2.0 - 1.0) * lateral_max
        yaw = (torch.rand(coarse.shape[0], 1, device=device, dtype=dtype) * 2.0 - 1.0) * yaw_max
        scale = 1.0 + (torch.rand(coarse.shape[0], 1, device=device, dtype=dtype) * 2.0 - 1.0) * scale_max
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        x, y = coarse[..., 0], coarse[..., 1]
        rotated = torch.stack([cosine * x - sine * y, sine * x + cosine * y], dim=-1)
        rotated = rotated * scale.unsqueeze(-1)
        _, _, normal, _, _ = FastTrajectoryPlannerFF._coarse_geometry(
            rotated, self.vlm.SAMPLE_DT
        )
        smooth_noise = torch.randn_like(x).cumsum(dim=1)
        divisor = torch.arange(
            1, x.shape[1] + 1, device=device, dtype=dtype
        ).sqrt().view(1, -1)
        smooth_noise = smooth_noise / divisor
        augmented = rotated + (lateral + point_noise * smooth_noise).unsqueeze(-1) * normal
        return coarse + active * (augmented - coarse)

    def _admlp_coarse(self, device, dtype, commands):
        del commands
        coarse = self._constant_motion_coarse(device, dtype)
        self._last_admlp_xy = coarse.detach()  # compatibility diagnostic name
        return self._augment_extrapolated_coarse(coarse, device, dtype)

    def planning(self, **kwargs):
        loss, prediction, final_traj, _, loss_dict = super().planning(**kwargs)
        device = final_traj.device
        loss_dict["ff_extrap_speed_mps"] = self.last_extrap_speed_mps.mean().to(device)
        loss_dict["ff_extrap_yaw_rate_deg_s"] = self.last_extrap_yaw_rate_deg_s.abs().mean().to(device)
        return loss, prediction, final_traj, "FAST_EXTRAP_FF", loss_dict
