from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ADMLPPlanner(nn.Module):
    COMMAND_TO_ID = {"LEFT": 0, "FORWARD": 1, "RIGHT": 2}

    def __init__(self, cfg, n_future: int):
        super().__init__()
        self.cfg = cfg
        self.n_future = int(n_future)
        self.n_output = self.n_future + 1
        self.feature_mode = str(getattr(cfg, "ADMLP_FEATURE_MODE", "past4_command"))
        self.past_frames = int(getattr(cfg, "ADMLP_PAST_FRAMES", 4))
        self.dt = float(getattr(cfg, "SAMPLE_INTERVAL", 0.5))
        hidden_dim = int(getattr(cfg, "ADMLP_HIDDEN_DIM", 512))
        if self.feature_mode == "past5_no_command":
            in_dim = self.past_frames * 3 + 3 + 3
        elif self.feature_mode == "past4_command":
            in_dim = self.past_frames * 3 + 3 + 3 + 3
        else:
            raise ValueError(f"Unsupported ADMLP_FEATURE_MODE={self.feature_mode!r}")

        # Same planner shape as AD-MLP: Linear-512-ReLU-Linear-512-ReLU-Linear.
        self.plan_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.n_output * 3),
        )

    def _command_one_hot(self, commands: List[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ids = []
        for command in commands:
            if command not in self.COMMAND_TO_ID:
                raise ValueError(f"Unsupported command {command!r}; expected LEFT, FORWARD, or RIGHT.")
            ids.append(self.COMMAND_TO_ID[command])
        return F.one_hot(torch.tensor(ids, device=device), num_classes=3).to(dtype=dtype)

    @staticmethod
    def _se2_from_vec(vec: torch.Tensor) -> torch.Tensor:
        b = vec.shape[0]
        dtype = vec.dtype
        device = vec.device
        yaw = vec[:, 5]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(b, 1, 1)
        mat[:, 0, 0] = cos_yaw
        mat[:, 0, 1] = -sin_yaw
        mat[:, 1, 0] = sin_yaw
        mat[:, 1, 1] = cos_yaw
        mat[:, 0, 2] = vec[:, 0]
        mat[:, 1, 2] = vec[:, 1]
        return mat

    @staticmethod
    def _yaw_from_se2(mat: torch.Tensor) -> torch.Tensor:
        return torch.atan2(mat[:, 1, 0], mat[:, 0, 0])

    def _pad_history(self, ego_motion: torch.Tensor) -> torch.Tensor:
        need_steps = self.past_frames
        if ego_motion.shape[1] >= need_steps:
            return ego_motion[:, -need_steps:]
        pad = ego_motion.new_zeros(ego_motion.shape[0], need_steps - ego_motion.shape[1], ego_motion.shape[2])
        return torch.cat([pad, ego_motion], dim=1)

    def build_ego_state(self, ego_motion: torch.Tensor, commands: List[str]) -> torch.Tensor:
        if ego_motion is None:
            raise ValueError("ADMLPPlanner requires ego_history_egomotion or future_egomotion.")
        if ego_motion.dim() != 3 or ego_motion.shape[-1] < 6:
            raise ValueError(f"ego_motion should be (B,T,6), got shape={tuple(ego_motion.shape)}")

        ego_motion = self._pad_history(ego_motion[..., :6].float())
        b, steps, _ = ego_motion.shape
        device = ego_motion.device
        dtype = ego_motion.dtype

        step_tf = [self._se2_from_vec(ego_motion[:, i]) for i in range(steps)]
        cumulative = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(b, 1, 1)
        poses_reversed = []
        for mat in reversed(step_tf):
            cumulative = cumulative @ mat
            poses_reversed.append(cumulative)
        poses = list(reversed(poses_reversed))[-self.past_frames:]

        past = torch.stack(
            [
                torch.stack([pose[:, 0, 2], pose[:, 1, 2], self._yaw_from_se2(pose)], dim=-1)
                for pose in poses
            ],
            dim=1,
        )

        dt = max(self.dt, 1e-6)
        vel = ego_motion[:, -1, [0, 1, 5]] / dt
        if steps >= 2:
            prev_vel = ego_motion[:, -2, [0, 1, 5]] / dt
            acc = (vel - prev_vel) / dt
        else:
            acc = torch.zeros_like(vel)
        if self.feature_mode == "past5_no_command":
            return torch.cat([past.flatten(1), vel, acc], dim=-1)

        command = self._command_one_hot(commands, device=device, dtype=dtype)
        return torch.cat([past.flatten(1), vel, acc, command], dim=-1)

    def forward(self, ego_motion: torch.Tensor, commands: List[str]) -> torch.Tensor:
        state = self.build_ego_state(ego_motion, commands)
        return self.forward_state(state)

    def forward_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.plan_head(state).view(state.shape[0], self.n_output, 3)


def admlp_l1_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    valid = targets > -5e3
    same_segment = torch.floor(inputs / 0.5) == torch.floor(targets / 0.5)
    weight = torch.where(same_segment & valid, 0.5, 1.0).to(dtype=inputs.dtype)
    return F.l1_loss(inputs[valid] * weight[valid], targets[valid] * weight[valid])


class VLM_STP3_Gen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.receptive_field = int(cfg.TIME_RECEPTIVE_FIELD)
        self.n_future = int(cfg.N_FUTURE_FRAMES)
        self.planner = ADMLPPlanner(cfg, self.n_future)
        self._last_ego_motion: Optional[torch.Tensor] = None
        self._last_admlp_input: Optional[torch.Tensor] = None

    def forward(
        self,
        image,
        intrinsics,
        extrinsics,
        future_egomotion,
        *,
        rgb_224_seq=None,
        seg_224_seq=None,
        seg_id_224_seq=None,
        depth_224_seq=None,
        ego_history_egomotion=None,
        admlp_input=None,
        **kwargs,
    ):
        self._last_admlp_input = admlp_input
        if admlp_input is not None:
            self._last_ego_motion = None
        elif ego_history_egomotion is not None:
            # Dataset builds ego_history_egomotion as past transitions plus a
            # final zero row at the current frame. AD-MLP needs only motion
            # history, so do not let that sentinel become the current velocity.
            self._last_ego_motion = ego_history_egomotion[:, :-1] if ego_history_egomotion.shape[1] > 1 else ego_history_egomotion
        else:
            hist_len = max(1, self.receptive_field - 1)
            self._last_ego_motion = future_egomotion[:, :hist_len]
        return {}, None

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
        **kwargs,
    ):
        device = gt_trajs.device
        if self._last_admlp_input is not None:
            state = self._last_admlp_input.to(device=device, dtype=torch.float32)
            pred_all = self.planner.forward_state(state).to(device=device, dtype=gt_trajs.dtype)
        else:
            if self._last_ego_motion is None:
                raise RuntimeError("Call forward before planning so AD-MLP input or ego history is cached.")
            ego_motion = self._last_ego_motion.to(device=device, dtype=torch.float32)
            pred_all = self.planner(ego_motion, commands).to(device=device, dtype=gt_trajs.dtype)
        final_traj = pred_all[:, 1:]

        if not self.training:
            tiny = final_traj[..., :2].norm(dim=-1, keepdim=True) < 1e-2
            final_traj = torch.where(tiny.expand_as(final_traj), torch.zeros_like(final_traj), final_traj)

        if gt_trajs.shape[1] == pred_all.shape[1]:
            gt_all = gt_trajs[..., : pred_all.shape[-1]]
        else:
            zero = torch.zeros(gt_trajs.shape[0], 1, gt_trajs.shape[-1], device=gt_trajs.device, dtype=gt_trajs.dtype)
            gt_all = torch.cat([zero, gt_trajs], dim=1)[..., : pred_all.shape[-1]]

        l1 = admlp_l1_loss(pred_all, gt_all)
        gt_future = gt_all[:, 1:]
        xy_l2_t = torch.sqrt(((final_traj[..., :2] - gt_future[..., :2]) ** 2).sum(dim=-1) + 1e-8)
        ade = xy_l2_t.mean()
        fde = xy_l2_t[:, -1].mean()

        pred_step = torch.cat([final_traj[:, :1, :2], final_traj[:, 1:, :2] - final_traj[:, :-1, :2]], dim=1)
        gt_step = torch.cat([gt_future[:, :1, :2], gt_future[:, 1:, :2] - gt_future[:, :-1, :2]], dim=1)
        vel_l1 = F.l1_loss(pred_step, gt_step)
        loss = l1

        loss_dict = {
            "admlp_l1": l1,
            "admlp_ade": ade,
            "admlp_fde": fde,
            "admlp_vel_l1": vel_l1,
        }
        return loss, final_traj, final_traj, "ADMLP", loss_dict
