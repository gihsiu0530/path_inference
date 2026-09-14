"""ASAP-FF with command-switched hybrid coarse trajectories.

FORWARD samples use direct constant-motion SE(2) extrapolation. LEFT and
RIGHT samples keep the frozen AD-MLP coarse planner so that a turn can be
anticipated before the measured yaw rate changes. The FF residual, route
attention, camera BEV, confidence gate, and footprint safety filter are
inherited unchanged.
"""

from typing import List

import torch

from stp3.model_vlm.codex_pure_ASAP import VLM_STP3_Gen as OriginalASAPVLM
from stp3.model_vlm.codex_pure_ASAP_extrap_ff import (
    VLM_STP3_Gen as ExtrapolationFFVLM,
)
from stp3.model_vlm.codex_pure_ASAP_ff import (
    VLM_STP3_Gen as BaseFFVLM,
)


class VLM_STP3_Gen(BaseFFVLM):
    """Use extrapolation for FORWARD and AD-MLP for LEFT/RIGHT."""

    _strip_current_zero_sentinel = staticmethod(
        ExtrapolationFFVLM._strip_current_zero_sentinel
    )
    _constant_motion_coarse = ExtrapolationFFVLM._constant_motion_coarse
    _augment_hybrid_coarse = ExtrapolationFFVLM._augment_extrapolated_coarse

    def __init__(self, cfg):
        # BaseFFVLM constructs and freezes the AD-MLP baseline, then replaces
        # only its trainable residual planner with FastTrajectoryPlannerFF.
        super().__init__(cfg)
        self._last_extrapolation_motion_ff = None
        self.last_extrap_speed_mps = torch.tensor(0.0)
        self.last_extrap_yaw_rate_deg_s = torch.tensor(0.0)
        self.last_hybrid_forward_mask_ff = None
        self.last_admlp_coarse_ff = None
        self.last_extrap_coarse_ff = None
        print("coarse override: FORWARD=ego-motion extrapolation, LEFT/RIGHT=frozen AD-MLP")

    def forward(self, image, intrinsics, extrinsics, future_egomotion, **kwargs):
        # Cache raw ego motion separately because the AD-MLP may consume the
        # precomputed admlp_input instead of ego_history_egomotion.
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
        self._last_extrapolation_motion_ff = extrapolation_motion
        return output

    def _admlp_coarse(
        self,
        device: torch.device,
        dtype: torch.dtype,
        commands: List[str],
    ) -> torch.Tensor:
        if len(commands) == 0:
            raise ValueError("Hybrid coarse planner received an empty command batch.")

        # Call the original method directly to obtain an unaugmented frozen
        # AD-MLP trajectory. Calling BaseFFVLM here would augment it before the
        # two sources are combined.
        admlp_coarse = OriginalASAPVLM._admlp_coarse(
            self, device, dtype, commands
        )
        extrap_coarse = self._constant_motion_coarse(device, dtype)
        if admlp_coarse.shape != extrap_coarse.shape:
            raise RuntimeError(
                "Hybrid coarse sources have different shapes: "
                f"AD-MLP={tuple(admlp_coarse.shape)}, "
                f"extrapolation={tuple(extrap_coarse.shape)}"
            )
        if len(commands) != admlp_coarse.shape[0]:
            raise ValueError(
                f"Got {len(commands)} commands for batch size {admlp_coarse.shape[0]}."
            )

        unsupported = [command for command in commands if command not in {"LEFT", "FORWARD", "RIGHT"}]
        if unsupported:
            raise ValueError(
                f"Unsupported hybrid commands {unsupported}; expected LEFT, FORWARD, or RIGHT."
            )
        forward_mask = torch.tensor(
            [command == "FORWARD" for command in commands],
            device=device,
            dtype=torch.bool,
        )
        coarse = torch.where(
            forward_mask.view(-1, 1, 1), extrap_coarse, admlp_coarse
        )

        self.last_hybrid_forward_mask_ff = forward_mask.detach()
        self.last_admlp_coarse_ff = admlp_coarse.detach()
        self.last_extrap_coarse_ff = extrap_coarse.detach()
        self._last_admlp_xy = coarse.detach()  # existing plotting/diagnostic API
        return self._augment_hybrid_coarse(coarse, device, dtype)

    def planning(self, **kwargs):
        loss, prediction, final_traj, _, loss_dict = super().planning(**kwargs)
        device = final_traj.device
        mask = self.last_hybrid_forward_mask_ff
        if mask is None:
            forward_ratio = final_traj.new_tensor(0.0)
        else:
            forward_ratio = mask.to(device=device, dtype=final_traj.dtype).mean()
        loss_dict["ff_hybrid_forward_ratio"] = forward_ratio
        loss_dict["ff_extrap_speed_mps"] = self.last_extrap_speed_mps.mean().to(device)
        loss_dict["ff_extrap_yaw_rate_deg_s"] = (
            self.last_extrap_yaw_rate_deg_s.abs().mean().to(device)
        )
        return loss, prediction, final_traj, "FAST_HYBRID_FF", loss_dict
