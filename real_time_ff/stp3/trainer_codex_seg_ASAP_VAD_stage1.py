import torch

from stp3.trainer_codex_seg_ASAP_VAD import TrainingModule as BaseTrainingModule


class TrainingModule(BaseTrainingModule):
    """Stage-1 trainer for ASAP_VAD perception/vector pretraining.

    This keeps the ASAP planner loaded for forward compatibility and validation,
    but the training loss only updates BEV/vector perception heads. Planning,
    residual, collision, and map-constraint losses are logged only.
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        self._set_stage1_trainable_params()
        self._force_stage1_runtime()

    def _is_stage1_trainable_name(self, name):
        prefixes = (
            "vlm.bev_input_proj",
            "vlm.bev_encoder",
            "vlm.bev_obstacle_head",
            "vlm.teacher_risk_head",
            "vlm.bev_teacher_adapter",
            "vlm.bev_scale_embed",
            "vlm.vad_agent_",
            "vlm.vad_map_",
        )
        return name.startswith(prefixes)

    def _set_stage1_trainable_params(self):
        trainable_names = []
        frozen_count = 0
        for name, param in self.model.named_parameters():
            trainable = self._is_stage1_trainable_name(name)
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
            else:
                frozen_count += param.numel()

        trainable_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("[ASAP_VAD stage1] trainable perception tensors:", len(trainable_names))
        print("[ASAP_VAD stage1] trainable parameters:", f"{trainable_count:,}")
        print("[ASAP_VAD stage1] frozen parameters:", f"{frozen_count:,}")
        for name in trainable_names[:80]:
            print("[ASAP_VAD stage1] train:", name)
        if len(trainable_names) > 80:
            print(f"[ASAP_VAD stage1] ... {len(trainable_names) - 80} more trainable tensors")

    def _force_stage1_runtime(self):
        vlm = self.model.vlm

        # Perception heads must run, but their context must not perturb planning.
        vlm.vad_vector_context_enabled = True
        vlm.vad_vector_context_scale = 0.0
        vlm.vad_traj_residual_enabled = False

        # Stage 1 should not inject BEV into the planner/residual path.
        vlm.bev_token_fusion_enabled = False
        vlm.bev_context_fusion_enabled = False
        vlm.bev_ego_risk_context_enabled = False
        vlm.bev_context_scale = 0.0
        vlm.bev_ego_context_scale = 0.0
        vlm.bev_residual_scale = 0.0

        # In stage1, default to training BEV features from perception losses.
        # Override by adding STAGE1_DETACH_BEV: True in the config if needed.
        vlm.vad_vector_detach_bev = bool(getattr(self.cfg, "STAGE1_DETACH_BEV", False))
        vlm.residual_gate_floor = 0.0
        vlm.residual_gate_free = False

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._force_stage1_runtime()
        if self.logger is not None:
            self.logger.experiment.add_scalar(
                "epoch_param_stage1_vad_vector_context_scale",
                0.0,
                global_step=self.training_step_count,
            )
            self.logger.experiment.add_scalar(
                "epoch_param_stage1_detach_bev",
                float(self.model.vlm.vad_vector_detach_bev),
                global_step=self.training_step_count,
            )

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._force_stage1_runtime()

    def _stage1_loss_weight(self, stage1_name, fallback_name, default=0.0):
        fallback = float(getattr(self.cfg, fallback_name, default))
        return float(getattr(self.cfg, stage1_name, fallback))

    def _make_stage1_loss(self, loss_dict):
        device = next(self.model.parameters()).device
        zero = torch.tensor(0.0, device=device)

        terms = {
            "vad_agent": (
                self._stage1_loss_weight("LOSS_STAGE1_VAD_AGENT_W", "LOSS_VAD_AGENT_W", 0.0),
                loss_dict.get("vad_agent_vector", zero),
            ),
            "vad_map": (
                self._stage1_loss_weight("LOSS_STAGE1_VAD_MAP_W", "LOSS_VAD_MAP_W", 0.0),
                loss_dict.get("vad_map_vector", zero),
            ),
            "bev_obstacle": (
                self._stage1_loss_weight(
                    "LOSS_STAGE1_BEV_OBSTACLE_W",
                    "LOSS_VAD_BEV_OBSTACLE_W",
                    0.0,
                ),
                loss_dict.get("bev_obstacle_aux", zero),
            ),
            "bev_teacher": (
                self._stage1_loss_weight("LOSS_STAGE1_BEV_TEACHER_W", "LOSS_BEV_TEACHER_W", 0.0),
                loss_dict.get("bev_teacher_distill", zero),
            ),
            "teacher_risk": (
                self._stage1_loss_weight("LOSS_STAGE1_TEACHER_RISK_W", "LOSS_TEACHER_RISK_W", 0.0),
                loss_dict.get("teacher_risk_distill", zero),
            ),
        }

        total = zero
        weighted_logs = {}
        active_weight = 0.0
        for name, (weight, value) in terms.items():
            weighted = weight * value
            total = total + weighted
            active_weight += abs(weight)
            weighted_logs[f"stage1_{name}_with_lam"] = weighted
            weighted_logs[f"stage1_{name}_lam"] = torch.tensor(weight, device=device)

        if active_weight <= 0.0:
            raise RuntimeError(
                "ASAP_VAD stage1 has no active perception loss. "
                "Set LOSS_VAD_AGENT_W / LOSS_VAD_MAP_W / LOSS_VAD_BEV_OBSTACLE_W / LOSS_BEV_TEACHER_W, "
                "or their LOSS_STAGE1_* overrides."
            )
        return total, weighted_logs

    def _shared_step_stage1_train(self, batch):
        self._force_stage1_runtime()

        image = batch["image"]
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]
        future_egomotion = batch["future_egomotion"]
        command = batch["command"]
        trajs = batch["sample_trajectory"]
        target_points = batch["target_point"]
        labels = self.prepare_future_labels(batch)

        gt = labels["gt_trajectory"][:, 1:, :2]
        self._epoch_gt_abs_x_train.append(gt[..., 0].abs().reshape(-1).detach().float().cpu())
        self._epoch_gt_step_dx_train.append(
            (gt[:, 1:, 0] - gt[:, :-1, 0]).abs().reshape(-1).detach().float().cpu()
        )

        output, bev_rgbs = self.model(
            image,
            intrinsics,
            extrinsics,
            future_egomotion,
            rgb_224_seq=batch["rgb_224_seq"],
            seg_224_seq=batch["seg_224_seq"],
            seg_id_224_seq=batch.get("seg_id_224_seq"),
            depth_224_seq=batch.get("depth_224_seq"),
            ego_history_egomotion=batch.get("ego_history_egomotion"),
            admlp_input=batch.get("admlp_input"),
        )

        receptive_field = self.model.receptive_field
        ped_lbl = labels["pedestrian"][:, receptive_field:].squeeze(2)
        occupancy = torch.logical_or(labels["segmentation"][:, receptive_field:].squeeze(2), ped_lbl)
        present_occupancy = torch.logical_or(
            labels["segmentation"][:, receptive_field - 1].squeeze(1),
            labels["pedestrian"][:, receptive_field - 1].squeeze(1),
        )
        drivable_mask = self._build_drivable_mask(labels, labels["gt_trajectory"][:, 1:].shape[1])
        drivable_aux = drivable_mask[:, 0] if drivable_mask is not None else None

        planning_loss, final_traj, _, _, loss_dict = self.model.planning(
            bev_rgbs=bev_rgbs,
            trajs=trajs,
            gt_trajs=labels["gt_trajectory"][:, 1:],
            commands=command,
            target_points=target_points,
            occupancy=occupancy,
            drivable_mask=drivable_mask,
            occupancy_aux=present_occupancy,
            drivable_aux=drivable_aux,
            teacher_scene_feature=batch.get("teacher_gpv_feature"),
            teacher_bev_feature=batch.get("teacher_bev_feature"),
            teacher_bev_valid=batch.get("teacher_bev_valid"),
            teacher_risk_heatmap=batch.get("teacher_risk_heatmap"),
            teacher_risk_valid=batch.get("teacher_risk_valid"),
        )

        stage1_loss, weighted_logs = self._make_stage1_loss(loss_dict)
        turn_ratio = sum(c in ["LEFT", "RIGHT"] for c in command) / max(1, len(command))

        loss = {
            "stage1_perception": stage1_loss,
            "planning_log": planning_loss.detach(),
            "turn_ratio_log": torch.tensor(turn_ratio, device=stage1_loss.device),
        }
        for key, value in loss_dict.items():
            loss[key + "_log"] = value.detach()
        for key, value in weighted_logs.items():
            loss[key + "_log"] = value.detach()

        output = {
            **output,
            "selected_traj": torch.cat(
                [torch.zeros((final_traj.shape[0], 1, 3), device=final_traj.device), final_traj],
                dim=1,
            ),
        }
        return output, labels, loss

    def training_step(self, batch, batch_idx):
        output, labels, loss = self._shared_step_stage1_train(batch)
        self.training_step_count += 1

        if self.logger is not None:
            for key, value in loss.items():
                log_value = value.detach() if torch.is_tensor(value) else value
                self.logger.experiment.add_scalar(
                    "step_train_loss_" + key,
                    log_value,
                    global_step=self.training_step_count,
                )

            opt = self.optimizers(use_pl_optimizer=True)
            if isinstance(opt, torch.optim.Optimizer):
                for i, group in enumerate(opt.param_groups):
                    self.logger.experiment.add_scalar(
                        f"lr/group_{i}",
                        group["lr"],
                        global_step=self.training_step_count,
                    )

        loss_total = loss["stage1_perception"]
        self.log(
            "stage1_train_perception",
            loss_total,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if not torch.isfinite(loss_total):
            raise RuntimeError("Non-finite ASAP_VAD stage1 perception loss.")
        return loss_total

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("ASAP_VAD stage1 found no trainable perception parameters.")

        lr = float(getattr(self.cfg.OPTIMIZER, "LR", 1e-4))
        wd = float(getattr(self.cfg.OPTIMIZER, "WEIGHT_DECAY", 0.0))
        print("[ASAP_VAD stage1] optimizer_lr:", lr)
        print("[ASAP_VAD stage1] optimizer_weight_decay:", wd)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

        if bool(getattr(self.cfg.OPTIMIZER, "FIXED_LR", False)):
            print("[ASAP_VAD stage1] fixed_lr_enabled: True")
            return optimizer

        fallback_steps_per_epoch = 1376
        estimated_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
        total_steps = estimated_steps if estimated_steps > 0 else fallback_steps_per_epoch * self.cfg.EPOCHS
        print("[ASAP_VAD stage1] onecycle_total_steps:", total_steps)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.15,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e3,
            three_phase=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "onecycle",
            },
        }
