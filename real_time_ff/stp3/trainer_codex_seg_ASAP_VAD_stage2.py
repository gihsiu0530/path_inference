from stp3.trainer_codex_seg_ASAP_VAD import TrainingModule as BaseTrainingModule


class TrainingModule(BaseTrainingModule):
    """Stage-2 trainer for learning to use the stage-1 VAD-like perception.

    By default this freezes the BEV/vector perception heads learned in stage 1
    and trains the planner/context path to use their small vector context. This
    keeps the perception signal stable while testing whether it helps collision.
    """

    _FIXED_PERCEPTION_PREFIXES = (
        "vlm.rgb_stem",
        "vlm.seg_rgb_stem",
        "vlm.seg_id_embedding",
        "vlm.seg_id_stem",
        "vlm.depth_stem",
        "vlm.mid_fusion",
        "vlm.fusion",
        "vlm.command_embed",
        "vlm.command_film",
        "vlm.spatial_pool",
        "vlm.temporal_gru",
        "vlm.spatial_pos_mlp",
        "vlm.spatial_token_ln",
        "vlm.frame_embed",
        "vlm.scale_embed",
        "vlm.bev_input_proj",
        "vlm.bev_encoder",
        "vlm.bev_obstacle_head",
        "vlm.teacher_risk_head",
        "vlm.bev_teacher_adapter",
        "vlm.bev_teacher_fusion_proj",
        "vlm.bev_teacher_gate",
        "vlm.bev_teacher_context_ln",
        "vlm.bev_scale_embed",
        "vlm.vad_agent_",
        "vlm.vad_map_",
    )

    _FIXED_PERCEPTION_MODULES = (
        "rgb_stem",
        "seg_rgb_stem",
        "seg_id_embedding",
        "seg_id_stem",
        "depth_stem",
        "mid_fusion",
        "fusion",
        "command_embed",
        "command_film",
        "spatial_pool",
        "temporal_gru",
        "spatial_pos_mlp",
        "spatial_token_ln",
        "bev_input_proj",
        "bev_encoder",
        "bev_obstacle_head",
        "teacher_risk_head",
        "bev_teacher_adapter",
        "bev_teacher_fusion_proj",
        "bev_teacher_gate",
        "bev_teacher_context_ln",
    )

    _VAD_ADAPTER_TRAINABLE_PREFIXES = (
        "vlm.vad_ego_agent_attn",
        "vlm.vad_ego_map_attn",
        "vlm.vad_ego_norm1",
        "vlm.vad_ego_norm2",
        "vlm.vad_traj_residual_head",
    )

    def __init__(self, hparams):
        super().__init__(hparams)
        self._stage2_fixed_perception = bool(getattr(self.cfg, "STAGE2_FREEZE_PERCEPTION", True))
        self._stage2_adapter_only = bool(getattr(self.cfg, "STAGE2_TRAIN_VAD_ADAPTER_ONLY", True))
        self._apply_stage2_runtime()
        self._apply_stage2_freeze_policy()

    def _apply_stage2_runtime(self):
        vlm = self.model.vlm
        vlm.vad_vector_context_enabled = True

        # Keep only the VAD-like vector context path open for the first stage-2
        # run. Other BEV injections already showed instability in earlier runs.
        vlm.bev_token_fusion_enabled = False
        vlm.bev_context_fusion_enabled = False
        vlm.bev_ego_risk_context_enabled = False
        vlm.bev_context_scale = 0.0
        vlm.bev_ego_context_scale = 0.0
        vlm.bev_residual_scale = 0.0
        vlm.vad_vector_query_seed_scale = float(getattr(self.cfg, "VAD_VECTOR_QUERY_SEED_SCALE", 1.0))
        vlm.vad_traj_residual_enabled = bool(getattr(self.cfg, "VAD_TRAJ_RESIDUAL_ENABLED", False))
        vlm.vad_traj_residual_scale = float(getattr(self.cfg, "VAD_TRAJ_RESIDUAL_SCALE", 0.5))

    def _apply_stage2_freeze_policy(self):
        freeze_perception = bool(getattr(self.cfg, "STAGE2_FREEZE_PERCEPTION", True))
        adapter_only = bool(getattr(self.cfg, "STAGE2_TRAIN_VAD_ADAPTER_ONLY", True))
        self._stage2_fixed_perception = freeze_perception
        self._stage2_adapter_only = adapter_only

        if adapter_only:
            trainable_names = []
            frozen_names = []
            for name, param in self.model.named_parameters():
                trainable = name.startswith(self._VAD_ADAPTER_TRAINABLE_PREFIXES)
                param.requires_grad_(trainable)
                if trainable:
                    trainable_names.append(name)
                else:
                    frozen_names.append(name)

            self._set_fixed_perception_eval()
            trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            frozen_count = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
            print("[ASAP_VAD stage2] adapter-only mode enabled")
            print("[ASAP_VAD stage2] trainable VAD adapter tensors:", len(trainable_names))
            print("[ASAP_VAD stage2] trainable VAD adapter parameters:", f"{trainable_count:,}")
            print("[ASAP_VAD stage2] frozen parameters:", f"{frozen_count:,}")
            for name in trainable_names:
                print("[ASAP_VAD stage2] train:", name)
            return

        if not freeze_perception:
            print("[ASAP_VAD stage2] perception freeze disabled")
            return

        frozen_names = []
        for name, param in self.model.named_parameters():
            if name.startswith(self._FIXED_PERCEPTION_PREFIXES):
                param.requires_grad_(False)
                frozen_names.append(name)

        frozen_count = sum(
            p.numel()
            for name, p in self.model.named_parameters()
            if name.startswith(self._FIXED_PERCEPTION_PREFIXES)
        )
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._set_fixed_perception_eval()
        print("[ASAP_VAD stage2] frozen fixed-perception tensors:", len(frozen_names))
        print("[ASAP_VAD stage2] frozen fixed-perception parameters:", f"{frozen_count:,}")
        print("[ASAP_VAD stage2] remaining trainable parameters:", f"{trainable_count:,}")
        for name in frozen_names[:80]:
            print("[ASAP_VAD stage2] frozen:", name)
        if len(frozen_names) > 80:
            print(f"[ASAP_VAD stage2] ... {len(frozen_names) - 80} more frozen tensors")

    def _set_fixed_perception_eval(self):
        if not bool(getattr(self, "_stage2_fixed_perception", False)):
            return
        vlm = self.model.vlm
        for module_name in self._FIXED_PERCEPTION_MODULES:
            module = getattr(vlm, module_name, None)
            if module is not None:
                module.eval()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._apply_stage2_runtime()
        self._set_fixed_perception_eval()
        if self.logger is not None:
            self.logger.experiment.add_scalar(
                "epoch_param_stage2_freeze_perception",
                float(bool(getattr(self.cfg, "STAGE2_FREEZE_PERCEPTION", True))),
                global_step=self.training_step_count,
            )
            self.logger.experiment.add_scalar(
                "epoch_param_stage2_train_vad_adapter_only",
                float(bool(getattr(self.cfg, "STAGE2_TRAIN_VAD_ADAPTER_ONLY", True))),
                global_step=self.training_step_count,
            )
            self.logger.experiment.add_scalar(
                "epoch_param_vad_vector_query_seed_scale",
                float(getattr(self.cfg, "VAD_VECTOR_QUERY_SEED_SCALE", 1.0)),
                global_step=self.training_step_count,
            )
            self.logger.experiment.add_scalar(
                "epoch_param_vad_traj_residual_enabled",
                float(bool(getattr(self.cfg, "VAD_TRAJ_RESIDUAL_ENABLED", False))),
                global_step=self.training_step_count,
            )
            self.logger.experiment.add_scalar(
                "epoch_param_vad_traj_residual_scale",
                float(getattr(self.cfg, "VAD_TRAJ_RESIDUAL_SCALE", 0.5)),
                global_step=self.training_step_count,
            )

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self._set_fixed_perception_eval()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._apply_stage2_runtime()
        self._set_fixed_perception_eval()
