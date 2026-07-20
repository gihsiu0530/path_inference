import stp3.trainer_ped_traj_ft as base_trainer

# from stp3.model_0512_graduate.codex_pure_new_super_ft import VLM_STP3_Gen as VLM_STP3
from stp3.model_0512_graduate.codex_pure_vlm import VLM_STP3_Gen as VLM_STP3


base_trainer.VLM_STP3 = VLM_STP3


class TrainingModule(base_trainer.TrainingModule):
    def _apply_finetune_trainable_mode(self):
        mode = str(getattr(self.cfg, 'PED_FINETUNE_TRAINABLE_MODE', 'ped_light'))
        self.ft_trainable_mode = mode

        if mode == 'ped_strict':
            trainable_substrings = [
                'vlm.ped_encoder',
                'vlm.ped_query_fusion',
                'vlm.ped_bev_encoder',
                'vlm.ped_bev_align_head',
                'vlm.ped_bev_fuse_gate',
            ]
            trainable_scalar_names = {
                'planning_weight',
            }
        elif mode == 'ped_light':
            trainable_substrings = [
                'vlm.query_context',
                'vlm.endpoint_head',
                'vlm.ped_encoder',
                'vlm.ped_query_fusion',
                'vlm.ped_bev_encoder',
                'vlm.ped_bev_align_head',
                'vlm.ped_bev_fuse_gate',
            ]
            trainable_scalar_names = {
                'planning_weight',
            }
        elif mode == 'ped_only':
            trainable_substrings = [
                'vlm.time_queries',
                'vlm.ego_mlp',
                'vlm.context_mlp',
                'vlm.query_context',
                'vlm.decoder_layers',
                'vlm.waypoint_gru',
                'vlm.waypoint_refine_ln',
                'vlm.traj_abs_head',
                'vlm.traj_delta_head',
                'vlm.endpoint_head',
                'vlm.ped_encoder',
                'vlm.ped_query_fusion',
                'vlm.ped_bev_encoder',
                'vlm.ped_bev_align_head',
                'vlm.ped_bev_fuse_gate',
            ]
            trainable_scalar_names = {
                'planning_weight',
            }
        else:
            super()._apply_finetune_trainable_mode()
            return

        trainable_names = []
        frozen_names = []
        trainable_params = 0
        frozen_params = 0

        for name, param in self.model.named_parameters():
            keep = any(substr in name for substr in trainable_substrings) or any(
                name.endswith(scalar_name) for scalar_name in trainable_scalar_names
            )
            param.requires_grad = bool(keep)
            if keep:
                trainable_names.append(name)
                trainable_params += param.numel()
            else:
                frozen_names.append(name)
                frozen_params += param.numel()

        print(f'[CODEX_SUPER_FT] trainable_mode={mode}')
        print(f'[CODEX_SUPER_FT] trainable param tensors={len(trainable_names)} params={trainable_params:,}')
        print(f'[CODEX_SUPER_FT] frozen param tensors={len(frozen_names)} params={frozen_params:,}')
        for name in trainable_names:
            print(f'[CODEX_SUPER_FT] trainable: {name}')
