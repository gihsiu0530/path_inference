import stp3.trainer_ped_traj_ft as base_trainer

from stp3.model_super_ft.gemini_bev_ped_traj_super_ft import VLM_STP3_Gen as VLM_STP3


base_trainer.VLM_STP3 = VLM_STP3


class TrainingModule(base_trainer.TrainingModule):
    def _apply_finetune_trainable_mode(self):
        mode = str(getattr(self.cfg, 'PED_FINETUNE_TRAINABLE_MODE', 'ped_strict'))
        self.ft_trainable_mode = mode

        if mode == 'ped_strict':
            ped_name_substrings = [
                'vlm.cross_attn',
                'vlm.ped_step_proj',
                'vlm.ped_ctx_proj',
                'vlm.ped_time_pe',
                'vlm.ped_agent_embed',
                'vlm.ped_fuse_gate',
                'vlm.ped_bev_encoder',
                'vlm.ped_bev_align_head',
                'vlm.ped_bev_fuse_gate',
            ]
            trainable_scalar_names = {
                'planning_weight',
            }

            trainable_names = []
            frozen_names = []
            trainable_params = 0
            frozen_params = 0

            for name, param in self.model.named_parameters():
                keep = any(substr in name for substr in ped_name_substrings) or any(
                    name.endswith(scalar_name) for scalar_name in trainable_scalar_names
                )
                param.requires_grad = bool(keep)
                if keep:
                    trainable_names.append(name)
                    trainable_params += param.numel()
                else:
                    frozen_names.append(name)
                    frozen_params += param.numel()

            print(f'[SUPER_FT] trainable_mode={mode}')
            print(f'[SUPER_FT] trainable param tensors={len(trainable_names)} params={trainable_params:,}')
            print(f'[SUPER_FT] frozen param tensors={len(frozen_names)} params={frozen_params:,}')
            for name in trainable_names:
                print(f'[SUPER_FT] trainable: {name}')
            return

        super()._apply_finetune_trainable_mode()
