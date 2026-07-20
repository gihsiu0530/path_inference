import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import numpy as np
import inspect

from stp3.config import get_cfg


# from stp3.model_vlm.vlm_planner import VLM_STP3  # 新


# from stp3.model_vlm.mult_planner import VLM_STP3


# from stp3.model_vlm.gemini_hybrid import VLM_STP3_Gen as VLM_STP3

# from stp3.model_vlm.gemini_autocast import VLM_STP3_Gen as VLM_STP3

# from stp3.model_vlm.vlm_come_on import VLM_STP3_Gen as VLM_STP3

# from stp3.model_vlm.image_forced_ped_traj import VLM_STP3_Gen as VLM_STP3

from stp3.model_0512_graduate.image_forced_ped_traj_codex import VLM_STP3_Gen as VLM_STP3


# from stp3.model_vlm.clean_loss_ped_traj import VLM_STP3_Gen as VLM_STP3



from stp3.losses import SpatialRegressionLoss, SegmentationLoss, HDmapLoss, DepthLoss
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from stp3.utils.geometry import cumulative_warp_features_reverse, cumulative_warp_features
from stp3.utils.instance import predict_instance_segmentation_and_trajectories
from stp3.utils.visualisation import visualise_output


class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # see config.py for details
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        # pytorch lightning does not support saving YACS CfgNone
        print("cfg_dict :", self.hparams)
        cfg = get_cfg(cfg_dict=self.hparams)
        self.cfg = cfg
        self.n_classes = len(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)
        self.hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS

        # Bird's-eye view extent in meters
        assert self.cfg.LIFT.X_BOUND[1] > 0 and self.cfg.LIFT.Y_BOUND[1] > 0
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

        # Model
        self.model = VLM_STP3(cfg)

        self.losses_fn = nn.ModuleDict()

        # Semantic segmentation
        self.losses_fn['segmentation'] = SegmentationLoss(
            class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS),
            use_top_k=self.cfg.SEMANTIC_SEG.VEHICLE.USE_TOP_K,
            top_k_ratio=self.cfg.SEMANTIC_SEG.VEHICLE.TOP_K_RATIO,
            future_discount=self.cfg.FUTURE_DISCOUNT,
        )
        self.model.segmentation_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.metric_vehicle_val = IntersectionOverUnion(self.n_classes)

        # Pedestrian segmentation
        if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            self.losses_fn['pedestrian'] = SegmentationLoss(
                class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.PEDESTRIAN.WEIGHTS),
                use_top_k=self.cfg.SEMANTIC_SEG.PEDESTRIAN.USE_TOP_K,
                top_k_ratio=self.cfg.SEMANTIC_SEG.PEDESTRIAN.TOP_K_RATIO,
                future_discount=self.cfg.FUTURE_DISCOUNT,
            )
            self.model.pedestrian_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.metric_pedestrian_val = IntersectionOverUnion(self.n_classes)

        # HD map
        if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            self.losses_fn['hdmap'] = HDmapLoss(
                class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.HDMAP.WEIGHTS),
                training_weights=self.cfg.SEMANTIC_SEG.HDMAP.TRAIN_WEIGHT,
                use_top_k=self.cfg.SEMANTIC_SEG.HDMAP.USE_TOP_K,
                top_k_ratio=self.cfg.SEMANTIC_SEG.HDMAP.TOP_K_RATIO,
            )
            self.metric_hdmap_val = []
            for i in range(len(self.hdmap_class)):
                self.metric_hdmap_val.append(IntersectionOverUnion(2, absent_score=1))
            self.model.hdmap_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.metric_hdmap_val = nn.ModuleList(self.metric_hdmap_val)

        # Depth
        if self.cfg.LIFT.GT_DEPTH:
            self.losses_fn['depths'] = DepthLoss()
            self.model.depths_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Instance segmentation
        if self.cfg.INSTANCE_SEG.ENABLED:
            self.losses_fn['instance_center'] = SpatialRegressionLoss(
                norm=2, future_discount=self.cfg.FUTURE_DISCOUNT
            )
            self.losses_fn['instance_offset'] = SpatialRegressionLoss(
                norm=1, future_discount=self.cfg.FUTURE_DISCOUNT, ignore_index=self.cfg.DATASET.IGNORE_INDEX
            )
            self.model.centerness_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.model.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.metric_panoptic_val = PanopticMetric(n_classes=self.n_classes)

        # Instance flow
        if self.cfg.INSTANCE_FLOW.ENABLED:
            self.losses_fn['instance_flow'] = SpatialRegressionLoss(
                norm=1, future_discount=self.cfg.FUTURE_DISCOUNT, ignore_index=self.cfg.DATASET.IGNORE_INDEX
            )
            self.model.flow_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Planning
        if self.cfg.PLANNING.ENABLED:
            self.metric_planning_val = PlanningMetric(self.cfg, self.cfg.N_FUTURE_FRAMES)
            self.metric_planning_ped_only_on_ped_traj_val = PlanningMetric(self.cfg, self.cfg.N_FUTURE_FRAMES)
            self.model.planning_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self._apply_finetune_trainable_mode()
        self.training_step_count = 0

                # === Val turn metrics accumulators ===
        self.val_cmd_total = 0
        self.val_turn_total = 0
        self.val_right_total = 0
        self.val_left_total = 0

        self.val_turn_ok = 0
        self.val_right_ok = 0
        self.val_left_ok = 0
        self._val_ped_traj_delta_sum_on_ped = 0.0
        self._val_ped_traj_delta_count_on_ped = 0
        self._val_ped_box_delta_sum_on_ped = 0.0
        self._val_ped_box_delta_count_on_ped = 0
        self._val_ped_traj_delta_sum_triggered = 0.0
        self._val_ped_traj_delta_count_triggered = 0
        self._val_ped_box_delta_sum_triggered = 0.0
        self._val_ped_box_delta_count_triggered = 0
        
        # ★ 新增：記錄訓練集的危險觸發次數
        self.train_danger_total = 0
        self.train_danger_radar_total = 0
        self.train_danger_gt_total = 0
        self.train_danger_both_total = 0
        self.train_fwd_total = 0

        # ===== Epoch-level quantile accumulators (store on CPU to save VRAM) =====
        self._epoch_gt_abs_x_train = []
        self._epoch_gt_step_dx_train = []
        self._epoch_gt_abs_x_val = []
        self._epoch_gt_step_dx_val = []

        self._epoch_risk_mask_sum = 0.0
        self._epoch_dy_exceed_sum = 0.0
        self._epoch_ratio_count = 0

    def _apply_finetune_trainable_mode(self):
        mode = str(getattr(self.cfg, 'PED_FINETUNE_TRAINABLE_MODE', 'ped_only'))
        self.ft_trainable_mode = mode
        if mode != 'ped_only':
            print(f'[PED_FT] trainable_mode={mode} (no freezing applied)')
            return

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
            'vlm.bev_decoder',
            'vlm.h0_proj',
            'vlm.ego_mlp',
            'vlm.xy_embedder',
            'vlm.g_proj',
            'vlm.xy_post_ln',
            'vlm.dec_in_ln',
            'vlm.decoder_gru',
            'vlm.delta_head',
            'vlm.mix_gate',
        ]
        trainable_scalar_names = {
            'planning_weight',
            'vlm.time_queries',
            'vlm.time_pe',
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

        print(f'[PED_FT] trainable_mode={mode}')
        print(f'[PED_FT] trainable param tensors={len(trainable_names)} params={trainable_params:,}')
        print(f'[PED_FT] frozen param tensors={len(frozen_names)} params={frozen_params:,}')
        for name in trainable_names:
            print(f'[PED_FT] trainable: {name}')

    def _model_forward_supports(self, arg_name):
        try:
            sig = inspect.signature(self.model.forward)
        except (TypeError, ValueError):
            return False
        return arg_name in sig.parameters



    

    def on_train_epoch_start(self):
        # Finetune: keep curriculum-related values fixed instead of changing with epoch.
        w = float(getattr(self.cfg, "LOSS_COARSE_W", 0.05))
        setattr(self.model.cfg, "LOSS_COARSE_W", w)

        current_tf = float(getattr(self.cfg, "AR_TF_RATIO", 0.0))
        self.model.vlm.AR_TF_RATIO = current_tf

        if self.logger is not None:
            self.logger.experiment.add_scalar("epoch_param_coarse_w", w, global_step=self.training_step_count)
            self.logger.experiment.add_scalar("epoch_param_tf_ratio", current_tf, global_step=self.training_step_count)

        # ★ 新增：每個 Epoch 開始時歸零
        self.train_danger_total = 0
        self.train_danger_radar_total = 0
        self.train_danger_gt_total = 0
        self.train_danger_both_total = 0
        self.train_fwd_total = 0

        self._epoch_risk_mask_sum = 0.0
        self._epoch_dy_exceed_sum = 0.0
        self._epoch_ratio_count = 0

    def on_validation_epoch_start(self):
        # reset accumulators
        self.val_cmd_total = 0
        self.val_turn_total = 0
        self.val_right_total = 0
        self.val_left_total = 0
        self.val_turn_ok = 0
        self.val_right_ok = 0
        self.val_left_ok = 0
        self._val_ped_traj_delta_sum_on_ped = 0.0
        self._val_ped_traj_delta_count_on_ped = 0
        self._val_ped_box_delta_sum_on_ped = 0.0
        self._val_ped_box_delta_count_on_ped = 0
        self._val_ped_traj_delta_sum_triggered = 0.0
        self._val_ped_traj_delta_count_triggered = 0
        self._val_ped_box_delta_sum_triggered = 0.0
        self._val_ped_box_delta_count_triggered = 0

    def on_validation_epoch_end(self):
        # avoid div0
        turn_total = max(1, self.val_turn_total)
        right_total = max(1, self.val_right_total)
        left_total = max(1, self.val_left_total)
        cmd_total = max(1, self.val_cmd_total)

        epoch_turn_ratio = self.val_turn_total / cmd_total
        epoch_acc_turn  = self.val_turn_ok / turn_total
        epoch_acc_right = self.val_right_ok / right_total
        epoch_acc_left  = self.val_left_ok / left_total

        gs = self.training_step_count  # 用同一個 global_step 方便對齊 train 曲線
        self.logger.experiment.add_scalar("epoch_val_turn_ratio", epoch_turn_ratio, global_step=gs)
        self.logger.experiment.add_scalar("epoch_val_acc_turn",  epoch_acc_turn,  global_step=gs)
        self.logger.experiment.add_scalar("epoch_val_acc_right", epoch_acc_right, global_step=gs)
        self.logger.experiment.add_scalar("epoch_val_acc_left",  epoch_acc_left,  global_step=gs)
        if self._val_ped_traj_delta_count_on_ped > 0:
            self.logger.experiment.add_scalar(
                "epoch_val_ped_bev_traj_delta_mean_on_ped",
                self._val_ped_traj_delta_sum_on_ped / self._val_ped_traj_delta_count_on_ped,
                global_step=gs,
            )
            self.logger.experiment.add_scalar(
                "epoch_val_ped_bev_box_delta_on_ped",
                self._val_ped_box_delta_sum_on_ped / self._val_ped_box_delta_count_on_ped,
                global_step=gs,
            )
        if self._val_ped_traj_delta_count_triggered > 0:
            self.logger.experiment.add_scalar(
                "epoch_val_ped_bev_traj_delta_mean_triggered",
                self._val_ped_traj_delta_sum_triggered / self._val_ped_traj_delta_count_triggered,
                global_step=gs,
            )
            self.logger.experiment.add_scalar(
                "epoch_val_ped_bev_box_delta_triggered",
                self._val_ped_box_delta_sum_triggered / self._val_ped_box_delta_count_triggered,
                global_step=gs,
            )
            
        self.shared_epoch_end(None, False)



    def shared_step(self, batch, is_train):
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']
        command = batch['command']
        trajs = batch['sample_trajectory']
        target_points = batch['target_point']
        B = trajs.size(0)

        turn_ratio = sum([(c in ['LEFT', 'RIGHT']) for c in command]) / max(1, len(command))
        

        
        receptive_field = self.model.receptive_field  # 訓練/驗證兩邊都加

        # Warp labels
        labels = self.prepare_future_labels(batch)


        # 單幀

        # output, bev_rgbs = self.model(
        #             image, intrinsics, extrinsics, future_egomotion,
        #             # 使用關鍵字參數傳遞
        #             rgb_224=batch['rgb_224'],
        #             seg_224=batch['seg_224']
        #         )
        # ===== Collect GT lateral stats for epoch-end quantile (x = lateral) =====
        gt = labels['gt_trajectory'][:, 1:, :2]     # (B,T,2)  跟你 planning 用的一致
        gt_x_abs = gt[..., 0].abs().reshape(-1)     # |x|
        gt_step_dx_abs = (gt[:, 1:, 0] - gt[:, :-1, 0]).abs().reshape(-1)  # |Δx|

        # 放 CPU，避免吃 VRAM
        gt_x_abs_cpu = gt_x_abs.detach().float().cpu()
        gt_step_dx_abs_cpu = gt_step_dx_abs.detach().float().cpu()

        if is_train:
            self._epoch_gt_abs_x_train.append(gt_x_abs_cpu)
            self._epoch_gt_step_dx_train.append(gt_step_dx_abs_cpu)
        else:
            self._epoch_gt_abs_x_val.append(gt_x_abs_cpu)
            self._epoch_gt_step_dx_val.append(gt_step_dx_abs_cpu)



        # 多幀
        model_kwargs = dict(
            rgb_224_seq=batch['rgb_224_seq'],
            seg_224_seq=batch['seg_224_seq'],
            ped_traj_preds=batch.get('ped_traj_preds'),
            ped_traj_mask=batch.get('ped_traj_mask'),
            ped_traj_valid_steps=batch.get('ped_traj_valid_steps'),
        )
        if self._model_forward_supports('seg_id_224_seq'):
            model_kwargs['seg_id_224_seq'] = batch.get('seg_id_224_seq')
        if self._model_forward_supports('depth_224_seq'):
            model_kwargs['depth_224_seq'] = batch.get('depth_224_seq')
        if self._model_forward_supports('ped_bev_map'):
            model_kwargs['ped_bev_map'] = batch.get('ped_bev_map')

        output, bev_rgbs = self.model(
            image, intrinsics, extrinsics, future_egomotion,
            **model_kwargs
        )

        # print("batch['depths']",batch['depths'])

        # output, bev_rgbs = self.model(
        #     image, intrinsics, extrinsics, future_egomotion,
        #     rgb_224_seq=batch['rgb_224_seq'],          # (B,T_rf,H,W,3) 或 (B,T_rf,3,H,W)
        #     seg_224_seq=batch['seg_224_seq'],          # 同上
        #     depth_224_seq=batch['depths']              # (B,T_rf,1,224,224) 或 (B,T_rf,N,1,224,224)
        # )

        


        #####
        # Loss computation
        #####
        loss = {}
        loss['turn_ratio_log'] = torch.tensor(turn_ratio, device=self.device)

        if is_train:


            # Planning
            if self.cfg.PLANNING.ENABLED:
                receptive_field = self.model.receptive_field
                planning_factor = 1 / (2 * torch.exp(self.model.planning_weight))



                ped_lbl = labels['pedestrian'][:, receptive_field:].squeeze(2)

                occupancy = torch.logical_or(labels['segmentation'][:, receptive_field:].squeeze(2), ped_lbl)

                # pl_loss, final_traj, loss_rank, loss_cost = self.model.planning(
                #     bev_rgbs=bev_rgbs,
                #     trajs=trajs[:, :, 1:],                     # 仍沿用你原本的 slice
                #     gt_trajs=labels['gt_trajectory'][:, 1:],
                #     commands=command,
                #     target_points=target_points
                # )

                pl_loss, final_traj, resample_traj, loss_rank, loss_dict = self.model.planning(
                    bev_rgbs=bev_rgbs,
                    # trajs=trajs[:, :, 1:],                 # 生成式不使用，但保留相容參數
                    trajs=trajs,                 # 生成式不使用，但保留相容參數
                    gt_trajs=labels['gt_trajectory'][:, 1:],
                    commands=command,
                    target_points=target_points,
                    occupancy=occupancy,                    # ★ 新增
                    ped_bev_points=batch.get('ped_bev_points'),
                    ped_bev_valid_steps=batch.get('ped_bev_valid_steps'),
                )


                # loss['planning'] = planning_factor * pl_loss
                s = self.model.planning_weight
                loss['planning'] = torch.exp(-s) * pl_loss + s
                # loss['loss_rank'] = loss_rank
                # loss['loss_cost'] = loss_cost
                for k, v in loss_dict.items():
                    loss[k + "_log"] = v.detach()

                output = {**output, 'selected_traj': torch.cat(
                    [torch.zeros((B, 1, 3), device=final_traj.device), final_traj], dim=1)}
            else:
                output = {**output, 'selected_traj': labels['gt_trajectory']}

        # Metrics
        else:
            n_present = self.model.receptive_field
            receptive_field = self.model.receptive_field


            # # semantic segmentation metric
            # seg_prediction = torch.argmax(output['segmentation'].detach(), dim=2)  # (B, S, H, W)
            # self.metric_vehicle_val(seg_prediction[:, n_present - 1:], labels['segmentation'][:, n_present - 1:])

            # pedestrian segmentation metric
            # if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            #     pedestrian_prediction = torch.zeros_like(seg_prediction)
            # else:
            #     pedestrian_prediction = torch.zeros_like(seg_prediction)

            # # hdmap metric
            # if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            #     for i in range(len(self.hdmap_class)):
            #         hdmap_prediction = output['hdmap'][:, 2 * i:2 * (i + 1)].detach()
            #         hdmap_prediction = torch.argmax(hdmap_prediction, dim=1, keepdim=True)
            #         self.metric_hdmap_val[i](hdmap_prediction, labels['hdmap'][:, i:i + 1])

            # # instance segmentation metric
            # if self.cfg.INSTANCE_SEG.ENABLED:
            #     pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
            #         output, compute_matched_centers=False
            #     )
            #     self.metric_panoptic_val(pred_consistent_instance_seg[:, n_present - 1:],
            #                              labels['instance'][:, n_present - 1:])

            # planning metric
            if self.cfg.PLANNING.ENABLED:

                # _, final_traj, loss_rank, loss_cost = self.model.planning(
                #     bev_rgbs=bev_rgbs,
                #     trajs=trajs[:, :, 1:],
                #     gt_trajs=labels['gt_trajectory'][:, 1:],
                #     commands=command,
                #     target_points=target_points
                # )

                occupancy = torch.logical_or(labels['segmentation'][:, n_present:].squeeze(2),
                                             labels['pedestrian'][:, n_present:].squeeze(2))
                

                _, final_traj, resample_traj, loss_rank, loss_dict = self.model.planning(
                    bev_rgbs=bev_rgbs,
                    # trajs=trajs[:, :, 1:],
                    trajs= trajs,
                    gt_trajs=labels['gt_trajectory'][:, 1:],
                    commands=command,
                    target_points=target_points,
                    occupancy=occupancy,                    # ★ 新增
                    ped_bev_points=batch.get('ped_bev_points'),
                    ped_bev_valid_steps=batch.get('ped_bev_valid_steps'),
                )

                # print("final_traj",final_traj)

                                    # === turn metrics on VAL ===
                device = final_traj.device
                pred_last_x = final_traj[:, -1, 0]  # (B,)

                # command 是 batch['command']，通常是 list[str] 長度 B
                commands = command
                cmd_right   = torch.tensor([c == 'RIGHT'   for c in commands], device=device)
                cmd_left    = torch.tensor([c == 'LEFT'    for c in commands], device=device)
                cmd_forward = torch.tensor([c == 'FORWARD' for c in commands], device=device)

                turn_mask = (cmd_right | cmd_left)
                B = pred_last_x.shape[0]

                # ratio
                turn_ratio = turn_mask.float().mean()  # (0~1)

                # success by threshold
                ok_right = (pred_last_x >= 2.0) & cmd_right
                ok_left  = (pred_last_x <= -2.0) & cmd_left
                ok_turn  = ok_right | ok_left

                # accuracies (avoid div0)
                turn_cnt  = turn_mask.sum().clamp(min=1)
                right_cnt = cmd_right.sum().clamp(min=1)
                left_cnt  = cmd_left.sum().clamp(min=1)

                acc_turn  = ok_turn.float().sum()  / turn_cnt
                acc_right = ok_right.float().sum() / right_cnt
                acc_left  = ok_left.float().sum()  / left_cnt

                # 1) 回填到 loss（用 detach，確保只是 log）
                loss["turn_ratio_log"] = turn_ratio.detach()
                loss["acc_turn_log"]   = acc_turn.detach()
                loss["acc_right_log"]  = acc_right.detach()
                loss["acc_left_log"]   = acc_left.detach()

                # 2) 累積 epoch 統計（用 python int 計算加權平均）
                self.val_cmd_total   += int(B)
                self.val_turn_total  += int(turn_mask.sum().item())
                self.val_right_total += int(cmd_right.sum().item())
                self.val_left_total  += int(cmd_left.sum().item())

                self.val_turn_ok  += int(ok_turn.sum().item())
                self.val_right_ok += int(ok_right.sum().item())
                self.val_left_ok  += int(ok_left.sum().item())





                self.metric_planning_val(final_traj, labels['gt_trajectory'][:, 1:], occupancy)
                ped_traj_mask = batch.get('ped_traj_mask')
                if ped_traj_mask is not None:
                    ped_traj_mask = ped_traj_mask.to(device=final_traj.device, dtype=torch.bool)
                    pred_has_ped_traj = ped_traj_mask.any(dim=1)
                    if pred_has_ped_traj.any():
                        self.metric_planning_ped_only_on_ped_traj_val(
                            final_traj[pred_has_ped_traj],
                            labels['gt_trajectory'][pred_has_ped_traj, 1:],
                            labels['pedestrian'][pred_has_ped_traj, n_present:].squeeze(2),
                        )
                output = {**output,
                          'selected_traj': torch.cat([torch.zeros((B, 1, 3), device=final_traj.device), final_traj],
                                                     dim=1)}
            else:
                output = {**output, 'selected_traj': labels['gt_trajectory']}

        return output, labels, loss

    def prepare_future_labels(self, batch):
        labels = {}

        segmentation_labels = batch['segmentation']
        
        future_egomotion = batch['future_egomotion']
        gt_trajectory = batch['gt_trajectory']

        # print("future_egomotion",future_egomotion)
        # present frame hd map gt
        # hdmap_labels = batch['hdmap']
        # labels['hdmap'] = hdmap_labels[:, self.model.receptive_field - 1].long().contiguous()

        # gt trajectory
        labels['gt_trajectory'] = gt_trajectory

        pedestrian_labels = batch['pedestrian']
        pedestrian_labels_past = cumulative_warp_features(
            pedestrian_labels[:, :self.model.receptive_field].float(),
            future_egomotion[:, :self.model.receptive_field],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()[:, :-1]
        pedestrian_labels = cumulative_warp_features_reverse(
            pedestrian_labels[:, (self.model.receptive_field - 1):].float(),
            future_egomotion[:, (self.model.receptive_field - 1):],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()
        labels['pedestrian'] = torch.cat([pedestrian_labels_past, pedestrian_labels], dim=1)


        # Warp labels to present's reference frame
        segmentation_labels_past = cumulative_warp_features(
            segmentation_labels[:, :self.model.receptive_field].float(),
            future_egomotion[:, :self.model.receptive_field],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()[:, :-1]
        segmentation_labels = cumulative_warp_features_reverse(
            segmentation_labels[:, (self.model.receptive_field - 1):].float(),
            future_egomotion[:, (self.model.receptive_field - 1):],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()
        labels['segmentation'] = torch.cat([segmentation_labels_past, segmentation_labels], dim=1)

        


        # Warp instance labels to present's reference frame
        if self.cfg.INSTANCE_SEG.ENABLED:
            gt_instance = batch['instance']
            instance_center_labels = batch['centerness']
            instance_offset_labels = batch['offset']
            gt_instance_past = cumulative_warp_features(
                gt_instance[:, :self.model.receptive_field].float().unsqueeze(2),
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).long().contiguous()[:, :-1, 0]
            gt_instance = cumulative_warp_features_reverse(
                gt_instance[:, (self.model.receptive_field - 1):].float().unsqueeze(2),
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).long().contiguous()[:, :, 0]
            labels['instance'] = torch.cat([gt_instance_past, gt_instance], dim=1)

            instance_center_labels_past = cumulative_warp_features(
                instance_center_labels[:, :self.model.receptive_field],
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()[:, :-1]
            instance_center_labels = cumulative_warp_features_reverse(
                instance_center_labels[:, (self.model.receptive_field - 1):],
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()
            labels['centerness'] = torch.cat([instance_center_labels_past, instance_center_labels], dim=1)

            instance_offset_labels_past = cumulative_warp_features(
                instance_offset_labels[:, :self.model.receptive_field],
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()[:, :-1]
            instance_offset_labels = cumulative_warp_features_reverse(
                instance_offset_labels[:, (self.model.receptive_field - 1):],
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()
            labels['offset'] = torch.cat([instance_offset_labels_past, instance_offset_labels], dim=1)

        if self.cfg.INSTANCE_FLOW.ENABLED:
            instance_flow_labels = batch['flow']
            instance_flow_labels_past = cumulative_warp_features(
                instance_flow_labels[:, :self.model.receptive_field],
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()[:, :-1]
            instance_flow_labels = cumulative_warp_features_reverse(
                instance_flow_labels[:, (self.model.receptive_field - 1):],
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()
            labels['flow'] = torch.cat([instance_flow_labels_past, instance_flow_labels], dim=1)

        return labels

    def visualise(self, labels, output, batch_idx, prefix='train'):
        visualisation_video = visualise_output(labels, output, self.cfg)
        name = f'{prefix}_outputs'
        if prefix == 'val':
            name = name + f'_{batch_idx}'
        self.logger.experiment.add_video(name, visualisation_video, global_step=self.training_step_count, fps=2)

    def _batch_debug_summary(self, batch):
        summary = {}
        for key in ['command', 'ped_traj_has_data', 'ped_traj_mask', 'gt_trajectory', 'sample_trajectory']:
            if key not in batch:
                continue
            value = batch[key]
            if torch.is_tensor(value):
                item = {'shape': tuple(value.shape), 'dtype': str(value.dtype)}
                if value.numel() > 0 and torch.is_floating_point(value):
                    finite = torch.isfinite(value)
                    item['finite_ratio'] = float(finite.float().mean().item())
                    if finite.any():
                        item['min'] = float(value[finite].min().item())
                        item['max'] = float(value[finite].max().item())
                summary[key] = item
            elif isinstance(value, (list, tuple)):
                summary[key] = {'len': len(value), 'sample': value[:4]}
            else:
                summary[key] = str(value)
        return summary

    def _assert_finite_loss_dict(self, loss, batch, stage, batch_idx):
        bad = []
        for key, value in loss.items():
            if torch.is_tensor(value):
                chk = value.detach().float()
                if not torch.isfinite(chk).all():
                    bad.append(key)
            else:
                try:
                    fv = float(value)
                except Exception:
                    bad.append(key)
                    continue
                if not np.isfinite(fv):
                    bad.append(key)

        loss_total = sum(v if torch.is_tensor(v) else torch.tensor(float(v), device=self.device) for v in loss.values())
        if torch.is_tensor(loss_total):
            loss_total_chk = loss_total.detach().float()
            total_ok = torch.isfinite(loss_total_chk).all()
        else:
            total_ok = np.isfinite(float(loss_total))

        if bad or (not total_ok):
            debug = self._batch_debug_summary(batch)
            raise RuntimeError(
                f'Non-finite loss detected at stage={stage}, batch_idx={batch_idx}, keys={bad}, '
                f'loss_total_finite={bool(total_ok)}\n'
                f'batch_debug={debug}'
            )

    def training_step(self, batch, batch_idx):

        # if self.global_step == 0:
        #     for n,p in self.model.named_parameters():
        #         if p.requires_grad:
        #             print("[LEARN]", n, tuple(p.shape))

        output, labels, loss = self.shared_step(batch, True)
        self.training_step_count += 1

        # ★ 從 loss_dict 裡面把次數挖出來累加
        if "is_danger_cnt_log" in loss:
            self.train_danger_total += int(loss["is_danger_cnt_log"].item())
        if "danger_radar_cnt_log" in loss:
            self.train_danger_radar_total += int(loss["danger_radar_cnt_log"].item())
        if "danger_gt_cnt_log" in loss:
            self.train_danger_gt_total += int(loss["danger_gt_cnt_log"].item())
        if "danger_both_cnt_log" in loss:
            self.train_danger_both_total += int(loss["danger_both_cnt_log"].item())
        if "fwd_cnt_log" in loss:
            self.train_fwd_total += int(loss["fwd_cnt_log"].item())
            
        self._assert_finite_loss_dict(loss, batch, stage='train', batch_idx=batch_idx)

        for key, value in loss.items():
            self.logger.experiment.add_scalar('step_train_loss_' + key, value, global_step=self.training_step_count)

        # if self.training_step_count % self.cfg.VIS_INTERVAL == 0:
        # if self.training_step_count % 1 == 0:
        #     print("可視化")
        #     self.visualise(labels, output, batch_idx, prefix='train')


        opt = self.optimizers(use_pl_optimizer=True)
        if isinstance(opt, torch.optim.Optimizer):
            lrs = [pg["lr"] for pg in opt.param_groups]
            for i, lr in enumerate(lrs):
                self.logger.experiment.add_scalar(f"lr/group_{i}", lr, global_step=self.training_step_count)

        train_loss = loss['planning'] if 'planning' in loss else next(v for v in loss.values() if torch.is_tensor(v) and v.requires_grad)
        if not torch.isfinite(train_loss.detach().float()).all():
            raise RuntimeError(f'Non-finite training loss at batch_idx={batch_idx}')
        # if not torch.isfinite(loss_total):
        #     print("Non-finite loss! dumping stats...")
        #     for n,p in self.model.named_parameters():
        #         if p.requires_grad and torch.isnan(p).any():
        #             print("NaN param:", n)
            # raise RuntimeError("Non-finite loss")

        def _to_float_scalar(v):
            # v 可能是 tensor / python number
            if torch.is_tensor(v):
                if not torch.isfinite(v.detach().float()).all():
                    raise RuntimeError('Non-finite metric encountered during epoch accumulation')
                v = v.detach().float()
                if v.numel() != 1:
                    v = v.mean()
                return float(v.item())
            return float(v)

        # 只在 loss dict 有這些 key 時才累積
        if "risk_mask_ratio_log" in loss:
            self._epoch_risk_mask_sum += _to_float_scalar(loss["risk_mask_ratio_log"])
        if "dy_exceed_ratio_on_risk_log" in loss:
            self._epoch_dy_exceed_sum += _to_float_scalar(loss["dy_exceed_ratio_on_risk_log"])

        # 以「有 risk_mask_ratio 就算一次」為準（你也可改成兩者都存在才 +1）
        if "risk_mask_ratio_log" in loss:
            self._epoch_ratio_count += 1

        return train_loss

    def validation_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, False)
        ped_bev_effect = self._compute_val_ped_bev_effect(batch, labels, output)
        loss.update(ped_bev_effect)
        self._assert_finite_loss_dict(loss, batch, stage='val', batch_idx=batch_idx)
         # ✅ 把 val 的 log 也寫進 tensorboard
        for key, value in loss.items():
            self.logger.experiment.add_scalar('step_val_loss_' + key, value, global_step=self.training_step_count)

        # scores = self.metric_vehicle_val.compute()
        # self.log('step_val_seg_iou_dynamic', scores[1])
        # self.log('step_predicted_traj_x', output['selected_traj'][0, -1, 0])
        # self.log('step_target_traj_x', labels['gt_trajectory'][0, -1, 0])
        # self.log('step_predicted_traj_y', output['selected_traj'][0, -1, 1])
        # self.log('step_target_traj_y', labels['gt_trajectory'][0, -1, 1])

        if batch_idx == 0:
            print("先不可視化")
            # self.visualise(labels, output, batch_idx, prefix='val')

    def _compute_val_ped_bev_effect(self, batch, labels, output):
        if not self._model_forward_supports('ped_bev_map'):
            return {}
        ped_bev_map = batch.get('ped_bev_map')
        ped_bev_valid_steps = batch.get('ped_bev_valid_steps')
        if (ped_bev_map is None and ped_bev_valid_steps is None) or not self.cfg.PLANNING.ENABLED:
            return {}

        if ped_bev_valid_steps is not None:
            if not torch.is_tensor(ped_bev_valid_steps):
                ped_bev_valid_steps = torch.as_tensor(ped_bev_valid_steps)
            ped_bev_valid_steps = ped_bev_valid_steps.to(device=self.device, dtype=torch.bool, non_blocking=True)
            has_ped = ped_bev_valid_steps.any(dim=-1).any(dim=-1)
        else:
            if not torch.is_tensor(ped_bev_map):
                ped_bev_map = torch.as_tensor(ped_bev_map)
            ped_bev_map = ped_bev_map.to(device=self.device, dtype=torch.float32, non_blocking=True)
            if ped_bev_map.dim() == 3:
                ped_bev_map = ped_bev_map.unsqueeze(1)
            has_ped = ped_bev_map.flatten(1).abs().amax(dim=1) > 0
        if not has_ped.any():
            zero = torch.tensor(0.0, device=self.device)
            return {
                'ped_bev_traj_delta_mean_log': zero,
                'ped_bev_box_delta_mean_log': zero,
                'ped_bev_box_delta_on_ped_log': zero,
                'ped_bev_active_ratio_log': zero,
            }

        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']
        command = batch['command']
        trajs = batch['sample_trajectory']

        n_present = self.model.receptive_field
        occupancy = torch.logical_or(
            labels['segmentation'][:, n_present:].squeeze(2),
            labels['pedestrian'][:, n_present:].squeeze(2),
        )

        with_ped_xy = output['selected_traj'][:, 1:, :2].detach()
        gt_xy = labels['gt_trajectory'][:, 1:, :2].detach()
        zero_ped_bev_map = None
        ped_bev_map = batch.get('ped_bev_map')
        if ped_bev_map is not None:
            if not torch.is_tensor(ped_bev_map):
                ped_bev_map = torch.as_tensor(ped_bev_map)
            ped_bev_map = ped_bev_map.to(device=self.device, dtype=torch.float32, non_blocking=True)
            if ped_bev_map.dim() == 3:
                ped_bev_map = ped_bev_map.unsqueeze(1)
            zero_ped_bev_map = torch.zeros_like(ped_bev_map)

        zero_ped_traj_mask = None
        if batch.get('ped_traj_mask') is not None:
            zero_ped_traj_mask = torch.zeros_like(batch['ped_traj_mask'], dtype=torch.bool, device=self.device)

        zero_ped_valid_steps = None
        if batch.get('ped_traj_valid_steps') is not None:
            zero_ped_valid_steps = torch.zeros_like(batch['ped_traj_valid_steps'], dtype=torch.bool, device=self.device)

        model_kwargs = dict(
            rgb_224_seq=batch['rgb_224_seq'],
            seg_224_seq=batch['seg_224_seq'],
            ped_traj_preds=None,
            ped_traj_mask=zero_ped_traj_mask,
            ped_traj_valid_steps=zero_ped_valid_steps,
            ped_bev_map=zero_ped_bev_map,
        )
        if self._model_forward_supports('seg_id_224_seq'):
            model_kwargs['seg_id_224_seq'] = batch.get('seg_id_224_seq')
        if self._model_forward_supports('depth_224_seq'):
            model_kwargs['depth_224_seq'] = batch.get('depth_224_seq')

        with torch.no_grad():
            _, bev_rgbs_off = self.model(
                image,
                intrinsics,
                extrinsics,
                future_egomotion,
                **model_kwargs,
            )
            _, pred_off, _, _, _ = self.model.planning(
                bev_rgbs=bev_rgbs_off,
                trajs=trajs,
                gt_trajs=labels['gt_trajectory'][:, 1:],
                commands=command,
                target_points=batch['target_point'],
                occupancy=occupancy,
                ped_bev_points=None,
                ped_bev_valid_steps=None,
            )

        without_ped_xy = pred_off[..., :2].detach()
        traj_delta_t = torch.norm(with_ped_xy - without_ped_xy, dim=-1)
        traj_delta_per_sample = traj_delta_t.mean(dim=1)

        box_with_t = self.model.box_collision_loss_soft(with_ped_xy, occupancy, return_per_t=True).detach()
        box_without_t = self.model.box_collision_loss_soft(without_ped_xy, occupancy, return_per_t=True).detach()
        box_delta_per_sample = (box_without_t - box_with_t).mean(dim=1)

        ped_bev_points = batch.get('ped_bev_points')
        ped_bev_valid_steps = batch.get('ped_bev_valid_steps')
        triggered = torch.zeros_like(has_ped)
        if ped_bev_points is not None and ped_bev_valid_steps is not None:
            if not torch.is_tensor(ped_bev_points):
                ped_bev_points = torch.as_tensor(ped_bev_points)
            if not torch.is_tensor(ped_bev_valid_steps):
                ped_bev_valid_steps = torch.as_tensor(ped_bev_valid_steps)
            ped_bev_points = ped_bev_points.to(device=self.device, dtype=with_ped_xy.dtype, non_blocking=True)
            ped_bev_valid_steps = ped_bev_valid_steps.to(device=self.device, dtype=torch.bool, non_blocking=True)
            t = gt_xy.shape[1]
            ped_xy = ped_bev_points[:, :, :t]
            ped_valid = ped_bev_valid_steps[:, :, :t]
            if ped_valid.any():
                dist = torch.norm(gt_xy.unsqueeze(1) - ped_xy, dim=-1)
                inf = torch.full_like(dist, float('inf'))
                dist_valid = torch.where(ped_valid, dist, inf)
                min_dist_t = dist_valid.amin(dim=1)
                has_valid_t = ped_valid.any(dim=1)
                safe_dist = float(getattr(self.cfg, 'LOSS_PED_REPULSE_SAFE_DIST', 6.0))
                triggered = ((min_dist_t < safe_dist) & has_valid_t).any(dim=1)

        if has_ped.any():
            self._val_ped_traj_delta_sum_on_ped += float(traj_delta_per_sample[has_ped].sum().item())
            self._val_ped_traj_delta_count_on_ped += int(has_ped.sum().item())
            self._val_ped_box_delta_sum_on_ped += float(box_delta_per_sample[has_ped].sum().item())
            self._val_ped_box_delta_count_on_ped += int(has_ped.sum().item())
        if triggered.any():
            self._val_ped_traj_delta_sum_triggered += float(traj_delta_per_sample[triggered].sum().item())
            self._val_ped_traj_delta_count_triggered += int(triggered.sum().item())
            self._val_ped_box_delta_sum_triggered += float(box_delta_per_sample[triggered].sum().item())
            self._val_ped_box_delta_count_triggered += int(triggered.sum().item())

        return {
            'ped_bev_traj_delta_mean_log': traj_delta_per_sample.mean(),
            'ped_bev_box_delta_mean_log': box_delta_per_sample.mean(),
            'ped_bev_box_delta_on_ped_log': box_delta_per_sample[has_ped].mean(),
            'ped_bev_active_ratio_log': has_ped.float().mean(),
        }

    def shared_epoch_end(self, step_outputs, is_train):
        if not is_train:
            # scores = self.metric_vehicle_val.compute()
            # self.logger.experiment.add_scalar('epoch_val_all_seg_iou_dynamic', scores[1],
            #                                   global_step=self.training_step_count)
            # self.metric_vehicle_val.reset()

            if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
                scores = self.metric_pedestrian_val.compute()
                self.logger.experiment.add_scalar('epoch_val_all_seg_iou_pedestrian', scores[1],
                                                  global_step=self.training_step_count)
                self.metric_pedestrian_val.reset()

            if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
                for i, name in enumerate(self.hdmap_class):
                    scores = self.metric_hdmap_val[i].compute()
                    self.logger.experiment.add_scalar('epoch_val_hdmap_iou_' + name, scores[1],
                                                      global_step=self.training_step_count)
                    self.metric_hdmap_val[i].reset()

            if self.cfg.INSTANCE_SEG.ENABLED:
                scores = self.metric_panoptic_val.compute()
                for key, value in scores.items():
                    self.logger.experiment.add_scalar(f'epoch_val_all_ins_{key}_vehicle', value[1].item(),
                                                      global_step=self.training_step_count)
                self.metric_panoptic_val.reset()

            if self.cfg.PLANNING.ENABLED:
                scores = self.metric_planning_val.compute()
                for key, value in scores.items():
                    metric_name = 'epoch_val_plan_' + key
                    metric_value = value.mean()
                    self.logger.experiment.add_scalar(metric_name, metric_value,
                                                      global_step=self.training_step_count)
                    self.log(
                        metric_name,
                        metric_value,
                        prog_bar=(key in ['ADE', 'FDE']),
                        on_epoch=True,
                        logger=True,
                        sync_dist=True,
                    )
                self.metric_planning_val.reset()

                ped_total = int(self.metric_planning_ped_only_on_ped_traj_val.total.item())
                self.logger.experiment.add_scalar(
                    'epoch_val_plan_ped_only_on_ped_traj_count',
                    ped_total,
                    global_step=self.training_step_count,
                )
                if ped_total > 0:
                    ped_scores = self.metric_planning_ped_only_on_ped_traj_val.compute()
                    for key, value in ped_scores.items():
                        metric_name = 'epoch_val_plan_ped_only_on_ped_traj_' + key
                        metric_value = value.mean()
                        self.logger.experiment.add_scalar(
                            metric_name,
                            metric_value,
                            global_step=self.training_step_count,
                        )
                        self.log(
                            metric_name,
                            metric_value,
                            on_epoch=True,
                            logger=True,
                            sync_dist=True,
                        )
                    self.metric_planning_ped_only_on_ped_traj_val.reset()

        self.logger.experiment.add_scalar('epoch_segmentation_weight',
                                          1 / (2 * torch.exp(self.model.segmentation_weight)),
                                          global_step=self.training_step_count)

        if self.cfg.PLANNING.ENABLED:
            self.logger.experiment.add_scalar('epoch_planning_weight', 1 / (2 * torch.exp(self.model.planning_weight)),
                                              global_step=self.training_step_count)

    # def training_epoch_end(self, step_outputs):
    #     self.shared_epoch_end(step_outputs, True)

    # def validation_epoch_end(self, step_outputs):
    #     self.shared_epoch_end(step_outputs, False)

    def on_train_epoch_end(self):
        self.shared_epoch_end(None, True)

        # ===== compute epoch-level quantiles for TRAIN =====
        if len(self._epoch_gt_abs_x_train) > 0:
            x = torch.cat(self._epoch_gt_abs_x_train, dim=0)  # CPU tensor
            dx = torch.cat(self._epoch_gt_step_dx_train, dim=0)

            gt_abs_x_p95 = torch.quantile(x, 0.95)
            gt_abs_x_mean = x.mean()

            gt_step_dx_p95 = torch.quantile(dx, 0.95)
            gt_step_dx_mean = dx.mean()

            gs = self.training_step_count
            self.logger.experiment.add_scalar("epoch_train_gt_abs_x_mean", float(gt_abs_x_mean), global_step=gs)
            self.logger.experiment.add_scalar("epoch_train_gt_abs_x_p95",  float(gt_abs_x_p95),  global_step=gs)
            self.logger.experiment.add_scalar("epoch_train_gt_step_dx_mean", float(gt_step_dx_mean), global_step=gs)
            self.logger.experiment.add_scalar("epoch_train_gt_step_dx_p95",  float(gt_step_dx_p95),  global_step=gs)

            print("epoch_train_gt_abs_x_mean",float(gt_abs_x_mean))
            print("epoch_train_gt_abs_x_p95",float(gt_abs_x_p95))
            print("epoch_train_gt_step_dx_mean",float(gt_step_dx_mean))
            print("epoch_train_gt_step_dx_p95",float(gt_step_dx_p95))

            if self._epoch_ratio_count > 0:
                risk_mask_ratio_epoch = self._epoch_risk_mask_sum / self._epoch_ratio_count
                dy_exceed_ratio_epoch = self._epoch_dy_exceed_sum / self._epoch_ratio_count

                gs = self.training_step_count  # 跟你其他 epoch 指標一致用 global_step
                self.logger.experiment.add_scalar("epoch_train_risk_mask_ratio", risk_mask_ratio_epoch, global_step=gs)
                self.logger.experiment.add_scalar("epoch_train_dy_exceed_ratio_on_risk", dy_exceed_ratio_epoch, global_step=gs)

                print("epoch_train_risk_mask_ratio",risk_mask_ratio_epoch)
                print("epoch_train_dy_exceed_ratio_on_risk",dy_exceed_ratio_epoch)

        # clear buffers
        self._epoch_gt_abs_x_train.clear()
        self._epoch_gt_step_dx_train.clear()

        # ★ 在 Epoch 結束時印出霸氣的總結報告
        if getattr(self, "train_fwd_total", 0) > 0:
            danger_ratio = self.train_danger_total / self.train_fwd_total
            print(f"\n{'='*60}")
            print(f"🎯 [Epoch {self.current_epoch} 避障觸發超詳細總結]")
            print(f"  - 收到 FORWARD 指令總數: {self.train_fwd_total} 筆")
            print(f"  - 總觸發強制避障 (is_danger): {self.train_danger_total} 筆 (佔比 {danger_ratio:.2%})")
            print(f"    ├─ 純因 [物理雷達] 觸發: {self.train_danger_radar_total - self.train_danger_both_total} 筆")
            print(f"    ├─ 純因 [人類示範] 觸發: {self.train_danger_gt_total - self.train_danger_both_total} 筆")
            print(f"    └─ [雷達 & 人類] 同時觸發: {self.train_danger_both_total} 筆")
            print(f"{'='*60}\n")
            
            
            # 同時寫進 TensorBoard 方便看趨勢
            self.logger.experiment.add_scalar('epoch_train_danger_ratio', danger_ratio, global_step=self.training_step_count)

    
    # def configure_optimizers(self):
    #     head_params = []
    #     clip_last_params = []
    #     misc_params = []
    #     for n, p in self.model.named_parameters():
    #         if not p.requires_grad:
    #             continue
    #         # CLIP 視覺塔最後幾層 + LN → 小學習率
    #         if ("vlm.model.visual.transformer.resblocks" in n) or ("vlm.model.visual.ln_" in n):
    #             clip_last_params.append(p)
    #         # Head 組：img_head（若保留）、traj_enc、cross_attn、score_mlp、txt_to_vis
    #         elif ("vlm.img_head" in n) or ("vlm.traj_enc" in n) or ("vlm.cross_attn" in n) \
    #             or ("vlm.score_mlp" in n) or ("vlm.txt_to_vis" in n):
    #             head_params.append(p)
    #         elif n.endswith("planning_weight"):
    #             misc_params.append(p)

    #     # 保障至少有 head 或 clip 組
    #     assert len(head_params) > 0, "找不到可訓練 head 參數（traj_enc / cross_attn / score_mlp / txt_to_vis / img_head）"
    #     assert len(clip_last_params) > 0, "找不到已解凍的 CLIP 視覺塔層。"

    #     head_lr = getattr(self.cfg.OPTIMIZER, "HEAD_LR", 1e-4)
    #     clip_lr = getattr(self.cfg.OPTIMIZER, "CLIP_LR", 1e-6)
    #     wd      = self.cfg.OPTIMIZER.WEIGHT_DECAY

    #     param_groups = [
    #         {"params": head_params,      "lr": head_lr, "weight_decay": wd},
    #         {"params": clip_last_params, "lr": clip_lr, "weight_decay": wd},
    #     ]
    #     if len(misc_params) > 0:
    #         param_groups.append({"params": misc_params, "lr": head_lr, "weight_decay": wd})

    #     optimizer = torch.optim.AdamW(param_groups)
    #     return optimizer

    def configure_optimizers(self):
        head_params = []
        clip_last_params = []
        misc_params = []
        print("CLIP 參數")
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            
            # 1. CLIP 視覺塔最後幾層 + LN
            if ("vlm.clip.visual.transformer.resblocks" in n) or ("vlm.clip.visual.ln_" in n):
                clip_last_params.append(p)
            
            # 2. 其他特殊的 scalar weight
            elif n.endswith("planning_weight"):
                misc_params.append(p)
                
            # 3. 剩下的所有 vlm 參數（包含 cross_attn, GRU, mix_gate 等所有你寫的結構）
            elif n.startswith("vlm.") and not n.startswith("vlm.clip."):
                head_params.append(p)
                
            else:
                print(f"[警告] 參數位歸檔，請檢查: {n}")
                # 為了安全，沒分類到的通通丟給 head_params
                head_params.append(p)


        assert len(head_params) > 0 or len(misc_params) > 0, "找不到可訓練參數"

        head_lr = getattr(self.cfg.OPTIMIZER, "LR", 2e-4)
        clip_lr = getattr(self.cfg.OPTIMIZER, "CLIP_LR", 2e-5)
        wd      = self.cfg.OPTIMIZER.WEIGHT_DECAY
        # clip_lr = 0.1

        print("head_lr : ", head_lr)
        print("clip_lr : ", clip_lr)

        param_groups = []
        if len(head_params) > 0:
            param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": wd})
        if len(clip_last_params) > 0:
            param_groups.append({"params": clip_last_params, "lr": clip_lr, "weight_decay": wd})
        if len(misc_params) > 0:
            param_groups.append({"params": misc_params, "lr": head_lr, "weight_decay": wd})

        optimizer = torch.optim.AdamW(param_groups)
        print("[PED_FT] constant LR finetune")
        return optimizer
