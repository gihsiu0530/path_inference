import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

from stp3.config import get_cfg


# from stp3.model_vlm.vlm_planner import VLM_STP3  # 新


# from stp3.model_vlm.mult_planner import VLM_STP3


from stp3.model_vlm.add_velocity import VLM_STP3_Gen as VLM_STP3

# from stp3.model_vlm.depth_planner import VLM_STP3_Gen as VLM_STP3




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
            self.metric_planning_val_speed = PlanningMetric(self.cfg, self.cfg.N_FUTURE_FRAMES)  # ★ 新增
            self.model.planning_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.training_step_count = 0


    

    def on_train_epoch_start(self):
        # self._update_tf_ratio()
        # print("start")
        # 課程式下降：coarse loss 從 0.3 緩降到 0.05
        start_w = float(getattr(self.cfg, "LOSS_CRS_START", 0.3))
        end_w   = float(getattr(self.cfg, "LOSS_CRS_END", 0.05))
        E       = int(getattr(self.cfg, "EPOCHS", 20))
        t = min(1.0, self.current_epoch / max(1, E-1))
        w = end_w + (start_w - end_w) * (1 - math.cos(math.pi * t)) * 0.5  # cosine decay
        setattr(self.model.cfg, "LOSS_COARSE_W", w)  # 直接覆蓋 cfg，planning() 會讀它
        self.log("coarse_w", w, prog_bar=False, on_epoch=True, logger=True)

    def shared_step(self, batch, is_train):
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']
        command = batch['command']
        trajs = batch['sample_trajectory']
        target_points = batch['target_point']
        B = trajs.size(0)

        
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


        # 多幀
        output, bev_rgbs = self.model(
            image, intrinsics, extrinsics, future_egomotion,
            rgb_224_seq=batch['rgb_224_seq'],   # (B,T_rf,H,W,3) uint8
            seg_224_seq=batch['seg_224_seq']    # (B,T_rf,H,W,3) uint8
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

                pl_loss, final_traj, final_traj_with_speed, loss_rank, loss_cost = self.model.planning(
                    bev_rgbs=bev_rgbs,
                    # trajs=trajs[:, :, 1:],                 # 生成式不使用，但保留相容參數
                    trajs=trajs,                 # 生成式不使用，但保留相容參數
                    gt_trajs=labels['gt_trajectory'][:, 1:],
                    commands=command,
                    target_points=target_points,
                    occupancy=occupancy                    # ★ 新增
                )


                # loss['planning'] = planning_factor * pl_loss
                s = self.model.planning_weight
                loss['planning'] = torch.exp(-s) * pl_loss + s
                loss['loss_rank'] = loss_rank
                loss['loss_cost'] = loss_cost

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
                
                with torch.no_grad():
                    _, final_traj, final_traj_with_speed, _, _ = self.model.planning(
                        bev_rgbs=bev_rgbs,
                        # trajs=trajs[:, :, 1:],
                        trajs= trajs,
                        gt_trajs=labels['gt_trajectory'][:, 1:],
                        commands=command,
                        target_points=target_points,
                        occupancy=occupancy                    # ★ 新增
                    )






                self.metric_planning_val(final_traj, labels['gt_trajectory'][:, 1:], occupancy)
                self.metric_planning_val_speed(final_traj_with_speed, labels['gt_trajectory'][:, 1:], occupancy)

                output = {**output,
                          'selected_traj': torch.cat([torch.zeros((B, 1, 3), device=final_traj.device), final_traj],
                                                     dim=1)}
            else:
                output = {**output, 'selected_traj': labels['gt_trajectory']}

        return output, labels, loss

    def prepare_future_labels(self, batch):
        labels = {}

        segmentation_labels = batch['segmentation']
        # hdmap_labels = batch['hdmap']
        future_egomotion = batch['future_egomotion']
        gt_trajectory = batch['gt_trajectory']

        # present frame hd map gt
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

    def training_step(self, batch, batch_idx):

        # if self.global_step == 0:
        #     for n,p in self.model.named_parameters():
        #         if p.requires_grad:
        #             print("[LEARN]", n, tuple(p.shape))

        output, labels, loss = self.shared_step(batch, True)
        self.training_step_count += 1
        for key, value in loss.items():
            self.logger.experiment.add_scalar('step_train_loss_' + key, value, global_step=self.training_step_count)
        if self.training_step_count % self.cfg.VIS_INTERVAL == 0:
            print("先不可視化")
            # self.visualise(labels, output, batch_idx, prefix='train')


        opt = self.optimizers(use_pl_optimizer=True)
        if isinstance(opt, torch.optim.Optimizer):
            lrs = [pg["lr"] for pg in opt.param_groups]
            for i, lr in enumerate(lrs):
                self.logger.experiment.add_scalar(f"lr/group_{i}", lr, global_step=self.training_step_count)
                
        for k, v in loss.items():
            if not torch.isfinite(v).all():
                print(f"[NON-FINITE LOSS] {k} = {v}")
                t = v.detach()
                mask = torch.isfinite(t)
                if mask.any():
                    print(f"  finite range: min={t[mask].min().item():.6g}, max={t[mask].max().item():.6g}")

        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, False)
        # scores = self.metric_vehicle_val.compute()
        # self.log('step_val_seg_iou_dynamic', scores[1])
        # self.log('step_predicted_traj_x', output['selected_traj'][0, -1, 0])
        # self.log('step_target_traj_x', labels['gt_trajectory'][0, -1, 0])
        # self.log('step_predicted_traj_y', output['selected_traj'][0, -1, 1])
        # self.log('step_target_traj_y', labels['gt_trajectory'][0, -1, 1])

        if batch_idx == 0:
            print("先不可視化")
            # self.visualise(labels, output, batch_idx, prefix='val')

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
                # 幾何軌跡（公平評估）
                scores_geom = self.metric_planning_val.compute()
                for k, v in scores_geom.items():
                    self.logger.experiment.add_scalar(f'epoch_val_plan_{k}', v.mean(), global_step=self.training_step_count)
                    # 2) 新增：寫到 Lightning callback metrics（給 ModelCheckpoint 用）
                self.metric_planning_val.reset()
                if 'L2' in scores_geom:
                    self.log('epoch_val_plan_L2',  scores_geom['L2'].mean(),  prog_bar=True, on_epoch=True, logger=True, sync_dist=False)
                if 'collision' in scores_geom:
                    self.log('epoch_val_plan_col', scores_geom['collision'].mean(), prog_bar=False, on_epoch=True, logger=True, sync_dist=False)


                # 含速度軌跡（re-timed 行為觀察）
                if hasattr(self, "metric_planning_val_speed"):
                    scores_speed = self.metric_planning_val_speed.compute()
                    for k, v in scores_speed.items():
                        self.logger.experiment.add_scalar(f'epoch_val_plan_speed_{k}', v.mean(), global_step=self.training_step_count)
                    self.metric_planning_val_speed.reset()

                

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

    def on_validation_epoch_end(self):
        self.shared_epoch_end(None, False)

    
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
            if ("vlm.clip.visual.transformer.resblocks" in n) or ("vlm.clip.visual.ln_" in n):
                print(n)
                clip_last_params.append(p)
            # elif ("vlm.img_head" in n) or ("vlm.traj_enc" in n) or ("vlm.cross_attn" in n) \
            # or ("vlm.score_mlp" in n) or ("vlm.txt_to_vis" in n) or ("vlm.ego_mlp" in n) \
            # or ("vlm.traj_head" in n) \
            # or ("vlm.decoder_gru" in n) or ("vlm.delta_head" in n) \
            # or ("vlm.h0_proj" in n) or ("vlm.time_pe" in n) or ("vlm.time_queries" in n)\
            # or ("vlm.xy_embedder" in n) or ("vlm.traj_coarse" in n):
            #     head_params.append(p)
            elif ("vlm.img_head" in n) or ("vlm.traj_enc" in n) or ("vlm.cross_attn" in n) \
                or ("vlm.score_mlp" in n) or ("vlm.txt_to_vis" in n) or ("vlm.ego_mlp" in n) \
                or ("vlm.traj_head" in n) or ("vlm.decoder_gru" in n) or ("vlm.delta_head" in n) \
                or ("vlm.h0_proj" in n) or ("vlm.time_pe" in n) or ("vlm.time_queries" in n) \
                or ("vlm.xy_embedder" in n) or ("vlm.traj_coarse" in n) \
                or ("vlm.depth_enc" in n) or ("vlm.gate_depth" in n) or ("vlm.depth_affine" in n):
                    head_params.append(p)


            # 無GRU
            # elif ("vlm.img_head" in n) or ("vlm.traj_enc" in n) or ("vlm.cross_attn" in n) \
            #     or ("vlm.score_mlp" in n) or ("vlm.txt_to_vis" in n) or ("vlm.ego_mlp" in n) \
            #     or ("vlm.traj_head" in n):   # ★ 新增
            #     head_params.append(p)
            elif n.endswith("planning_weight"):
                misc_params.append(p)
            # else :
            #     print("n",n)
            #     raise RuntimeError("參數位歸檔")

        assert len(head_params) > 0, "找不到可訓練 head 參數"
        assert len(clip_last_params) > 0, "找不到已解凍的 CLIP 視覺塔層。"

        head_lr = getattr(self.cfg.OPTIMIZER, "LR", 2e-4)
        clip_lr = getattr(self.cfg.OPTIMIZER, "CLIP_LR", 2e-5)
        wd      = self.cfg.OPTIMIZER.WEIGHT_DECAY
        # clip_lr = 0.1

        print("head_lr : ", head_lr)
        print("clip_lr : ", clip_lr)

        optimizer = torch.optim.AdamW([
            {"params": head_params,      "lr": head_lr, "weight_decay": wd},
            {"params": clip_last_params, "lr": clip_lr, "weight_decay": wd},
            # {"params": clip_last_params, "lr": 0.1, "weight_decay": wd},
            {"params": misc_params,      "lr": head_lr, "weight_decay": wd},
        ])

        # === OneCycleLR ===
        # -------------直接用看的，有改資料集記得改-----------
        # steps_per_epoch = 1376    2923
        steps_per_epoch = 5847
        total_steps = steps_per_epoch * self.cfg.EPOCHS

        print("epoch", self.cfg.EPOCHS)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[head_lr, clip_lr, head_lr],  # 對應 param groups
            total_steps=total_steps,
            pct_start=0.15,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e3,
            three_phase=False
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "onecycle"
            }
        }

