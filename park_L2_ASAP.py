# evaluate.py — drop-in replacement with pluggable model swapping
from argparse import ArgumentParser
from PIL import Image, ImageDraw
import torch
import torch.utils.data
import numpy as np
import torchvision
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import pathlib
import datetime
import importlib
import inspect
import csv
import time


from stp3.data_0512_graduate.NuscenesData_0624_ASAP import FuturePredictionDataset


from stp3.config import get_cfg as get_cfg_codex
import stp3.trainer_codex_seg_ASAP as trainer_module
from stp3.trainer_codex_seg_ASAP import TrainingModule
# 使用你提供的本地 metrics.py（相容 Lightning/torchmetrics）
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric

from stp3.utils.network import preprocess_batch, NormalizeInverse
from stp3.utils.geometry import mat2pose_vec, pose_vec2mat
from stp3.utils.instance import predict_instance_segmentation_and_trajectories
from stp3.utils.visualisation import make_contour


# python park_L2.py   --checkpoint /home/cyc/ST-P3_please/tensorboard_logs/12May2026at02_21_56CST_letmesleep_Planning_super_ft/lightning_logs/version_0/checkpoints/epoch=04-pedonly-epoch_val_plan_ped_only_on_ped_traj_obj_box_col=0.0009.ckpt --dataroot /home/cyc/dataset/0504_what_up/resample/video6


def mk_save_dir():
    now = datetime.datetime.now()
    string = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
    save_path = pathlib.Path('inference/imgs') / string
    save_path.mkdir(parents=True, exist_ok=False)
    return save_path

def _load_trainer_for_eval(checkpoint_path, strict=True, model_module=None, model_class=None):
    """
    直接從 Lightning checkpoint 建立 trainer，並在需要時切換到自訂 model class。
    這樣可以評估在其他 workspace 訓練、且 model class 與目前 trainer 預設不同的權重。
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters")
    if hparams is None:
        raise RuntimeError(f"checkpoint 缺少 hyper_parameters: {checkpoint_path}")

    trainer_module.get_cfg = get_cfg_codex
    if hasattr(trainer_module, "base_trainer"):
        trainer_module.base_trainer.get_cfg = get_cfg_codex
    trainer = TrainingModule(hparams)
    if model_module and model_class:
        trainer = _swap_in_custom_model(trainer, model_module, model_class)

    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise RuntimeError(f"checkpoint 缺少 state_dict: {checkpoint_path}")

    incompatible = trainer.load_state_dict(state_dict, strict=strict)
    if incompatible.missing_keys:
        print(f"[WARN] missing keys: {len(incompatible.missing_keys)}")
        for key in incompatible.missing_keys:
            print(f"  missing: {key}")
    if incompatible.unexpected_keys:
        print(f"[WARN] unexpected keys: {len(incompatible.unexpected_keys)}")
        for key in incompatible.unexpected_keys:
            print(f"  unexpected: {key}")

    print(f"Loaded weights from \n {checkpoint_path}")
    trainer.eval()
    return trainer

def _swap_in_custom_model(trainer, model_module: str, model_class: str):
    """
    用自訂模型替換 trainer.model，但保留 trainer 的資料前處理與評估流程。
    要求你的模型建構子接受 cfg：__init__(self, cfg)。
    """
    if not model_module or not model_class:
        return trainer  # 沒有指定就不替換

    mod = importlib.import_module(model_module)
    klass = getattr(mod, model_class)
    try:
        new_model = klass(trainer.model.cfg)
    except TypeError as e:
        raise RuntimeError(
            f"無法用 cfg 建構 {model_module}.{model_class}，請確認你的模型 __init__(self, cfg) 介面。"
        ) from e

    # 將 trainer 內的 model 指到你的模型
    trainer.model = new_model
    print(f"[INFO] 已替換為自訂模型：{model_module}.{model_class}")
    return trainer

def _build_valid_occupancy(labels, n_present):
    seg = labels.get('segmentation')
    ped = labels.get('pedestrian')
    if seg is None or ped is None:
        return None
    if seg.numel() == 0 or ped.numel() == 0:
        return None
    if seg.dim() < 5 or ped.dim() < 5:
        return None
    if seg.shape[-1] <= 1 or seg.shape[-2] <= 1:
        return None
    if ped.shape[-1] <= 1 or ped.shape[-2] <= 1:
        return None
    if seg.shape[1] <= n_present or ped.shape[1] <= n_present:
        return None
    return torch.logical_or(
        labels['segmentation'][:, n_present:].squeeze(2),
        labels['pedestrian'][:, n_present:].squeeze(2)
    )

def _prepare_l2_labels(batch):
    labels = {
        'gt_trajectory': batch['gt_trajectory'],
    }
    if 'segmentation' in batch:
        labels['segmentation'] = batch['segmentation'].long().contiguous()
    if 'pedestrian' in batch:
        labels['pedestrian'] = batch['pedestrian'].long().contiguous()
    if 'hdmap' in batch and torch.is_tensor(batch['hdmap']) and batch['hdmap'].numel() > 0:
        labels['hdmap'] = batch['hdmap'].long().contiguous()
    return labels

def _sync_cuda_if_needed(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def _normalize_ablation_modalities(modalities):
    if modalities is None:
        return set()
    if isinstance(modalities, str):
        modalities = [modalities]
    normalized = set()
    aliases = {
        "rgb": "image",
        "img": "image",
        "image": "image",
        "seg": "seg",
        "semantic": "seg",
        "segmentation": "seg",
    }
    for item in modalities:
        for token in str(item).replace(",", " ").split():
            token = token.strip().lower()
            if not token:
                continue
            if token not in aliases:
                raise ValueError(
                    f"未知 ablation modality: {token}. 可用: image/rgb/img, seg/semantic/segmentation"
                )
            normalized.add(aliases[token])
    return normalized

def _zero_batch_tensor(batch, key):
    tensor = batch.get(key)
    if torch.is_tensor(tensor):
        batch[key] = torch.zeros_like(tensor)

def _apply_input_ablation(batch, modalities):
    """
    Black out selected model inputs for ablation experiments.

    Only input tensors are modified. Ground-truth trajectory and evaluation
    labels are left untouched so L2 remains comparable across runs.
    """
    modalities = _normalize_ablation_modalities(modalities)
    if not modalities:
        return

    if "image" in modalities:
        _zero_batch_tensor(batch, "rgb_224_seq")
        _zero_batch_tensor(batch, "image")
    if "seg" in modalities:
        _zero_batch_tensor(batch, "seg_224_seq")
        _zero_batch_tensor(batch, "seg_id_224_seq")

def _ablation_output_tag(modalities, ablation_name=None):
    if ablation_name:
        return str(ablation_name).strip()
    modalities = _normalize_ablation_modalities(modalities)
    if not modalities:
        return ""
    ordered = [name for name in ("image", "seg") if name in modalities]
    return "ablate_" + "_".join(ordered)

def _trajectory_l2(pred_traj: torch.Tensor, gt_trajectory: torch.Tensor) -> torch.Tensor:
    """
    pred_traj:      (B,Tp,2/3), model output, starts at future step 1
    gt_trajectory:  (B,Tg,3), dataset GT, gt[:,0] is current (0,0,0)
    return:         (B,T) per-step L2 in the same relative x/y frame
    """
    gt_future = gt_trajectory[:, 1:, :2]
    pred_xy = pred_traj[:, :, :2]
    T = min(pred_xy.shape[1], gt_future.shape[1])
    if T <= 0:
        return torch.empty(pred_xy.shape[0], 0, device=pred_xy.device)
    return torch.linalg.norm(pred_xy[:, :T] - gt_future[:, :T].to(pred_xy.device), dim=-1)


def _trajectory_xy_error(pred_traj: torch.Tensor, gt_trajectory: torch.Tensor) -> torch.Tensor:
    """
    Returns signed per-step error (pred - gt) in model coordinates:
    x is lateral, y is longitudinal.
    """
    gt_future = gt_trajectory[:, 1:, :2]
    pred_xy = pred_traj[:, :, :2]
    T = min(pred_xy.shape[1], gt_future.shape[1])
    if T <= 0:
        return torch.empty(pred_xy.shape[0], 0, 2, device=pred_xy.device)
    return pred_xy[:, :T] - gt_future[:, :T].to(pred_xy.device)


def _new_error_stats():
    return {
        "count": 0,
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "abs_min": float("inf"),
        "abs_max": 0.0,
        "max_location": None,
    }


def _update_error_stats(stats, values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    vals = values.detach().float().reshape(-1).cpu()
    abs_vals = vals.abs()
    stats["count"] += int(vals.numel())
    stats["abs_sum"] += float(abs_vals.sum().item())
    stats["sq_sum"] += float((vals * vals).sum().item())
    stats["abs_min"] = min(stats["abs_min"], float(abs_vals.min().item()))
    stats["abs_max"] = max(stats["abs_max"], float(abs_vals.max().item()))


def _update_error_stats_max_location(stats, abs_error: float, location: dict) -> None:
    if abs_error >= stats["abs_max"]:
        stats["abs_max"] = abs_error
        stats["max_location"] = dict(location)


def _finalize_error_stats(stats):
    count = stats["count"]
    if count <= 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan"), None
    return (
        count,
        stats["abs_min"],
        stats["abs_max"],
        stats["abs_sum"] / count,
        float(np.sqrt(stats["sq_sum"] / count)),
        stats["max_location"],
    )

def _input_history_from_egomotion(future_egomotion: torch.Tensor):
    """
    Reconstruct input past-frame positions relative to the current frame.
    Returns points in plot coordinates: x_left, y_front, plus relative yaw.
    Last point is current ego (0, 0, 0).
    """
    fego = future_egomotion.detach().float().cpu()
    t_rf = fego.shape[0]
    if t_rf <= 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    mats = pose_vec2mat(fego)
    points = []
    yaws = []
    for t in range(t_rf):
        if t == t_rf - 1:
            points.append([0.0, 0.0])
            yaws.append(0.0)
            continue
        transform = mats[t]
        for k in range(t + 1, t_rf - 1):
            transform = torch.mm(transform, mats[k])
        pose = mat2pose_vec(transform)
        x_forward = float(pose[0])
        y_left = float(pose[1])
        points.append([y_left, x_forward])
        yaws.append(float(pose[5]))
    return np.asarray(points, dtype=np.float32), np.asarray(yaws, dtype=np.float32)

def _input_history_xy(future_egomotion: torch.Tensor) -> np.ndarray:
    return _input_history_from_egomotion(future_egomotion)[0]

def save_inference_plot(rgb_224: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                        l2: np.ndarray, t_ref: int, seq_idx: int, out_dir: pathlib.Path,
                        input_xy: np.ndarray = None, input_yaw: np.ndarray = None):
    if pred.shape[1] < 2 or gt.shape[1] < 2:
        return

    img = Image.fromarray(rgb_224.astype(np.uint8), mode="RGB")
    canvas_size = 512
    margin = 48
    traj_panel = Image.new("RGB", (canvas_size, canvas_size), (250, 250, 250))
    draw = ImageDraw.Draw(traj_panel)

    origin_xy = np.zeros((1, 2), dtype=np.float32)
    pred_xy = pred[:, :2].astype(np.float32)
    gt_xy = gt[:, :2].astype(np.float32)
    gt_future_xy = gt_xy[1:] if gt_xy.shape[0] > 1 else gt_xy
    pred_plot_xy = np.concatenate([origin_xy, pred_xy], axis=0) if pred_xy.size else pred_xy
    gt_plot_xy = np.concatenate([origin_xy, gt_future_xy], axis=0) if gt_future_xy.size else gt_future_xy
    input_xy = np.asarray(input_xy, dtype=np.float32) if input_xy is not None else np.zeros((0, 2), dtype=np.float32)
    plot_parts = [pred_plot_xy]
    if gt_plot_xy.size:
        plot_parts.append(gt_plot_xy)
    if input_xy.size:
        plot_parts.append(input_xy)
    all_xy = np.concatenate(plot_parts, axis=0)

    center_x = canvas_size // 2
    lateral_extent = float(np.max(np.abs(all_xy[:, 0]))) if all_xy.size else 1.0
    x_extent = max(lateral_extent + 1.0, 5.0)
    y_min = min(float(np.min(all_xy[:, 1])) if all_xy.size else 0.0, 0.0) - 1.0
    y_max = max(float(np.max(all_xy[:, 1])) if all_xy.size else 0.0, 0.0) + 1.0
    if y_max - y_min < 5.0:
        pad = (5.0 - (y_max - y_min)) / 2.0
        y_min -= pad
        y_max += pad
    scale_x = (canvas_size - 2 * margin) / (2.0 * x_extent)
    scale_y = (canvas_size - 2 * margin) / (y_max - y_min)
    scale = min(scale_x, scale_y)
    origin_y = (canvas_size - margin) + y_min * scale

    def to_pixel(x_left: float, y_front: float):
        return (
            center_x - float(x_left) * scale,
            origin_y - float(y_front) * scale,
        )

    def choose_tick_step(extent_m: float) -> float:
        if extent_m <= 6.0:
            return 1.0
        if extent_m <= 12.0:
            return 2.0
        if extent_m <= 30.0:
            return 5.0
        return 10.0

    tick_step = choose_tick_step(max(x_extent, y_max - y_min))
    x_tick = -np.floor(x_extent / tick_step) * tick_step
    while x_tick <= x_extent + 1e-6:
        px, _ = to_pixel(x_tick, 0.0)
        if margin <= px <= canvas_size - margin:
            color = (210, 210, 210) if abs(x_tick) > 1e-6 else (120, 120, 120)
            width = 1 if abs(x_tick) > 1e-6 else 2
            draw.line((px, margin, px, canvas_size - margin + 8), fill=color, width=width)
            if abs(x_tick) > 1e-6:
                draw.text((px - 12, canvas_size - margin + 12), f"{x_tick:g}", fill=(70, 70, 70))
        x_tick += tick_step

    y_tick = np.floor(y_min / tick_step) * tick_step
    while y_tick <= y_max + 1e-6:
        _, py = to_pixel(0.0, y_tick)
        if margin <= py <= canvas_size - margin:
            color = (210, 210, 210) if abs(y_tick) > 1e-6 else (120, 120, 120)
            width = 1 if abs(y_tick) > 1e-6 else 2
            draw.line((margin - 8, py, canvas_size - margin, py), fill=color, width=width)
            draw.text((8, py - 6), f"{y_tick:g}", fill=(70, 70, 70))
        y_tick += tick_step

    draw.text((center_x + 10, margin + 4), "y_front (m)", fill=(40, 40, 40))
    draw.text((canvas_size - margin - 82, canvas_size - margin + 30), "x_left (m)", fill=(40, 40, 40))
    draw.text((margin, 8), f"seq={seq_idx} | ts={t_ref}", fill=(30, 30, 30))
    if l2.size:
        draw.text((margin, 28), f"L2 mean={float(np.mean(l2)):.2f}m final={float(l2[-1]):.2f}m", fill=(30, 30, 30))
    draw.text((margin, 48), "Input 3s green | GT blue | Pred red", fill=(30, 30, 30))

    ego_r = 6
    draw.ellipse(
        (center_x - ego_r, origin_y - ego_r, center_x + ego_r, origin_y + ego_r),
        fill=(220, 40, 40),
        outline=(120, 0, 0),
    )

    def draw_traj(points_xy: np.ndarray, line_color, dot_color, width: int):
        pts = [to_pixel(float(x), float(y)) for x, y in points_xy[:, :2]]
        if len(pts) == 1:
            px, py = pts[0]
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=dot_color)
        elif len(pts) > 1:
            draw.line(pts, fill=line_color, width=width)
            for px, py in pts:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=dot_color)

    if input_xy.size:
        draw_traj(input_xy, (40, 150, 90), (20, 105, 55), 4)
        if input_yaw is not None:
            arrow_len_m = 0.7
            for (x_left, y_front), yaw in zip(input_xy, np.asarray(input_yaw).reshape(-1)):
                end_x = float(x_left) + np.sin(float(yaw)) * arrow_len_m
                end_y = float(y_front) + np.cos(float(yaw)) * arrow_len_m
                draw.line((*to_pixel(float(x_left), float(y_front)), *to_pixel(end_x, end_y)),
                          fill=(0, 90, 30), width=2)
    draw_traj(gt_plot_xy, (40, 120, 255), (20, 80, 180), 4)
    draw_traj(pred_plot_xy, (220, 50, 50), (150, 20, 20), 3)

    combo = Image.new("RGB", (img.width + traj_panel.width, max(img.height, traj_panel.height)), (255, 255, 255))
    combo.paste(img, (0, 0))
    combo.paste(traj_panel, (img.width, 0))

    out_path = out_dir / f"{seq_idx:06d}_{int(t_ref)}.png"
    combo.save(out_path)

def _model_forward_supports(model, name: str) -> bool:
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters

def _model_planning_supports(model, name: str) -> bool:
    try:
        sig = inspect.signature(model.planning)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters

def _call_model_forward(model, batch, device):
    """
    與原 STP3 相容的 forward 呼叫；若是你的 VLM 生成式模型（gen_planner.VLM_STP3_Gen），
    會要求 batch 具備 rgb_224_seq / seg_224_seq，並走相容分支。
    """
    image = batch['image']
    intrinsics = batch['intrinsics']
    extrinsics = batch['extrinsics']
    future_egomotion = batch['future_egomotion']
    # print("future_egomotion",future_egomotion)

    # 偵測是否是你的 VLM 生成式 wrapper（有 vlm attribute 且 planning 參數包含 bev/occupancy）
    is_vlm_gen = hasattr(model, "vlm")
    # print("future_egomotion",future_egomotion)
    if is_vlm_gen:
        # 你的 forward 介面需要額外序列（見 gen_planner.py）
        if ('rgb_224_seq' not in batch) or ('seg_224_seq' not in batch):
            raise KeyError("需要 batch 提供 'rgb_224_seq' 與 'seg_224_seq' 以支援 VLM 生成式模型。")
        rgb_224_seq = batch['rgb_224_seq'].to(device, non_blocking=True)
        seg_224_seq = batch['seg_224_seq'].to(device, non_blocking=True)
        ped_traj_preds = batch.get('ped_traj_preds')
        ped_traj_mask = batch.get('ped_traj_mask')
        ped_traj_valid_steps = batch.get('ped_traj_valid_steps')
        ped_bev_map = batch.get('ped_bev_map')

        forward_kwargs = {
            'rgb_224_seq': rgb_224_seq,
            'seg_224_seq': seg_224_seq,
        }
        if 'seg_id_224_seq' in batch and _model_forward_supports(model, 'seg_id_224_seq'):
            forward_kwargs['seg_id_224_seq'] = batch['seg_id_224_seq'].to(device, non_blocking=True)
        # 深度：由 cfg.USE_DEPTH 決定 batch 內是真實 depth 或零深度佔位，這裡照實傳給模型
        if 'depth_224_seq' in batch and _model_forward_supports(model, 'depth_224_seq'):
            forward_kwargs['depth_224_seq'] = batch['depth_224_seq'].to(device, non_blocking=True)
        if 'admlp_input' in batch and _model_forward_supports(model, 'admlp_input'):
            forward_kwargs['admlp_input'] = batch['admlp_input'].to(device, non_blocking=True)
        if ped_traj_preds is not None and _model_forward_supports(model, 'ped_traj_preds'):
            forward_kwargs['ped_traj_preds'] = ped_traj_preds.to(device, non_blocking=True)
        if ped_traj_mask is not None and _model_forward_supports(model, 'ped_traj_mask'):
            forward_kwargs['ped_traj_mask'] = ped_traj_mask.to(device, non_blocking=True)
        if ped_traj_valid_steps is not None and _model_forward_supports(model, 'ped_traj_valid_steps'):
            forward_kwargs['ped_traj_valid_steps'] = ped_traj_valid_steps.to(device, non_blocking=True)
        if ped_bev_map is not None and _model_forward_supports(model, 'ped_bev_map'):
            forward_kwargs['ped_bev_map'] = ped_bev_map.to(device, non_blocking=True)

        # 走相容 forward（gen_planner 會回傳 ({}, cache)；我們只需觸發內部快取）
        with torch.no_grad():
            _ = model(image, intrinsics, extrinsics, future_egomotion, **forward_kwargs)
        # 用最小字典回傳下游不會用到的鍵
        return {
            'segmentation': torch.zeros(image.size(0), 1, 200, 200, device=device),  # 只為了佔位
            'pedestrian': torch.zeros(image.size(0), 1, 200, 200, device=device),
            'hdmap': torch.zeros(image.size(0), 4, 200, 200, device=device),
            'cam_front': getattr(model, 'fake_cam_front', torch.zeros(1, 64, 60, 28, device=device)).expand(image.size(0), -1, -1, -1),
            'costvolume': torch.zeros(image.size(0), getattr(model, 'receptive_field', 4)+1, 1, 1, device=device)
        }, True
    else:
        raise RuntimeError("現在版本不該來這")
        with torch.no_grad():
            out = model(image, intrinsics, extrinsics, future_egomotion)
        return out, False

def _call_model_planning(model, output, labels, batch, n_present, device, is_vlm_gen: bool):
    """
    呼叫規劃分支：支援
    - 原 STP3: planning(cam_front=..., cost_volume=..., semantic_pred=..., hd_map=..., ...)
    - 你的 VLM 生成式: planning(bev_rgbs=..., trajs=..., gt_trajs=..., commands=..., target_points=..., occupancy=...)
    """
    trajs = batch['sample_trajectory']
    command = batch['command']
    target_points = batch['target_point']
    # print("command",command)

    if is_vlm_gen:
        # 準備 occupancy（依照 evaluate.py 現行邏輯）
        seg_pred = output['segmentation']
        ped_pred = output['pedestrian']
        seg_prediction = torch.argmax(seg_pred, dim=2, keepdim=True)
        ped_prediction = torch.argmax(ped_pred, dim=2, keepdim=True) if ped_pred.numel() > 0 else torch.zeros_like(seg_prediction)
        occupancy = _build_valid_occupancy(labels, n_present)
        ped_bev_points = batch.get('ped_bev_points')
        ped_bev_valid_steps = batch.get('ped_bev_valid_steps')

        # 你的規劃介面（見 gen_planner.py）
        planning_kwargs = {
            'bev_rgbs': None,
            'trajs': trajs.to(device),
            'gt_trajs': labels['gt_trajectory'][:, 1:].to(device),
            'commands': command,
            'target_points': target_points,
            'occupancy': occupancy,
        }
        if ped_bev_points is not None:
            planning_kwargs['ped_bev_points'] = ped_bev_points.to(device, non_blocking=True)
        if ped_bev_valid_steps is not None:
            planning_kwargs['ped_bev_valid_steps'] = ped_bev_valid_steps.to(device, non_blocking=True)
        if 'timestamp_us' in batch and _model_planning_supports(model, 'clip_sign_timestamps'):
            planning_kwargs['clip_sign_timestamps'] = batch['timestamp_us']
        if 'filename' in batch and _model_planning_supports(model, 'clip_sign_filenames'):
            planning_kwargs['clip_sign_filenames'] = batch['filename']

        planning_ret = model.planning(**planning_kwargs)
        if len(planning_ret) == 5:
            loss, _, final_traj, _, _ = planning_ret
        elif len(planning_ret) == 4:
            loss, final_traj, _, _ = planning_ret
        else:
            raise RuntimeError(f"不支援的 planning 回傳格式，長度={len(planning_ret)}")
        # print("final_traj",final_traj)
        return loss, final_traj

    else:
        raise RuntimeError("現在版本不該來這")
        # 原 STP3 規劃介面
        seg_prediction = torch.argmax(output['segmentation'].detach(), dim=2, keepdim=True)
        if 'pedestrian' in output and output['pedestrian'] is not None and output['pedestrian'].numel() > 0:
            pedestrian_prediction = torch.argmax(output['pedestrian'].detach(), dim=2, keepdim=True)
        else:
            pedestrian_prediction = torch.zeros_like(seg_prediction)
        occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
        _, final_traj = model.planning(
            cam_front=output['cam_front'].detach(),
            trajs=trajs[:, :, 1:],
            gt_trajs=labels['gt_trajectory'][:, 1:],
            cost_volume=output['costvolume'][:, n_present:].detach(),
            semantic_pred=occupancy[:, n_present:].squeeze(2),
            hd_map=output.get('hdmap', torch.zeros_like(output['segmentation'])).detach(),
            commands=command,
            target_points=target_points
        )
        return torch.tensor(0.0, device=device), final_traj

def eval(checkpoint_path, dataroot, *, strict=True, model_module=None, model_class=None,
         measure_inference_time=False, ablation_modalities=None, ablation_name=None,
         admlp_fit_past_frames=None, admlp_fit_degree=None,
         admlp_fit_yaw_degree=None, use_depth=True):
    save_path = mk_save_dir()
    plot_dir = save_path / "inference_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ablation_modalities = _normalize_ablation_modalities(ablation_modalities)
    ablation_tag = _ablation_output_tag(ablation_modalities, ablation_name)
    csv_suffix = f"_{ablation_tag}" if ablation_tag else ""
    if ablation_modalities:
        print(f"[Ablation] blacked out inputs: {', '.join(sorted(ablation_modalities))}")

    trainer = _load_trainer_for_eval(
        checkpoint_path,
        strict=strict,
        model_module=model_module,
        model_class=model_class,
    )

    # === 建立 CSV 檔，用來存輸入軌跡 / GT / 最佳軌跡（與圖輸出到同一個執行資料夾）===
    csv_path = save_path / f"trajectories{csv_suffix}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    l2_csv_path = save_path / f"l2_errors{csv_suffix}.csv"
    l2_csv_file = open(l2_csv_path, "w", newline="")
    l2_writer = csv.writer(l2_csv_file)
    l2_summary_csv_path = save_path / f"l2_error_summary{csv_suffix}.csv"

    # 每一列是一個「某種軌跡（input/gt/pred）」在某個時間步 t_step 的一點
    # kind = 'input' 代表 future_egomotion（dx,dy,dyaw）
    # kind = 'gt'    代表 gt_trajectory（相對於當前車體座標的 x,y,yaw）
    # kind = 'pred'  代表 final_traj（模型預測的軌跡）
    writer.writerow([
        "seq_idx",          # evaluate 裡的樣本 index（方便 debug）
        "timestamp_us",     # 你 CSV 的時間戳
        "filename",         # 對應影像檔名
        "curr_x", "curr_y", "curr_yaw",   # 當前全域座標（map/frame）
        "kind",             # 'input' / 'gt' / 'pred'
        "t_step",           # 該軌跡內的第幾個點
        "x", "y", "yaw"     # 對 input：x,y,yaw = dx,dy,dyaw；對 gt/pred：x,y,yaw = 相對位置
    ])
    l2_writer.writerow([
        "seq_idx",
        "timestamp_us",
        "filename",
        "t_step",
        "gt_x",
        "gt_y",
        "pred_x",
        "pred_y",
        "lateral_error",
        "longitudinal_error",
        "abs_lateral_error",
        "abs_longitudinal_error",
        "l2",
    ])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trainer.to(device)
    model = trainer.model

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1
    cfg.LIFT.GT_DEPTH = False
    # 深度開關：開啟時載入器讀 depth_infer/*.npy；關閉時改餵零深度（沒有深度資料也能執行）
    cfg.USE_DEPTH = use_depth
    if dataroot is None:
        dataroot = "/home/cyc/dataset/0504_what_up/resample/video1"
    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.MAP_FOLDER = dataroot
    if admlp_fit_past_frames is not None:
        cfg.ADMLP_FIT_PAST_FRAMES = int(admlp_fit_past_frames)
    if admlp_fit_degree is not None:
        cfg.ADMLP_FIT_DEGREE = int(admlp_fit_degree)
    if admlp_fit_yaw_degree is not None:
        cfg.ADMLP_FIT_YAW_DEGREE = int(admlp_fit_yaw_degree)

    dataroot = cfg.DATASET.DATAROOT
    nworkers = cfg.N_WORKERS
    valdata = FuturePredictionDataset(cfg)  # 自訂場域資料版本不需要 NuScenes handle
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False
    )

    n_classes = len(cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)
    hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS
    metric_vehicle_val = IntersectionOverUnion(n_classes).to(device)
    future_second = int(cfg.N_FUTURE_FRAMES / 2)

    # if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
    #     metric_pedestrian_val = IntersectionOverUnion(n_classes).to(device)

    # if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
    #     metric_hdmap_val = [IntersectionOverUnion(2, absent_score=1).to(device) for _ in range(len(hdmap_class))]

    # if cfg.INSTANCE_SEG.ENABLED:
    #     metric_panoptic_val = PanopticMetric(n_classes=n_classes).to(device)

    metric_planning_val = []
    if cfg.PLANNING.ENABLED:
        for i in range(future_second):
            metric_planning_val.append(PlanningMetric(cfg, 2 * (i + 1)).to(device))

    inference_time_total = 0.0
    inference_count = 0
    l2_sum = 0.0
    l2_count = 0
    fde_sum = 0.0
    fde_count = 0
    error_stats = {
        "lateral_x": _new_error_stats(),
        "longitudinal_y": _new_error_stats(),
        "euclidean_l2": _new_error_stats(),
    }
    global_seq_idx = 0

    for index, batch in enumerate(tqdm(valloader)):
        t1 = time.time()
        preprocess_batch(batch, device)
        _apply_input_ablation(batch, ablation_modalities)
        labels = _prepare_l2_labels(batch)

        if measure_inference_time:
            _sync_cuda_if_needed(device)
            infer_t0 = time.perf_counter()
        output, is_vlm_gen = _call_model_forward(model, batch, device)
        n_present = getattr(model, 'receptive_field', 4)

        # # semantic seg metrics（若有）
        # if 'segmentation' in output and output['segmentation'].numel() > 0:
        #     seg_prediction = torch.argmax(output['segmentation'].detach(), dim=2, keepdim=True)
        #     metric_vehicle_val(seg_prediction[:, n_present - 1:], labels['segmentation'][:, n_present - 1:])

        # if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED and 'pedestrian' in output:
        #     ped_pred = output['pedestrian'].detach()
        #     ped_pred = torch.argmax(ped_pred, dim=2, keepdim=True) if ped_pred.numel() > 0 else torch.zeros_like(seg_prediction)
        #     metric_pedestrian_val(ped_pred[:, n_present - 1:], labels['pedestrian'][:, n_present - 1:])

        # if cfg.SEMANTIC_SEG.HDMAP.ENABLED and 'hdmap' in output:
        #     for i in range(len(hdmap_class)):
        #         hdmap_prediction = output['hdmap'][:, 2 * i:2 * (i + 1)].detach()
        #         hdmap_prediction = torch.argmax(hdmap_prediction, dim=1, keepdim=True)
        #         metric_hdmap_val[i](hdmap_prediction, labels['hdmap'][:, i:i + 1])

        # if cfg.INSTANCE_SEG.ENABLED and ('instance' in labels):
        #     pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
        #         output, compute_matched_centers=False, make_consistent=True
        #     )
        #     metric_panoptic_val(pred_consistent_instance_seg[:, n_present - 1:], labels['instance'][:, n_present - 1:])

        if cfg.PLANNING.ENABLED:
            t2 = time.time()
            _, final_traj = _call_model_planning(model, output, labels, batch, n_present, device, is_vlm_gen)
            final_traj = final_traj.clone()
            final_traj[..., 0] *= -1.0

            occupancy = _build_valid_occupancy(labels, n_present)
            xy_error = _trajectory_xy_error(final_traj.detach(), labels['gt_trajectory'].detach())
            l2_per_step = torch.linalg.norm(xy_error, dim=-1)
            if l2_per_step.numel() > 0:
                l2_sum += float(l2_per_step.sum().item())
                l2_count += int(l2_per_step.numel())
                fde_sum += float(l2_per_step[:, -1].sum().item())
                fde_count += int(l2_per_step.shape[0])
                _update_error_stats(error_stats["lateral_x"], xy_error[..., 0])
                _update_error_stats(error_stats["longitudinal_y"], xy_error[..., 1])
                _update_error_stats(error_stats["euclidean_l2"], l2_per_step)

            # print("final_traj",final_traj)

            # === CSV 部分：把輸入 / GT / 最佳軌跡都寫進去 ===
            # batch size（理論上你現在是 1，但這樣寫可以支援 B>1）
            B = batch['future_egomotion'].shape[0]

            for b in range(B):
                # 1) 輸入軌跡：future_egomotion，shape (T_rf, 6)
                if 'plot_input_xy_3s' in batch:
                    input_xy = batch['plot_input_xy_3s'][b].detach().cpu().numpy()
                    input_yaw = batch['plot_input_yaw_3s'][b].detach().cpu().numpy()
                else:
                    input_xy, input_yaw = _input_history_from_egomotion(batch['future_egomotion'][b])
                fe = batch['future_egomotion'][b].detach().cpu().numpy()   # (T_rf, 6)
                rgb_224 = batch['rgb_224_seq'][b, -1].detach().cpu().numpy()

                # 2) GT 未來軌跡：gt_trajectory，shape (T_gt, 3)，第一個點是 (0,0,0)
                gt = labels['gt_trajectory'][b].detach().cpu().numpy()     # (T_gt, 3)

                # 3) 模型最佳軌跡：final_traj，shape (T_pred, 3) 或 (T_pred, 2)
                pred = final_traj[b].detach().cpu().numpy()                # (T_pred, D)
                if pred.shape[1] == 2:
                    # 如果沒有 yaw，就補 0
                    pred = np.concatenate([pred, np.zeros((pred.shape[0], 1), dtype=pred.dtype)], axis=1)
                l2_np = l2_per_step[b].detach().cpu().numpy() if l2_per_step.numel() > 0 else np.zeros((0,), dtype=np.float32)
                xy_err_np = xy_error[b].detach().cpu().numpy() if xy_error.numel() > 0 else np.zeros((0, 2), dtype=np.float32)

                # 4) 當前全域座標（你在 Dataset.__getitem__ 裡塞的）
                curr = batch['curr_pose'][b].detach().cpu().numpy()        # (3,) = (x,y,yaw)

                # 5) 時間戳與檔名
                ts = int(batch['timestamp_us'][b].item())
                # DataLoader 會把字串 collate 成 list[str]
                fname = batch['filename'][b]

                # 給這個 sample 一個連續的 seq_idx；最後一個 partial batch 也不會跳號。
                seq_idx = global_seq_idx
                global_seq_idx += 1

                # ---- 5.1 寫入輸入軌跡（future_egomotion）----
                # 這裡我們只取 dx, dy, dyaw (假設 pose_vec = [x, y, z, roll, pitch, yaw])
                for t_step in range(fe.shape[0]):
                    dx = float(fe[t_step, 0])
                    dy = float(fe[t_step, 1])
                    dyaw = float(fe[t_step, 5]) if fe.shape[1] >= 6 else 0.0

                    writer.writerow([
                        seq_idx,
                        ts,
                        fname,
                        float(curr[0]), float(curr[1]), float(curr[2]),
                        "input",
                        t_step,
                        dx, dy, dyaw
                    ])

                # ---- 5.2 寫入 GT 未來軌跡（gt_trajectory）----
                # gt[t] 是當前座標系下的相對位置 (x,y,yaw)
                for t_step in range(gt.shape[0]):
                    gx, gy, gyaw = gt[t_step, 0], gt[t_step, 1], gt[t_step, 2]
                    writer.writerow([
                        seq_idx,
                        ts,
                        fname,
                        float(curr[0]), float(curr[1]), float(curr[2]),
                        "gt",
                        t_step,
                        float(gx), float(gy), float(gyaw)
                    ])

                # ---- 5.3 寫入模型預測的軌跡 ----
                for t_step in range(pred.shape[0]):
                    px, py, pyaw = pred[t_step, 0], pred[t_step, 1], pred[t_step, 2]
                    writer.writerow([
                        seq_idx,
                        ts,
                        fname,
                        float(curr[0]), float(curr[1]), float(curr[2]),
                        "pred",
                        t_step,
                        float(px), float(py), float(pyaw)
                    ])

                for t_step in range(l2_np.shape[0]):
                    gx, gy = gt[t_step + 1, 0], gt[t_step + 1, 1]
                    px, py = pred[t_step, 0], pred[t_step, 1]
                    lateral_error = float(xy_err_np[t_step, 0])
                    longitudinal_error = float(xy_err_np[t_step, 1])
                    l2_error = float(l2_np[t_step])
                    location = {
                        "seq_idx": seq_idx,
                        "timestamp_us": ts,
                        "filename": fname,
                        "t_step": t_step + 1,
                        "gt_x": float(gx),
                        "gt_y": float(gy),
                        "pred_x": float(px),
                        "pred_y": float(py),
                        "lateral_error": lateral_error,
                        "longitudinal_error": longitudinal_error,
                        "l2": l2_error,
                    }
                    _update_error_stats_max_location(error_stats["lateral_x"], abs(lateral_error), location)
                    _update_error_stats_max_location(error_stats["longitudinal_y"], abs(longitudinal_error), location)
                    _update_error_stats_max_location(error_stats["euclidean_l2"], l2_error, location)
                    l2_writer.writerow([
                        seq_idx,
                        ts,
                        fname,
                        t_step + 1,
                        float(gx),
                        float(gy),
                        float(px),
                        float(py),
                        lateral_error,
                        longitudinal_error,
                        abs(lateral_error),
                        abs(longitudinal_error),
                        l2_error,
                    ])

                save_inference_plot(
                    rgb_224=rgb_224,
                    pred=pred,
                    gt=gt,
                    l2=l2_np,
                    t_ref=ts,
                    seq_idx=seq_idx,
                    out_dir=plot_dir,
                    input_xy=input_xy,
                    input_yaw=input_yaw,
                )
            # === CSV 部分結束 ===
            # for i in range(future_second):
            #     cur_time = (i + 1) * 2
            #     metric_planning_val[i](final_traj[:, :cur_time].detach(),
            #                            labels['gt_trajectory'][:, 1:cur_time + 1],
            #                            occupancy[:, :cur_time])
                
        if measure_inference_time:
            _sync_cuda_if_needed(device)
            inference_time_total += (time.perf_counter() - infer_t0)
            inference_count += batch['future_egomotion'].shape[0]

        # if index % 100 == 0:
            # save(output, labels, batch, n_present, index, save_path)
            # save(output, labels, batch, n_present, index, save_path, pred_traj=final_traj)
        t3 = time.time()
        # print(f"total : {(t3 - t1) * 1000:.2f} ms", f"planning : {(t2 - t1) * 1000:.2f} ms")
    # 關掉 CSV 檔
    csv_file.close()
    l2_csv_file.close()
    with open(l2_summary_csv_path, "w", newline="") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow([
            "axis",
            "count",
            "min_abs_error_m",
            "max_abs_error_m",
            "avg_abs_error_m",
            "rmse_m",
            "max_seq_idx",
            "max_timestamp_us",
            "max_filename",
            "max_t_step",
            "max_gt_x",
            "max_gt_y",
            "max_pred_x",
            "max_pred_y",
            "max_lateral_error",
            "max_longitudinal_error",
            "max_l2",
        ])
        for axis in ("lateral_x", "longitudinal_y", "euclidean_l2"):
            count, min_abs, max_abs, avg_abs, rmse, location = _finalize_error_stats(error_stats[axis])
            location = location or {}
            summary_writer.writerow([
                axis,
                count,
                min_abs,
                max_abs,
                avg_abs,
                rmse,
                location.get("seq_idx", ""),
                location.get("timestamp_us", ""),
                location.get("filename", ""),
                location.get("t_step", ""),
                location.get("gt_x", ""),
                location.get("gt_y", ""),
                location.get("pred_x", ""),
                location.get("pred_y", ""),
                location.get("lateral_error", ""),
                location.get("longitudinal_error", ""),
                location.get("l2", ""),
            ])
    print(f"[INFO] 已將軌跡輸出到 {csv_path}")
    print(f"[INFO] 已將 L2 誤差輸出到 {l2_csv_path}")
    print(f"[INFO] 已將 L2 統計輸出到 {l2_summary_csv_path}")
    print(f"[INFO] 已將每次推論可視化輸出到 {plot_dir}")
    if l2_count > 0:
        print(f"[RESULT] mean L2/ADE: {l2_sum / l2_count:.4f} m over {l2_count} points")
    if fde_count > 0:
        print(f"[RESULT] final L2/FDE: {fde_sum / fde_count:.4f} m over {fde_count} samples")
    if measure_inference_time and inference_count > 0:
        avg_ms = (inference_time_total / inference_count) * 1000.0
        print(f"[INFO] 平均推論時間: {avg_ms:.2f} ms/sample over {inference_count} samples")

# def save(output, labels, batch, n_present, frame, save_path):
def save(output, labels, batch, n_present, frame, save_path, pred_traj=None):
    # --- 1. 準備數據 (自動適配有無預測的情況) ---
    
    # HD Map: 如果模型沒輸出(例如 VLM)，改用 GT 地圖 (labels['hdmap']) 當背景
    # Fallback：使用真實地圖
    hdmap_tensor = labels['hdmap']
    use_gt_map = True

    # Segmentation: 如果模型沒輸出，用全黑代替
    has_seg = False

    # Pedestrian
    has_ped = False

    gt_trajs = labels['gt_trajectory']

    # --- 2. 設定畫布：只留一張 BEV 的大小 ---
    plt.figure(1, figsize=(5, 5)) # 5x5 inch，正方形
    plt.clf() # 清除舊圖

    # --- 3. 製作 BEV 圖像內容 ---
    # 初始化灰色背景
    showing = torch.zeros((200, 200, 3)).numpy()
    showing[:, :] = np.array([219 / 255, 215 / 255, 215 / 255]) 

    # 繪製地圖 (Road & Lane)
    # 注意：GT map 和 Pred map 的維度處理方式不同
    if use_gt_map:
        # labels['hdmap'] 形狀通常是 (B, 2, 200, 200)，0:Lane, 1:Road
        # 這裡假設 batch=0
        road_mask = hdmap_tensor[0, 1].cpu().numpy() > 0
        lane_mask = hdmap_tensor[0, 0].cpu().numpy() > 0
    else:
        # output['hdmap'] 原本邏輯 (0:2 Lane, 2:4 Road)
        road_mask = torch.argmax(hdmap_tensor[0, 2:4], dim=0).cpu().numpy() > 0
        lane_mask = torch.argmax(hdmap_tensor[0, 0:2], dim=0).cpu().numpy() > 0
    
    showing[road_mask] = np.array([161 / 255, 158 / 255, 158 / 255]) # 深灰路面
    showing[lane_mask] = np.array([84 / 255, 70 / 255, 70 / 255])   # 車道線

    # 繪製語意分割 (Semantic Segmentation)
    if has_seg:
        semantic_seg = torch.argmax(segmentation[0], dim=0).cpu().numpy()
        showing[semantic_seg > 0] = np.array([255 / 255, 128 / 255, 0 / 255]) # 車輛 (橘色)

    if has_ped:
        pedestrian_seg = torch.argmax(pedestrian[0], dim=0).cpu().numpy()
        showing[pedestrian_seg > 0] = np.array([28 / 255, 81 / 255, 227 / 255]) # 行人 (藍色)

    # 顯示圖片
    plt.imshow(make_contour(showing))
    plt.axis('off')

    # --- 4. 畫自車 (Ego Vehicle) ---
    bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
    dx = np.array([0.5, 0.5])
    w, h = 1.85, 4.084
    pts = np.array([
        [-h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, -w / 2.],
        [-h / 2. + 0.5, -w / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900') # 綠色車身

    # --- 5. 畫軌跡 (Trajectory) ---
    plt.xlim((200, 0))
    plt.ylim((0, 200))
    
    # 複製出來處理，避免改到原始 Tensor 影響計算
    traj_np = gt_trajs[0, :, :2].cpu().numpy().copy()
    traj_np[:, :1] = traj_np[:, :1] * -1 # 翻轉 X 軸 (原本代碼邏輯)
    traj_np = (traj_np - bx) / dx
    
    # 畫 GT 軌跡 (藍色粗線)
    plt.plot(traj_np[:, 0], traj_np[:, 1], linewidth=3.0, color='blue', label='GT')

    # ★ 新增：畫預測軌跡 (紅色)
    if pred_traj is not None:
        # pred_traj 形狀通常是 (B, T, 3) 或 (B, T, 2)
        # 取 batch 0
        p_traj = pred_traj[0, :, :2].detach().cpu().numpy().copy()

        # 座標轉換 (跟 GT 一樣的邏輯)
        p_traj[:, :1] = p_traj[:, :1] * -1  # 翻轉 X
        p_traj = (p_traj - bx) / dx

        plt.plot(p_traj[:, 0], p_traj[:, 1], linewidth=3.0, color='red', label='Pred')

    # 存檔
    plt.savefig(save_path / ('%04d.png' % frame), bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation (with pluggable model)')
    parser.add_argument('--checkpoint', default='last.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default=None, type=str)
    parser.add_argument('--strict', action='store_true', help='strict=True 讀取 checkpoint')
    # parser.add_argument('--model-module', type=str, default='stp3.model_ped_traj.image_forced_ped_traj_codex',
    #                     help='自訂模型的模組路徑')
    parser.add_argument('--model-module', type=str, default= None,
                        help='自訂模型的模組路徑')
    parser.add_argument('--model-class', type=str, default='VLM_STP3_Gen',
                        help='自訂模型的類別名稱')
    parser.add_argument('--measure-inference-time', action='store_true',
                        help='計算平均推論時間；開啟時強制 batch_size=1，計時範圍為 forward + planning')
    parser.add_argument('--no-depth', action='store_true',
                        help='關閉深度：不讀 depth_infer，改餵零深度（沒有深度資料也能執行）。預設為使用深度。')
    parser.add_argument('--ablate-image', action='store_true',
                        help='消融影像輸入：將 rgb_224_seq 與 image 塗黑為 0')
    parser.add_argument('--ablate-seg', action='store_true',
                        help='消融語意分割輸入：將 seg_224_seq 與 seg_id_224_seq 塗黑為 0')
    parser.add_argument('--ablate-modalities', nargs='*', default=None,
                        help='一次指定要消融的輸入，可用 image/rgb/img、seg/semantic/segmentation，也可逗號分隔')
    parser.add_argument('--ablation-name', default=None, type=str,
                        help='指定輸出 CSV 後綴，例如 no_rgb；未指定時會自動用 ablate_image_seg')
    parser.add_argument('--admlp-fit-past-frames', default=None, type=int,
                        help='速度/加速度擬合使用的歷史幀數；4 代表加上 t0 共五點')
    parser.add_argument('--admlp-fit-degree', default=None, type=int,
                        help='x/y polynomial 階數；1=平均速度且 ax/ay=0，預設 1')
    parser.add_argument('--admlp-fit-yaw-degree', default=None, type=int,
                        help='yaw polynomial 階數；1=平均 yaw rate 且 yaw_acc=0，預設 1')
    args = parser.parse_args()

    ablation_modalities = []
    if args.ablate_modalities:
        ablation_modalities.extend(args.ablate_modalities)
    if args.ablate_image:
        ablation_modalities.append("image")
    if args.ablate_seg:
        ablation_modalities.append("seg")

    eval(args.checkpoint, args.dataroot, strict=args.strict,
         model_module=args.model_module, model_class=args.model_class,
         measure_inference_time=args.measure_inference_time,
         ablation_modalities=ablation_modalities,
         ablation_name=args.ablation_name,
         admlp_fit_past_frames=args.admlp_fit_past_frames,
         admlp_fit_degree=args.admlp_fit_degree,
         admlp_fit_yaw_degree=args.admlp_fit_yaw_degree,
         use_depth=not args.no_depth)
