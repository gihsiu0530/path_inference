"""Runtime helpers for the additive ASAP FF ROS node.

This module intentionally has no ROS dependency, so checkpoint loading and the
model-call contract can be tested in the training environment.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
import torch

from stp3.utils.geometry import mat2pose_vec, pose_vec2mat


SUPPORTED_TAGS = {
    "planning_asap_ff": "all_admlp",
    "planning_asap_hybrid_ff": "hybrid",
}


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_trainer_for_eval(checkpoint_path, strict=True, model_module=None, model_class=None):
    """Load exactly the FF trainer selected by checkpoint hyperparameter TAG."""
    if model_module is not None or model_class is not None:
        raise ValueError("FF runtime selects the model from checkpoint TAG; custom model swapping is disabled.")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = _torch_load(checkpoint_path)
    hparams = checkpoint.get("hyper_parameters")
    state_dict = checkpoint.get("state_dict")
    if hparams is None or state_dict is None:
        raise RuntimeError(f"Checkpoint must contain hyper_parameters and state_dict: {checkpoint_path}")

    hparams = dict(hparams)
    tag = str(hparams.get("TAG", "")).strip().lower()
    variant = SUPPORTED_TAGS.get(tag)
    if variant == "hybrid":
        from stp3.trainer_codex_seg_ASAP_hybrid_ff import TrainingModule
    elif variant == "all_admlp":
        from stp3.trainer_codex_seg_ASAP_ff import TrainingModule
    else:
        raise RuntimeError(
            f"Unsupported checkpoint TAG={hparams.get('TAG')!r}. "
            "Expected Planning_ASAP_ff or Planning_ASAP_hybrid_ff."
        )

    # The full checkpoint has all six model.admlp_baseline tensors. Avoid a
    # non-portable dependency on the training machine's separate baseline path.
    hparams["FF_SKIP_ADMLP_BASELINE_INIT"] = True
    trainer = TrainingModule(hparams)
    incompatible = trainer.load_state_dict(state_dict, strict=strict)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    trainer.eval()
    trainer.ff_checkpoint_path = str(checkpoint_path)
    trainer.ff_checkpoint_tag = str(hparams.get("TAG"))
    trainer.ff_model_variant = variant
    print(
        f"[FF loader] checkpoint={checkpoint_path}\n"
        f"[FF loader] TAG={trainer.ff_checkpoint_tag}, variant={variant}"
    )
    return trainer


def _prepare_l2_labels(batch):
    labels = {"gt_trajectory": batch["gt_trajectory"]}
    for key in ("segmentation", "pedestrian", "hdmap"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.numel() > 0:
            labels[key] = value.long().contiguous()
    return labels


def _to_device(value, device):
    return value.to(device, non_blocking=True) if torch.is_tensor(value) else value


def _call_model_forward(model, batch, device):
    """Populate every cache required by both all-ADMLP and hybrid FF models."""
    required = (
        "rgb_224_seq",
        "seg_224_seq",
        "seg_id_224_seq",
        "depth_224_seq",
        "future_egomotion",
        "ego_history_egomotion",
        "admlp_input",
        "intrinsics",
        "extrinsics",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"FF realtime batch is missing: {missing}")

    image = _to_device(batch.get("image", torch.empty(1, 0)), device)
    intrinsics = _to_device(batch["intrinsics"], device)
    extrinsics = _to_device(batch["extrinsics"], device)
    future_egomotion = _to_device(batch["future_egomotion"], device)
    with torch.inference_mode():
        model(
            image,
            intrinsics,
            extrinsics,
            future_egomotion,
            rgb_224_seq=_to_device(batch["rgb_224_seq"], device),
            seg_224_seq=_to_device(batch["seg_224_seq"], device),
            seg_id_224_seq=_to_device(batch["seg_id_224_seq"], device),
            depth_224_seq=_to_device(batch["depth_224_seq"], device),
            ego_history_egomotion=_to_device(batch["ego_history_egomotion"], device),
            admlp_input=_to_device(batch["admlp_input"], device),
        )
    return {}, True


def _call_model_planning(model, output, labels, batch, n_present, device, is_vlm_gen):
    """Run normal FF planning, including its inference-only safety filter."""
    del output, n_present
    if not is_vlm_gen:
        raise RuntimeError("FF realtime runtime only supports VLM_STP3_Gen models.")
    gt = labels["gt_trajectory"]
    gt_future = gt[:, 1:].to(device, non_blocking=True)
    with torch.inference_mode():
        loss, _, final_traj, planning_tag, loss_dict = model.planning(
            bev_rgbs=_to_device(batch["rgb_224_seq"], device),
            trajs=_to_device(batch["sample_trajectory"], device),
            gt_trajs=gt_future,
            commands=batch["command"],
            target_points=_to_device(batch["target_point"], device),
            occupancy=None,
            drivable_mask=None,
        )
    model.last_runtime_planning_tag_ff = planning_tag
    model.last_runtime_loss_dict_ff = loss_dict
    return loss, final_traj


def _trajectory_xy_error(pred_traj: torch.Tensor, gt_trajectory: torch.Tensor) -> torch.Tensor:
    gt_future = gt_trajectory[:, 1:, :2]
    pred_xy = pred_traj[:, :, :2]
    horizon = min(pred_xy.shape[1], gt_future.shape[1])
    if horizon <= 0:
        return torch.empty(pred_xy.shape[0], 0, 2, device=pred_xy.device)
    return pred_xy[:, :horizon] - gt_future[:, :horizon].to(pred_xy.device)


def _input_history_from_egomotion(future_egomotion: torch.Tensor):
    """Return past input positions in public plot coordinates (left, forward)."""
    motion = future_egomotion.detach().float().cpu()
    time_steps = motion.shape[0]
    if time_steps <= 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
    matrices = pose_vec2mat(motion)
    points, yaws = [], []
    for index in range(time_steps):
        if index == time_steps - 1:
            points.append([0.0, 0.0])
            yaws.append(0.0)
            continue
        transform = matrices[index]
        for later in range(index + 1, time_steps - 1):
            transform = torch.mm(transform, matrices[later])
        pose = mat2pose_vec(transform)
        points.append([float(pose[1]), float(pose[0])])
        yaws.append(float(pose[5]))
    return np.asarray(points, np.float32), np.asarray(yaws, np.float32)


def save_inference_plot(
    rgb_224: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    l2: np.ndarray,
    t_ref: int,
    seq_idx: int,
    out_dir: Path,
    input_xy: Optional[np.ndarray] = None,
    input_yaw: Optional[np.ndarray] = None,
    extra_panels: Optional[list] = None,
    lateral_limit_m: Optional[float] = None,
    smoothed_xy: Optional[np.ndarray] = None,
):
    """Small dependency-free replacement for the legacy offline plot helper.

    `smoothed_xy` is the post-processed path that was actually published —
    already including its own start point, and no longer one point per
    sample_interval. Drawn as a second line so the raw prediction and what the
    MPC receives can be compared in one image; the L2 numbers stay on `pred`.
    """
    del input_yaw
    image = Image.fromarray(np.asarray(rgb_224, dtype=np.uint8), mode="RGB")
    size, margin = 512, 48
    panel = Image.new("RGB", (size, size), (250, 250, 250))
    draw = ImageDraw.Draw(panel)
    origin = np.zeros((1, 2), np.float32)
    pred_xy = np.asarray(pred, np.float32)[:, :2]
    gt_xy = np.asarray(gt, np.float32)[:, :2]
    gt_future = gt_xy[1:] if len(gt_xy) > 1 else gt_xy
    history = np.asarray(input_xy, np.float32) if input_xy is not None else np.zeros((0, 2), np.float32)
    smoothed = (np.asarray(smoothed_xy, np.float32)[:, :2]
                if smoothed_xy is not None else np.zeros((0, 2), np.float32))
    parts = [origin, pred_xy]
    if gt_future.size:
        parts.append(gt_future)
    if history.size:
        parts.append(history)
    if smoothed.size:
        parts.append(smoothed)
    all_xy = np.concatenate(parts, axis=0)
    if lateral_limit_m is None:
        x_extent = max(float(np.abs(all_xy[:, 0]).max()) + 1.0, 5.0)
    else:
        x_extent = max(float(lateral_limit_m), 1e-3)
    y_min = min(float(all_xy[:, 1].min()), 0.0) - 1.0
    y_max = max(float(all_xy[:, 1].max()), 0.0) + 1.0
    if y_max - y_min < 5.0:
        padding = (5.0 - (y_max - y_min)) / 2.0
        y_min, y_max = y_min - padding, y_max + padding
    scale_x = (size - 2 * margin) / (2.0 * x_extent)
    scale_y = (size - 2 * margin) / max(y_max - y_min, 1e-6)
    if lateral_limit_m is None:
        # Preserve metric aspect ratio for the legacy/offline-style view.
        scale_x = scale_y = min(scale_x, scale_y)
    center_x = size // 2
    origin_y = size - margin + y_min * scale_y

    def pixel(point):
        return center_x - float(point[0]) * scale_x, origin_y - float(point[1]) * scale_y

    draw.line((center_x, margin, center_x, size - margin), fill=(150, 150, 150), width=2)
    draw.line((margin, origin_y, size - margin, origin_y), fill=(150, 150, 150), width=2)
    draw.text((margin, 8), f"seq={seq_idx} | ts={t_ref}", fill=(30, 30, 30))
    if np.asarray(l2).size:
        draw.text((margin, 27), f"L2 mean={float(np.mean(l2)):.2f} m", fill=(30, 30, 30))
    legend = "history green | prediction red"
    if gt_future.size:
        legend = "history green | GT blue | prediction red"
    if smoothed.size:
        legend += " | published orange"
    draw.text((margin, 46), legend, fill=(30, 30, 30))
    if lateral_limit_m is not None:
        draw.text((margin, size - 20), f"LEFT +{x_extent:g} m", fill=(30, 30, 30))
        right_label = f"RIGHT -{x_extent:g} m"
        right_width = draw.textbbox((0, 0), right_label)[2]
        draw.text((size - margin - right_width, size - 20), right_label, fill=(30, 30, 30))

    def trajectory(points, line_color, dot_color, width):
        if not np.asarray(points).size:
            return
        pixels = [pixel(point) for point in points]
        if len(pixels) > 1:
            draw.line(pixels, fill=line_color, width=width)
        for px, py in pixels:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=dot_color)

    trajectory(history, (40, 150, 90), (20, 105, 55), 4)
    if gt_future.size:
        trajectory(np.concatenate([origin, gt_future], axis=0), (40, 120, 255), (20, 80, 180), 4)
    trajectory(np.concatenate([origin, pred_xy], axis=0), (220, 50, 50), (150, 20, 20), 3)
    # Last, so the published path stays readable where it overlaps the raw one.
    trajectory(smoothed, (245, 140, 20), (170, 90, 0), 3)

    named_extra = extra_panels or []
    panels = [image] + [Image.fromarray(np.asarray(value, np.uint8), mode="RGB") for _, value in named_extra] + [panel]
    canvas = Image.new("RGB", (sum(item.width for item in panels), max(item.height for item in panels)), "white")
    offset = 0
    for item in panels:
        canvas.paste(item, (offset, 0))
        offset += item.width
    canvas_draw = ImageDraw.Draw(canvas)
    offset = image.width
    for (label, _), item in zip(named_extra, panels[1:]):
        canvas_draw.text((offset + 3, min(item.height + 4, canvas.height - 14)), label, fill=(30, 30, 30))
        offset += item.width
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / f"{seq_idx:06d}_{int(t_ref)}.png")
