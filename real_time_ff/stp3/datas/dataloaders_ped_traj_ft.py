import numpy as np
import torch
import torch.utils.data
from nuscenes.nuscenes import NuScenes
from torch.utils.data import WeightedRandomSampler

from stp3.datas.NuscenesData_ped_traj import FuturePredictionDataset


def _build_ped_presence_mask(dataset):
    has_ped = np.zeros(len(dataset), dtype=np.bool_)
    for i in range(len(dataset)):
        keyframe_name = dataset._get_present_keyframe_name(i)
        has_ped[i] = bool(dataset.ped_pred_index.get(keyframe_name))
    return has_ped


def _compute_min_dist_per_t(gt_xy: torch.Tensor, ped_xy: torch.Tensor, ped_valid: torch.Tensor):
    if ped_xy.numel() == 0 or not ped_valid.any():
        return None, None

    t = gt_xy.shape[0]
    ped_xy = ped_xy[:, :t]
    ped_valid = ped_valid[:, :t]
    if not ped_valid.any():
        return None, None

    dist = torch.norm(gt_xy.unsqueeze(0) - ped_xy, dim=-1)
    inf = torch.full_like(dist, float('inf'))
    dist_valid = torch.where(ped_valid, dist, inf)
    min_dist_t = dist_valid.amin(dim=0)
    has_valid_t = ped_valid.any(dim=0)
    return min_dist_t, has_valid_t


def _build_triggered_mask(dataset, safe_dist):
    triggered = np.zeros(len(dataset), dtype=np.bool_)
    for i in range(len(dataset)):
        ref_index = dataset.indices[i][dataset.receptive_field - 1]
        rec = dataset.ixes[ref_index]
        gt_traj_np, _ = dataset.get_gt_trajectory(rec, ref_index)
        gt_xy = torch.from_numpy(gt_traj_np[1:, :2]).float()

        keyframe_name = dataset._get_present_keyframe_name(i)
        persons = dataset.ped_pred_index.get(keyframe_name, [])[:dataset.ped_max_agents]
        if not persons:
            continue

        ped_xy, ped_valid = dataset._build_ped_bev_points(rec, persons)
        ped_xy = ped_xy.float()
        ped_valid = ped_valid.bool()
        min_dist_t, has_valid_t = _compute_min_dist_per_t(gt_xy, ped_xy, ped_valid)
        if min_dist_t is None:
            continue
        if bool(((min_dist_t < float(safe_dist)) & has_valid_t).any().item()):
            triggered[i] = True
    return triggered


def _build_ped_heavy_weights(has_ped, target_ratio):
    has_ped = has_ped.astype(np.bool_)
    n_total = len(has_ped)
    n_ped = int(has_ped.sum())
    n_nonped = int(n_total - n_ped)

    if n_ped == 0 or n_nonped == 0:
        return np.ones(n_total, dtype=np.float32)

    target_ratio = float(np.clip(target_ratio, 1e-3, 1.0 - 1e-3))

    # Expected ped sampling mass = target_ratio
    # Per-sample weights chosen so:
    #   n_ped * w_ped : n_nonped * w_nonped = target_ratio : (1-target_ratio)
    w_ped = target_ratio / n_ped
    w_nonped = (1.0 - target_ratio) / n_nonped

    weights = np.where(has_ped, w_ped, w_nonped).astype(np.float32)
    weights /= weights.mean()
    return weights


def prepare_dataloaders_ped_traj_ft(cfg, return_dataset=False):
    if cfg.DATASET.NAME != 'nuscenes':
        raise NotImplementedError('ped_traj_ft currently only supports nuscenes')

    dataroot = cfg.DATASET.DATAROOT
    nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
    traindata = FuturePredictionDataset(nusc, 0, cfg)
    valdata = FuturePredictionDataset(nusc, 1, cfg)

    nworkers = cfg.N_WORKERS
    ped_target_ratio = float(getattr(cfg, 'PED_FINETUNE_TARGET_RATIO', 0.4))
    ped_sampling_mode = str(getattr(cfg, 'PED_FINETUNE_SAMPLING', 'triggered'))
    ped_safe_dist = float(getattr(cfg, 'PED_FINETUNE_SAFE_DIST', getattr(cfg, 'LOSS_PED_REPULSE_SAFE_DIST', 6.0)))

    has_ped = _build_ped_presence_mask(traindata)
    triggered = _build_triggered_mask(traindata, ped_safe_dist)

    if ped_sampling_mode == 'triggered':
        positives = triggered
    else:
        positives = has_ped

    if not positives.any():
        print(f'[PED_FT] no positive samples found for mode={ped_sampling_mode}; fallback to has_ped')
        positives = has_ped
        ped_sampling_mode = 'has_ped'

    n_total = len(positives)
    n_pos = int(positives.sum())
    n_neg = int(n_total - n_pos)

    weights = _build_ped_heavy_weights(positives, ped_target_ratio)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )

    print(
        f'[PED_FT] train total={n_total}, has_ped={int(has_ped.sum())} ({has_ped.mean():.2%}), '
        f'triggered={int(triggered.sum())} ({triggered.mean():.2%})'
    )
    print(
        f'[PED_FT] sampler positives(mode={ped_sampling_mode})={n_pos} ({n_pos / max(1, n_total):.2%}), '
        f'negatives={n_neg}'
    )
    print(f'[PED_FT] target ped sampling ratio={ped_target_ratio:.2%}')
    print(f'[PED_FT] ped trigger safe dist={ped_safe_dist:.2f} m')
    print(
        f'[PED_FT] sampler weights: positive={weights[positives][0] if n_pos > 0 else 0:.6f}, '
        f'negative={weights[~positives][0] if n_neg > 0 else 0:.6f}'
    )

    trainloader = torch.utils.data.DataLoader(
        traindata,
        batch_size=cfg.BATCHSIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=nworkers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    valloader = torch.utils.data.DataLoader(
        valdata,
        batch_size=cfg.BATCHSIZE,
        shuffle=False,
        num_workers=nworkers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
    )

    if return_dataset:
        return trainloader, valloader, traindata, valdata
    return trainloader, valloader
