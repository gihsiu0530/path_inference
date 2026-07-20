import torch
import torch.utils.data
from nuscenes.nuscenes import NuScenes
from stp3.datas.NuscenesData_ped_traj import FuturePredictionDataset
from stp3.datas.CarlaData import CarlaDataset
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import numpy as np

from collections import Counter

def print_command_distribution(dataset, name="train"):
    cmds = dataset.commands_per_index
    cnt = Counter(cmds)
    total = sum(cnt.values())

    print(f"\n[DATASET STATS] {name} command distribution:")
    for k in ["FORWARD", "LEFT", "RIGHT"]:
        n = cnt.get(k, 0)
        ratio = n / max(1, total)
        print(f"  {k:8s}: {n:6d}  ({ratio*100:6.2f}%)")
    print(f"  TOTAL   : {total:6d}\n")


def prepare_dataloaders(cfg, return_dataset=False):
    if cfg.DATASET.NAME == 'nuscenes':
        # 28130 train and 6019 val
        dataroot = cfg.DATASET.DATAROOT
        nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
        traindata = FuturePredictionDataset(nusc, 0, cfg)
        valdata = FuturePredictionDataset(nusc, 1, cfg)

        print_command_distribution(traindata, name="train")


        if cfg.DATASET.VERSION == 'mini':
            traindata.indices = traindata.indices[:10]
            raise RuntimeError
            # valdata.indices = valdata.indices[:10]

        nworkers = cfg.N_WORKERS


        balance = getattr(cfg.DATASET, "BALANCE_COMMANDS", False)
        turn_boost = float(getattr(cfg.DATASET, "TURN_BOOST", 1.0)) # 1.2是40％

        if balance:
            cmds = traindata.commands_per_index
            cnt = Counter(cmds)

            # 基本的 class-balanced 權重：1 / freq
            alpha = 0.5
            class_w = {c: 1.0 / (cnt[c] ** alpha) for c in cnt}
            # class_w = {c: 1.0 / cnt[c] for c in cnt}

            # 額外加強 turn（LEFT/RIGHT）
            class_w["LEFT"]  = class_w.get("LEFT", 0.0)  * turn_boost
            class_w["RIGHT"] = class_w.get("RIGHT", 0.0) * turn_boost

            weights = np.array([class_w[c] for c in cmds], dtype=np.float32)
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(weights),
                num_samples=len(weights),   # 每個 epoch 抽 len(dataset) 次
                replacement=True
            )

            print("[BALANCE] command counts:", dict(cnt))
            print("[BALANCE] class weights:", class_w)

            trainloader = torch.utils.data.DataLoader(
                traindata,
                batch_size=cfg.BATCHSIZE,
                sampler=sampler,
                shuffle=False,              # 有 sampler 就不能 shuffle
                num_workers=nworkers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                drop_last=True,
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                traindata,
                batch_size=cfg.BATCHSIZE,
                shuffle=True,
                num_workers=nworkers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                drop_last=True,
            )


        print("BATCHSIZE : ",cfg.BATCHSIZE)
        # trainloader = torch.utils.data.DataLoader(
        #     traindata,
        #     batch_size=cfg.BATCHSIZE,
        #     shuffle=True,
        #     num_workers=nworkers,
        #     pin_memory=True,
        #     persistent_workers=True,   # ★ 建議加
        #     prefetch_factor=2,         # ★ 建議加（PyTorch>=1.8）
        #     drop_last=True,
        # )
        # valloader = torch.utils.data.DataLoader(
        #     valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)
        valloader = torch.utils.data.DataLoader(
            valdata,
            batch_size=cfg.BATCHSIZE,
            shuffle=False,
            num_workers=nworkers,
            pin_memory=True,
            persistent_workers=True,   # ★
            prefetch_factor=2,         # ★
            drop_last=False
        )
        

    
    elif cfg.DATASET.NAME == 'carla':
        dataroot = cfg.DATASET.DATAROOT
        traindata = CarlaDataset(dataroot, True, cfg)
        valdata = CarlaDataset(dataroot, False, cfg)
        nworkers = cfg.N_WORKERS
        trainloader = torch.utils.data.DataLoader(
            traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
        )
        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)
    else:
        raise NotImplementedError

    if return_dataset:
        return trainloader, valloader, traindata, valdata
    else:
        return trainloader, valloader