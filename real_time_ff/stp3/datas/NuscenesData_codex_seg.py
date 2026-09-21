import torch
import pathlib
import os

from stp3.datas.NuscenesData import FuturePredictionDataset as BaseFuturePredictionDataset


class FuturePredictionDataset(BaseFuturePredictionDataset):
    """
    Codex-seg dataset variant.

    It keeps the current NuscenesData.py data path unchanged and only restores
    BEV HD-map labels for offroad-loss supervision. The HD map is not a model
    input; trainer_codex_seg.py consumes it only to build a drivable mask.
    """

    def __init__(self, nusc, is_train, cfg):
        super().__init__(nusc, is_train, cfg)
        cache_root = getattr(cfg.DATASET, 'HDMAP_CACHE_DIR', None)
        if cache_root is None:
            cache_root = pathlib.Path(getattr(cfg.DATASET, 'SAVE_DIR', 'datas')) / 'drivable_cache_codex_seg'
        self.hdmap_cache_dir = pathlib.Path(cache_root)
        self.hdmap_cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path_for_rec(self, rec):
        return self.hdmap_cache_dir / f"{rec['token']}.pt"

    def _load_or_build_present_drivable(self, rec):
        cache_path = self._cache_path_for_rec(rec)
        if cache_path.exists():
            return torch.load(cache_path, map_location='cpu')

        hdmap = self.voxelize_hd_map(rec).squeeze(0).cpu()
        drivable_channel = int(getattr(self.cfg, 'OFFROAD_DRIVABLE_CHANNEL', 1))
        if hdmap.dim() == 3 and hdmap.shape[0] > drivable_channel:
            drivable = (hdmap[drivable_channel] > 0.5).to(torch.uint8)
        elif hdmap.dim() == 3:
            drivable = (hdmap.amax(dim=0) > 0.5).to(torch.uint8)
        else:
            drivable = (hdmap > 0.5).to(torch.uint8)

        tmp_path = cache_path.with_suffix(f'.{os.getpid()}.tmp')
        torch.save(drivable, tmp_path)
        if not cache_path.exists():
            tmp_path.replace(cache_path)
        elif tmp_path.exists():
            tmp_path.unlink()
        return drivable

    def __getitem__(self, index):
        data = super().__getitem__(index)

        present_i = self.receptive_field - 1
        present_index = self.indices[index][present_i]
        rec = self.ixes[present_index]

        # Only the present-frame map is needed for offroad loss. Computing the
        # full sequence map is much slower and the trainer expands this mask to
        # the planning horizon.
        data['hdmap'] = self._load_or_build_present_drivable(rec)
        return data
