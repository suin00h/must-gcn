"""LightningDataModule wrapping `MultiviewFeeder` for X-Sub train + val."""

from __future__ import annotations

import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from feeders.feeder_ntu_multiview import MultiviewFeeder


_DATASET_TO_PKL = {
    'ntu60':  'ntu60_hrnet.pkl',
    'ntu120': 'ntu120_hrnet.pkl',
}


def _default_pkl(dataset: str = 'ntu60') -> str:
    if dataset not in _DATASET_TO_PKL:
        raise ValueError(f'unknown dataset {dataset!r}; expected one of {list(_DATASET_TO_PKL)}')
    return os.path.join(
        os.environ.get('DATA_ROOT', 'data'),
        'ntu-rgbd', 'processed', 'hrnet', _DATASET_TO_PKL[dataset],
    )


class MUSTGCNDataModule(pl.LightningDataModule):
    """Wraps the PYSKL HRNet pkl as (train, val) multi-view dataloaders."""

    def __init__(
        self,
        dataset: str = 'ntu60',                       # 'ntu60' | 'ntu120'
        pkl_path: Optional[str] = None,               # overrides `dataset` if given
        train_split: str = 'xsub_train',
        val_split: str = 'xsub_val',
        batch_size: int = 64,
        num_workers: int = 8,
        window_size: int = 64,
        max_person: int = 2,
        with_score: bool = True,
        # augmentations — train-only; val/test always run un-augmented
        random_rot: bool = False,
        random_flip: bool = False,
        random_shear: bool = False,
        rot_range: float = 0.2617993877991494,        # π/12 ≈ ±15°
        shear_range: float = 0.3,
        # single-view baseline mode
        select_camera: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pkl_path = pkl_path or _default_pkl(dataset)

    # ------------------------------------------------------------------ Lightning hooks

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams
        self.train_set = MultiviewFeeder(
            pkl_path=self.pkl_path,
            split=hp.train_split,
            window_size=hp.window_size,
            max_person=hp.max_person,
            with_score=hp.with_score,
            random_crop=True,
            random_rot=hp.random_rot,
            random_flip=hp.random_flip,
            random_shear=hp.random_shear,
            rot_range=hp.rot_range,
            shear_range=hp.shear_range,
            select_camera=hp.select_camera,
        )
        self.val_set = MultiviewFeeder(
            pkl_path=self.pkl_path,
            split=hp.val_split,
            window_size=hp.window_size,
            max_person=hp.max_person,
            with_score=hp.with_score,
            random_crop=False,
            # No augmentation on val/test.
            select_camera=hp.select_camera,
        )

    def _loader(self, ds, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True,  drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set,   shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Identical to `val_dataloader` — NTU has no separate test split;
        PYSKL's `xsub_val` IS the official held-out test set used in every
        skeleton-HAR paper."""
        return self._loader(self.val_set,   shuffle=False, drop_last=False)
