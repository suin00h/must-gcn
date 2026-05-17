"""Multi-view 2D skeleton feeder for NTU RGB+D (X-Sub).

Each item returns the simultaneous camera views of one (S, P, R, A) instance,
ready to feed into MUST-GCN.

Returned tensor shape:  (V_views, C, T, V_joints=17, M=2)
where V_views == 1 if `select_camera` is set, otherwise the number of complete views.

Channel layout (C):
  - channel 0 : x   (normalised)
  - channel 1 : y   (normalised)
  - channel 2 : HRNet keypoint score in [0, 1]   (only if `with_score=True`)

Augmentation
------------
Training-time augmentations operate in normalised coordinate space (after
the image-centred / shorter-side normalisation) and use the **same random
parameters across all views of a single sample** so cross-view geometric
consistency is preserved.  Available (enable via flags):

    random_rot    Uniform(-rot_range, +rot_range), about origin (2D z-rotation)
    random_flip   Bernoulli(0.5) — negate x then swap COCO-17 L/R joint pairs
    random_shear  Uniform(-shear_range, +shear_range) — x' = x + s·y

Temporal cropping is also shared: a single `[start:start+window_size]`
window is sampled once per item and applied to every view, so frame `t`
of view `i` always corresponds (modulo HRNet detection drops) to the same
wall-clock instant across views.

Single-view mode
----------------
Pass `select_camera ∈ {1, 2, 3}` to fetch only that camera's data from each
3-view-complete group.  Returned shape becomes `(V_views=1, ...)`, suitable
for the single-view baseline ablation.  Groups are still filtered to those
that have all 3 cameras available, so the train/val sample lists match the
multi-view setting exactly.
"""

from __future__ import annotations

import os
import pickle
import re
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import Dataset


DEFAULT_PKL = os.path.join(
    os.environ.get('DATA_ROOT', 'data'),
    'ntu-rgbd', 'processed', 'hrnet', 'ntu60_hrnet.pkl',
)


_NAME_PAT = re.compile(r'^S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})$')


def parse_name(name: str):
    m = _NAME_PAT.match(name)
    if m is None:
        return None
    return tuple(int(x) for x in m.groups())  # (S, C, P, R, A)


# COCO-17 left-right joint pairs, for horizontal flip
# (eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
COCO17_FLIP_PAIRS = [
    (1, 2),  (3, 4),
    (5, 6),  (7, 8),  (9, 10),
    (11, 12), (13, 14), (15, 16),
]


class MultiviewFeeder(Dataset):
    NUM_JOINTS = 17                # COCO-17

    def __init__(
        self,
        pkl_path: str,
        split: str = 'xsub_train',
        window_size: int = 64,
        max_person: int = 2,
        with_score: bool = True,
        random_crop: bool = True,
        # ---- augmentation (training only — leave all False on val/test)
        random_rot: bool = False,
        random_flip: bool = False,
        random_shear: bool = False,
        rot_range: float = np.pi / 12,     # ±15°
        shear_range: float = 0.3,
        # ---- single-view mode for the baseline ablation
        select_camera: Optional[int] = None,    # 1 | 2 | 3, or None for all 3
        # ---- baseline-experiment modes
        squeeze_view: bool = False,             # drop the leading V_views=1 axis → (C,T,V,M)
        expand_views: bool = False,             # enumerate every (group, camera) as its own item
        # ---- temporal sampling: 'uniform' (PYSKL UniformSample) or 'crop' (contiguous window)
        temporal_sampling: str = 'uniform',
    ):
        self.window_size = window_size
        self.max_person = max_person
        self.with_score = with_score
        self.random_crop = random_crop

        self.random_rot = random_rot
        self.random_flip = random_flip
        self.random_shear = random_shear
        self.rot_range = float(rot_range)
        self.shear_range = float(shear_range)

        if select_camera is not None and select_camera not in (1, 2, 3):
            raise ValueError(f'select_camera must be 1, 2, 3, or None; got {select_camera}')
        self.select_camera = select_camera
        self.squeeze_view = squeeze_view
        self.expand_views = expand_views

        if temporal_sampling not in ('uniform', 'crop'):
            raise ValueError(f"temporal_sampling must be 'uniform' or 'crop'; got {temporal_sampling!r}")
        self.temporal_sampling = temporal_sampling

        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f)

        if split not in raw['split']:
            raise KeyError(f'split {split!r} not in pkl: {list(raw["split"])}')

        wanted = set(raw['split'][split])
        anno_by_name = {a['frame_dir']: a for a in raw['annotations']
                        if a['frame_dir'] in wanted}

        # group by (S, P, R, A); keep only complete 3-view groups even when
        # we'll only consume one camera — this keeps the sample list fair.
        groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
        for name, anno in anno_by_name.items():
            parsed = parse_name(name)
            if parsed is None:
                continue
            s, c, p, r, a = parsed
            groups[(s, p, r, a)][c] = anno

        self.groups = sorted(
            (key, [v[1], v[2], v[3]])
            for key, v in groups.items()
            if {1, 2, 3}.issubset(v.keys())
        )

        # expand_views: a flat list of (group_idx, cam_idx) — each camera of
        # each group becomes an independent dataset item (3× the length).
        if self.expand_views:
            self.flat_index = [(g, c) for g in range(len(self.groups))
                               for c in range(3)]

    # ------------------------------------------------------------------ Dataset

    def __len__(self) -> int:
        if self.expand_views:
            return len(self.flat_index)
        return len(self.groups)

    @property
    def num_views(self) -> int:
        return 1 if (self.select_camera is not None or self.expand_views) else 3

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, tuple]:
        if self.expand_views:
            group_idx, cam_idx = self.flat_index[idx]
            key, all_views = self.groups[group_idx]
            view_annos = [all_views[cam_idx]]
        else:
            key, view_annos = self.groups[idx]
            if self.select_camera is not None:
                view_annos = [view_annos[self.select_camera - 1]]

        labels = {a['label'] for a in view_annos}
        assert len(labels) == 1, f'inconsistent labels across views at {key}: {labels}'
        label = next(iter(labels))

        # ---- shared temporal index across views (frame i of view A == frame i of view B)
        min_T_in = min(a['total_frames'] for a in view_annos)
        if self.temporal_sampling == 'uniform':
            time_slice = self._uniform_indices(min_T_in, training=self.random_crop)
        elif min_T_in >= self.window_size:
            if self.random_crop:
                start = np.random.randint(0, min_T_in - self.window_size + 1)
            else:
                start = (min_T_in - self.window_size) // 2
            time_slice = slice(start, start + self.window_size)
        else:
            # take everything and pad downstream
            time_slice = slice(0, min_T_in)

        # ---- shared augmentation params (sampled ONCE; applied identically to every view)
        aug = self._sample_aug() if self._augment_enabled() else None

        views = np.stack([self._process_one(a, time_slice, aug) for a in view_annos])
        views = views.astype(np.float32)               # (n_views, C, T, V, M)

        # squeeze_view: drop the leading singleton view axis → (C, T, V, M).
        # Used by single-view models (CTR-GCN baseline) that expect no view axis.
        if self.squeeze_view and views.shape[0] == 1:
            views = views[0]

        return views, int(label), key

    # ------------------------------------------------------------------ helpers

    def _augment_enabled(self) -> bool:
        return self.random_rot or self.random_flip or self.random_shear

    def _sample_aug(self) -> dict:
        return {
            'theta': float(np.random.uniform(-self.rot_range, self.rot_range))
                     if self.random_rot else 0.0,
            'flip':  bool(np.random.rand() < 0.5) if self.random_flip else False,
            'shear': float(np.random.uniform(-self.shear_range, self.shear_range))
                     if self.random_shear else 0.0,
        }

    def _uniform_indices(self, num_frames: int, training: bool) -> np.ndarray:
        """PYSKL-style UniformSample: split [0, num_frames) into window_size
        equal bins and take one frame per bin — a random frame within the bin
        when training, the bin centre otherwise.  Always returns exactly
        window_size indices, so short clips are evenly oversampled (no
        zero-padding) and long clips evenly subsampled (no truncation)."""
        L = self.window_size
        bids = (np.arange(L + 1, dtype=np.int64) * max(num_frames, 1)) // L
        bsize = np.diff(bids)
        if training:
            offset = (np.random.rand(L) * bsize).astype(np.int64)
        else:
            offset = bsize // 2
        return bids[:L] + offset

    def _process_one(self, anno, time_slice, aug: Optional[dict]) -> np.ndarray:
        kp = anno['keypoint'].astype(np.float32)              # (M, T, V, 2)
        score = anno['keypoint_score'].astype(np.float32)     # (M, T, V)
        H, W = anno['img_shape']

        kp, score = self._fix_person(kp, score)
        kp = kp[:, time_slice]
        score = score[:, time_slice]
        kp, score = self._pad_time(kp, score)

        # normalise to roughly [-1, 1] (image-centred, scaled by min(H,W)/2)
        scale = min(H, W) / 2.0
        kp[..., 0] = (kp[..., 0] - W / 2.0) / scale
        kp[..., 1] = (kp[..., 1] - H / 2.0) / scale

        # augmentation operates on normalised coords so rotation/shear are
        # about the image centre (origin).
        if aug is not None:
            kp = self._apply_aug(kp, aug)

        if self.with_score:
            data = np.concatenate([kp, score[..., None]], axis=-1)   # (M, T, V, 3)
        else:
            data = kp                                                  # (M, T, V, 2)

        # (M, T, V, C) → (C, T, V, M)
        return data.transpose(3, 1, 2, 0)

    def _fix_person(self, kp, score):
        M = kp.shape[0]
        if M == self.max_person:
            return kp, score
        if M < self.max_person:
            pad = self.max_person - M
            T, V = kp.shape[1], kp.shape[2]
            kp = np.concatenate([kp, np.zeros((pad, T, V, 2), dtype=kp.dtype)], axis=0)
            score = np.concatenate([score, np.zeros((pad, T, V), dtype=score.dtype)], axis=0)
            return kp, score
        return kp[: self.max_person], score[: self.max_person]

    def _pad_time(self, kp, score):
        """Zero-pad the time axis up to window_size if shorter."""
        T_in, T_out = kp.shape[1], self.window_size
        if T_in >= T_out:
            return kp, score
        M, _, V, _ = kp.shape
        pad = T_out - T_in
        kp = np.concatenate([kp, np.zeros((M, pad, V, 2), dtype=kp.dtype)], axis=1)
        score = np.concatenate([score, np.zeros((M, pad, V), dtype=score.dtype)], axis=1)
        return kp, score

    @staticmethod
    def _apply_aug(kp: np.ndarray, aug: dict) -> np.ndarray:
        theta = aug['theta']
        flip  = aug['flip']
        shear = aug['shear']

        # rotation about origin (image centre, after normalisation)
        if theta != 0.0:
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x = kp[..., 0].copy()
            y = kp[..., 1].copy()
            kp[..., 0] = cos_t * x - sin_t * y
            kp[..., 1] = sin_t * x + cos_t * y

        # shear along x-axis (x' = x + s·y)
        if shear != 0.0:
            kp[..., 0] = kp[..., 0] + shear * kp[..., 1]

        # horizontal flip: negate x then swap COCO-17 L/R joint pairs
        if flip:
            kp[..., 0] *= -1.0
            for a, b in COCO17_FLIP_PAIRS:
                kp[:, :, [a, b]] = kp[:, :, [b, a]]

        return kp


# ---------------------------------------------------------------- self-test

if __name__ == '__main__':
    print('=== full 3-view ===')
    ds = MultiviewFeeder(
        pkl_path=DEFAULT_PKL,
        split='xsub_val',
        window_size=64,
        random_crop=False,
    )
    print(f'len(dataset) = {len(ds)}    num_views = {ds.num_views}')
    data, label, key = ds[0]
    print(f'data.shape   = {data.shape}    # (V_views, C, T, V_joints=17, M=2)')
    print(f'data range   = [{data.min():.3f}, {data.max():.3f}]    label={label}    key={key}')

    print('\n=== single-view (camera 2 only) ===')
    ds1 = MultiviewFeeder(
        pkl_path=DEFAULT_PKL, split='xsub_val', window_size=64,
        random_crop=False, select_camera=2,
    )
    print(f'len(dataset) = {len(ds1)}    num_views = {ds1.num_views}')
    data, label, key = ds1[0]
    print(f'data.shape   = {data.shape}    label={label}    key={key}')

    print('\n=== train with augmentation ===')
    np.random.seed(0)
    ds2 = MultiviewFeeder(
        pkl_path=DEFAULT_PKL, split='xsub_val', window_size=64,
        random_crop=True, random_rot=True, random_flip=True, random_shear=True,
    )
    data, label, key = ds2[0]
    print(f'augmented data.shape = {data.shape}    data range = [{data.min():.3f}, {data.max():.3f}]')
