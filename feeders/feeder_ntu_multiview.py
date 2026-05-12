"""Multi-view 2D skeleton feeder for NTU RGB+D (X-Sub).

Each item returns 3 simultaneous camera views of one (S, P, R, A) instance,
ready to feed into a shared CTR-GCN backbone.

Returned tensor shape:  (V_views=3, C, T, V_joints=17, M=2)

Where C = 3 if `with_score` else 2:
  - channel 0: x   (normalised)
  - channel 1: y   (normalised)
  - channel 2: HRNet keypoint score in [0, 1]   (optional)
"""

from __future__ import annotations

import os
import pickle
import re
from collections import defaultdict
from typing import Tuple

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


class MultiviewFeeder(Dataset):
    NUM_VIEWS = 3
    NUM_JOINTS = 17                # COCO-17

    def __init__(
        self,
        pkl_path: str,
        split: str = 'xsub_train',
        window_size: int = 64,
        max_person: int = 2,
        with_score: bool = True,
        random_crop: bool = True,
    ):
        self.window_size = window_size
        self.max_person = max_person
        self.with_score = with_score
        self.random_crop = random_crop

        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f)

        if split not in raw['split']:
            raise KeyError(f'split {split!r} not in pkl: {list(raw["split"])}')

        wanted = set(raw['split'][split])
        anno_by_name = {a['frame_dir']: a for a in raw['annotations']
                        if a['frame_dir'] in wanted}

        # group by (S, P, R, A); keep only complete 3-view groups
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

    # ------------------------------------------------------------------ Dataset

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, tuple]:
        key, view_annos = self.groups[idx]
        labels = {a['label'] for a in view_annos}
        assert len(labels) == 1, f'inconsistent labels across views at {key}: {labels}'
        label = next(iter(labels))

        views = np.stack([self._process_one(a) for a in view_annos])  # (3, C, T, V, M)
        return views.astype(np.float32), int(label), key

    # ------------------------------------------------------------------ helpers

    def _process_one(self, anno) -> np.ndarray:
        kp = anno['keypoint'].astype(np.float32)              # (M, T, V, 2)
        score = anno['keypoint_score'].astype(np.float32)     # (M, T, V)
        H, W = anno['img_shape']

        kp, score = self._fix_person(kp, score)
        kp, score = self._fix_time(kp, score)

        # normalise to roughly [-1, 1] using image centre / shorter side
        scale = min(H, W) / 2.0
        kp[..., 0] = (kp[..., 0] - W / 2.0) / scale
        kp[..., 1] = (kp[..., 1] - H / 2.0) / scale

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

    def _fix_time(self, kp, score):
        T_in, T_out = kp.shape[1], self.window_size
        if T_in == T_out:
            return kp, score
        if T_in > T_out:
            start = np.random.randint(0, T_in - T_out + 1) if self.random_crop \
                    else (T_in - T_out) // 2
            return kp[:, start:start + T_out], score[:, start:start + T_out]
        # pad
        pad = T_out - T_in
        M, _, V, _ = kp.shape
        kp = np.concatenate([kp, np.zeros((M, pad, V, 2), dtype=kp.dtype)], axis=1)
        score = np.concatenate([score, np.zeros((M, pad, V), dtype=score.dtype)], axis=1)
        return kp, score


# ---------------------------------------------------------------- self-test

if __name__ == '__main__':
    ds = MultiviewFeeder(
        pkl_path=DEFAULT_PKL,
        split='xsub_val',
        window_size=64,
        random_crop=False,
    )
    print(f'len(dataset) = {len(ds)}')
    data, label, key = ds[0]
    print(f'data.shape   = {data.shape}    # (V_views=3, C, T, V_joints=17, M=2)')
    print(f'data.dtype   = {data.dtype}')
    print(f'label        = {label}')
    print(f'key (S,P,R,A)= {key}')
    print(f'data range   = [{data.min():.3f}, {data.max():.3f}]')
    print(f'#zero frames : {int((data.sum(axis=(1, 3, 4)) == 0).sum())}/{3 * 64}')
