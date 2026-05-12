"""Inspect a PYSKL HRNet pkl: print top-level structure, splits, and one sample.

Usage:
    python data_prep/inspect_pkl.py <path-to-pkl>
    python data_prep/inspect_pkl.py     # uses $DATA_ROOT/ntu-rgbd/processed/hrnet/ntu60_hrnet.pkl
"""

import os
import pickle
import sys

import numpy as np


DEFAULT_PKL = os.path.join(
    os.environ.get('DATA_ROOT', 'data'),
    'ntu-rgbd', 'processed', 'hrnet', 'ntu60_hrnet.pkl',
)


def main(path: str) -> None:
    with open(path, 'rb') as f:
        data = pickle.load(f)

    print(f'=== {path} ===')
    print(f'top-level type: {type(data).__name__}')
    if isinstance(data, dict):
        print(f'top-level keys: {list(data.keys())}')

    # PYSKL format: {'split': {...}, 'annotations': [...]}
    splits = data.get('split', {})
    annos = data.get('annotations', [])

    print(f'\nsplits ({len(splits)} protocols):')
    for k, v in splits.items():
        print(f'  {k:20s}  {len(v):>7d} samples')

    print(f'\nannotations: {len(annos)} entries')
    if annos:
        sample = annos[0]
        print(f'sample keys: {list(sample.keys())}')
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                print(f'  {k:14s}  ndarray  shape={v.shape}  dtype={v.dtype}')
            elif isinstance(v, (list, tuple)):
                print(f'  {k:14s}  {type(v).__name__:8s} len={len(v)}')
            else:
                print(f'  {k:14s}  {type(v).__name__:8s} {v!r}')

        kp = sample.get('keypoint')
        if kp is not None:
            print(f'\nkeypoint stats (first sample):')
            print(f'  shape       = {kp.shape}   # (M, T, V, C)')
            print(f'  dtype       = {kp.dtype}')
            print(f'  range x     = [{kp[..., 0].min():.1f}, {kp[..., 0].max():.1f}]')
            print(f'  range y     = [{kp[..., 1].min():.1f}, {kp[..., 1].max():.1f}]')
            print(f'  img_shape   = {sample.get("img_shape")}')
            print(f'  total_frames= {sample.get("total_frames")}')
            print(f'  label       = {sample.get("label")}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PKL)
