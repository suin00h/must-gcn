# MUST-GCN

**Mu**lti-view **S**patio-**T**emporal GCN — multi-view 2D skeleton action recognition with cross-view attention.

## Overview

NTU RGB+D records every action from three simultaneous camera views, but most skeleton-based HAR models still consume a single view, or sidestep the problem entirely by taking Kinect 3D coordinates as input.  MUST-GCN takes 2D pose sequences from all three NTU views and fuses them through a **Cross-View Attention** block — spatial cross-attention plus temporal cross-attention across views — inserted into a shared CTR-GCN backbone.

The hypothesis: complementary spatial and temporal information across views narrows the gap between 2D- and 3D-skeleton HAR, without depending on depth sensors.

## Status

- [x] HRNet 2D pose data downloaded (PYSKL release)
- [x] 3-view coverage verified — 98.5% of X-Sub samples have all 3 cameras
- [ ] COCO-17 graph definition
- [ ] Multi-view dataloader (group by `(S, P, R, A)`)
- [ ] Cross-View Attention block
- [ ] Training pipeline + ablations

## Data

HRNet-inferenced 2D pose for NTU60 / NTU120, released by [PYSKL](https://github.com/kennymckormick/pyskl) — Faster-RCNN (ResNet-50) for human detection, HRNet-w32 for pose estimation.  Keypoints are COCO-17 in pixel coordinates.

Place (or symlink) the pkls under `<data_root>/ntu-rgbd/processed/hrnet/`:

```
<data_root>/ntu-rgbd/processed/hrnet/
├── ntu60_hrnet.pkl     56,578 samples · 4 protocols (xsub/xview × train/val)
└── ntu120_hrnet.pkl
```

Each annotation:

```python
{
    'frame_dir'     : 'S001C001P001R001A001',
    'label'         : int,                        # 0..59 (NTU60) / 0..119 (NTU120)
    'img_shape'     : (1080, 1920),
    'total_frames'  : int,
    'keypoint'      : np.float16 (M, T, 17, 2),  # (x, y) pixel coords, COCO-17
    'keypoint_score': np.float16 (M, T, 17),
}
```

`M` is 1 or 2 depending on whether the action involves one actor or a pair.

### Multi-view setup (X-Sub)

Samples are grouped by `(setup, performer, replication, action)`; complete groups have one entry per camera.

| split | groups | 3-view | 2-view | 1-view |
|-------|------:|------:|------:|------:|
| `xsub_train` | 13,433 | **98.5%** | 1.5% | 2 |
| `xsub_val`   |  5,520 | **98.7%** | 1.3% | 0 |

X-View is incompatible with 3-view inference (cam 1 is held out at test time), so MUST-GCN uses X-Sub only.

## Repository layout

```
must-gcn/
├── data_prep/
│   ├── download_hrnet.sh      fetch PYSKL HRNet pkls
│   ├── inspect_pkl.py         dump structure + first-sample stats
│   └── check_views.py         verify per-split 3-view coverage
└── README.md
```

## Citations

PYSKL / PoseConv3D — HRNet 2D pose annotations:

> Duan H., Zhao Y., Chen K., Lin D., Dai B. *Revisiting Skeleton-based Action Recognition*. CVPR 2022.

CTR-GCN — backbone architecture:

> Chen Y., Zhang Z., Yuan C., Li B., Deng Y., Hu W. *Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition*. ICCV 2021.
