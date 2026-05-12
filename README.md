# MUST-GCN
![Pose](https://img.shields.io/badge/SkeletonPose-4E56C0?style=flat-square&logo=fastlane&logoColor=8388d2&labelColor=212030)
![HAR](https://img.shields.io/badge/ActionRecognition-D13A4F?style=flat-square&logo=odysee&logoColor=da6172&labelColor=212030)

**Mu**lti-view **S**patio-**T**emporal GCN — multi-view 2D skeleton action recognition with cross-view attention.

## Overview

NTU RGB+D captures every action from three simultaneous camera views, but most skeleton-based HAR models still consume a single view, or fall back on Kinect 3D coordinates.  MUST-GCN keeps the three 2D pose streams as an explicit tensor axis and fuses them at every depth of the backbone via multi-head attention.

Each MUST-GCN block is a `GCN → SA → TA` cycle:

- **GCN** — CTR-GCN's channel-wise topology refinement, applied per view with shared weights (handles joint structure).
- **SA** — cross-view spatial attention: at every spatio-temporal cell, each view's feature attends to the other two views.
- **TA** — temporal self-attention within view, replacing CTR-GCN's multi-scale temporal convolution.

The hypothesis: complementary spatial and temporal information across views narrows the gap between 2D- and 3D-skeleton HAR, without depending on depth sensors.

## Status

- [x] HRNet 2D pose data downloaded (PYSKL release)
- [x] 3-view coverage verified — 98.5 % of X-Sub samples have all 3 cameras
- [x] COCO-17 graph definition
- [x] Multi-view dataloader (group by `(S, P, R, A)`)
- [x] GCN-SA-TA block + full MUST-GCN model (~3.86 M params)
- [x] Training pipeline with WandB logging, AdamW + SGD-Nesterov, bf16 mixed precision
- [ ] Full NTU60 X-Sub training run
- [ ] NTU120 X-Sub training run
- [ ] Ablations (heads, aux-loss weight, no-score, insertion variants)

## Setup

### Environment

- Python 3.10+, PyTorch 2.x with CUDA, PyTorch Lightning, WandB.
- OpenCV is only required for `sanity_view.py` (RGB overlay).
- Tested on PyTorch 2.11 / CUDA 13 / RTX 5090.

```bash
pip install pytorch-lightning wandb matplotlib pyyaml opencv-python-headless
```

### Data

HRNet-inferenced 2D pose annotations for NTU60 / NTU120, released by [PYSKL](https://github.com/kennymckormick/pyskl) — Faster-RCNN (ResNet-50) for human detection, HRNet-w32 for pose estimation.  Keypoints are COCO-17 in pixel coordinates.

Expected layout under `$DATA_ROOT` (defaults to `./data`):

```
<data_root>/ntu-rgbd/
├── videos/                            raw NTU RGB videos (optional, sanity_view.py only)
└── processed/hrnet/
    ├── ntu60_hrnet.pkl                56,578 samples · 4 protocols
    └── ntu120_hrnet.pkl
```

Either symlink your data into the project (one-time):

```bash
ln -s /path/to/data ./data
```

…or export the environment variable each session:

```bash
export DATA_ROOT=/path/to/data
```

Download the pkls (re-uses `$DATA_ROOT`):

```bash
bash data_prep/download_hrnet.sh
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
| `xsub_train` | 13,433 | **98.5 %** | 1.5 % | 2 |
| `xsub_val`   |  5,520 | **98.7 %** | 1.3 % | 0 |

X-View is incompatible with 3-view inference (cam 1 is held out at test time), so MUST-GCN uses X-Sub only.

## Repository layout

```
must-gcn/
├── configs/                       YAML training recipes
│   ├── smoke.yaml                  pipeline smoke (~30 s)
│   ├── ntu60_xsub_adamw.yaml       AdamW · NTU60 · 65 ep  (primary)
│   ├── ntu60_xsub_sgd.yaml         SGD-Nesterov · NTU60 · 65 ep  (CTR-GCN-faithful)
│   ├── ntu60_xsub_adamw_long.yaml  AdamW · NTU60 · 100 ep
│   └── ntu120_xsub_adamw.yaml      AdamW · NTU120 · 65 ep
├── data_prep/
│   ├── download_hrnet.sh           fetch PYSKL HRNet pkls into $DATA_ROOT
│   ├── inspect_pkl.py              dump structure + first-sample stats
│   └── check_views.py              verify per-split 3-view coverage
├── feeders/
│   └── feeder_ntu_multiview.py     multi-view Dataset → (B, 3, C, T, V, M)
├── graph/
│   ├── coco17.py                   17-joint adjacency (3 partitions)
│   └── tools.py                    graph utilities
├── model/
│   ├── attention.py                cross-view SA + temporal TA (multi-head)
│   ├── block.py                    GCN-SA-TA block (with optional strided TCN)
│   ├── ctrgcn_blocks.py            CTR-GCN building blocks (vendored)
│   └── must_gcn.py                 top-level MUSTGCN module
├── mustgcn_module.py               LightningModule (loss, metrics, optimisers)
├── mustgcn_datamodule.py           LightningDataModule (X-Sub train / val)
├── train.py                        training entry point (--config YAML support)
├── run_experiments.sh              sequential queue runner
└── sanity_view.py                  HRNet keypoint overlay on RGB frames
```

## Usage

### Pipeline smoke test (~30 s)

```bash
python train.py --config configs/smoke.yaml
```

### Train

```bash
# Primary run: AdamW, NTU60 X-Sub, 65 epochs
python train.py --config configs/ntu60_xsub_adamw.yaml

# SGD-Nesterov variant
python train.py --config configs/ntu60_xsub_sgd.yaml

# NTU120 X-Sub
python train.py --config configs/ntu120_xsub_adamw.yaml

# Override any setting from the CLI (CLI > YAML > defaults)
python train.py --config configs/ntu60_xsub_adamw.yaml --batch_size 32
```

Checkpoints (top-3 by `val/acc1` + `last.ckpt`) land under `lightning_logs/`; metrics stream to WandB under the project named in each YAML's `wandb_project`.

### Multiple runs back-to-back

`run_experiments.sh` runs configs in order, logs each to its own file, and continues on failure:

```bash
chmod +x run_experiments.sh        # one-time
./run_experiments.sh               # or: tmux new -s exp; ./run_experiments.sh
```

### Visual sanity (RGB overlay of HRNet keypoints on all 3 cameras)

```bash
python sanity_view.py --frame_dir S001P001R001A050     # any (S,P,R,A) string
```

### Inspect data

```bash
python data_prep/inspect_pkl.py                        # default: ntu60 pkl
python data_prep/check_views.py                        # per-split 3-view coverage
```

## Resuming a run

```bash
python train.py --config configs/ntu60_xsub_adamw.yaml \
                --ckpt_path lightning_logs/.../checkpoints/last.ckpt
```

## Citations

PYSKL / PoseConv3D — HRNet 2D pose annotations:

> Duan H., Zhao Y., Chen K., Lin D., Dai B. *Revisiting Skeleton-based Action Recognition*. CVPR 2022.

CTR-GCN — backbone architecture (the GCN component is reused, the temporal-conv component is replaced):

> Chen Y., Zhang Z., Yuan C., Li B., Deng Y., Hu W. *Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition*. ICCV 2021.
