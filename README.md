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

The hypothesis was: complementary spatial and temporal information across views narrows the gap between 2D- and 3D-skeleton HAR.  Empirically, **dense per-block cross-view attention contaminates and consistently underperforms late logit fusion** on both NTU60 and NTU120 (12+ variants tested, including the proper Perceiver gather-scatter bottleneck).  The variant that finally *does* beat late fusion is **confidence-gated, occlusion-driven *selective* cross-view transmission** — sparse fusion that fires only on joints HRNet flagged as uncertain (preliminary, single dataset).

## Status

- [x] HRNet 2D pose data downloaded (PYSKL release)
- [x] 3-view coverage verified — 98.5 % of X-Sub samples have all 3 cameras
- [x] COCO-17 graph definition
- [x] Multi-view dataloader — UniformSample temporal sampling, single-view modes
- [x] GCN-SA-TA block + full MUST-GCN model
- [x] Training pipeline — WandB, SGD / AdamW, bf16, YAML configs, gradient accumulation
- [x] Test phase with per-class CSV / confusion-matrix PNG / per-class-bar PNG / raw-logits npz artifacts
- [x] CTR-GCN 2D single-view baseline
- [x] Cross-view fusion variants — mha, cross_pair, bottleneck, joint / input / post / sparse / CLS
- [x] NTU60 / NTU120 X-Sub training on the corrected uniform-100 pipeline
- [x] Cross-view fusion ablation (NTU60 + NTU120) — every dense fusion variant loses to `none` (late logit averaging); even the proper Perceiver gather-scatter bottleneck loses
- [x] Confidence-gated selective cross-view transmission (NTU60, preliminary) — **92.33 % > `none` 92.20 %**; topology-only sparsity *hurts*, so confidence (occlusion) is the active ingredient
- [x] Headline on the corrected pipeline: **NTU60 92.67 %** (`none` @ 100 ep); NTU120 100-ep run in progress
- [ ] NTU120 confirmation of confidence-gated transmission
- [ ] Bone + motion multi-stream
- [ ] Paper writeup

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
├── configs/                              YAML training recipes
│   ├── smoke.yaml                         pipeline smoke (~30 s)
│   ├── ntu60_xsub_adamw.yaml              AdamW · NTU60 · 65 ep
│   ├── ntu60_xsub_sgd.yaml                SGD-Nesterov · NTU60 · 65 ep
│   ├── ntu60_xsub_adamw_long.yaml         AdamW · NTU60 · 100 ep
│   ├── ntu120_xsub_adamw.yaml             AdamW · NTU120 · 65 ep  (no aug)
│   ├── ntu120_xsub_adamw_aug.yaml         + train-time aug (rot / flip / shear)
│   ├── ntu120_xsub_adamw_aug_100ep.yaml   + 100 ep, step decay [50, 80]
│   ├── ntu120_xsub_adamw_aug_sa1.yaml     SA num_heads = 1
│   ├── ntu120_xsub_adamw_aug_sa2.yaml     SA num_heads = 2
│   ├── ntu120_xsub_adamw_aug_pvbn.yaml    per-view data_bn (3 separate BN1d)
│   ├── ntu120_xsub_adamw_aug_wsum.yaml    SA = learnable weighted sum (no MHA)
│   └── ntu120_xsub_1view.yaml             single-view baseline (camera 2 only)
├── data_prep/
│   ├── download_hrnet.sh                  fetch PYSKL HRNet pkls into $DATA_ROOT
│   ├── inspect_pkl.py                     dump structure + first-sample stats
│   └── check_views.py                     verify per-split 3-view coverage
├── feeders/
│   └── feeder_ntu_multiview.py            multi-view Dataset → (B, V_views, C, T, V, M)
├── graph/
│   ├── coco17.py                          17-joint adjacency (3 partitions)
│   └── tools.py                           graph utilities
├── model/
│   ├── attention.py                       cross-view fusion modules (mha / pairwise / bottleneck / joint / CLS) + temporal TA
│   ├── block.py                           GCN-SA-TA block (with optional strided TCN)
│   ├── ctrgcn_blocks.py                   CTR-GCN building blocks (vendored + split)
│   └── must_gcn.py                        top-level MUSTGCN module
├── mustgcn_module.py                      LightningModule (loss, metrics, optimisers, test artifacts)
├── ctrgcn_module.py                       CTR-GCN 2D single-view baseline LightningModule
├── mustgcn_datamodule.py                  LightningDataModule (X-Sub train / val / test)
├── train.py                               training entry point (--config YAML, --model_arch, --eval_only)
├── analyze_params.py                      hierarchical parameter-count breakdown per variant
├── sanity_view.py                         HRNet skeleton overlay / multi-view animation / trajectory plots
└── run_experiments.sh                     sequential queue runner (template — edit CONFIGS array)
```

## Usage

### Pipeline smoke test (~30 s)

```bash
python train.py --config configs/smoke.yaml
```

### Train

```bash
# Primary AdamW run on NTU120 X-Sub with augmentation (recommended)
python train.py --config configs/ntu120_xsub_adamw_aug.yaml

# SGD-Nesterov variant on NTU60
python train.py --config configs/ntu60_xsub_sgd.yaml

# Single-view baseline (camera 2 only)
python train.py --config configs/ntu120_xsub_1view.yaml

# Override any setting from the CLI (CLI > YAML > defaults)
python train.py --config configs/ntu120_xsub_adamw_aug.yaml --batch_size 32 --num_heads 1
```

`train.py` accepts the usual CLI flags including `--random_rot / --random_flip / --random_shear`, `--select_camera {1,2,3}` (auto-forces `num_views=1`), `--data_bn_mode {shared,per_view}`, `--temporal_sampling {uniform,crop}`, `--accumulate_grad_batches N`, `--model_arch {mustgcn,ctrgcn}`, `--num_heads N`, and `--sa_mode {mha,cross_pair,cross_bottle,cross_bottle_learn,cross_bottle_gather,joint_cross,weighted_sum,none}`.

### Test only (no training)

Re-run the test phase on a saved checkpoint without retraining — useful for getting test artifacts on a previously-completed run, or for ensembling later:

```bash
python train.py --config configs/ntu120_xsub_adamw.yaml \
                --eval_only \
                --ckpt_path lightning_logs/MUST-GCN/<run_id>/checkpoints/<best>.ckpt \
                --wandb_name "ntu120-baseline-test"
```

Eval-only uses ~3–6 GB VRAM (no backward pass), so it's safe to run alongside a training job.

### Test artifacts

Each completed test phase writes to `lightning_logs/test_output/<wandb_id>/`:

| file | content |
|------|---------|
| `predictions.csv` | per-sample `idx, pred, label, correct, top1_prob, top2_class, top2_prob, top3_class, top3_prob` |
| `per_class_acc.csv` | `class, total, correct, acc` |
| `confusion_matrix.png` | row-normalised heat-map |
| `per_class_acc.png` | bar chart with mean line (red = below mean, blue = at/above) |
| `logits.npz` | raw `logits (N, K)` + `per_view (N, V_views, K)` + `labels` + `preds` |

Both PNGs are also uploaded to WandB as `wandb.Image()`s under `test/cm_image` and `test/per_class_acc_image`.

### Multiple runs back-to-back

`run_experiments.sh` runs configs in order, logs each to its own file, and continues on failure.  Edit the `CONFIGS` array at the top to choose which experiments to run:

```bash
chmod +x run_experiments.sh        # one-time
./run_experiments.sh               # or: tmux new -s exp; ./run_experiments.sh
```

The runner defaults to `CUDA_VISIBLE_DEVICES=2`; override at invocation time:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_experiments.sh
```

### Inspect model parameters

```bash
python analyze_params.py                                  # baseline (NTU120, 3 views, mha, h=4, shared BN)
python analyze_params.py --num_views 1                    # single-view variant
python analyze_params.py --sa_mode weighted_sum
python analyze_params.py --data_bn_mode per_view
python analyze_params.py --num_heads 2                    # SA-heads ablation
python analyze_params.py --num_class 60                   # NTU60 — only `fc` changes
```

Prints a top-level summary, per-block breakdown, and a zoom into one representative block.

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
python train.py --config configs/ntu120_xsub_adamw_aug.yaml \
                --ckpt_path lightning_logs/MUST-GCN/<run_id>/checkpoints/last.ckpt
```

## Citations

PYSKL / PoseConv3D — HRNet 2D pose annotations:

> Duan H., Zhao Y., Chen K., Lin D., Dai B. *Revisiting Skeleton-based Action Recognition*. CVPR 2022.

CTR-GCN — backbone architecture (the GCN component is reused, the temporal-conv component is replaced):

> Chen Y., Zhang Z., Yuan C., Li B., Deng Y., Hu W. *Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition*. ICCV 2021.
