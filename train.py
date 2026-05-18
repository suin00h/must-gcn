"""MUST-GCN training entry point.

Examples
--------
With a YAML config (recommended — see configs/):
    python train.py --config configs/ntu60_xsub_adamw.yaml

Override config values from the CLI (CLI > YAML > argparse defaults):
    python train.py --config configs/ntu60_xsub_adamw.yaml --batch_size 32

Without a config (full CLI):
    python train.py --wandb_project mustgcn --wandb_tags ntu60 xsub adamw

Quick smoke run:
    python train.py --config configs/smoke.yaml

Resume from a checkpoint:
    python train.py --config configs/ntu60_xsub_adamw.yaml \\
                    --ckpt_path lightning_logs/.../checkpoints/last.ckpt

Data resolution
---------------
`mustgcn_datamodule` resolves the pkl via `$DATA_ROOT` (default `./data`).
Use `DATA_ROOT=/path/to/data python train.py ...` if your data lives
elsewhere.
"""

from __future__ import annotations

import argparse
import os
import sys

import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# DataLoader workers default to passing batches via /dev/shm.  Many containers
# limit /dev/shm to 64 MB, which overflows during training with 8 workers ×
# batch 64.  Using file-descriptor-based sharing is independent of /dev/shm
# size and costs only a few % throughput.
mp.set_sharing_strategy('file_system')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mustgcn_datamodule import MUSTGCNDataModule
from mustgcn_module import MUSTGCNModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None,
                   help='YAML config file; values are used as defaults and may be overridden by other CLI flags')

    # ---------------- data
    p.add_argument('--data_root', default=None,
                   help='overrides $DATA_ROOT for this run')
    p.add_argument('--dataset',   default='ntu60', choices=['ntu60', 'ntu120'],
                   help='picks the pkl name under $DATA_ROOT/ntu-rgbd/processed/hrnet/')
    p.add_argument('--pkl_path',  default=None,
                   help='explicit pkl path (overrides --dataset)')
    p.add_argument('--train_split', default='xsub_train')
    p.add_argument('--val_split',   default='xsub_val')
    p.add_argument('--batch_size',  type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--window_size', type=int, default=64,
                   help='output clip length T (= UniformSample clip_len)')
    p.add_argument('--temporal_sampling', default='uniform', choices=['uniform', 'crop'],
                   help='uniform = PYSKL UniformSample (whole-clip, bin-per-frame); crop = contiguous window')
    p.add_argument('--max_person',  type=int, default=2)
    p.add_argument('--no_score',    action='store_true',
                   help='drop the HRNet keypoint score channel (in_channels = 2 instead of 3)')
    # augmentations
    p.add_argument('--random_rot',   action='store_true', help='train-time z-axis rotation (±15°)')
    p.add_argument('--random_flip',  action='store_true', help='train-time horizontal flip with COCO-17 L/R swap')
    p.add_argument('--random_shear', action='store_true', help='train-time x-shear (s ~ Uniform(-0.3, 0.3))')
    # single-view baseline
    p.add_argument('--select_camera', type=int, default=None, choices=[1, 2, 3],
                   help='use only one camera; auto-forces --num_views 1')
    # baseline-experiment modes
    p.add_argument('--squeeze_view', action='store_true',
                   help='Exp 1: drop the leading V_views=1 axis so the feeder emits (C,T,V,M) — required by the CTR-GCN baseline.')
    p.add_argument('--expand_views', action='store_true',
                   help='Exp 2: TRAIN feeder enumerates every (group, camera) as an independent sample (3× the data); val/test stay single-camera.')

    # ---------------- model
    p.add_argument('--model_arch',   default='mustgcn', choices=['mustgcn', 'ctrgcn'],
                   help='mustgcn (default) = GCN-SA-TA architecture; ctrgcn = original CTR-GCN (GCN+TCN) 2D baseline.')
    p.add_argument('--num_class',    type=int, default=60)
    p.add_argument('--num_point',    type=int, default=17)
    p.add_argument('--num_views',    type=int, default=3)
    p.add_argument('--graph',        default='graph.coco17.Graph')
    p.add_argument('--base_channel', type=int, default=64)
    p.add_argument('--num_heads',    type=int, default=4)
    p.add_argument('--drop_out',     type=float, default=0.0)
    p.add_argument('--attn_dropout', type=float, default=0.0)
    p.add_argument('--no_adaptive',  action='store_true')
    p.add_argument('--data_bn_mode', default='shared', choices=['shared', 'per_view'],
                   help='shared = one BN with V_views in channel index; per_view = 3 BNs')
    p.add_argument('--sa_mode',      default='mha',
                   choices=['mha', 'cross_pair', 'cross_bottle',
                            'cross_bottle_learn', 'cross_bottle_gather',
                            'joint_cross', 'weighted_sum', 'none'],
                   help='cross-view fusion: '
                        'mha (self-attn, default), '
                        'cross_pair (pairwise cross-attn over views), '
                        'cross_bottle (bottleneck cross-attn, mean latent), '
                        'cross_bottle_learn (bottleneck, STATIC learnable latent — per-view only), '
                        'cross_bottle_gather (proper Perceiver bottleneck, gather-then-scatter), '
                        'joint_cross (pairwise cross-attn, tokens = V_joints), '
                        'weighted_sum (learnable softmax avg), '
                        'or none (identity).')
    p.add_argument('--input_level_cva', action='store_true',
                   help='Option 1: prepend a cross-view attention module at the raw C_in level, before any GCN.  Adds ~17 k params; zero-init so initially identity.')
    p.add_argument('--view_weights', default='shared', choices=['shared', 'separate'],
                   help='Option 0: shared (single GCN/TA across views — default) or separate (per-view GCN+TA; SA stays shared).  "separate" ~ 3× the GCN+TA params.')
    p.add_argument('--sa_start_block', type=int, default=0,
                   help='Phase-5 Option B: blocks before this index are forced to sa_mode=none; only blocks >= this index run cross-view SA.  e.g. 8 → SA only at blocks 8,9.')
    p.add_argument('--post_backbone_ca', action='store_true',
                   help='Phase-5 Option A: one cross-view attention on pooled features, between the (T,V) pool and the FC head.')
    p.add_argument('--cls_cross_view', action='store_true',
                   help='Phase-5 Option C: per-view CLS-token cross-view exchange after the backbone.')

    # ---------------- loss
    p.add_argument('--aux_weight', type=float, default=0.3,
                   help='weight on the per-view auxiliary CE loss (O9 default = 0.3)')

    # ---------------- optimisation
    p.add_argument('--optimizer',     choices=['adamw', 'sgd'], default='adamw')
    p.add_argument('--base_lr',       type=float, default=None,
                   help='None ⇒ optimiser-specific default (adamw: 3e-4, sgd: 0.1)')
    p.add_argument('--weight_decay',  type=float, default=None,
                   help='None ⇒ optimiser-specific default (adamw: 1e-2, sgd: 4e-4)')
    p.add_argument('--max_epochs',    type=int, default=65)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--step_epochs',   type=int, default=[35, 55], nargs='+')
    p.add_argument('--lr_decay_rate', type=float, default=0.1)
    p.add_argument('--lr_schedule',   default='step', choices=['step', 'cosine'],
                   help='LR schedule for the CTR-GCN baseline (model_arch=ctrgcn); MUST-GCN always uses step.')
    p.add_argument('--grad_clip',     type=float, default=1.0,
                   help='gradient L2 clip norm (0 to disable)')

    # ---------------- trainer
    p.add_argument('--gpus',      type=int, default=1)
    p.add_argument('--accumulate_grad_batches', type=int, default=1,
                   help='accumulate gradients over N batches; effective batch = batch_size * N')
    p.add_argument('--precision', default='bf16-mixed',
                   help='one of: 32, 16-mixed, bf16-mixed   (bf16-mixed is recommended on RTX 5090)')
    p.add_argument('--log_dir',   default='./lightning_logs')
    p.add_argument('--ckpt_path', default=None, help='resume training from a checkpoint, OR (with --eval_only) load weights for test phase')
    p.add_argument('--eval_only', action='store_true',
                   help='skip training; run only the test phase on --ckpt_path')
    p.add_argument('--seed',      type=int, default=1)
    p.add_argument('--limit_train_batches', type=float, default=None,
                   help='Lightning passthrough — useful for smoke tests')
    p.add_argument('--limit_val_batches',   type=float, default=None)
    p.add_argument('--limit_test_batches',  type=float, default=None)

    # ---------------- WandB
    p.add_argument('--wandb_project', default='mustgcn',
                   help='set to empty string ("") to disable WandB and fall back to TensorBoard')
    p.add_argument('--wandb_entity',  default=None)
    p.add_argument('--wandb_name',    default=None,
                   help='WandB run name; auto-generated from optimiser + split if unset')
    p.add_argument('--wandb_tags',    default=[], nargs='*')

    # ---- two-pass parse: CLI > YAML > argparse defaults
    args = p.parse_args()
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        unknown = set(cfg) - set(vars(args))
        if unknown:
            raise SystemExit(f'unknown keys in {args.config}: {sorted(unknown)}')
        # raise YAML values to the level of defaults, then re-parse so CLI overrides them
        p.set_defaults(**cfg)
        args = p.parse_args()
    return args


def main():
    args = parse_args()

    if args.data_root is not None:
        os.environ['DATA_ROOT'] = args.data_root

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision('high')          # safe on RTX 5090 / Blackwell

    in_channels = 2 if args.no_score else 3

    # single-view mode collapses the view axis to 1; force consistency.
    if (args.select_camera is not None or args.expand_views) and args.num_views != 1:
        print(f'[train] single-view mode  →  forcing num_views=1 (was {args.num_views})')
        args.num_views = 1

    # ---------------- model
    if args.model_arch == 'ctrgcn':
        # Original CTR-GCN 2D baseline (Exp 1) — single-view, no per-view logits.
        from ctrgcn_module import CTRGCNModule
        sgd_lr = args.base_lr if args.base_lr is not None else 0.1
        sgd_wd = args.weight_decay if args.weight_decay is not None else 4e-4
        model = CTRGCNModule(
            num_class=args.num_class,
            num_point=args.num_point,
            num_person=args.max_person,
            graph=args.graph,
            in_channels=in_channels,
            base_channel=args.base_channel,
            drop_out=args.drop_out,
            adaptive=not args.no_adaptive,
            optimizer=args.optimizer,
            base_lr=sgd_lr,
            weight_decay=sgd_wd,
            max_epochs=args.max_epochs,
            warmup_epochs=args.warmup_epochs,
            lr_schedule=args.lr_schedule,
            step_epochs=tuple(args.step_epochs),
            lr_decay_rate=args.lr_decay_rate,
        )
    else:
        model = MUSTGCNModule(
            num_class=args.num_class,
            num_point=args.num_point,
            num_person=args.max_person,
            num_views=args.num_views,
            graph=args.graph,
            in_channels=in_channels,
            base_channel=args.base_channel,
            num_heads=args.num_heads,
            drop_out=args.drop_out,
            adaptive=not args.no_adaptive,
            attn_dropout=args.attn_dropout,
            data_bn_mode=args.data_bn_mode,
            sa_mode=args.sa_mode,
            input_level_cva=args.input_level_cva,
            view_weights=args.view_weights,
            sa_start_block=args.sa_start_block,
            post_backbone_ca=args.post_backbone_ca,
            cls_cross_view=args.cls_cross_view,
            aux_weight=args.aux_weight,
            optimizer=args.optimizer,
            base_lr=args.base_lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            warmup_epochs=args.warmup_epochs,
            step_epochs=tuple(args.step_epochs),
            lr_decay_rate=args.lr_decay_rate,
        )

    # ---------------- data
    dm = MUSTGCNDataModule(
        dataset=args.dataset,
        pkl_path=args.pkl_path,
        train_split=args.train_split,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_size=args.window_size,
        max_person=args.max_person,
        with_score=not args.no_score,
        random_rot=args.random_rot,
        random_flip=args.random_flip,
        random_shear=args.random_shear,
        select_camera=args.select_camera,
        squeeze_view=args.squeeze_view,
        expand_views=args.expand_views,
        temporal_sampling=args.temporal_sampling,
    )

    # ---------------- logger
    logger = True                                       # default: TensorBoard
    if args.wandb_project:
        run_name = args.wandb_name or f'{args.optimizer}_{args.train_split}'
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=list(args.wandb_tags),
            save_dir=args.log_dir,
            log_model=False,
        )
        logger.log_hyperparams(vars(args))

    # ---------------- callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val/acc1',
            mode='max',
            save_top_k=3,
            save_last=True,
            filename='epoch{epoch:03d}-valacc{val/acc1:.4f}',
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    # ---------------- trainer
    trainer_kwargs = dict(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=args.log_dir,
        log_every_n_steps=50,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=False,    # MHA + bf16 isn't fully deterministic — accept it
    )
    if args.limit_train_batches is not None:
        trainer_kwargs['limit_train_batches'] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs['limit_val_batches']   = args.limit_val_batches
    if args.limit_test_batches is not None:
        trainer_kwargs['limit_test_batches']  = args.limit_test_batches

    trainer = pl.Trainer(**trainer_kwargs)

    if args.eval_only:
        if args.ckpt_path is None:
            raise SystemExit('--eval_only requires --ckpt_path pointing at a saved checkpoint')
        trainer.test(model, datamodule=dm, ckpt_path=args.ckpt_path)
        return

    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)

    # Final test phase on the best checkpoint.  In NTU's protocol, the
    # held-out test set is what PYSKL labels `xsub_val`, so test_dataloader
    # and val_dataloader expose the same data — but this phase additionally
    # writes test_output/{predictions.csv, per_class_acc.csv,
    # confusion_matrix.png, logits.npz} and logs WandB CM + per-class bar.
    trainer.test(model, datamodule=dm, ckpt_path='best')


if __name__ == '__main__':
    main()
