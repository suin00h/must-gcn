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
    p.add_argument('--window_size', type=int, default=64)
    p.add_argument('--max_person',  type=int, default=2)
    p.add_argument('--no_score',    action='store_true',
                   help='drop the HRNet keypoint score channel (in_channels = 2 instead of 3)')

    # ---------------- model
    p.add_argument('--num_class',    type=int, default=60)
    p.add_argument('--num_point',    type=int, default=17)
    p.add_argument('--num_views',    type=int, default=3)
    p.add_argument('--graph',        default='graph.coco17.Graph')
    p.add_argument('--base_channel', type=int, default=64)
    p.add_argument('--num_heads',    type=int, default=4)
    p.add_argument('--drop_out',     type=float, default=0.0)
    p.add_argument('--attn_dropout', type=float, default=0.0)
    p.add_argument('--no_adaptive',  action='store_true')

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
    p.add_argument('--grad_clip',     type=float, default=1.0,
                   help='gradient L2 clip norm (0 to disable)')

    # ---------------- trainer
    p.add_argument('--gpus',      type=int, default=1)
    p.add_argument('--precision', default='bf16-mixed',
                   help='one of: 32, 16-mixed, bf16-mixed   (bf16-mixed is recommended on RTX 5090)')
    p.add_argument('--log_dir',   default='./lightning_logs')
    p.add_argument('--ckpt_path', default=None, help='resume from a checkpoint')
    p.add_argument('--seed',      type=int, default=1)
    p.add_argument('--limit_train_batches', type=float, default=None,
                   help='Lightning passthrough — useful for smoke tests')
    p.add_argument('--limit_val_batches',   type=float, default=None)

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

    # ---------------- model
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
        deterministic=False,    # MHA + bf16 isn't fully deterministic — accept it
    )
    if args.limit_train_batches is not None:
        trainer_kwargs['limit_train_batches'] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs['limit_val_batches']   = args.limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    trainer.validate(model, datamodule=dm, ckpt_path='best')


if __name__ == '__main__':
    main()
