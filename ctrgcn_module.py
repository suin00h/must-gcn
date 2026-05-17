"""LightningModule for the plain CTR-GCN 2D baseline (Exp 1).

Wraps `model.ctrgcn_blocks.CTRGCN` — the original CTR-GCN architecture
(GCN + multi-scale temporal conv, NO attention, NO per-view logits).

Single-view input: the feeder is run with `select_camera` + `squeeze_view`,
so each item is `(C, T, V, M)` and the batched tensor is `(N, C, T, V, M)` —
exactly what `CTRGCN.forward` expects.

Loss: plain CrossEntropy (no per-view aux loss — there is only one logit head).

LR schedule: linear warmup then cosine annealing (PYSKL CTR-GCN recipe) or
step decay, selected by `lr_schedule`.

Test-phase artifacts (written to `<log_dir>/test_output/<run_id>/`):
    predictions.csv, per_class_acc.csv, confusion_matrix.png,
    per_class_acc.png, logits.npz
plus `test/acc1`, `test/acc5`, `test/macro_f1`, `test/mean_per_class_acc`.
"""

from __future__ import annotations

import csv
import math
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from model.ctrgcn_blocks import CTRGCN


def _topk_correct(logits: torch.Tensor, label: torch.Tensor, k: int) -> torch.Tensor:
    _, topk = logits.topk(k, dim=-1)
    return topk.eq(label.unsqueeze(-1)).any(dim=-1).float().mean()


class CTRGCNModule(pl.LightningModule):
    def __init__(
        self,
        num_class: int = 60,
        num_point: int = 17,
        num_person: int = 2,
        graph: str = 'graph.coco17.Graph',
        in_channels: int = 3,
        base_channel: int = 64,
        drop_out: float = 0.0,
        adaptive: bool = True,
        # optimisation
        optimizer: str = 'sgd',
        base_lr: float = 0.1,
        weight_decay: float = 4e-4,
        max_epochs: int = 80,
        warmup_epochs: int = 5,
        lr_schedule: str = 'cosine',                 # 'cosine' | 'step'
        step_epochs: tuple = (35, 55),
        lr_decay_rate: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = CTRGCN(
            num_class=num_class, num_point=num_point, num_person=num_person,
            graph=graph, in_channels=in_channels, base_channel=base_channel,
            drop_out=drop_out, adaptive=adaptive,
        )
        self.ce = nn.CrossEntropyLoss()

    # --------------------------------------------------------------------- API

    def forward(self, x):
        return self.model(x)                          # (N, C, T, V, M) → (N, num_class)

    def _step(self, batch, stage: str):
        data, label, _key = batch
        logits = self.model(data)                     # (N, num_class)
        loss = self.ce(logits, label)
        acc1 = _topk_correct(logits, label, k=1)
        acc5 = _topk_correct(logits, label, k=5)

        bs = data.size(0)
        on_step = (stage == 'train')
        self.log(f'{stage}/loss', loss, on_step=on_step, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f'{stage}/acc1', acc1, on_step=on_step, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f'{stage}/acc5', acc5, on_step=on_step, on_epoch=True, batch_size=bs)
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._step(batch, 'val')
        return loss

    # ----------------------------------------------------------- test phase

    def on_test_epoch_start(self):
        self._test_logits = []
        self._test_labels = []

    def test_step(self, batch, batch_idx):
        loss, logits = self._step(batch, 'test')
        _, label, _ = batch
        self._test_logits.append(logits.detach().float().cpu())
        self._test_labels.append(label.detach().cpu())
        return loss

    def on_test_epoch_end(self):
        if not self._test_logits:
            return
        logits = torch.cat(self._test_logits)         # (N, K)
        labels = torch.cat(self._test_labels)         # (N,)
        preds  = logits.argmax(dim=1)
        K      = self.hparams.num_class

        tp = torch.zeros(K); fp = torch.zeros(K); fn = torch.zeros(K)
        total = torch.zeros(K)
        for p, l in zip(preds.tolist(), labels.tolist()):
            total[l] += 1
            if p == l: tp[l] += 1
            else: fp[p] += 1; fn[l] += 1
        acc  = tp / total.clamp(min=1)
        prec = tp / (tp + fp).clamp(min=1)
        rec  = tp / (tp + fn).clamp(min=1)
        f1   = 2 * prec * rec / (prec + rec).clamp(min=1e-8)
        seen = total > 0
        mean_pc  = acc[seen].mean().item()
        macro_f1 = f1[seen].mean().item()
        self.log('test/mean_per_class_acc', mean_pc)
        self.log('test/macro_f1', macro_f1)

        self._save_artifacts(logits, labels, preds, tp, total, acc, prec, rec, f1,
                             mean_pc, macro_f1)

    def _test_output_dir(self):
        from pathlib import Path
        base = Path(self.trainer.default_root_dir) / 'test_output'
        try:
            from pytorch_lightning.loggers import WandbLogger
            if isinstance(self.logger, WandbLogger):
                rid = getattr(self.logger.experiment, 'id', None)
                if rid:
                    return base / str(rid)
        except Exception:
            pass
        from datetime import datetime
        return base / datetime.now().strftime('%Y%m%d_%H%M%S')

    def _save_artifacts(self, logits, labels, preds, tp, total, acc, prec, rec, f1,
                        mean_pc, macro_f1):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        K = self.hparams.num_class
        out_dir = self._test_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        preds_np, labels_np = preds.numpy(), labels.numpy()

        probs = torch.softmax(logits, dim=1).numpy()
        with open(out_dir / 'predictions.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['idx', 'pred', 'label', 'correct', 'top1_prob'])
            for i in range(len(preds_np)):
                w.writerow([i, int(preds_np[i]), int(labels_np[i]),
                            int(preds_np[i] == labels_np[i]),
                            f'{probs[i, preds_np[i]]:.4f}'])

        with open(out_dir / 'per_class_acc.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['class', 'total', 'correct', 'acc', 'precision', 'recall', 'f1'])
            for c in range(K):
                w.writerow([c, int(total[c]), int(tp[c]),
                            f'{float(acc[c]):.4f}', f'{float(prec[c]):.4f}',
                            f'{float(rec[c]):.4f}', f'{float(f1[c]):.4f}'])

        cm = np.zeros((K, K), dtype=np.float32)
        for p, l in zip(preds_np, labels_np):
            cm[l, p] += 1
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        side = float(max(6, min(K * 0.15 + 3, 22)))
        fig, ax = plt.subplots(figsize=(side, side))
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_xlabel('predicted'); ax.set_ylabel('true')
        ax.set_title(f'CTR-GCN 2D — confusion matrix (row-norm, K={K})')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / 'confusion_matrix.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

        acc_np = acc.numpy()
        bar_w = float(max(8, min(K * 0.12 + 3, 22)))
        fig, ax = plt.subplots(figsize=(bar_w, 4))
        colors = ['steelblue' if a >= mean_pc else 'salmon' for a in acc_np]
        ax.bar(np.arange(K), acc_np, color=colors, width=0.85)
        ax.axhline(mean_pc, color='black', ls='--', lw=1.0, label=f'mean = {mean_pc:.3f}')
        ax.set_xlim(-0.5, K - 0.5); ax.set_ylim(0, 1)
        ax.set_xlabel('class'); ax.set_ylabel('accuracy')
        ax.set_title(f'CTR-GCN 2D — per-class accuracy (K={K})')
        ax.legend(loc='lower right', fontsize=9); ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / 'per_class_acc.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

        np.savez(out_dir / 'logits.npz',
                 logits=logits.numpy(), labels=labels_np, preds=preds_np)

        top1 = float((preds == labels).float().mean())
        if isinstance(self.logger, type(self.logger)):
            try:
                from pytorch_lightning.loggers import WandbLogger
                if isinstance(self.logger, WandbLogger):
                    import wandb
                    wandb.log({
                        'test/cm_image': wandb.Image(str(out_dir / 'confusion_matrix.png')),
                        'test/per_class_acc_image': wandb.Image(str(out_dir / 'per_class_acc.png')),
                    })
            except Exception as e:
                print(f'[warn] wandb image upload failed: {e}')

        print(f'\n[CTR-GCN test artifacts] {out_dir}')
        print(f'  predictions.csv      {len(preds_np)} samples  (top-1 = {top1:.4f})')
        print(f'  per_class_acc.csv    {K} classes  (mean = {mean_pc:.4f}, macro F1 = {macro_f1:.4f})')

    # --------------------------------------------------------------- opt / sched

    def configure_optimizers(self):
        hp = self.hparams
        if hp.optimizer == 'sgd':
            opt = torch.optim.SGD(self.parameters(), lr=hp.base_lr, momentum=0.9,
                                  weight_decay=hp.weight_decay, nesterov=True)
        elif hp.optimizer == 'adamw':
            opt = torch.optim.AdamW(self.parameters(), lr=hp.base_lr,
                                    weight_decay=hp.weight_decay)
        else:
            raise ValueError(f'unknown optimizer {hp.optimizer!r}')

        warmup, total = hp.warmup_epochs, hp.max_epochs

        if hp.lr_schedule == 'cosine':
            def lr_lambda(epoch):
                if epoch < warmup:
                    return (epoch + 1) / max(1, warmup)
                progress = (epoch - warmup) / max(1, total - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        elif hp.lr_schedule == 'step':
            steps = list(hp.step_epochs)
            rate = hp.lr_decay_rate
            def lr_lambda(epoch):
                if epoch < warmup:
                    return (epoch + 1) / max(1, warmup)
                factor = 1.0
                for s in steps:
                    if epoch >= s:
                        factor *= rate
                return factor
        else:
            raise ValueError(f'unknown lr_schedule {hp.lr_schedule!r}')

        sched = LambdaLR(opt, lr_lambda)
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'}}
