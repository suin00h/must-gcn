"""LightningModule wrapping the MUST-GCN model.

Loss:
    L = CE(logits) + aux_weight · mean_v CE(per_view_logits[:, v, :])

Logged metrics (Lightning on_step / on_epoch aggregation):
    {train,val,test}/loss              total loss
    {train,val,test}/loss_main         CE on view-aggregated logits
    {train,val,test}/loss_aux          mean of per-view CE
    {train,val,test}/acc1, /acc5       top-1 / top-5 on aggregated logits
    {val,test}/per_view_acc_{0,1,2}    top-1 from each view in isolation
    test/mean_per_class_acc            mean over classes (unweighted)
    test/macro_f1                      mean over per-class F1 — auxiliary diagnostic
                                       for class-imbalance behaviour (NOT a paper metric)
    lr                                  current learning rate

Test-phase artifacts (written under `<log_dir>/test_output/<run_id>/`):
    predictions.csv       per-sample pred / label / top-3 probs
    per_class_acc.csv     class · total · correct · acc
    confusion_matrix.png  row-normalised heat-map
    per_class_acc.png     per-class accuracy bar chart (mean line, red/blue by sign)
    logits.npz            raw logits, per-view logits, labels, preds

WandB extras (logged when using WandbLogger):
    test/cm_image              PNG image of the confusion matrix
    test/per_class_acc_image   PNG image of the per-class accuracy bar chart
    (uploaded as static images via wandb.Image — the interactive
    wandb.plot.* variants are too slow to render at K=120.  These keys
    are distinct from the historical `test/confusion_matrix` /
    `test/per_class_acc` so freshly-added panels don't collide with the
    old laggy Vega-plot runs.)

Test == official held-out NTU split (PYSKL labels it `xsub_val` for legacy
reasons; we expose the same dataloader as `test_dataloader`).

Optimisers:
    `adamw` (default):  lr=3e-4, wd=1e-2 — good defaults for the MHA layers
    `sgd`            :  lr=0.1, wd=4e-4, momentum=0.9, nesterov — CTR-GCN style

LR schedule (both optimisers):
    linear warmup over `warmup_epochs`, then step decay at `step_epochs`.
"""

from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from model.must_gcn import MUSTGCN


def _topk_correct(logits: torch.Tensor, label: torch.Tensor, k: int) -> torch.Tensor:
    """Fraction of samples whose true label is in the top-k of `logits`."""
    _, topk = logits.topk(k, dim=-1)
    return topk.eq(label.unsqueeze(-1)).any(dim=-1).float().mean()


class MUSTGCNModule(pl.LightningModule):
    def __init__(
        self,
        # model
        num_class: int = 60,
        num_point: int = 17,
        num_person: int = 2,
        num_views: int = 3,
        graph: str = 'graph.coco17.Graph',
        in_channels: int = 3,
        base_channel: int = 64,
        num_heads: int = 4,
        drop_out: float = 0.0,
        adaptive: bool = True,
        attn_dropout: float = 0.0,
        data_bn_mode: str = 'shared',                # 'shared' | 'per_view'
        sa_mode: str = 'mha',                        # see MUSTGCN docstring for all options
        input_level_cva: bool = False,               # Option 1 — pre-backbone cross-view fusion
        view_weights: str = 'shared',                # Option 0 — 'shared' (default) or 'separate' per-view GCN+TA
        sa_start_block: int = 0,                     # Phase-5 Option B — sparse SA from this block index
        post_backbone_ca: bool = False,              # Phase-5 Option A — post-backbone cross-attention
        cls_cross_view: bool = False,                # Phase-5 Option C — CLS-token cross-view exchange
        # loss
        aux_weight: float = 0.3,
        # optimisation
        optimizer: str = 'adamw',                    # 'adamw' | 'sgd'
        base_lr: Optional[float] = None,             # None → optimiser-specific default
        weight_decay: Optional[float] = None,
        # schedule
        max_epochs: int = 65,
        warmup_epochs: int = 5,
        step_epochs: tuple = (35, 55),
        lr_decay_rate: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MUSTGCN(
            num_class=num_class,
            num_point=num_point,
            num_person=num_person,
            num_views=num_views,
            graph=graph,
            in_channels=in_channels,
            base_channel=base_channel,
            num_heads=num_heads,
            drop_out=drop_out,
            adaptive=adaptive,
            attn_dropout=attn_dropout,
            data_bn_mode=data_bn_mode,
            sa_mode=sa_mode,
            input_level_cva=input_level_cva,
            view_weights=view_weights,
            sa_start_block=sa_start_block,
            post_backbone_ca=post_backbone_ca,
            cls_cross_view=cls_cross_view,
        )
        self.ce = nn.CrossEntropyLoss()

    # --------------------------------------------------------------------- API

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _step(self, batch, stage: str):
        """Returns (loss, logits, per_view) so test_step can accumulate without a re-forward."""
        data, label, _key = batch
        logits, per_view = self.model(data)        # (B, C) , (B, Vv, C)

        loss_main = self.ce(logits, label)
        v_count = per_view.shape[1]
        loss_aux = sum(self.ce(per_view[:, v, :], label)
                       for v in range(v_count)) / v_count
        loss = loss_main + self.hparams.aux_weight * loss_aux

        acc1 = _topk_correct(logits, label, k=1)
        acc5 = _topk_correct(logits, label, k=5)

        bs = data.size(0)
        on_step = (stage == 'train')
        self.log(f'{stage}/loss',      loss,      on_step=on_step, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f'{stage}/loss_main', loss_main, on_step=on_step, on_epoch=True, batch_size=bs)
        self.log(f'{stage}/loss_aux',  loss_aux,  on_step=on_step, on_epoch=True, batch_size=bs)
        self.log(f'{stage}/acc1',      acc1,      on_step=on_step, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f'{stage}/acc5',      acc5,      on_step=on_step, on_epoch=True, batch_size=bs)

        # per-view top-1: epoch-aggregated for both val and test (cheap)
        if stage in ('val', 'test'):
            for v in range(v_count):
                acc_v = _topk_correct(per_view[:, v, :], label, k=1)
                self.log(f'{stage}/per_view_acc_{v}', acc_v, on_step=False, on_epoch=True, batch_size=bs)

        return loss, logits, per_view

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, stage='train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, stage='val')
        return loss

    # ----------------------------------------------------------- test phase

    def on_test_epoch_start(self):
        # Buffers for end-of-epoch aggregation.
        self._test_logits = []
        self._test_per_view = []
        self._test_labels = []

    def test_step(self, batch, batch_idx):
        loss, logits, per_view = self._step(batch, stage='test')
        _, label, _ = batch
        self._test_logits.append(logits.detach().float().cpu())
        self._test_per_view.append(per_view.detach().float().cpu())
        self._test_labels.append(label.detach().cpu())
        return loss

    def on_test_epoch_end(self):
        if not self._test_logits:
            return

        logits   = torch.cat(self._test_logits)            # (N, K)
        per_view = torch.cat(self._test_per_view)          # (N, V_views, K)
        labels   = torch.cat(self._test_labels)            # (N,)
        preds    = logits.argmax(dim=1)                    # (N,)
        K        = self.hparams.num_class

        # Per-class TP / FP / FN  →  accuracy / precision / recall / F1
        tp = torch.zeros(K)
        fp = torch.zeros(K)
        fn = torch.zeros(K)
        per_class_total = torch.zeros(K)
        for p, l in zip(preds.tolist(), labels.tolist()):
            per_class_total[l] += 1
            if p == l:
                tp[l] += 1
            else:
                fp[p] += 1
                fn[l] += 1
        per_class_correct = tp                                  # alias
        per_class_acc       = tp / per_class_total.clamp(min=1)
        per_class_precision = tp / (tp + fp).clamp(min=1)
        per_class_recall    = tp / (tp + fn).clamp(min=1)
        per_class_f1        = 2 * per_class_precision * per_class_recall \
                              / (per_class_precision + per_class_recall).clamp(min=1e-8)
        seen                = per_class_total > 0
        mean_per_class_acc  = per_class_acc[seen].mean().item()
        macro_f1            = per_class_f1[seen].mean().item()
        self.log('test/mean_per_class_acc', mean_per_class_acc)
        self.log('test/macro_f1',           macro_f1)
        # auxiliary metric for behaviour inspection — diagnoses class imbalance:
        # if test/acc1 is high but macro_f1 is low, the model is "winning" by
        # over-predicting popular classes and failing on rare ones.

        # 1) Save local artifacts FIRST so the generated PNGs exist on disk
        #    before we upload them to WandB.
        out_dir = self._save_test_artifacts(
            logits, per_view, labels, preds,
            per_class_correct, per_class_total, per_class_acc,
            per_class_precision, per_class_recall, per_class_f1,
            mean_per_class_acc, macro_f1,
        )

        # 2) Upload the PNGs as wandb.Image instead of the interactive
        #    wandb.plot.* variants.  Interactive plots render client-side
        #    as Vega specs and choke the browser at 120 classes (the
        #    confusion matrix is then 120×120).  Static images are
        #    instant to render.
        self._log_wandb_images(out_dir)

    # ------------------------------------- private helpers for the test phase

    def _log_wandb_images(self, out_dir):
        """Upload pre-rendered PNGs from `out_dir` to WandB as Images.

        NOTE on key names: the historical keys `test/confusion_matrix` and
        `test/per_class_acc` are intentionally NOT reused because earlier
        runs logged interactive Vega plots under those names, and any panel
        targeting them re-renders the laggy historical data alongside the
        new images.  We log to fresh keys so a freshly-added panel only
        loads the static image-style entries from this run onward.
        """
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError:
            return
        if not isinstance(self.logger, WandbLogger):
            return
        import wandb
        try:
            wandb.log({
                'test/cm_image':            wandb.Image(str(out_dir / 'confusion_matrix.png'),
                                                       caption='Test confusion matrix (row-normalised)'),
                'test/per_class_acc_image': wandb.Image(str(out_dir / 'per_class_acc.png'),
                                                       caption='Per-class accuracy'),
            })
        except Exception as e:
            print(f'[warn] WandB image upload failed: {e}')

    def _test_output_dir(self):
        """Per-run subdirectory under `<log_dir>/test_output/` so different runs
        don't clobber each other.  Resolution order:
            1. WandB run id  (e.g. `qrixw10m`)  — when a WandbLogger is active
            2. Lightning logger version          — `version_<n>`
            3. wall-clock timestamp              — `YYYYMMDD_HHMMSS`
        """
        from pathlib import Path
        base = Path(self.trainer.default_root_dir) / 'test_output'
        try:
            from pytorch_lightning.loggers import WandbLogger
            if isinstance(self.logger, WandbLogger):
                run_id = getattr(self.logger.experiment, 'id', None)
                if run_id:
                    return base / str(run_id)
        except Exception:
            pass
        if self.logger is not None and getattr(self.logger, 'version', None) is not None:
            return base / f'version_{self.logger.version}'
        from datetime import datetime
        return base / datetime.now().strftime('%Y%m%d_%H%M%S')

    def _save_test_artifacts(self, logits, per_view, labels, preds,
                             per_class_correct, per_class_total, per_class_acc,
                             per_class_precision, per_class_recall, per_class_f1,
                             mean_per_class_acc, macro_f1):
        import csv
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        K = self.hparams.num_class
        out_dir = self._test_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        preds_np  = preds.numpy()
        labels_np = labels.numpy()

        # sample-level predictions (incl. top-3 probabilities)
        probs = torch.softmax(logits, dim=1).numpy()
        with open(out_dir / 'predictions.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['idx', 'pred', 'label', 'correct',
                        'top1_prob', 'top2_class', 'top2_prob', 'top3_class', 'top3_prob'])
            top3 = np.argsort(-probs, axis=1)[:, :3]
            for i in range(len(preds_np)):
                t = top3[i]
                w.writerow([
                    i, int(preds_np[i]), int(labels_np[i]),
                    int(preds_np[i] == labels_np[i]),
                    f'{probs[i, t[0]]:.4f}',
                    int(t[1]), f'{probs[i, t[1]]:.4f}',
                    int(t[2]), f'{probs[i, t[2]]:.4f}',
                ])

        # per-class CSV — includes precision / recall / F1 for behaviour inspection
        with open(out_dir / 'per_class_acc.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['class', 'total', 'correct', 'acc', 'precision', 'recall', 'f1'])
            for c in range(K):
                w.writerow([
                    c, int(per_class_total[c]),
                    int(per_class_correct[c]),
                    f'{float(per_class_acc[c]):.4f}',
                    f'{float(per_class_precision[c]):.4f}',
                    f'{float(per_class_recall[c]):.4f}',
                    f'{float(per_class_f1[c]):.4f}',
                ])

        # confusion matrix PNG (row-normalised)
        cm = np.zeros((K, K), dtype=np.float32)
        for p, l in zip(preds_np, labels_np):
            cm[l, p] += 1
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

        side = float(max(6, min(K * 0.15 + 3, 22)))
        fig, ax = plt.subplots(figsize=(side, side))
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        ax.set_xlabel('predicted'); ax.set_ylabel('true')
        ax.set_title(f'Confusion matrix (row-normalised, K={K})')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / 'confusion_matrix.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

        # per-class accuracy PNG (bar chart) — uploaded to WandB instead of
        # the interactive wandb.plot.bar (which is slow at K=120).
        per_class_np = per_class_acc.numpy() if hasattr(per_class_acc, 'numpy') else np.asarray(per_class_acc)
        bar_w = float(max(8, min(K * 0.12 + 3, 22)))
        fig, ax = plt.subplots(figsize=(bar_w, 4))
        colors = ['steelblue' if a >= mean_per_class_acc else 'salmon' for a in per_class_np]
        ax.bar(np.arange(K), per_class_np, color=colors, width=0.85)
        ax.axhline(mean_per_class_acc, color='black', linestyle='--', linewidth=1.0,
                   label=f'mean = {mean_per_class_acc:.3f}')
        ax.set_xlim(-0.5, K - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('class'); ax.set_ylabel('accuracy')
        ax.set_title(f'Per-class accuracy  (K={K})')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / 'per_class_acc.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

        # raw outputs for late ensembling / further analysis
        np.savez(out_dir / 'logits.npz',
                 logits=logits.numpy(),
                 per_view=per_view.numpy(),
                 labels=labels_np,
                 preds=preds_np)

        top1 = float((preds == labels).float().mean())
        print(f'\n[test artifacts] {out_dir}')
        print(f'  predictions.csv       {len(preds_np)} samples  (top-1 = {top1:.4f})')
        print(f'  per_class_acc.csv     {K} classes  (mean acc = {mean_per_class_acc:.4f}, '
              f'macro F1 = {macro_f1:.4f})')
        print(f'  confusion_matrix.png  row-normalised heat-map')
        print(f'  per_class_acc.png     bar chart vs mean')
        print(f'  logits.npz            raw logits + per-view logits + labels')

        return out_dir

    # --------------------------------------------------------------------- opt / sched

    def _optimiser_defaults(self):
        if self.hparams.optimizer == 'adamw':
            return dict(base_lr=3e-4, weight_decay=1e-2)
        if self.hparams.optimizer == 'sgd':
            return dict(base_lr=0.1,  weight_decay=4e-4)
        raise ValueError(f'unknown optimizer {self.hparams.optimizer!r}')

    def configure_optimizers(self):
        defaults = self._optimiser_defaults()
        lr = self.hparams.base_lr if self.hparams.base_lr is not None else defaults['base_lr']
        wd = self.hparams.weight_decay if self.hparams.weight_decay is not None else defaults['weight_decay']

        if self.hparams.optimizer == 'adamw':
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        else:                                                # sgd
            opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9,
                                  weight_decay=wd, nesterov=True)

        warmup = self.hparams.warmup_epochs
        steps  = list(self.hparams.step_epochs)
        rate   = self.hparams.lr_decay_rate

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:                                # linear warmup
                return (epoch + 1) / max(1, warmup)
            factor = 1.0
            for s in steps:
                if epoch >= s:
                    factor *= rate
            return factor

        sched = LambdaLR(opt, lr_lambda)
        return {
            'optimizer': opt,
            'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'},
        }
