"""LightningModule wrapping the MUST-GCN model.

Loss:
    L = CE(logits) + aux_weight · mean_v CE(per_view_logits[:, v, :])

Logged metrics (per Lightning's on_step / on_epoch aggregation):
    {train,val}/loss              total loss
    {train,val}/loss_main         CE on view-aggregated logits
    {train,val}/loss_aux          mean of per-view CE
    {train,val}/acc1, /acc5       top-1 / top-5 on aggregated logits
    val/per_view_acc_{0,1,2}      top-1 from each view in isolation
    lr                            current learning rate

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
        # loss
        aux_weight: float = 0.3,
        # optimisation
        optimizer: str = 'adamw',                   # 'adamw' | 'sgd'
        base_lr: Optional[float] = None,            # None → optimiser-specific default
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
        )
        self.ce = nn.CrossEntropyLoss()

    # --------------------------------------------------------------------- API

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _step(self, batch, stage: str) -> torch.Tensor:
        data, label, _key = batch
        logits, per_view = self.model(data)        # (B, C) , (B, Vv, C)

        loss_main = self.ce(logits, label)
        # aux: mean of per-view CE
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

        # per-view accuracy: only on val (cheap to compute, but only meaningful as epoch agg)
        if stage == 'val':
            for v in range(v_count):
                acc_v = _topk_correct(per_view[:, v, :], label, k=1)
                self.log(f'val/per_view_acc_{v}', acc_v, on_step=False, on_epoch=True, batch_size=bs)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, stage='val')

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
