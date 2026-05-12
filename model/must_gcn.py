"""MUST-GCN — Multi-view Spatio-Temporal Graph Convolutional Network.

Top-level model.  See `design-discussion.md` for the full design rationale.

Pipeline:
    input  (B, V_views=3, C_in=3, T=64, V=17, M=2)        <-- feeder convention
        │
        ▼   permute to canonical layout (B, V_views, M, C, T, V)
        ▼   view-aware data_bn  (single BN1d, V_views folded into feature index)
        ▼
    10 × MUSTGCNBlock (GCN → optional strided TCN → SA → TA)
        channel schedule: 3 → 64 → 64 → 64 → 64 → 128 → 128 → 128 → 256 → 256 → 256
        T   schedule    : 64 → 64 → 64 → 64 → 64 → 32  → 32  → 32  → 16  → 16  → 16
        │
        ▼   pool over (M, T, V)                            (B, V_views, C_out)
        ▼   shared FC (drop optional)                      (B, V_views, num_class)
        ▼   mean over view axis                            (B, num_class)
        ▼
    logits

Returns a tuple `(logits, per_view_logits)`.  Training uses
    L = CE(logits) + aux_weight · mean[CE(per_view_logits[:, i, :])]
to push each view to be individually discriminative; see design O9.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .block import MUSTGCNBlock
from .ctrgcn_blocks import bn_init, import_class


# Block schedule: (in_c, out_c, stride) — matches CTR-GCN's channel/stride plan
def _build_block_specs(in_channels: int, base_channel: int = 64):
    c = base_channel
    return [
        (in_channels, c,     1),    # 0
        (c,           c,     1),    # 1
        (c,           c,     1),    # 2
        (c,           c,     1),    # 3
        (c,           c * 2, 2),    # 4 — stride 2
        (c * 2,       c * 2, 1),    # 5
        (c * 2,       c * 2, 1),    # 6
        (c * 2,       c * 4, 2),    # 7 — stride 2
        (c * 4,       c * 4, 1),    # 8
        (c * 4,       c * 4, 1),    # 9
    ]


class MUSTGCN(nn.Module):
    """Multi-view CTR-GCN with cross-view SA and temporal TA at every block."""

    def __init__(
        self,
        num_class: int,
        num_point: int = 17,           # COCO-17
        num_person: int = 2,
        num_views: int = 3,
        graph: str = 'graph.coco17.Graph',
        graph_args: Optional[dict] = None,
        in_channels: int = 3,          # (x, y, score)
        base_channel: int = 64,
        num_heads: int = 4,
        drop_out: float = 0.0,
        adaptive: bool = True,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        Graph = import_class(graph)
        self.graph = Graph(**(graph_args or {}))
        A = self.graph.A                                          # (num_subset, V, V)

        self.num_views = num_views
        self.num_person = num_person
        self.num_point = num_point
        self.in_channels = in_channels
        self.base_channel = base_channel

        # 1) view-aware data_bn — V_views is part of the channel index (O1)
        self.data_bn = nn.BatchNorm1d(
            num_views * num_person * in_channels * num_point
        )
        bn_init(self.data_bn, 1)

        # 2) 10 GCN-SA-TA blocks
        specs = _build_block_specs(in_channels, base_channel)
        self.blocks = nn.ModuleList([
            MUSTGCNBlock(in_c, out_c, A, stride=s,
                         num_heads=num_heads, adaptive=adaptive,
                         attn_dropout=attn_dropout)
            for in_c, out_c, s in specs
        ])
        self.out_channels = specs[-1][1]                          # = base_channel * 4

        # 3) shared FC head (applied per view)
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        nn.init.constant_(self.fc.bias, 0)

    # --------------------------------------------------------------------- API

    def _apply_data_bn(self, x: torch.Tensor) -> torch.Tensor:
        """Per-(view, person, channel, joint) BN along the time axis.

        Input/Output: (B, V_views, M, C, T, V)
        """
        B, Vv, M, C, T, V = x.shape
        # (B, Vv, M, C, T, V) → (B, Vv, M, V, C, T)  so axes group as (V_view·M·V·C, T)
        x_bn = x.permute(0, 1, 2, 5, 3, 4).contiguous()
        x_bn = x_bn.view(B, Vv * M * V * C, T)
        x_bn = self.data_bn(x_bn)
        # back: (B, Vv, M, V, C, T) → (B, Vv, M, C, T, V)
        x_bn = x_bn.view(B, Vv, M, V, C, T)
        return x_bn.permute(0, 1, 2, 4, 5, 3).contiguous()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input  : feeder convention `(B, V_views, C, T, V, M)`
        Output : (logits, per_view_logits)
                 logits          : (B, num_class)
                 per_view_logits : (B, V_views, num_class)
        """
        # feeder convention → canonical (B, V_views, M, C, T, V)
        B, Vv, C, T, V, M = x.shape
        assert Vv == self.num_views,  f'V_views: expected {self.num_views}, got {Vv}'
        assert C  == self.in_channels, f'C: expected {self.in_channels}, got {C}'
        assert V  == self.num_point,   f'V: expected {self.num_point}, got {V}'
        assert M  == self.num_person,  f'M: expected {self.num_person}, got {M}'

        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()              # (B, Vv, M, C, T, V)
        x = self._apply_data_bn(x)

        for blk in self.blocks:
            x = blk(x)
        # x: (B, Vv, M, C_out, T_out, V)

        # per-view pool: mean over (M, T, V) → (B, Vv, C_out)
        x = x.mean(dim=(2, 4, 5))

        # shared FC head per view
        x = self.drop_out(x)
        per_view_logits = self.fc(x)                              # (B, Vv, num_class)

        # view aggregation: mean over views
        logits = per_view_logits.mean(dim=1)                      # (B, num_class)
        return logits, per_view_logits


# ------------------------------------------------------------------ smoke test


if __name__ == '__main__':
    import os
    from feeders.feeder_ntu_multiview import MultiviewFeeder, DEFAULT_PKL
    from torch.utils.data import DataLoader

    torch.manual_seed(0)

    # 1) random-tensor sanity (no data required)
    model = MUSTGCN(num_class=60, num_point=17, num_views=3,
                    in_channels=3, num_heads=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'#parameters     : {n_params:,}')
    print(f'data_bn features: {model.data_bn.num_features}')
    print(f'blocks          : {len(model.blocks)}')
    print(f'out channels    : {model.out_channels}')

    x = torch.randn(2, 3, 3, 64, 17, 2)                            # feeder shape
    logits, per_view = model(x)
    print(f'\nrandom input    : {tuple(x.shape)}')
    print(f'logits          : {tuple(logits.shape)}')
    print(f'per-view logits : {tuple(per_view.shape)}')

    # backward sanity
    logits.sum().backward()
    grad_norm = sum(p.grad.abs().sum().item()
                    for p in model.parameters() if p.grad is not None)
    print(f'∑|grad|         : {grad_norm:.1f}')

    # 2) real-data sanity — only if the HRNet pkl is reachable
    if os.path.exists(DEFAULT_PKL):
        model.zero_grad()
        ds = MultiviewFeeder(DEFAULT_PKL, split='xsub_val',
                             window_size=64, random_crop=False)
        dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(dl))
        data, label, key = batch
        print(f'\nreal data       : {tuple(data.shape)}  labels={label.tolist()}')
        logits, per_view = model(data)
        print(f'logits          : {tuple(logits.shape)}')
        print(f'per-view logits : {tuple(per_view.shape)}')
        # aux loss as O9 prescribes
        ce = nn.CrossEntropyLoss()
        loss_main = ce(logits, label)
        loss_aux  = sum(ce(per_view[:, v, :], label)
                        for v in range(per_view.shape[1])) / per_view.shape[1]
        loss = loss_main + 0.3 * loss_aux
        print(f'loss_main={loss_main.item():.3f}  '
              f'loss_aux={loss_aux.item():.3f}  '
              f'total={loss.item():.3f}')
        loss.backward()
        print('backward OK')
    else:
        print(f'\n[skip] real-data sanity — {DEFAULT_PKL} not found')
