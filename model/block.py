"""MUST-GCN block: GCN → (optional strided TCN) → SA → TA, with outer residual.

The block consumes and emits the canonical tensor shape

    (B, V_views, M, C, T, V)

and corresponds to one level of the 10-block stack defined in
`design-discussion.md` §4.

Internal sub-steps (`stride` decides whether the strided TCN is active):

    in (B, Vv, M, C_in, T_in, V)
        │
        │  outer residual  ─────────────────────────┐
        ▼                                           │
    fold (Vv, M) into batch  → (B·Vv·M, C_in, T_in, V)
        │                                           │
    GCN  : unit_gcn(C_in → C_out)                   │
        │  shape: (B·Vv·M, C_out, T_in, V)          │
        │                                           │
    if stride > 1:                                  │
        unit_tcn(C_out, C_out, stride=stride)       │
        shape: (B·Vv·M, C_out, T_out, V)            │
        │                                           │
    unfold → (B, Vv, M, C_out, T_out, V)            │
        │                                           │
    SA  (cross-view attention, internal residual)   │
    TA  (temporal self-attention, internal residual)│
        │                                           │
        ▼                                           │
        + outer_residual ◄──────────────────────────┘
        │
    ReLU
        │
    out (B, Vv, M, C_out, T_out, V)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import (
    BottleneckCrossViewAttention,
    CrossViewSpatialAttention,
    PairwiseCrossViewAttention,
    TemporalSelfAttention,
    ViewFusionWeightedSum,
)
from .ctrgcn_blocks import unit_gcn, unit_tcn


class MUSTGCNBlock(nn.Module):
    """One iteration of the GCN-SA-TA cycle."""

    # Cross-view SA options:
    #   mha                  — 3-token self-attention over all views (default)
    #   cross_pair           — pairwise cross-attention; view i ← {j, k} only (Option A)
    #   cross_bottle         — bottleneck cross-attn, latent = mean(views)    (Option B)
    #   cross_bottle_learn   — bottleneck cross-attn, latent = learnable param (Option B-learn)
    #   weighted_sum         — learnable softmax-weighted average across views
    #   none                 — identity (no cross-view fusion at this stage)
    SA_MODES = ('mha', 'cross_pair', 'cross_bottle', 'cross_bottle_learn',
                'weighted_sum', 'none')

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A,                                       # numpy (num_subset, V, V)
        stride: int = 1,
        num_heads: int = 4,
        adaptive: bool = True,
        attn_dropout: float = 0.0,
        sa_mode: str = 'mha',                    # see SA_MODES
        num_views: int = 3,
    ):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.stride = stride
        self.sa_mode = sa_mode

        # 1) GCN — intra-view joint structure, handles channel change
        self.gcn = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)

        # 2) Strided TCN — only when stride > 1 (blocks 4 and 7)
        if stride != 1:
            self.tcn_stride = unit_tcn(out_channels, out_channels,
                                       kernel_size=9, stride=stride)
            self.tcn_relu = nn.ReLU(inplace=True)
        else:
            self.tcn_stride = None
            self.tcn_relu = None

        # 3) Cross-view spatial fusion — picked by `sa_mode`.
        #    Single-view runs (num_views == 1) make any cross-view op a no-op,
        #    so we collapse to Identity for speed and to avoid wasted params.
        if num_views == 1:
            self.sa = nn.Identity()
        elif sa_mode == 'mha':
            self.sa = CrossViewSpatialAttention(out_channels, num_heads=num_heads,
                                                dropout=attn_dropout)
        elif sa_mode == 'cross_pair':
            self.sa = PairwiseCrossViewAttention(out_channels, num_heads=num_heads,
                                                 dropout=attn_dropout)
        elif sa_mode == 'cross_bottle':
            self.sa = BottleneckCrossViewAttention(out_channels, num_heads=num_heads,
                                                   dropout=attn_dropout,
                                                   use_learnable_bottleneck=False)
        elif sa_mode == 'cross_bottle_learn':
            self.sa = BottleneckCrossViewAttention(out_channels, num_heads=num_heads,
                                                   dropout=attn_dropout,
                                                   use_learnable_bottleneck=True)
        elif sa_mode == 'weighted_sum':
            self.sa = ViewFusionWeightedSum(num_views=num_views)
        elif sa_mode == 'none':
            self.sa = nn.Identity()
        else:
            raise ValueError(f'sa_mode must be one of {self.SA_MODES}; got {sa_mode!r}')

        # 4) Temporal self-attention (within view, over T tokens)
        self.ta = TemporalSelfAttention(out_channels, num_heads=num_heads,
                                        dropout=attn_dropout)

        # 5) Outer block-level residual (skip around GCN-strided_TCN-SA-TA)
        if in_channels == out_channels and stride == 1:
            self.outer_residual = nn.Identity()
        else:
            self.outer_residual = unit_tcn(in_channels, out_channels,
                                           kernel_size=1, stride=stride)
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V_views, M, C_in, T_in, V)
        B, Vv, M, C_in, T_in, V = x.shape

        # fold (V_views, M) into batch for 4D convs / GCN
        x_flat = x.reshape(B * Vv * M, C_in, T_in, V)

        # outer residual
        res = self.outer_residual(x_flat)                # (B·Vv·M, C_out, T_out, V)

        # GCN
        out = self.gcn(x_flat)                            # (B·Vv·M, C_out, T_in, V)

        # optional strided temporal conv
        if self.tcn_stride is not None:
            out = self.tcn_stride(out)                    # (B·Vv·M, C_out, T_out, V)
            out = self.tcn_relu(out)

        # unfold views and persons for attention
        C_out, T_out = out.shape[1], out.shape[2]
        out = out.reshape(B, Vv, M, C_out, T_out, V)

        # cross-view spatial attention
        out = self.sa(out)
        # temporal self-attention within view
        out = self.ta(out)

        # add outer residual (also un-flatten it)
        res = res.reshape(B, Vv, M, C_out, T_out, V)
        return self.out_relu(out + res)


# ------------------------------------------------------------------ smoke test


if __name__ == '__main__':
    from .ctrgcn_blocks import import_class

    Graph = import_class('graph.coco17.Graph')
    A = Graph().A

    test_configs = [
        # (label, in_c, out_c, stride, in_T)
        ('block 0', 3,   64,  1, 64),
        ('block 1', 64,  64,  1, 64),
        ('block 4', 64,  128, 2, 64),
        ('block 7', 128, 256, 2, 32),
        ('block 9', 256, 256, 1, 16),
    ]

    print(f'{"label":<8}  {"in_shape":<28}  {"out_shape":<28}  #params')
    for label, in_c, out_c, s, T_in in test_configs:
        blk = MUSTGCNBlock(in_c, out_c, A, stride=s, num_heads=4)
        x = torch.randn(2, 3, 2, in_c, T_in, 17)
        y = blk(x)
        n = sum(p.numel() for p in blk.parameters())
        print(f'{label:<8}  {str(tuple(x.shape)):<28}  {str(tuple(y.shape)):<28}  {n:>10,}')
        if s == 1:
            assert y.shape == (2, 3, 2, out_c, T_in, 17)
        else:
            assert y.shape == (2, 3, 2, out_c, T_in // s, 17)
