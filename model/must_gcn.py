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

    DATA_BN_MODES = ('shared', 'per_view')

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
        data_bn_mode: str = 'shared',  # 'shared' | 'per_view'
        sa_mode: str = 'mha',          # see MUSTGCNBlock.SA_MODES
        input_level_cva: bool = False, # Option 1 — prepend an InputLevelCVA between data_bn and the first block
        view_weights: str = 'shared',  # 'shared' | 'separate'  (Option 0)
        sa_start_block: int = 0,       # Phase-5 Option B — blocks before this index use sa_mode='none'
        post_backbone_ca: bool = False,# Phase-5 Option A — one cross-view attn on pooled features
        cls_cross_view: bool = False,  # Phase-5 Option C — CLS-token cross-view exchange after backbone
        gated_cross_view: str = 'off', # I-6 — gated sparse cross-view transmission: off|conf|topo|both
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
        self.data_bn_mode = data_bn_mode

        # 1) data_bn — either ONE shared BN with V_views folded into the channel
        #    index, or ONE BN PER VIEW.  Per-view BN matches the intuition that
        #    NTU cameras have different x/y distributions (frontal vs lateral).
        if data_bn_mode == 'shared':
            self.data_bn = nn.BatchNorm1d(
                num_views * num_person * in_channels * num_point
            )
            bn_init(self.data_bn, 1)
        elif data_bn_mode == 'per_view':
            self.data_bn = nn.ModuleList([
                nn.BatchNorm1d(num_person * in_channels * num_point)
                for _ in range(num_views)
            ])
            for bn in self.data_bn:
                bn_init(bn, 1)
        else:
            raise ValueError(
                f'data_bn_mode must be one of {self.DATA_BN_MODES}; got {data_bn_mode!r}'
            )

        # 2) Optional input-level cross-view fusion (before the first block).
        #    Fires only when num_views > 1.  Operates at C_in raw channels.
        self.input_level_cva = None
        if input_level_cva and num_views > 1:
            from .attention import InputLevelCVA
            self.input_level_cva = InputLevelCVA(
                in_channels=in_channels,
                work_dim=max(64, base_channel),
                num_heads=num_heads if num_heads <= 4 else 2,
                dropout=attn_dropout,
            )

        # 3) 10 GCN-SA-TA blocks.
        #    Phase-5 Option B (`sa_start_block`): blocks with index < sa_start_block
        #    are forced to sa_mode='none'; only the deeper blocks run cross-view SA.
        specs = _build_block_specs(in_channels, base_channel)
        self.blocks = nn.ModuleList([
            MUSTGCNBlock(in_c, out_c, A, stride=s,
                         num_heads=num_heads, adaptive=adaptive,
                         attn_dropout=attn_dropout,
                         sa_mode=(sa_mode if i >= sa_start_block else 'none'),
                         num_views=num_views,
                         view_weights=view_weights)
            for i, (in_c, out_c, s) in enumerate(specs)
        ])
        self.out_channels = specs[-1][1]                          # = base_channel * 4

        # Phase-5 Option C — CLS-token cross-view exchange (operates on the
        # full post-backbone feature map, before head pooling).
        self.cls_cross_view = None
        if cls_cross_view and num_views > 1:
            from .attention import CLSCrossViewExchange
            self.cls_cross_view = CLSCrossViewExchange(
                self.out_channels, num_views=num_views,
                num_heads=num_heads if num_heads <= 4 else 2,
                dropout=attn_dropout,
            )

        # Phase-5 Option A — one cross-view attention on pooled features,
        # between the (T, V) pool and the FC head.
        self.post_backbone_ca = None
        if post_backbone_ca and num_views > 1:
            from .attention import PostBackboneCrossAttn
            self.post_backbone_ca = PostBackboneCrossAttn(
                self.out_channels,
                num_heads=num_heads if num_heads <= 4 else 2,
                dropout=attn_dropout,
            )

        # I-6 — gated sparse cross-view transmission on per-joint post-T-pool
        # features, with a per-(view, joint) gate (HRNet confidence) and/or a
        # skeletal-adjacency mask.  See attention.GatedCrossViewTransmission.
        self.gated_cross_view = None
        if gated_cross_view != 'off' and num_views > 1:
            from .attention import GatedCrossViewTransmission
            self.gated_cross_view = GatedCrossViewTransmission(A, gate_kind=gated_cross_view)

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

        if self.data_bn_mode == 'shared':
            # one BN with V_views in the channel index
            x_bn = x.permute(0, 1, 2, 5, 3, 4).contiguous()
            x_bn = x_bn.view(B, Vv * M * V * C, T)
            x_bn = self.data_bn(x_bn)
            x_bn = x_bn.view(B, Vv, M, V, C, T)
            return x_bn.permute(0, 1, 2, 4, 5, 3).contiguous()

        # per-view BN: run each view through its own BN1d
        out = []
        for v in range(Vv):
            xv = x[:, v]                                                # (B, M, C, T, V)
            xv = xv.permute(0, 1, 4, 2, 3).contiguous().view(B, M * V * C, T)
            xv = self.data_bn[v](xv)
            xv = xv.view(B, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()  # (B, M, C, T, V)
            out.append(xv)
        return torch.stack(out, dim=1)                                   # (B, Vv, M, C, T, V)

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

        # Extract HRNet confidence from channel 2 *before* data_bn, for
        # confidence-gated cross-view transmission (I-6).  Pool over T to
        # get a per-(view, person, joint) confidence summary.
        conf_pool = None
        if self.gated_cross_view is not None and C >= 3:
            # x is still (B, Vv, C, T, V, M); channel 2 = HRNet score
            conf_pool = x[:, :, 2, :, :, :].mean(dim=2)            # → (B, Vv, V, M)
            conf_pool = conf_pool.permute(0, 1, 3, 2).contiguous() # → (B, Vv, M, V)

        x = x.permute(0, 1, 5, 2, 3, 4).contiguous()              # (B, Vv, M, C, T, V)
        x = self._apply_data_bn(x)

        # Optional pre-backbone cross-view fusion (input-level CVA — Option 1).
        if self.input_level_cva is not None:
            x = self.input_level_cva(x)

        for blk in self.blocks:
            x = blk(x)
        # x: (B, Vv, M, C_out, T_out, V)

        # Phase-5 Option C — CLS cross-view exchange on the full feature map.
        if self.cls_cross_view is not None:
            x = self.cls_cross_view(x)

        # pool over T → (B, Vv, M, C_out, V).  If gated_cross_view is active
        # we apply it per joint here, then pool V.  Otherwise we pool (T, V)
        # together as before — numerically identical to the original head.
        if self.gated_cross_view is not None:
            x = x.mean(dim=4)                                      # (B, Vv, M, C_out, V)
            x = self.gated_cross_view(x, conf_pool)
            x = x.mean(dim=4)                                      # (B, Vv, M, C_out)
        else:
            x = x.mean(dim=(4, 5))                                 # original pool(T, V)

        # Phase-5 Option A — one cross-view attention on the pooled features.
        if self.post_backbone_ca is not None:
            x = self.post_backbone_ca(x)

        # pool over persons M → (B, Vv, C_out)
        # (mean over (T,V) then over M is numerically identical to mean over
        #  (M,T,V), so with both options off this matches the original head.)
        x = x.mean(dim=2)

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
