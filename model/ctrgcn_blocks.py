"""CTR-GCN building blocks, adapted from the official CTR-GCN reference
implementation (Chen et al., *Channel-wise Topology Refinement Graph
Convolution for Skeleton-Based Action Recognition*, ICCV 2021 —
https://github.com/Uason-Chen/CTR-GCN).

Refactored so the 10 `TCN_GCN_unit` blocks are individually addressable via
`CTRGCNBackbone.blocks[i]`, which is what MUST-GCN needs in order to insert
the Cross-View Attention module between any two blocks.

The original monolithic `Model` (data_bn + 10 blocks + fc) is split here into
`CTRGCNBackbone` (data_bn + 10 blocks) and `CTRGCNHead` (pool + fc).
A convenience `CTRGCN` = Backbone + Head is provided as a single-view
baseline.
"""

from __future__ import annotations

import importlib
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------------ helpers


def import_class(name: str):
    mod_path, _, class_name = name.rpartition('.')
    return getattr(importlib.import_module(mod_path), class_name)


def conv_branch_init(conv, branches):
    n = conv.weight.size(0)
    k1, k2 = conv.weight.size(1), conv.weight.size(2)
    nn.init.normal_(conv.weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    name = m.__class__.__name__
    if 'Conv' in name:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif 'BatchNorm' in name:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


# ------------------------------------------------------------------ building blocks


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1), padding=(pad, 0),
            stride=(stride, 1), dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScale_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilations=(1, 2, 3, 4), residual=True, residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, \
            '# out channels must be a multiple of # branches'

        branch_channels = out_channels // (len(dilations) + 2)
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels,
                             kernel_size=ks, stride=stride, dilation=d),
            )
            for ks, d in zip(kernel_size, dilations)
        ])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels),
        ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0,
                      stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels),
        ))

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels,
                                         kernel_size=residual_kernel_size, stride=stride)
        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return out + res


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # for very small in_channels (raw input dims), use a fixed minimum
        if in_channels in (2, 3, 9):
            self.rel_channels, self.mid_channels = 8, 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction

        self.conv1 = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1.0):
        x1 = self.conv1(x).mean(-2)
        x2 = self.conv2(x).mean(-2)
        x3 = self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
        return torch.einsum('ncuv,nctv->nctu', x1, x3)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4,
                 adaptive=True, residual=True):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.inter_c = out_channels // coff_embedding
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList(
            [CTRGC(in_channels, out_channels) for _ in range(self.num_subset)]
        )
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.register_buffer('A', torch.from_numpy(A.astype(np.float32)))

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        A = self.PA if self.adaptive else self.A
        y = None
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y = y + self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    """One CTR-GCN block: GCN → multi-scale TCN, with residual."""

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True,
                 adaptive=True, kernel_size=5, dilations=(1, 2)):
        super().__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels,
                                            kernel_size=kernel_size, stride=stride,
                                            dilations=list(dilations), residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


# ------------------------------------------------------------------ refactored modules


class CTRGCNBackbone(nn.Module):
    """data_bn + 10 TCN_GCN_unit blocks (no classification head).

    The 10 blocks are kept in a `ModuleList` so a multi-view wrapper can run
    them up to any insertion point, splice in a Cross-View Attention block,
    then resume.

    Channel schedule (base_channel = 64):
        block 0       : in_channels → 64    (T, V)
        blocks 1-3    : 64 → 64              (T, V)
        block 4       : 64 → 128             (T/2, V)   stride=2
        blocks 5-6    : 128 → 128            (T/2, V)
        block 7       : 128 → 256            (T/4, V)   stride=2
        blocks 8-9    : 256 → 256            (T/4, V)
    """

    NUM_BLOCKS = 10

    def __init__(self,
                 num_point: int = 17,
                 num_person: int = 2,
                 graph: str = 'graph.coco17.Graph',
                 graph_args: Optional[dict] = None,
                 in_channels: int = 3,
                 base_channel: int = 64,
                 adaptive: bool = True):
        super().__init__()
        if graph is None:
            raise ValueError('graph must be specified, e.g. "graph.coco17.Graph"')
        Graph = import_class(graph)
        self.graph = Graph(**(graph_args or {}))
        A = self.graph.A                              # (3, V, V)

        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.base_channel = base_channel

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        c = base_channel
        self.blocks = nn.ModuleList([
            TCN_GCN_unit(in_channels, c,    A, residual=False, adaptive=adaptive),  # 0
            TCN_GCN_unit(c,           c,    A, adaptive=adaptive),                  # 1
            TCN_GCN_unit(c,           c,    A, adaptive=adaptive),                  # 2
            TCN_GCN_unit(c,           c,    A, adaptive=adaptive),                  # 3
            TCN_GCN_unit(c,           2 * c, A, stride=2, adaptive=adaptive),       # 4
            TCN_GCN_unit(2 * c,       2 * c, A, adaptive=adaptive),                 # 5
            TCN_GCN_unit(2 * c,       2 * c, A, adaptive=adaptive),                 # 6
            TCN_GCN_unit(2 * c,       4 * c, A, stride=2, adaptive=adaptive),       # 7
            TCN_GCN_unit(4 * c,       4 * c, A, adaptive=adaptive),                 # 8
            TCN_GCN_unit(4 * c,       4 * c, A, adaptive=adaptive),                 # 9
        ])
        bn_init(self.data_bn, 1)

    @property
    def out_channels(self) -> int:
        return self.base_channel * 4

    # --------------------------------------------------------------------- API

    def reshape_input(self, x: torch.Tensor):
        """(N, C, T, V, M) → (N*M, C, T, V) via data_bn + permute."""
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        return x.view(N * M, C, T, V), N, M

    def forward_blocks(self, x: torch.Tensor, start: int = 0,
                       end: Optional[int] = None) -> torch.Tensor:
        """Run blocks[start:end] on x. `end=None` runs until the last block."""
        if end is None:
            end = self.NUM_BLOCKS
        assert 0 <= start <= end <= self.NUM_BLOCKS
        for i in range(start, end):
            x = self.blocks[i](x)
        return x

    def forward(self, x: torch.Tensor):
        """Full backbone: (N, C, T, V, M) → ((N*M, C', T', V), N, M)."""
        x, N, M = self.reshape_input(x)
        x = self.forward_blocks(x)
        return x, N, M


class CTRGCNHead(nn.Module):
    """Pool over (M, T, V) → dropout → fc."""

    def __init__(self, in_channels: int, num_class: int, drop_out: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        nn.init.constant_(self.fc.bias, 0)
        self.drop = nn.Dropout(drop_out) if drop_out > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, N: int, M: int) -> torch.Tensor:
        # x: (N*M, C, T, V) → (N, C)
        c = x.size(1)
        x = x.view(N, M, c, -1).mean(3).mean(1)
        return self.fc(self.drop(x))


class CTRGCN(nn.Module):
    """Single-view CTR-GCN (Backbone + Head). Equivalent to the original Model."""

    def __init__(self, num_class: int, num_point: int = 17, num_person: int = 2,
                 graph: str = 'graph.coco17.Graph',
                 graph_args: Optional[dict] = None,
                 in_channels: int = 3, base_channel: int = 64,
                 drop_out: float = 0.0, adaptive: bool = True):
        super().__init__()
        self.backbone = CTRGCNBackbone(
            num_point=num_point, num_person=num_person, graph=graph,
            graph_args=graph_args, in_channels=in_channels,
            base_channel=base_channel, adaptive=adaptive,
        )
        self.head = CTRGCNHead(self.backbone.out_channels, num_class, drop_out=drop_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, N, M = self.backbone(x)
        return self.head(feat, N, M)


# ------------------------------------------------------------------ smoke test


if __name__ == '__main__':
    backbone = CTRGCNBackbone(num_point=17, in_channels=3,
                              graph='graph.coco17.Graph')
    print(f'#blocks      : {len(backbone.blocks)}')
    print(f'out_channels : {backbone.out_channels}')

    # one mock NTU-style mini-batch: B=2, C=3, T=64, V=17, M=2
    x = torch.randn(2, 3, 64, 17, 2)
    feat, N, M = backbone(x)
    print(f'full forward : input {tuple(x.shape)}  →  feat {tuple(feat.shape)}  '
          f'(N={N}, M={M})')

    # split forward at block 7 (= insertion point for CVA)
    h, N, M = backbone.reshape_input(x)
    print(f'reshaped     : {tuple(h.shape)}')
    h7 = backbone.forward_blocks(h, end=7)
    print(f'after blk 0-6: {tuple(h7.shape)}')
    h_end = backbone.forward_blocks(h7, start=7)
    print(f'after blk 7-9: {tuple(h_end.shape)}')

    head = CTRGCNHead(backbone.out_channels, num_class=60)
    logits = head(h_end, N, M)
    print(f'logits       : {tuple(logits.shape)}    (expected (2, 60))')

    # full single-view model
    model = CTRGCN(num_class=60, num_point=17, in_channels=3,
                   graph='graph.coco17.Graph')
    out = model(x)
    print(f'CTRGCN()     : {tuple(out.shape)}')
    print(f'#params      : {sum(p.numel() for p in model.parameters()):,}')
