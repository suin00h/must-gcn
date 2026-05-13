"""Hierarchical parameter count for MUST-GCN — per-block + inside-block.

Examples
--------
Default (NTU120, 3 views, mha SA, shared BN, 4 heads):
    python analyze_params.py

Single-view ablation:
    python analyze_params.py --num_views 1

SA = weighted_sum ablation:
    python analyze_params.py --sa_mode weighted_sum

Per-view BN ablation:
    python analyze_params.py --data_bn_mode per_view
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch.nn as nn

from model.must_gcn import MUSTGCN


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def fmt(n: int) -> str:
    if n == 0:
        return '—'
    if n >= 1_000_000:
        return f'{n / 1e6:>6.2f} M'
    if n >= 1_000:
        return f'{n / 1e3:>6.2f} k'
    return f'{n:>6d}  '


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num_class',    type=int, default=120)
    p.add_argument('--num_views',    type=int, default=3)
    p.add_argument('--num_heads',    type=int, default=4)
    p.add_argument('--in_channels',  type=int, default=3)
    p.add_argument('--base_channel', type=int, default=64)
    p.add_argument('--sa_mode',      default='mha', choices=['mha', 'weighted_sum', 'none'])
    p.add_argument('--data_bn_mode', default='shared', choices=['shared', 'per_view'])
    args = p.parse_args()

    model = MUSTGCN(
        num_class=args.num_class,
        num_views=args.num_views,
        in_channels=args.in_channels,
        base_channel=args.base_channel,
        num_heads=args.num_heads,
        sa_mode=args.sa_mode,
        data_bn_mode=args.data_bn_mode,
        graph='graph.coco17.Graph',
    )

    total = n_params(model)
    n_db = n_params(model.data_bn)
    n_blocks = n_params(model.blocks)
    n_fc = n_params(model.fc)

    print()
    print('================================================================')
    print(f'  MUST-GCN params · num_views={args.num_views}  num_heads={args.num_heads}'
          f'  sa_mode={args.sa_mode}  data_bn_mode={args.data_bn_mode}')
    print(f'                  num_class={args.num_class}  in_channels={args.in_channels}'
          f'  base_channel={args.base_channel}')
    print('================================================================')

    print(f'\n  top-level                  params      % of total')
    print(f'  -----------------------------------------------------')
    print(f'  data_bn                  {fmt(n_db)}    {100*n_db/total:>5.2f} %')
    print(f'  blocks[10]               {fmt(n_blocks)}    {100*n_blocks/total:>5.2f} %')
    print(f'  fc                       {fmt(n_fc)}    {100*n_fc/total:>5.2f} %')
    print(f'  -----------------------------------------------------')
    print(f'  TOTAL                    {fmt(total)}   100.00 %')

    # per-block
    print(f'\n  per-block breakdown')
    header = f'  {"#":<3}{"in→out":<10}{"T":<4}{"gcn":<11}{"strd_tcn":<11}{"sa":<11}{"ta":<11}{"resid":<11}{"block":<11}'
    print(f'\n{header}')
    print('  ' + '-' * (len(header) - 2))

    # Channel and time schedule (mirror of MUSTGCN._build_block_specs)
    T_schedule = [64, 64, 64, 64, 32, 32, 32, 16, 16, 16]

    sums = [0, 0, 0, 0, 0]   # gcn, tcn, sa, ta, res
    for i, blk in enumerate(model.blocks):
        n_gcn = n_params(blk.gcn)
        n_tcn = n_params(blk.tcn_stride) if blk.tcn_stride is not None else 0
        n_sa  = n_params(blk.sa)
        n_ta  = n_params(blk.ta)
        n_res = 0 if isinstance(blk.outer_residual, nn.Identity) else n_params(blk.outer_residual)
        n_blk = n_gcn + n_tcn + n_sa + n_ta + n_res
        sums = [a+b for a, b in zip(sums, (n_gcn, n_tcn, n_sa, n_ta, n_res))]
        c_arrow = f'{blk.in_c}→{blk.out_c}'
        print(f'  {i:<3}{c_arrow:<10}{T_schedule[i]:<4}'
              f'{fmt(n_gcn):<11}{fmt(n_tcn):<11}{fmt(n_sa):<11}{fmt(n_ta):<11}{fmt(n_res):<11}{fmt(n_blk):<11}')

    print('  ' + '-' * (len(header) - 2))
    sum_total = sum(sums)
    print(f'  {"Σ":<3}{"":<10}{"":<4}{fmt(sums[0]):<11}{fmt(sums[1]):<11}{fmt(sums[2]):<11}'
          f'{fmt(sums[3]):<11}{fmt(sums[4]):<11}{fmt(sum_total):<11}')

    # zoom into one representative block (block 7 = transition 128→256)
    print()
    print('  inside a single block (block 7: 128→256, stride=2)')
    print('  ------------------------------------------------------------------')
    blk7 = model.blocks[7]
    for name, sub in [
        ('gcn (unit_gcn)',                          blk7.gcn),
        ('  └ convs (3 × CTRGC + PA + alpha + bn)', blk7.gcn.convs),
        ('  └ down (1×1 conv + BN)',                blk7.gcn.down if not callable(blk7.gcn.down) or isinstance(blk7.gcn.down, nn.Module) else None),
        ('tcn_stride (unit_tcn, kernel=9, stride=2)', blk7.tcn_stride),
        ('sa  (CrossViewSpatialAttention)',         blk7.sa),
        ('  └ norm (LayerNorm)',                    getattr(blk7.sa, 'attn', None) and blk7.sa.attn.norm),
        ('  └ mha  (nn.MultiheadAttention)',        getattr(blk7.sa, 'attn', None) and blk7.sa.attn.mha),
        ('ta  (TemporalSelfAttention)',             blk7.ta),
        ('  └ norm (LayerNorm)',                    blk7.ta.attn.norm),
        ('  └ mha  (nn.MultiheadAttention)',        blk7.ta.attn.mha),
        ('outer_residual (unit_tcn, kernel=1, stride=2)', blk7.outer_residual if not isinstance(blk7.outer_residual, nn.Identity) else None),
    ]:
        if sub is None:
            print(f'  {name:<60}  —')
            continue
        if not isinstance(sub, nn.Module):
            continue
        nn_ = n_params(sub)
        print(f'  {name:<60}  {fmt(nn_)}')


if __name__ == '__main__':
    main()
