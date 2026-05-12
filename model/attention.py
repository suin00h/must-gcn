"""Cross-View Spatial Attention (SA) and Temporal Self-Attention (TA).

Both modules consume and emit the canonical MUST-GCN tensor shape

    (B, V_views, M, C, T, V)

They apply *pre-norm + multi-head attention + residual* on a designated axis
without changing any other dim.

SA — `CrossViewSpatialAttention`
    sequence axis: `V_views` (= 3)
    one independent attention map per `(B, M, T, V)` cell
    semantics: each view's feature at a fixed joint and frame attends to the
    other two views' features at the same joint and frame.  This is the
    sole cross-view fusion stage in the block.

TA — `TemporalSelfAttention`
    sequence axis: `T_l` (= 64 / 32 / 16 depending on block depth)
    one independent attention map per `(B, V_views, M, V)` cell
    semantics: temporal self-attention *within* each view, replacing the
    multi-scale temporal convolution that CTR-GCN used for time modelling.

Both classes wrap `torch.nn.MultiheadAttention` in self-attention mode
(Q = K = V = input).  With self-attention, the QKV projections are
automatically shared across the sequence axis — so for SA, the same Q/K/V
projection applies to every view's token, satisfying the
"shared QKV across views" requirement (O6).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


# Math / efficient-attention backends always handle our shapes (especially
# SA's seq_len=3 in bf16, which Flash-Attention rejects with
# "CUDA error: invalid argument" on PyTorch 2.11 + Blackwell).
_SAFE_SDP_BACKENDS = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]


class _SelfAttnBlock(nn.Module):
    """Pre-norm + nn.MultiheadAttention + residual on (B*, L, C) tokens."""

    def __init__(self, in_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if in_dim % num_heads != 0:
            raise ValueError(
                f'in_dim ({in_dim}) must be divisible by num_heads ({num_heads})'
            )
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(in_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B*, L, C)
        x_n = self.norm(tokens)
        # Restrict SDPA to math / efficient kernels — see comment on _SAFE_SDP_BACKENDS.
        with sdpa_kernel(_SAFE_SDP_BACKENDS):
            attn_out, _ = self.mha(x_n, x_n, x_n, need_weights=False)
        return tokens + attn_out          # residual


class CrossViewSpatialAttention(nn.Module):
    """MHA over views at every spatio-temporal cell.

    Input  : (B, V_views, M, C, T, V)
    Output : same shape

    Sequence length per call = V_views (3).
    `B * M * T * V` independent attention maps are computed in parallel.
    """

    def __init__(self, in_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = _SelfAttnBlock(in_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Vv, M, C, T, V = x.shape

        # (B, V_views, M, C, T, V) → (B, M, T, V, V_views, C)
        tokens = x.permute(0, 2, 4, 5, 1, 3).contiguous()
        # fold (B, M, T, V) into batch:  (B*M*T*V, V_views, C)
        tokens = tokens.reshape(B * M * T * V, Vv, C)

        tokens = self.attn(tokens)

        # unfold and permute back
        tokens = tokens.reshape(B, M, T, V, Vv, C)
        return tokens.permute(0, 4, 1, 5, 2, 3).contiguous()


class TemporalSelfAttention(nn.Module):
    """MHA over time, independently per view.

    Input  : (B, V_views, M, C, T, V)
    Output : same shape

    Sequence length per call = T.
    `B * V_views * M * V` independent attention maps are computed in parallel.
    """

    def __init__(self, in_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = _SelfAttnBlock(in_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Vv, M, C, T, V = x.shape

        # (B, V_views, M, C, T, V) → (B, V_views, M, V, T, C)
        tokens = x.permute(0, 1, 2, 5, 4, 3).contiguous()
        # fold (B, V_views, M, V) into batch:  (B*V_views*M*V, T, C)
        tokens = tokens.reshape(B * Vv * M * V, T, C)

        tokens = self.attn(tokens)

        # unfold and permute back  (the permutation is its own inverse here)
        tokens = tokens.reshape(B, Vv, M, V, T, C)
        return tokens.permute(0, 1, 2, 5, 4, 3).contiguous()


# ------------------------------------------------------------------ smoke test


if __name__ == '__main__':
    torch.manual_seed(0)

    # Try all three (C_l, T_l) settings from the channel schedule.
    for C, T in [(64, 64), (128, 32), (256, 16)]:
        x = torch.randn(2, 3, 2, C, T, 17, requires_grad=True)
        sa = CrossViewSpatialAttention(C, num_heads=4)
        ta = TemporalSelfAttention(C, num_heads=4)

        y = sa(x)
        z = ta(y)

        n_sa = sum(p.numel() for p in sa.parameters())
        n_ta = sum(p.numel() for p in ta.parameters())
        z.sum().backward()
        grad_norm = x.grad.norm().item()

        print(
            f'C={C:>3}  T={T:>2}  '
            f'in {tuple(x.shape)}  '
            f'→ SA {tuple(y.shape)}  '
            f'→ TA {tuple(z.shape)}  |  '
            f'#params SA={n_sa:,}  TA={n_ta:,}  |  '
            f'x.grad ‖·‖={grad_norm:.3f}'
        )
        assert y.shape == x.shape
        assert z.shape == x.shape
