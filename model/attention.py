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


class PairwiseCrossViewAttention(nn.Module):
    """Cross-view attention where each view conditions ONLY on the other views.

    Option A in `docs/sa_crossattn_design.md`.  Selected via
    ``sa_mode='cross_pair'`` in the model / block constructors.

    Input  : (B, V_views, M, C, T, V)
    Output : same shape — drop-in replacement for `CrossViewSpatialAttention`.

    Mechanism (per spatial-temporal cell):
        For each view i ∈ {0, …, V_views-1}:
            Q_i = W_Q · F_i                       # (1, C)
            K   = W_K · concat(F_j for j ≠ i)     # (V_views-1, C)
            V   = W_V · concat(F_j for j ≠ i)
            F_i'= F_i + MHA(Q_i, K, V)            # pre-norm + residual

    Unlike `CrossViewSpatialAttention` (self-attention over all V_views tokens
    with Q = K = V = input), there is NO self-attention path: view i never
    attends to itself, so the cross-view information flow is explicit.

    Shared Q / K / V projections across views (same `nn.MultiheadAttention`
    instance applied to every view's (q, kv) pair in a loop — view-agnostic).

    Default `num_heads=2`: KV sequence length is V_views-1 = 2 at V_views=3, so
    head counts ≥ 4 fragment a 2-token attention map and waste capacity.  At
    V_views > 3, more heads may help.
    """

    def __init__(self, in_dim: int, num_heads: int = 2, dropout: float = 0.0):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Vv, M, C, T, V = x.shape

        # (B, V_views, M, C, T, V) → (B, M, T, V, V_views, C)
        tokens = x.permute(0, 2, 4, 5, 1, 3).contiguous()
        # fold (B, M, T, V) into batch:  (B*M*T*V, V_views, C)
        N = B * M * T * V
        tokens = tokens.reshape(N, Vv, C)

        # Pre-norm once; the same norm is shared for Q and KV (per spec).
        tokens_n = self.norm(tokens)

        # For each view i, attend to the other V_views-1 views.
        # The loop is over V_views (typically 3) so it's cheap; vectorising
        # would require constructing a leave-one-out gather which buys little.
        out_pieces = []
        for i in range(Vv):
            q  = tokens_n[:, i:i + 1]                          # (N, 1, C)
            other_idx = [j for j in range(Vv) if j != i]
            kv = tokens_n[:, other_idx]                        # (N, V_views-1, C)

            with sdpa_kernel(_SAFE_SDP_BACKENDS):
                attn_out, _ = self.mha(q, kv, kv, need_weights=False)

            # Residual: ORIGINAL (un-normed) view-i token + attention output.
            out_pieces.append(tokens[:, i:i + 1] + attn_out)   # (N, 1, C)

        out = torch.cat(out_pieces, dim=1)                     # (N, V_views, C)
        # unfold and permute back: (N, V_views, C) → (B, V_views, M, C, T, V)
        out = out.reshape(B, M, T, V, Vv, C)
        return out.permute(0, 4, 1, 5, 2, 3).contiguous()


class BottleneckCrossViewAttention(nn.Module):
    """Cross-view attention via a single shared latent token.

    Option B in `docs/sa_crossattn_design.md`.  A 1-token latent mediates all
    cross-view information exchange; each view's Q attends to this single
    K=V=latent.  More parameter-efficient than `PairwiseCrossViewAttention`
    (KV seq=1 instead of V_views-1) and scales O(V) instead of O(V²) with the
    number of views.

    The latent is either:
      • the **mean** of the per-cell view features         (`use_learnable_bottleneck=False`)
      • a **learnable** `nn.Parameter` of shape (1, 1, C)  (`use_learnable_bottleneck=True`)

    Both modes are ablated in the design plan.

    Input  : (B, V_views, M, C, T, V)
    Output : same shape — drop-in replacement for `CrossViewSpatialAttention`.

    Default `num_heads=2`: KV seq=1 so head-fragmentation is the only concern.
    Multi-head still helps by letting different heads project the latent
    differently; ≥ 2 is reasonable, ≥ 4 starts wasting capacity.
    """

    def __init__(self, in_dim: int, num_heads: int = 2, dropout: float = 0.0,
                 use_learnable_bottleneck: bool = False):
        super().__init__()
        if in_dim % num_heads != 0:
            raise ValueError(
                f'in_dim ({in_dim}) must be divisible by num_heads ({num_heads})'
            )
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.use_learnable_bottleneck = use_learnable_bottleneck

        self.norm = nn.LayerNorm(in_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        if use_learnable_bottleneck:
            # Single global latent token, broadcast across batch and spatial.
            self.bottleneck = nn.Parameter(torch.zeros(1, 1, in_dim))
            nn.init.normal_(self.bottleneck, mean=0.0, std=0.02)
        else:
            self.bottleneck = None    # latent = mean of views per cell, computed live

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Vv, M, C, T, V = x.shape

        # (B, V_views, M, C, T, V) → (N, V_views, C)  with N = B·M·T·V
        tokens = x.permute(0, 2, 4, 5, 1, 3).contiguous()
        N = B * M * T * V
        tokens = tokens.reshape(N, Vv, C)
        tokens_n = self.norm(tokens)

        # Latent: (N, 1, C)
        if self.use_learnable_bottleneck:
            latent = self.bottleneck.expand(N, 1, C)
        else:
            latent = tokens_n.mean(dim=1, keepdim=True)

        # Each view's Q attends to the same 1-token latent.  Loop over views.
        out_pieces = []
        for i in range(Vv):
            q = tokens_n[:, i:i + 1]                                # (N, 1, C)
            with sdpa_kernel(_SAFE_SDP_BACKENDS):
                attn_out, _ = self.mha(q, latent, latent, need_weights=False)
            out_pieces.append(tokens[:, i:i + 1] + attn_out)        # residual

        out = torch.cat(out_pieces, dim=1)                          # (N, V_views, C)
        out = out.reshape(B, M, T, V, Vv, C)
        return out.permute(0, 4, 1, 5, 2, 3).contiguous()


class ViewFusionWeightedSum(nn.Module):
    """Lightweight alternative to `CrossViewSpatialAttention` for the SA stage.

    Cross-view fusion via a single learnable softmax-weighted sum, with a
    residual connection so each view keeps its own information:

        fused[b, m, c, t, v]   =  Σ_v'  softmax(w)[v']  ·  X[b, v', m, c, t, v]
        out  [b, v, m, c, t, v] = X[b, v, m, c, t, v] + fused[b, m, c, t, v]

    The view dimension is preserved (so downstream TA + further blocks still
    see V_views > 1).  Useful for the SA-heads ablation: 4 heads over 3
    tokens is mostly redundant, and a single 3-parameter learned average
    may give a stronger early-training signal.

    Cost: a single (num_views,) parameter per block — negligible.
    """

    def __init__(self, num_views: int = 3):
        super().__init__()
        self.num_views = num_views
        self.view_weights = nn.Parameter(torch.ones(num_views) / num_views)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V_views, M, C, T, V)
        w = self.view_weights.softmax(0).view(1, -1, 1, 1, 1, 1)
        fused = (x * w).sum(dim=1, keepdim=True)                  # (B, 1, M, C, T, V)
        return x + fused.expand_as(x)


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
