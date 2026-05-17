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


class JointCrossViewAttention(nn.Module):
    """Cross-view attention where TOKENS ARE JOINTS, not view-channel vectors.

    Diagnosis (post-phase-2): the existing `mha` SA operates **after** GCN's
    channel-wise topology refinement, so each view's C-dim feature is already
    an *abstraction* of the joint — direct cross-view information at the
    geometric level (which joint is occluded in view A but visible in view B)
    is already lost.

    This module restores joint identity to the cross-view stage by treating
    each view's `V_joints` joint embeddings as the attention sequence:

        For each view i:
            Q_i  = view i's V_joints joint tokens          (V_joints, C)
            K, V = concat(other views' joints)            ((V_views-1)·V_joints, C)
            F_i' = F_i + Attn(Q_i, K, V)                  per-view residual

    So "view 0 joint 9 (left-wrist)" can attend to "view 1 joint 10 (right-wrist)"
    etc.  The cross-view information flow is explicit AND retains joint-level
    spatial structure that GCN-abstracted features have washed out.

    Input  : (B, V_views, M, C, T, V_joints)
    Output : same shape.
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
        B, Vv, M, C, T, Vj = x.shape       # Vj = V_joints

        # (B, V_views, M, C, T, V_joints) → (B, M, T, V_views, V_joints, C)
        tokens = x.permute(0, 2, 4, 1, 5, 3).contiguous()
        N = B * M * T
        tokens = tokens.reshape(N, Vv, Vj, C)             # (N, V_views, V_joints, C)

        # Pre-norm once (shared for Q and KV).
        tokens_n = self.norm(tokens.reshape(N, Vv * Vj, C)).reshape(N, Vv, Vj, C)

        out_pieces = []
        for i in range(Vv):
            q = tokens_n[:, i, :, :]                       # (N, V_joints, C)
            other_idx = [j for j in range(Vv) if j != i]
            kv = tokens_n[:, other_idx, :, :].reshape(
                N, (Vv - 1) * Vj, C
            )                                              # (N, (V_views-1)·V_joints, C)

            with sdpa_kernel(_SAFE_SDP_BACKENDS):
                attn_out, _ = self.mha(q, kv, kv, need_weights=False)

            # Residual onto ORIGINAL un-normed view-i joints
            out_pieces.append(
                (tokens[:, i, :, :] + attn_out).unsqueeze(1)   # (N, 1, V_joints, C)
            )

        out = torch.cat(out_pieces, dim=1)                 # (N, V_views, V_joints, C)
        out = out.reshape(B, M, T, Vv, Vj, C)
        return out.permute(0, 3, 1, 5, 2, 4).contiguous()  # back to canonical


class InputLevelCVA(nn.Module):
    """Cross-view attention applied at the RAW INPUT level, before any GCN.

    Diagnosis (post-phase-2): cross-view complementarity is most explicit at
    the raw 2D-coordinate level — different cameras give literally different
    `(x, y, score)` per joint.  Once GCN abstracts joints into C-dim features,
    that direct geometric complementarity is mixed away.

    This module sits between `data_bn` and the first block.  It projects the
    raw 3-channel input to a higher working dimension, applies cross-view
    attention there, and projects back with a residual.

    Input  : (B, V_views, M, C_in, T, V)         e.g. C_in = 3 (x, y, score)
    Output : same shape

    The output projection is zero-initialised so the module starts as
    Identity — gradients gradually turn it on, never disturbs training start.
    """

    def __init__(self, in_channels: int = 3, work_dim: int = 64,
                 num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.work_dim = work_dim

        self.proj_in  = nn.Linear(in_channels, work_dim)
        self.attn     = PairwiseCrossViewAttention(work_dim,
                                                   num_heads=num_heads,
                                                   dropout=dropout)
        self.proj_out = nn.Linear(work_dim, in_channels)
        # Zero-init out projection → module is initially Identity (after residual).
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V_views, M, C_in, T, V) — same shape in/out
        B, Vv, M, C_in, T, V = x.shape

        # Move C_in axis to last for the Linear projection
        x_p = x.permute(0, 1, 2, 4, 5, 3).contiguous()     # (B, Vv, M, T, V, C_in)
        x_w = self.proj_in(x_p)                            # (B, Vv, M, T, V, work_dim)
        # Bring it back to canonical (B, Vv, M, C, T, V)
        x_w = x_w.permute(0, 1, 2, 5, 3, 4).contiguous()

        # Cross-view fusion in the working dim
        x_w = self.attn(x_w)

        # Project back to C_in
        x_w = x_w.permute(0, 1, 2, 4, 5, 3).contiguous()   # (B, Vv, M, T, V, work_dim)
        x_back = self.proj_out(x_w)                        # (B, Vv, M, T, V, C_in)
        x_back = x_back.permute(0, 1, 2, 5, 3, 4).contiguous()

        return x + x_back                                  # residual; zero-init → 0 at t=0


class PostBackboneCrossAttn(nn.Module):
    """Single cross-view attention AFTER the backbone, on pooled features.

    Phase-5 Option A.  The backbone runs `sa_mode=none` (each view fully
    independent through all 10 blocks).  Cross-view interaction happens
    exactly once, at the pooled-feature level — after each view has
    developed a stable representation, before the FC head.

    Input  : (B, V_views, M, C)   — features already pooled over (T, V)
    Output : same shape

    Each view's pooled feature (Q) attends to the other views (K, V),
    pre-norm + per-view residual, shared MHA weights across views.
    """

    def __init__(self, dim: int, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                         batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Vv, M, C = x.shape
        tokens = x.permute(0, 2, 1, 3).reshape(B * M, Vv, C)   # (B*M, V_views, C)
        tokens_n = self.norm(tokens)

        out_pieces = []
        for i in range(Vv):
            q = tokens_n[:, i:i + 1]                            # (B*M, 1, C)
            other = [j for j in range(Vv) if j != i]
            kv = tokens_n[:, other]                             # (B*M, V_views-1, C)
            with sdpa_kernel(_SAFE_SDP_BACKENDS):
                attn_out, _ = self.mha(q, kv, kv, need_weights=False)
            out_pieces.append(tokens[:, i:i + 1] + attn_out)    # residual

        out = torch.cat(out_pieces, dim=1)                      # (B*M, V_views, C)
        return out.reshape(B, M, Vv, C).permute(0, 2, 1, 3).contiguous()


class CLSCrossViewExchange(nn.Module):
    """Per-view CLS-token cross-view exchange — Phase-5 Option C.

    NOTE — deviation from the literal spec: the spec's `cls_tokens` are a bare
    `nn.Parameter`, which would make the CLS-to-CLS attention input-independent
    (just a learned bias).  To make the CLS token an actual *compressed view
    representation*, this implementation uses the learnable parameter as a
    *query* that first attention-pools its own view's feature map.  The flow:

        1. aggregate : learnable CLS query[i]  attends to view-i's (T·V) tokens
                       → per-sample summary cls_i
        2. exchange  : cls_i cross-attends to the other views' cls tokens
        3. broadcast : the exchanged cls_i' is added back over view-i's map

    Input  : (B, V_views, M, C, T, V)
    Output : same shape
    """

    def __init__(self, dim: int, num_views: int = 3,
                 num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_views = num_views
        self.cls_query = nn.Parameter(torch.zeros(num_views, 1, dim))
        nn.init.normal_(self.cls_query, std=0.02)
        self.norm_feat = nn.LayerNorm(dim)
        self.norm_cls  = nn.LayerNorm(dim)
        self.aggregate = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                               batch_first=True)
        self.exchange  = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                               batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Vv, M, C, T, V = x.shape
        N = B * M

        # 1) aggregate — each view's learnable CLS query attention-pools its
        #    own (T·V) feature tokens into a per-sample summary.
        cls = []
        for i in range(Vv):
            fi = x[:, i].permute(0, 1, 3, 4, 2).reshape(N, T * V, C)   # (N, T·V, C)
            fi_n = self.norm_feat(fi)
            q = self.cls_query[i].expand(N, 1, C)
            with sdpa_kernel(_SAFE_SDP_BACKENDS):
                ci, _ = self.aggregate(q, fi_n, fi_n, need_weights=False)
            cls.append(ci)                                             # (N, 1, C)
        cls = torch.cat(cls, dim=1)                                    # (N, V_views, C)

        # 2) exchange — CLS tokens cross-attend across views.
        cls_n = self.norm_cls(cls)
        exchanged = []
        for i in range(Vv):
            q = cls_n[:, i:i + 1]
            other = [j for j in range(Vv) if j != i]
            kv = cls_n[:, other]
            with sdpa_kernel(_SAFE_SDP_BACKENDS):
                ei, _ = self.exchange(q, kv, kv, need_weights=False)
            exchanged.append(cls[:, i:i + 1] + ei)                     # residual
        exchanged = torch.cat(exchanged, dim=1)                        # (N, V_views, C)

        # 3) broadcast — add the exchanged CLS back over view-i's (T, V) map.
        bcast = exchanged.reshape(B, M, Vv, C).permute(0, 2, 1, 3)     # (B, Vv, M, C)
        bcast = bcast.reshape(B, Vv, M, C, 1, 1)
        return x + bcast


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
