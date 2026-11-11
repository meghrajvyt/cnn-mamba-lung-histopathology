"""
Fusion Module — Multi-Scale Interaction and Cross-Attention Fusion.

Fuses local CNN features with global Mamba features through bidirectional
cross-attention over *spatial tokens*, followed by gated fusion:

    F_fused = σ(W_g[F_local ∥ F_global]) ⊙ F_local
            + (1 - σ(W_g[F_local ∥ F_global])) ⊙ F_global

WHY TOKENS AND NOT POOLED VECTORS
---------------------------------
An earlier version of this module attended between the two pathways *after*
each had been global-average-pooled to a single (B, D) vector. Attention over a
sequence of length 1 is vacuous: the score matrix is (B, heads, 1, 1) and
softmax over a single element returns exactly 1.0 for any input. The attention
therefore contributed no learnable routing at all — `attended_local` reduced to
a linear map of `f_global` — and the "cross-attention" label described nothing
the module actually did. (Information still crossed between pathways via those
value projections and the gate, so it was a gated linear fusion, not a no-op.)

Attention only means something once there are multiple positions to distribute
mass over. So fusion now runs before pooling:

    CNN    (B, 49,  D)   7x7 multi-scale feature map, flattened
    Mamba  (B, 196, D)   14x14 patch tokens

Each CNN position attends over all 196 Mamba positions (softmax over 196), and
each Mamba position attends over all 49 CNN positions (softmax over 49). That
lets a cell-level location query the tissue-level context around it, which is
the interaction the dual-pathway design is premised on. Pooling happens after.

Parameter count is unchanged: the same nine projections, applied to sequences
rather than singleton vectors.

Reference: MambaVesselNet++ (Xu et al., 2025) — arXiv:2507.19931
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention between CNN and Mamba spatial tokens.

        - CNN tokens attend to Mamba tokens (local → global context)
        - Mamba tokens attend to CNN tokens (global → local detail)

    The two sequences may have different lengths; only the feature dimension
    must match.

    Args:
        dim: Feature dimension (same for both pathways).
        num_heads: Number of attention heads. Must divide ``dim``.
        qkv_bias: Whether to use bias in QKV projections.
        attn_drop: Dropout applied to the attention weights.
        proj_drop: Dropout applied to the output projections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"fusion dim {dim} is not divisible by num_heads {num_heads}."
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop_p = attn_drop

        # CNN → queries; Mamba → keys, values (and vice versa)
        self.q_local = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_global = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_global = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_global = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_local = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_local = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        # Output projections
        self.proj_local = nn.Linear(dim, dim)
        self.proj_global = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _attend(
        self,
        q_src: torch.Tensor,
        kv_src: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        """One direction of cross-attention.

        Args:
            q_src: Sequence supplying queries (B, N_q, D).
            kv_src: Sequence supplying keys and values (B, N_kv, D).
            q_proj, k_proj, v_proj: Projections for this direction.
            out_proj: Output projection for this direction.

        Returns:
            Attended sequence (B, N_q, D).
        """
        B, N_q, _ = q_src.shape
        N_kv = kv_src.shape[1]

        q = self._heads(q_proj(q_src), B, N_q)      # (B, heads, N_q, head_dim)
        k = self._heads(k_proj(kv_src), B, N_kv)    # (B, heads, N_kv, head_dim)
        v = self._heads(v_proj(kv_src), B, N_kv)

        # Fused SDPA kernel: fewer materialised intermediates than an explicit
        # softmax(q @ k.T) @ v, which matters because this now runs over
        # 49x196 rather than 1x1.
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N_q, self.dim)
        return self.proj_drop(out_proj(out))

    def forward(
        self, tokens_local: torch.Tensor, tokens_global: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-attention over spatial tokens.

        Args:
            tokens_local: CNN tokens (B, N_local, D).
            tokens_global: Mamba tokens (B, N_global, D).

        Returns:
            Tuple of (attended_local, attended_global), shaped
            (B, N_local, D) and (B, N_global, D).
        """
        if tokens_local.dim() != 3 or tokens_global.dim() != 3:
            raise ValueError(
                "CrossAttentionFusion expects token sequences (B, N, D); got "
                f"{tuple(tokens_local.shape)} and {tuple(tokens_global.shape)}. "
                "Attention over length-1 sequences is degenerate — that is the "
                "bug this signature change fixes."
            )

        attended_local = self._attend(
            tokens_local, tokens_global,
            self.q_local, self.k_global, self.v_global, self.proj_local,
        )
        attended_global = self._attend(
            tokens_global, tokens_local,
            self.q_global, self.k_local, self.v_local, self.proj_global,
        )
        return attended_local, attended_global

    def _heads(self, x: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """Reshape (B, N, D) → (B, heads, N, head_dim)."""
        return x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    @torch.no_grad()
    def attention_maps(
        self, tokens_local: torch.Tensor, tokens_global: torch.Tensor
    ) -> dict:
        """Explicit attention weights, for inspection and XAI.

        Recomputed on demand rather than cached during forward, so the training
        path never retains these (B, heads, N_q, N_kv) tensors.

        Returns:
            ``{"local_to_global": (B, heads, N_local, N_global),
               "global_to_local": (B, heads, N_global, N_local)}``
        """
        B, N_l, _ = tokens_local.shape
        N_g = tokens_global.shape[1]

        q_l = self._heads(self.q_local(tokens_local), B, N_l)
        k_g = self._heads(self.k_global(tokens_global), B, N_g)
        q_g = self._heads(self.q_global(tokens_global), B, N_g)
        k_l = self._heads(self.k_local(tokens_local), B, N_l)

        return {
            "local_to_global": ((q_l @ k_g.transpose(-2, -1)) * self.scale).softmax(-1),
            "global_to_local": ((q_g @ k_l.transpose(-2, -1)) * self.scale).softmax(-1),
        }


class GatedFusion(nn.Module):
    """
    Gated fusion of local and global features.

    Computes a learnable gate to dynamically balance the contribution
    of CNN (local) and Mamba (global) features:

        gate = σ(W_g · [F_local ∥ F_global] + b_g)
        F_fused = gate ⊙ F_local + (1 - gate) ⊙ F_global

    Args:
        dim: Feature dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(
        self, f_local: torch.Tensor, f_global: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            f_local: CNN features (B, D).
            f_global: Mamba features (B, D).

        Returns:
            Fused features (B, D).
        """
        gate_input = torch.cat([f_local, f_global], dim=-1)  # (B, 2D)
        g = self.gate(gate_input)  # (B, D), values in [0, 1]
        return g * f_local + (1 - g) * f_global


class FusionModule(nn.Module):
    """
    Complete fusion module combining cross-attention and gated fusion.

    Pipeline:
        1. Cross-attention between spatial token sequences (both directions)
        2. Residual: add attended tokens back to the originals
        3. Mean-pool each pathway over its own token axis
        4. Gated fusion: dynamically blend local and global
        5. Output projection to classification dimension

    Pooling deliberately happens at step 3, after attention. Pooling first is
    what made the previous version's attention degenerate; see the module
    docstring.

    Args:
        dim: Feature dimension (shared by both pathways).
        num_heads: Number of cross-attention heads.
        dropout: Dropout rate.
        output_dim: Final output dimension.
    """

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: int = 384,
        mode: str = "token",
    ):
        super().__init__()
        if mode not in ("token", "pooled"):
            raise ValueError(f"mode must be 'token' or 'pooled', got {mode!r}")
        self.mode = mode

        # Normalization layers
        self.norm_local = nn.LayerNorm(dim)
        self.norm_global = nn.LayerNorm(dim)

        # Cross-attention
        self.cross_attention = CrossAttentionFusion(
            dim=dim,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        # Gated fusion
        self.gated_fusion = GatedFusion(dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, tokens_local: torch.Tensor, tokens_global: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse CNN spatial tokens with Mamba spatial tokens.

        Args:
            tokens_local: CNN pathway tokens (B, N_local, D), e.g. 7x7 -> 49.
            tokens_global: Mamba pathway tokens (B, N_global, D), e.g. 14x14 -> 196.

        Returns:
            Fused feature vector (B, output_dim).
        """
        if self.mode == "pooled":
            # Ablation path: pool to one vector per pathway *before* attending,
            # reproducing the original degenerate behaviour exactly. Attention
            # over length 1 has a softmax of 1.0, so this measures the hybrid
            # model without any functioning cross-attention.
            tokens_local = tokens_local.mean(dim=1, keepdim=True)    # (B, 1, D)
            tokens_global = tokens_global.mean(dim=1, keepdim=True)  # (B, 1, D)

        # LayerNorm acts on the feature axis, so it applies per token unchanged.
        tl_n = self.norm_local(tokens_local)
        tg_n = self.norm_global(tokens_global)

        # Cross-attention with residual connections, still at token resolution.
        att_local, att_global = self.cross_attention(tl_n, tg_n)
        tokens_local = tokens_local + att_local     # (B, N_local, D)
        tokens_global = tokens_global + att_global  # (B, N_global, D)

        # Pool each pathway over its own token axis. Lengths differ (49 vs 196),
        # so this is also what makes the two comparable for the gate.
        f_local = tokens_local.mean(dim=1)   # (B, D)
        f_global = tokens_global.mean(dim=1)  # (B, D)

        # Gated fusion, then output projection
        f_fused = self.gated_fusion(f_local, f_global)
        return self.output_proj(f_fused)
