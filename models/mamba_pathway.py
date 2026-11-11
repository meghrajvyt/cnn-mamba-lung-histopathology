"""
Vision Mamba Pathway — Global context extraction via bidirectional SSMs.

Implements the Mamba pathway of the dual-pathway architecture:
  Input Image → Patch Embedding → Positional Encoding → Vision Mamba Blocks → F_global

Each Vision Mamba block applies:
  1. LayerNorm
  2. Bidirectional SS2D selective scan (4 directions)
  3. Residual connection
  4. LayerNorm + MLP
  5. Residual connection

References:
  - Vision Mamba (Vim): arXiv:2401.09417
  - VMamba (SS2D): arXiv:2401.10166
  - MedMamba: arXiv:2403.03849
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.ssm_block import SS2D


class PatchEmbedding(nn.Module):
    """
    Convert image into a sequence of patch embeddings.

    Splits the image into non-overlapping patches of size (patch_size × patch_size)
    and linearly projects each patch to the embedding dimension.

    Args:
        image_size: Input image size (assumes square images).
        patch_size: Size of each patch.
        in_channels: Number of input channels (3 for RGB).
        embed_dim: Embedding dimension.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.grid_size = image_size // patch_size

        # Linear projection via convolution
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x: Input image (B, C, H, W).

        Returns:
            Tuple of (patch_tokens, H_patches, W_patches):
                - patch_tokens: (B, num_patches, embed_dim)
                - H_patches: Number of patches along height
                - W_patches: Number of patches along width
        """
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, (
            f"Input size ({H}×{W}) doesn't match expected ({self.image_size}×{self.image_size})"
        )

        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        H_p, W_p = x.shape[2], x.shape[3]
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.norm(x)

        return x, H_p, W_p


class VisionMambaBlock(nn.Module):
    """
    Single Vision Mamba block with bidirectional SS2D and MLP.

    Architecture:
        x → LN → SS2D (bidirectional) → + → LN → MLP → +
        └──────────────────────────────────┘    └────────┘
                   residual                     residual

    Args:
        dim: Feature dimension.
        d_state: SSM state dimension.
        d_conv: Local convolution width.
        expand: SSM expansion factor.
        dt_rank: Rank of Δ projection.
        num_scan_directions: Number of scan directions for SS2D.
        mlp_ratio: MLP hidden dimension ratio.
        drop_path: Drop path rate for stochastic depth.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        num_scan_directions: int = 4,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        scan_chunk_size: int = 32,
    ):
        super().__init__()

        # SS2D branch
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            num_directions=num_scan_directions,
            scan_chunk_size=scan_chunk_size,
        )

        # MLP branch
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(0.0),
        )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: Token sequence (B, L, D).
            H: Height of the 2D grid.
            W: Width of the 2D grid.

        Returns:
            Output tensor (B, L, D).
        """
        # SS2D with residual
        x = x + self.drop_path(self.ss2d(self.norm1(x), H, W))
        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MambaPathway(nn.Module):
    """
    Complete Mamba pathway for global context extraction.

    Pipeline:
        Image → PatchEmbed → [CLS] + Pos Encoding → VisionMambaBlock ×N → GAP → F_global

    Args:
        image_size: Input image size.
        patch_size: Patch size for tokenization.
        in_channels: Number of input channels.
        embed_dim: Embedding/hidden dimension.
        depth: Number of VisionMambaBlock layers.
        d_state: SSM state dimension.
        d_conv: Local convolution width.
        expand: SSM expansion factor.
        dt_rank: Rank of Δ projection.
        num_scan_directions: Number of scan directions.
        mlp_ratio: MLP hidden ratio.
        drop_path_rate: Maximum drop path rate (linearly increased per block).
        output_dim: Output feature vector dimension.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        num_scan_directions: int = 4,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        output_dim: int = 384,
        scan_chunk_size: int = 32,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        # Learnable 2D positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=0.0)

        # Stochastic depth: linearly increasing drop rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Stack of Vision Mamba blocks
        self.blocks = nn.ModuleList([
            VisionMambaBlock(
                dim=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                num_scan_directions=num_scan_directions,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                scan_chunk_size=scan_chunk_size,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output projection (if output_dim != embed_dim)
        self.proj = (
            nn.Linear(embed_dim, output_dim)
            if output_dim != embed_dim
            else nn.Identity()
        )

        # Store intermediate features for XAI. Only populated in eval mode:
        # during training these tensors would pin activations (and, for
        # "final", the whole autograd graph) alive between steps for no reason.
        self._features = {}
        self.store_features = True

        # Initialize positional encoding
        self._init_pos_embed()

    def _init_pos_embed(self):
        """Initialize positional embedding with sinusoidal pattern."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self, x: torch.Tensor, return_tokens: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image (B, C, H, W).
            return_tokens: Also return the per-patch tokens before pooling,
                which the fusion module needs for genuine cross-attention.

        Returns:
            ``f_global`` (B, output_dim), or ``(f_global, tokens)`` with tokens
            shaped (B, num_patches, output_dim) when ``return_tokens``.
        """
        # Patch embedding
        x, H, W = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Process through Vision Mamba blocks
        cache = self.store_features and not self.training
        for i, block in enumerate(self.blocks):
            if self.grad_checkpoint and self.training:
                # Recompute block activations during backward instead of
                # storing them. Roughly 30% slower per step, but cuts the
                # dominant memory term (the scan's (B, L, d_inner, d_state)
                # intermediates), which is what caps batch size.
                x = torch.utils.checkpoint.checkpoint(
                    block, x, H, W, use_reentrant=False
                )
            else:
                x = block(x, H, W)
            if cache:
                self._features[f"block_{i}"] = x.detach()

        # Store final features before pooling (for XAI — can be reshaped to 2D)
        if cache:
            self._features["final"] = x

        # Final normalization
        x = self.norm(x)

        if return_tokens:
            # Project per token, then pool. `proj` is Linear or Identity, both
            # of which commute with the mean, so f_global matches the
            # pooling-first path below (exactly when Identity, to float rounding
            # ~1e-7 when Linear).
            tokens = self.proj(x)          # (B, num_patches, output_dim)
            return tokens.mean(dim=1), tokens

        # Global average pooling over spatial tokens
        x = x.mean(dim=1)  # (B, embed_dim)

        # Output projection
        x = self.proj(x)  # (B, output_dim)

        return x

    def get_spatial_features(self) -> torch.Tensor:
        """
        Get the last block's features reshaped to 2D spatial map.
        Used by the XAI module for Mamba pathway attribution.

        Returns:
            Spatial features (B, D, H, W) where H, W are patch grid dimensions.
        """
        features = self._features.get("final")
        if features is None:
            return None
        grid_size = self.patch_embed.grid_size
        return rearrange(
            features, "b (h w) d -> b d h w", h=grid_size, w=grid_size
        )


class DropPath(nn.Module):
    """
    Drop paths (stochastic depth) per sample.

    During training, randomly drops entire residual branches with probability
    `drop_prob`, scaling surviving paths by 1/(1-p) to maintain expected values.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x.div(keep_prob) * random_tensor
