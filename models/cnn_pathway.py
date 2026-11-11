"""
CNN Pathway — Local feature extraction via EfficientNet-B3.

Implements the CNN pathway of the dual-pathway architecture:
  Input Image → EfficientNet-B3 → Multi-Scale Feature Maps → F_local

Extracts features from multiple intermediate layers to capture multi-scale
cellular morphology (nuclear texture, chromatin patterns, glandular structure).

Reference: EfficientNet (Tan & Le, 2019) — arXiv:1905.11946
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CNNPathway(nn.Module):
    """
    EfficientNet-B3 backbone for local feature extraction.

    Uses a pretrained EfficientNet as a multi-scale feature extractor.
    Features are extracted from 3 intermediate stages, unified to a common
    spatial grid and channel count, and exposed both as spatial tokens
    (for cross-attention fusion) and as their pooled mean F_local.

    EfficientNet-B3 feature dimensions at each stage:
        Stage 0: (B, 24,  112, 112)   — low-level edges
        Stage 1: (B, 32,  56,  56)    — textures
        Stage 2: (B, 48,  28,  28)    — cellular patterns
        Stage 3: (B, 96,  14,  14)    — structural features
        Stage 4: (B, 136, 14,  14)    — complex patterns
        Stage 5: (B, 232, 7,   7)     — high-level features
        Stage 6: (B, 384, 7,   7)     — semantic features

    Args:
        backbone: Name of the timm backbone model.
        pretrained: Whether to use ImageNet pretrained weights.
        extract_layers: Indices of stages to extract features from.
        output_dim: Output feature dimension for F_local.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        pretrained: bool = True,
        extract_layers: List[int] = None,
        output_dim: int = 384,
        image_size: int = 224,
        pool_size: Optional[int] = None,
    ):
        super().__init__()

        if extract_layers is None:
            extract_layers = [2, 4, 6]  # Low, mid, high-level features

        # Create feature extractor using timm
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=extract_layers,
        )

        # Probe at the ACTUAL input size. This was previously hardcoded to 224,
        # which silently reported wrong feature sizes at any other resolution.
        self.image_size = image_size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            features = self.backbone(dummy)
            self.feature_channels = [f.shape[1] for f in features]
            self.feature_sizes = [f.shape[2:] for f in features]

        # Adaptive pooling to unify spatial dimensions.
        #
        # pool_size cannot be a fixed 7. The MPS backend implements adaptive
        # pooling only when every input dimension is divisible by the output, so
        # a hardcoded 7 restricts Apple silicon to 224px (feature maps 56/14/7).
        # Choosing the largest common divisor instead unlocks reduced resolution
        # there, which is the difference between ~52 and ~20 min/epoch:
        #
        #   224px -> maps 56,14,7  -> pool 7
        #   160px -> maps 40,10,5  -> pool 5
        #   128px -> maps 32,8,4   -> pool 4
        #    96px -> maps 24,6,3   -> pool 3
        #
        # CPU and CUDA accept any pool size, so this only ever helps.
        fmap_sizes = [int(s[0]) for s in self.feature_sizes]
        if pool_size is None:
            candidates = [p for p in range(min(fmap_sizes), 0, -1)
                          if all(s % p == 0 for s in fmap_sizes)]
            self.pool_size = min(candidates[0], 7) if candidates else 1
            # Prefer the largest valid size at or below 7.
            for p in candidates:
                if p <= 7:
                    self.pool_size = p
                    break
        else:
            self.pool_size = pool_size

        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(self.pool_size) for _ in extract_layers
        ])

        # 1×1 convolution to unify channel dimensions
        unified_channels = output_dim // len(extract_layers)
        self.channel_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, unified_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(unified_channels),
                nn.ReLU(inplace=True),
            )
            for ch in self.feature_channels
        ])

        # Per-position projection producing the spatial tokens. F_local is the
        # mean of these tokens, which mirrors how MambaPathway derives F_global
        # from its patch tokens (norm -> proj -> mean).
        #
        # This replaces an earlier `final_proj` (GAP -> Linear -> LayerNorm).
        # Once the fusion module consumes tokens rather than pooled vectors,
        # that head sat outside the gradient path of the logits: ~37k parameters
        # that were saved in every checkpoint and never trained.
        total_channels = unified_channels * len(extract_layers)
        self.token_proj = (
            nn.Identity() if total_channels == output_dim
            else nn.Linear(total_channels, output_dim)
        )
        self.token_norm = nn.LayerNorm(output_dim)
        self.num_tokens = self.pool_size ** 2

        # Store intermediate feature maps for XAI (Grad-CAM needs these)
        self._feature_maps: Dict[str, torch.Tensor] = {}

    def forward(
        self, x: torch.Tensor, return_tokens: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Extract multi-scale local features.

        Args:
            x: Input image (B, 3, H, W).
            return_tokens: Also return the pre-pooling spatial tokens, which the
                fusion module needs for genuine cross-attention.

        Returns:
            ``f_local`` (B, output_dim), or ``(f_local, tokens)`` with tokens
            shaped (B, pool_size**2, output_dim) when ``return_tokens``.
        """
        # Extract multi-scale features
        features = self.backbone(x)  # List of feature maps

        # Store raw feature maps for XAI
        for i, feat in enumerate(features):
            self._feature_maps[f"stage_{i}"] = feat

        # Unify spatial dimensions and channel counts
        projected = []
        for feat, pool, proj in zip(
            features, self.adaptive_pools, self.channel_projections
        ):
            feat_pooled = pool(feat)  # (B, C_i, pool_size, pool_size)
            feat_proj = proj(feat_pooled)  # (B, unified_ch, pool_size, pool_size)
            projected.append(feat_proj)

        # Concatenate along channel dimension
        fused = torch.cat(projected, dim=1)  # (B, total_channels, pool_size, pool_size)

        # Store fused feature map for XAI
        self._feature_maps["fused"] = fused

        # (B, C, 7, 7) → (B, 49, C) → project/normalise per position.
        tokens = fused.flatten(2).transpose(1, 2)  # (B, pool_size**2, C)
        tokens = self.token_norm(self.token_proj(tokens))

        # F_local is the pooled token representation.
        f_local = tokens.mean(dim=1)  # (B, output_dim)

        if not return_tokens:
            return f_local
        return f_local, tokens

    def get_feature_maps(self) -> Dict[str, torch.Tensor]:
        """
        Get stored feature maps for XAI visualization.

        Returns:
            Dictionary of feature maps at different scales.
        """
        return self._feature_maps

    def get_last_feature_map(self) -> torch.Tensor:
        """
        Get the fused feature map for Grad-CAM computation.

        Returns:
            Fused feature map (B, C, H, W).
        """
        return self._feature_maps.get("fused")
