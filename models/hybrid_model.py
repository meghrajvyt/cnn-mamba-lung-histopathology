"""
Hybrid CNN-Mamba Model — Main architecture class.

Orchestrates the dual-pathway architecture:
    Input → CNN Pathway (local) ─┐
                                  ├→ Fusion Module → Classifier → Prediction
    Input → Mamba Pathway (global)┘

This is the core model class that:
  1. Processes input images through both pathways in parallel
  2. Fuses local (CNN) and global (Mamba) features
  3. Classifies the fused representation
  4. Provides hooks for XAI attribution

Architecture follows the research proposal for:
  "Hybrid CNN-Mamba Architecture for Histopathology Image Classification"
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from models.cnn_pathway import CNNPathway
from models.mamba_pathway import MambaPathway
from models.fusion_module import FusionModule


class HybridCNNMamba(nn.Module):
    """
    Dual-pathway Hybrid CNN-Mamba model for histopathology classification.

    Args:
        config: Model configuration dataclass.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()

        if config is None:
            config = ModelConfig()
        self.config = config

        # ─── Pathway 1: CNN (Local Features) ────────────────────────
        self.cnn_pathway = CNNPathway(
            backbone=config.cnn_backbone,
            pretrained=config.cnn_pretrained,
            extract_layers=config.cnn_extract_layers,
            output_dim=config.cnn_feature_dim,
            image_size=config.image_size,
        )

        # ─── Pathway 2: Mamba (Global Context) ──────────────────────
        self.mamba_pathway = MambaPathway(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.mamba_dim,
            depth=config.mamba_depth,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            dt_rank=config.mamba_dt_rank,
            num_scan_directions=config.mamba_num_scan_directions,
            drop_path_rate=config.drop_path_rate,
            output_dim=config.cnn_feature_dim,  # Match CNN output dim
            scan_chunk_size=config.scan_chunk_size,
            grad_checkpoint=config.grad_checkpoint,
        )

        # ─── Fusion Module ──────────────────────────────────────────
        self.fusion = FusionModule(
            dim=config.fusion_dim,
            num_heads=config.fusion_heads,
            dropout=config.fusion_dropout,
            output_dim=config.fusion_dim,
            mode=getattr(config, "fusion_mode", "token"),
        )

        # ─── Classification Head ────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden_dim, config.num_classes),
        )

        # ─── Feature storage for XAI ───────────────────────────────
        self._features: Dict[str, torch.Tensor] = {}

        # Initialize weights for non-pretrained layers
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for fusion and classification layers."""
        for m in [self.fusion, self.classifier]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the dual-pathway hybrid architecture.

        Args:
            x: Input image tensor (B, C, H, W).
            return_features: If True, also return intermediate features.

        Returns:
            logits: Class logits (B, num_classes).
            features: (optional) Dict of intermediate features for XAI.
        """
        # ─── Parallel pathway processing ───────────────────────────
        # Tokens are requested alongside the pooled vectors: the fusion module
        # cross-attends at token resolution and pools afterwards. f_local and
        # f_global are the means of those same tokens, so reporting them costs
        # two reductions and introduces no untrained parameters.
        f_local, tokens_local = self.cnn_pathway(x, return_tokens=True)
        f_global, tokens_global = self.mamba_pathway(x, return_tokens=True)

        # Store features
        self._features["f_local"] = f_local
        self._features["f_global"] = f_global

        # ─── Fusion ────────────────────────────────────────────────
        f_fused = self.fusion(tokens_local, tokens_global)  # (B, fusion_dim)
        self._features["f_fused"] = f_fused

        # ─── Classification ────────────────────────────────────────
        logits = self.classifier(f_fused)  # (B, num_classes)

        if return_features:
            features = {
                "f_local": f_local,
                "f_global": f_global,
                "f_fused": f_fused,
                "tokens_local": tokens_local,
                "tokens_global": tokens_global,
                "cnn_feature_maps": self.cnn_pathway.get_feature_maps(),
                "mamba_spatial": self.mamba_pathway.get_spatial_features(),
            }
            return logits, features

        return logits

    def get_pathway_features(self) -> Dict[str, torch.Tensor]:
        """Get the last computed pathway features."""
        return self._features

    def freeze_cnn_backbone(self):
        """Freeze CNN backbone parameters (for fine-tuning only new layers)."""
        for param in self.cnn_pathway.backbone.parameters():
            param.requires_grad = False

    def unfreeze_cnn_backbone(self):
        """Unfreeze CNN backbone parameters."""
        for param in self.cnn_pathway.backbone.parameters():
            param.requires_grad = True

    def get_parameter_groups(self, lr: float, weight_decay: float) -> list:
        """
        Get parameter groups with different learning rates.

        CNN backbone uses lower LR (lr/10), new layers use full LR.

        Args:
            lr: Base learning rate.
            weight_decay: Weight decay.

        Returns:
            List of parameter group dicts for optimizer.
        """
        backbone_params = list(self.cnn_pathway.backbone.parameters())
        backbone_ids = set(id(p) for p in backbone_params)

        new_params = [
            p for p in self.parameters()
            if id(p) not in backbone_ids and p.requires_grad
        ]

        return [
            {
                "params": backbone_params,
                "lr": lr * 0.1,
                "weight_decay": weight_decay,
                "name": "cnn_backbone",
            },
            {
                "params": new_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "new_layers",
            },
        ]

    @torch.no_grad()
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "cnn_pathway": _count(self.cnn_pathway),
            "mamba_pathway": _count(self.mamba_pathway),
            "fusion": _count(self.fusion),
            "classifier": _count(self.classifier),
            "total": _count(self),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
