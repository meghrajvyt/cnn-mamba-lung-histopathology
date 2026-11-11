"""Residual feature-level hybrid: Mamba over pretrained CNN features.

WHY THIS EXISTS
---------------
On the leakage-free LC25000 split the original hybrid (HybridCNNMamba) scored
significantly below its own CNN pathway (94.74% vs 96.15%, McNemar p < 0.001,
seed 42). The diagnosis: its Mamba pathway learns from raw 16x16 pixel patches
with no pretraining (alone it reaches only ~80% validation accuracy), and the
fusion mixes that weak signal into the strong pretrained CNN features from the
first step of training.

This design removes both causes:

1. **Mamba runs on CNN features, not pixels.** The SS2D blocks model global
   context over the pretrained backbone's stride-16 feature map (14x14 = 196
   tokens at 224px), so the state-space component starts from good features.
2. **Zero-initialised residual.** The Mamba contribution enters as
   f = f_cnn + W_out(mamba), with W_out initialised to zero. At initialisation
   the model is functionally CNN-only, and the Mamba branch contributes only if
   training finds that it lowers the loss.

Variants (``fusion``):
    "sum"    global average of the Mamba tokens, added residually
    "xattn"  token-level cross-attention between CNN tokens and Mamba tokens
             (the existing FusionModule), added residually

Exposes ``cnn_pathway`` and ``mamba_pathway`` (with ``.blocks``) so the
dual-pathway Grad-CAM++ in models/xai.py works unchanged, and
``ablate_mamba`` / ``last_contributions`` for pathway-attribution analysis.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from config import ModelConfig
from models.baselines import _BaseModel
from models.cnn_pathway import CNNPathway
from models.fusion_module import FusionModule
from models.mamba_pathway import VisionMambaBlock


class FeatureMamba(nn.Module):
    """SS2D Vision Mamba blocks over a CNN feature map (B, C, H, W)."""

    def __init__(self, in_channels: int, grid_hw: tuple, config: ModelConfig):
        super().__init__()
        dim = config.mamba_dim
        h, w = grid_hw
        self.grid_hw = (h, w)
        self.grad_checkpoint = config.grad_checkpoint

        self.proj = nn.Linear(in_channels, dim)
        self.norm_in = nn.LayerNorm(dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate,
                                                config.mamba_depth)]
        self.blocks = nn.ModuleList([
            VisionMambaBlock(
                dim=dim,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dt_rank=config.mamba_dt_rank,
                num_scan_directions=config.mamba_num_scan_directions,
                drop_path=dpr[i],
                scan_chunk_size=config.scan_chunk_size,
            )
            for i in range(config.mamba_depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self._features: Dict[str, torch.Tensor] = {}

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        B, C, H, W = fmap.shape
        x = fmap.flatten(2).transpose(1, 2)            # (B, H*W, C)
        x = self.norm_in(self.proj(x)) + self.pos_embed
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, H, W,
                                                      use_reentrant=False)
            else:
                x = blk(x, H, W)
        x = self.norm(x)
        if not self.training:
            self._features["final"] = x
        return x                                        # (B, H*W, dim)


class ResidualHybrid(_BaseModel):
    """CNN backbone + Mamba over its features, combined by a zero-init residual."""

    def __init__(self, config: ModelConfig, fusion: str = "sum"):
        super().__init__()
        if fusion not in ("sum", "xattn"):
            raise ValueError(f"fusion must be 'sum' or 'xattn', got {fusion!r}")
        self.fusion_type = fusion
        dim = config.cnn_feature_dim

        self.cnn_pathway = CNNPathway(
            backbone=config.cnn_backbone,
            pretrained=config.cnn_pretrained,
            extract_layers=config.cnn_extract_layers,
            output_dim=dim,
            image_size=config.image_size,
        )

        # The stage whose resolution is closest to stride 16 (14x14 at 224px):
        # enough tokens for global context, cheap enough for the scan.
        target = config.image_size // 16
        sizes = [int(s[0]) for s in self.cnn_pathway.feature_sizes]
        self.mamba_stage = min(range(len(sizes)), key=lambda i: abs(sizes[i] - target))
        fs = self.cnn_pathway.feature_sizes[self.mamba_stage]
        self.mamba_pathway = FeatureMamba(
            in_channels=self.cnn_pathway.feature_channels[self.mamba_stage],
            grid_hw=(int(fs[0]), int(fs[1])),
            config=config,
        )

        gdim = config.mamba_dim
        if fusion == "xattn":
            # FusionModule attends between token sets of equal width.
            self.global_to_local = (nn.Identity() if gdim == dim
                                    else nn.Linear(gdim, dim))
            self.fusion = FusionModule(dim=dim, num_heads=config.fusion_heads,
                                       dropout=config.fusion_dropout,
                                       output_dim=dim, mode="token")
            out_in = dim
        else:
            self.global_to_local = None
            self.fusion = None
            out_in = gdim

        # Zero-init: at step 0 the model is exactly CNN-only.
        self.mamba_out = nn.Linear(out_in, dim)
        nn.init.zeros_(self.mamba_out.weight)
        nn.init.zeros_(self.mamba_out.bias)

        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden_dim, config.num_classes),
        )

        # Pathway attribution hooks (eval only).
        self.ablate_mamba = False
        self.last_contributions: Dict[str, torch.Tensor] = {}

    def _backbone_module(self) -> Optional[nn.Module]:
        return self.cnn_pathway.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_local, tokens_local = self.cnn_pathway(x, return_tokens=True)
        fmap = self.cnn_pathway.get_feature_maps()[f"stage_{self.mamba_stage}"]
        tokens_global = self.mamba_pathway(fmap)

        if self.fusion is None:
            g = tokens_global.mean(dim=1)
        else:
            g = self.fusion(tokens_local, self.global_to_local(tokens_global))
        delta = self.mamba_out(g)
        if self.ablate_mamba:
            delta = torch.zeros_like(delta)

        if not self.training:
            self.last_contributions = {"f_local": f_local.detach(),
                                       "delta": delta.detach()}
        return self.classifier(self.norm(f_local + delta))

    @torch.no_grad()
    def count_parameters(self) -> Dict[str, int]:
        base = super().count_parameters()
        n = lambda m: sum(p.numel() for p in m.parameters()) if m is not None else 0
        base.update({
            "cnn_pathway": n(self.cnn_pathway),
            "mamba_pathway": n(self.mamba_pathway),
            "fusion": n(self.fusion) + n(self.mamba_out),
            "classifier": n(self.classifier),
        })
        return base
