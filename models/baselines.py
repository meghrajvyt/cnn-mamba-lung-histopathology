"""Baselines and single-pathway ablations.

WHY THIS EXISTS
---------------
Reporting an accuracy for the hybrid model says nothing on its own. A reviewer
cannot tell whether the Mamba pathway earns its parameters, whether fusion does
anything, or whether a plain fine-tuned CNN would match it. Those questions are
answered by running the *same* training pipeline over different architectures on
the *same* grouped split, which is what this module enables.

Every model here exposes the interface train.py expects — `forward`,
`count_parameters`, `get_parameter_groups` — so no experiment needs a bespoke
training script, and nothing about the comparison is confounded by pipeline
differences.

Architectures:
    hybrid       CNN + Mamba pathways, token-level cross-attention fusion
    hybrid_pooled  as above but fusion on pooled vectors (the degenerate variant,
                 kept to justify the token-level rewrite)
    cnn_only     CNN pathway alone, isolating its contribution
    mamba_only   Mamba pathway alone, isolating the SSM contribution
    effnet_b0    fine-tuned EfficientNet-B0: the "would a plain CNN do?" control
    vit_small    ViT-Small/16: attention baseline for the O(N) efficiency claim
    swin_tiny    Swin-Tiny: hierarchical attention baseline
    vmamba_tiny  VMamba-T: a standalone state-space classifier, added after the
                 pre-submission review (docs/added_analyses_protocol.md)
"""

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import timm
import torch
import torch.nn as nn

from config import ModelConfig
from models.cnn_pathway import CNNPathway
from models.mamba_pathway import MambaPathway

ARCHITECTURES = (
    "hybrid",
    "hybrid_pooled",
    "cnn_only",
    "mamba_only",
    "effnet_b0",
    "vit_small",
    "swin_tiny",
    "hybrid_res",
    "hybrid_res_xattn",
    "vmamba_tiny",
)

# The authors' ImageNet-1K weights for VMamba-T, configuration vssm1_tiny_0230s
# (release #v2cls of github.com/MzeroMiko/VMamba), and the SHA-256 of that file.
VMAMBA_T_URL = ("https://github.com/MzeroMiko/VMamba/releases/download/"
                "%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth")
VMAMBA_T_FILE = "vssm1_tiny_0230s_ckpt_epoch_264.pth"
VMAMBA_T_SHA256 = "d855b57224a652bc692804932f88dc3a39b9028e64be99f745a7fd6792ca672a"


class _BaseModel(nn.Module):
    """Shared plumbing so train.py can treat every architecture identically."""

    @torch.no_grad()
    def count_parameters(self) -> Dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def _backbone_module(self) -> Optional[nn.Module]:
        """Pretrained submodule that should train at a reduced learning rate."""
        return None

    def get_parameter_groups(self, lr: float, weight_decay: float) -> List[Dict[str, Any]]:
        """Differential LR: pretrained weights at lr/10, new layers at lr.

        Mirrors HybridCNNMamba.get_parameter_groups so the optimiser treatment is
        identical across architectures and cannot explain a difference in results.
        """
        backbone = self._backbone_module()
        if backbone is None:
            return [{
                "params": [p for p in self.parameters() if p.requires_grad],
                "lr": lr, "weight_decay": weight_decay, "name": "all",
            }]

        backbone_ids = {id(p) for p in backbone.parameters()}
        new_params = [p for p in self.parameters()
                      if id(p) not in backbone_ids and p.requires_grad]
        return [
            {"params": list(backbone.parameters()), "lr": lr * 0.1,
             "weight_decay": weight_decay, "name": "backbone"},
            {"params": new_params, "lr": lr,
             "weight_decay": weight_decay, "name": "new_layers"},
        ]


class SinglePathway(_BaseModel):
    """One pathway plus a classifier — the CNN-only / Mamba-only ablations.

    Classifies from the pathway's pooled token representation, which is the same
    quantity the fusion module would have received. The only thing removed is the
    other pathway and the fusion, so the comparison isolates exactly that.
    """

    def __init__(self, config: ModelConfig, pathway: str):
        super().__init__()
        if pathway not in ("cnn", "mamba"):
            raise ValueError(f"pathway must be 'cnn' or 'mamba', got {pathway!r}")
        self.pathway_name = pathway
        dim = config.cnn_feature_dim

        if pathway == "cnn":
            self.pathway = CNNPathway(
                backbone=config.cnn_backbone,
                pretrained=config.cnn_pretrained,
                extract_layers=config.cnn_extract_layers,
                output_dim=dim,
                image_size=config.image_size,
            )
        else:
            self.pathway = MambaPathway(
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
                output_dim=dim,
                scan_chunk_size=config.scan_chunk_size,
                grad_checkpoint=config.grad_checkpoint,
            )

        # Same head shape as the hybrid model, so head capacity is not a
        # confound when comparing against it.
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden_dim, config.num_classes),
        )

    def _backbone_module(self) -> Optional[nn.Module]:
        return getattr(self.pathway, "backbone", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.pathway(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        return self.classifier(self.norm(feats))

    @torch.no_grad()
    def count_parameters(self) -> Dict[str, int]:
        base = super().count_parameters()
        base[f"{self.pathway_name}_pathway"] = sum(
            p.numel() for p in self.pathway.parameters()
        )
        base["classifier"] = sum(p.numel() for p in self.classifier.parameters())
        return base


class TimmClassifier(_BaseModel):
    """A stock timm backbone with a fresh head.

    These are the external controls. `effnet_b0` answers whether the hybrid model
    beats a plain fine-tuned CNN at all; `vit_small` and `swin_tiny` provide the
    attention-based comparison the O(N) efficiency argument needs.
    """

    def __init__(self, model_name: str, config: ModelConfig,
                 pretrained: Optional[bool] = None):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=config.cnn_pretrained if pretrained is None else pretrained,
            num_classes=config.num_classes,
            drop_rate=config.head_dropout,
        )

    def _backbone_module(self) -> Optional[nn.Module]:
        # The classifier head is new; everything else is pretrained. timm exposes
        # the head by name, so exclude it from the reduced-LR group.
        head_names = ("classifier", "head", "fc")
        for name in head_names:
            if hasattr(self.backbone, name):
                head = getattr(self.backbone, name)
                head_ids = {id(p) for p in head.parameters()}
                pretrained = nn.ModuleList([
                    m for n, m in self.backbone.named_children() if n != name
                ])
                # Only return it if it actually excludes something.
                if any(id(p) not in head_ids for p in pretrained.parameters()):
                    return pretrained
        return self.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def count_parameters(self) -> Dict[str, int]:
        base = super().count_parameters()
        base["backbone"] = sum(p.numel() for p in self.backbone.parameters())
        return base


def vmamba_t_weights_path() -> Path:
    """Local copy of the VMamba-T weights, downloaded once into the torch hub cache."""
    path = Path(torch.hub.get_dir()) / "checkpoints" / VMAMBA_T_FILE
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(VMAMBA_T_URL, str(path), progress=False)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_vmamba_t_weights() -> Dict[str, torch.Tensor]:
    """The authors' ImageNet-1K state dict for VMamba-T, after checking its hash."""
    path = vmamba_t_weights_path()
    digest = sha256_file(path)
    if digest != VMAMBA_T_SHA256:
        raise RuntimeError(f"{path}: SHA-256 {digest} is not that of the authors' "
                           f"release ({VMAMBA_T_SHA256}); delete the file to re-download it")
    return torch.load(path, map_location="cpu", weights_only=True)["model"]


class VMambaClassifier(_BaseModel):
    """VMamba-T, the standalone state-space comparator.

    The authors' implementation (models/vmamba_vendor) in configuration
    vssm1_tiny_0230s (their builder ``vmamba_tiny_s1l8``) with their ImageNet-1K
    weights. As for the timm comparators, the 1000-class layer is replaced by
    dropout and a new linear layer for the task, the implementation keeps its
    own stochastic depth (0.2), and the pretrained parameters train at lr/10.
    """

    def __init__(self, config: ModelConfig, pretrained: Optional[bool] = None):
        super().__init__()
        from models.vmamba_vendor import vmamba as vm  # local: optional kernel import

        self.backbone = vm.vmamba_tiny_s1l8(pretrained=False)
        if config.cnn_pretrained if pretrained is None else pretrained:
            self.backbone.load_state_dict(load_vmamba_t_weights(), strict=True)

        old = self.backbone.classifier
        head = nn.Linear(old.head.in_features, config.num_classes)
        self.backbone._init_weights(head)  # the implementation's own initialisation
        self.backbone.classifier = nn.Sequential(OrderedDict(
            norm=old.norm, permute=old.permute, avgpool=old.avgpool, flatten=old.flatten,
            head_drop=nn.Dropout(config.head_dropout), head=head,
        ))
        self.backbone.num_classes = config.num_classes

    def _backbone_module(self) -> Optional[nn.Module]:
        # Every module except the new linear layer is pretrained, including the
        # final norm inside `classifier`.
        return nn.ModuleList(
            [m for n, m in self.backbone.named_children() if n != "classifier"]
            + [m for n, m in self.backbone.classifier.named_children() if n != "head"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def count_parameters(self) -> Dict[str, int]:
        base = super().count_parameters()
        base["backbone"] = sum(p.numel() for p in self.backbone.parameters())
        return base


def build_model(arch: str, config: ModelConfig) -> nn.Module:
    """Construct one of ``ARCHITECTURES``.

    Args:
        arch: Architecture name.
        config: Model configuration. ``num_classes`` and ``image_size`` are
            honoured by every architecture.

    Returns:
        A module exposing forward / count_parameters / get_parameter_groups.

    Raises:
        ValueError: If ``arch`` is unknown.
    """
    from models.hybrid_model import HybridCNNMamba  # local: avoids a cycle

    if arch == "hybrid":
        return HybridCNNMamba(config=config)

    if arch == "hybrid_pooled":
        # The degenerate fusion variant, reachable only through this flag so the
        # rewrite can be justified with a number instead of an argument.
        cfg = ModelConfig(**vars(config))
        cfg.fusion_mode = "pooled"
        return HybridCNNMamba(config=cfg)

    if arch in ("hybrid_res", "hybrid_res_xattn"):
        from models.hybrid_res import ResidualHybrid  # local: avoids a cycle
        return ResidualHybrid(config, fusion="xattn" if arch.endswith("xattn") else "sum")

    if arch == "cnn_only":
        return SinglePathway(config, "cnn")

    if arch == "mamba_only":
        return SinglePathway(config, "mamba")

    timm_names = {
        "effnet_b0": "efficientnet_b0",
        "vit_small": "vit_small_patch16_224",
        "swin_tiny": "swin_tiny_patch4_window7_224",
    }
    if arch in timm_names:
        return TimmClassifier(timm_names[arch], config)

    if arch == "vmamba_tiny":
        return VMambaClassifier(config)

    raise ValueError(f"Unknown arch {arch!r}. Available: {', '.join(ARCHITECTURES)}")
