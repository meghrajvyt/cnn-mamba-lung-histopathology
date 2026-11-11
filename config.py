# Hybrid CNN-Mamba for Lung Cancer Histopathology Classification
# Configuration
#
# Dataset: LC25000 (Borkowski et al., 2019, arXiv:1912.12142)
#   - lung_n   : Normal lung tissue
#   - lung_aca : Lung adenocarcinoma
#   - lung_scc : Lung squamous cell carcinoma

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for the Hybrid CNN-Mamba model."""

    # Input
    image_size: int = 224
    in_channels: int = 3
    num_classes: int = 3  # LC25000 lung cancer: lung_n, lung_aca, lung_scc

    # CNN Pathway (EfficientNet-B0 — lightweight for M2 Mac)
    cnn_backbone: str = "efficientnet_b0"
    cnn_pretrained: bool = True
    cnn_feature_dim: int = 192  # Output feature dimension
    cnn_extract_layers: List[int] = field(default_factory=lambda: [1, 3, 4])

    # Mamba Pathway
    patch_size: int = 16
    mamba_dim: int = 192  # Hidden dimension
    mamba_depth: int = 4  # Number of Vision Mamba blocks
    mamba_d_state: int = 16  # SSM state dimension
    mamba_d_conv: int = 4  # Local convolution width
    mamba_expand: int = 2  # Inner dimension expansion factor
    mamba_dt_rank: str = "auto"  # Rank of Δ projection
    mamba_bidirectional: bool = True
    mamba_num_scan_directions: int = 4  # SS2D directions

    # Chunk length for the parallel selective scan. Larger = fewer sequential
    # steps but more activation memory retained for backward. 0 = whole
    # sequence in one chunk. See models/ssm_block.py::_selective_scan.
    scan_chunk_size: int = 32

    # Recompute Mamba block activations during backward instead of storing
    # them. ~30% slower per step but roughly halves peak memory, which is the
    # binding constraint on 16 GB GPUs. Enable to train at batch_size 32.
    grad_checkpoint: bool = False

    # Fusion
    fusion_dim: int = 192
    fusion_heads: int = 4  # Cross-attention heads
    fusion_dropout: float = 0.1
    # "token"  cross-attend over spatial tokens (49 <-> 196). Correct.
    # "pooled" pool each pathway first, then attend over length-1 sequences,
    #          which makes the softmax identically 1.0. Retained ONLY as an
    #          ablation, to quantify what the token-level rewrite bought.
    fusion_mode: str = "token"

    # Classification Head
    head_hidden_dim: int = 128
    head_dropout: float = 0.3

    # Regularization
    drop_path_rate: float = 0.1
    label_smoothing: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.999)

    # Scheduler
    scheduler: str = "cosine"  # cosine | step | plateau
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Training
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0

    # Early Stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Knowledge Distillation (optional)
    use_distillation: bool = False
    teacher_model: Optional[str] = None
    distill_alpha: float = 0.5
    distill_temperature: float = 4.0

    # Paths
    data_root: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Reproducibility
    seed: int = 42


@dataclass
class DataConfig:
    """Configuration for data pipeline.

    Every field here is read by the pipeline. Where a field was previously
    declared but ignored, the default has been set to the behaviour that was
    actually in force, so enabling it is an explicit choice rather than a
    silent change to what past runs measured.
    """

    # Dataset
    dataset: str = "breakhis"  # breakhis | camelyon16
    magnification: str = "200X"  # 40X | 100X | 200X | 400X (BreakHis)

    # Preprocessing
    stain_normalize: bool = True
    # Only "macenko" is implemented; anything else raises rather than silently
    # falling back to Macenko (see data/dataset.py::create_dataloaders).
    stain_method: str = "macenko"

    # ── Per-sample augmentation (data/transforms.py) ──
    random_rotation: int = 90
    random_flip: bool = True
    color_jitter: float = 0.2
    # Elastic deformation was declared True but never applied. Defaulting to
    # False keeps the augmentation identical to every run made so far; it also
    # costs real loader throughput, so turn it on deliberately.
    elastic_deform: bool = False
    elastic_alpha: float = 50.0
    elastic_sigma: float = 5.0

    # ── Batch-level mixing (applied in train.py, not the transform pipeline) ──
    # Strengths are kept at their original values, but mixing is gated behind
    # mix_prob and off by default: switching it on changes what the loss
    # measures, so it should not happen implicitly.
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.0  # 0 = disabled; 0.5 is a common starting point

    # Splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # WSI Patch Extraction (Camelyon16)
    patch_size: int = 256
    patch_stride: int = 256
    tissue_threshold: float = 0.5
