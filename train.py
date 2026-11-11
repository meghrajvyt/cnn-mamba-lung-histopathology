#!/usr/bin/env python3
"""
train.py — Complete PyTorch training pipeline for a Hybrid CNN-Mamba
histopathology classifier.

Usage:
    python train.py                        # Run with all defaults from config.py
    python train.py --epochs 50 --lr 3e-4  # Override specific hyperparameters
    python train.py --use_distillation --teacher_checkpoint ./checkpoints/teacher.pth

Features:
    - Mixed-precision training (FP16 via torch.amp)
    - Cosine annealing LR schedule with linear warmup
    - AdamW with differential learning rates (backbone × 0.1)
    - Weighted cross-entropy with optional label smoothing
    - Gradient clipping (max-norm)
    - Optional knowledge-distillation from a pretrained CNN teacher
    - Early stopping on validation loss
    - Best-model + last-model checkpointing
    - TensorBoard logging
    - Full reproducibility (seed everything)
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `config`, `models`, and
# `data` packages are importable regardless of the working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DataConfig, ModelConfig, TrainingConfig  # noqa: E402
from data.dataset import create_dataloaders  # noqa: E402
from data.transforms import apply_batch_mixing, mixed_criterion  # noqa: E402
from models.baselines import ARCHITECTURES, build_model  # noqa: E402
from models.hybrid_model import HybridCNNMamba  # noqa: E402

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("train")


def _setup_logging(log_dir: str, verbose: bool = True) -> None:
    """Configure root + module logger to write to console and a file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file, mode="a"),
    ]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,  # override any prior basicConfig
    )


# ========================================================================== #
#                          REPRODUCIBILITY                                    #
# ========================================================================== #

def seed_everything(seed: int) -> None:
    """Set seeds for full reproducibility across Python, NumPy, and PyTorch.

    Also configures cuDNN for deterministic behaviour at the cost of a small
    performance penalty.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # Deterministic algorithms — may slow down training slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d (deterministic mode ON)", seed)


# ========================================================================== #
#                          DEVICE SELECTION                                   #
# ========================================================================== #

def get_device(prefer: Optional[str] = None) -> torch.device:
    """Pick the best available accelerator: CUDA > MPS (Apple silicon) > CPU.

    Args:
        prefer: Force a specific device string (e.g. ``"cpu"``). If the
            requested accelerator is unavailable a warning is logged and the
            normal fallback order applies.

    Returns:
        The selected ``torch.device``.
    """
    if prefer:
        prefer = prefer.lower()
        if prefer == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable — falling back.")
        elif prefer == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but unavailable — falling back.")
        else:
            return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_device_info(device: torch.device) -> None:
    """Log human-readable details about the selected device."""
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("CUDA version: %s", torch.version.cuda)
    elif device.type == "mps":
        logger.info("Apple Metal Performance Shaders (unified memory).")
    else:
        logger.warning(
            "Running on CPU. The selective scan is memory-bandwidth bound; "
            "expect training to be very slow. Use a CUDA GPU for real runs."
        )


def configure_cpu_threads(
    num_threads: int, device: torch.device, num_workers: int = 0
) -> None:
    """Set intra-op thread count for CPU training.

    PyTorch's default is frequently far below the available core count, which
    leaves most of a many-core Xeon idle. Dataloader workers compete for the same
    cores, so reserve capacity for them rather than claiming every core.

    Args:
        num_threads: Requested threads; 0 selects automatically.
        device: Training device. Ignored unless this is CPU training.
        num_workers: Dataloader worker count, so their load is accounted for.
    """
    if device.type != "cpu":
        # Threads still matter a little for host-side work, but the accelerator
        # dominates; leave the default alone.
        return

    total = os.cpu_count() or 1
    if num_threads > 0:
        chosen = num_threads
    else:
        # Leave one core per dataloader worker, and always keep at least one.
        chosen = max(1, total - max(1, num_workers))

    torch.set_num_threads(chosen)
    # Some kernels read these directly; set them for consistency. They only take
    # effect for libraries initialised after this point, hence the log line.
    os.environ.setdefault("OMP_NUM_THREADS", str(chosen))
    os.environ.setdefault("MKL_NUM_THREADS", str(chosen))

    logger.info(
        "CPU threads: %d of %d cores (%d dataloader workers). "
        "Set --num_threads to override.", chosen, total, num_workers,
    )
    try:
        import torch.backends.mkldnn as mkldnn
        logger.info(
            "oneDNN/MKL-DNN available: %s%s", mkldnn.is_available(),
            "" if mkldnn.is_available() else
            " — expect substantially slower CPU training without it",
        )
    except Exception:
        pass


def amp_enabled(requested: bool, device: torch.device) -> bool:
    """Decide whether automatic mixed precision should actually be used.

    AMP is supported on CUDA and (since torch 2.4) on MPS. On CPU it offers no
    benefit for this model, so it is disabled there regardless of the flag.
    """
    return bool(requested) and device.type in ("cuda", "mps")


# ========================================================================== #
#                       LR SCHEDULER WITH WARMUP                              #
# ========================================================================== #

class CosineAnnealingWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing schedule preceded by a linear warmup phase.

    During warmup (epoch < warmup_epochs), the LR increases linearly from
    ``eta_min`` to ``base_lr``.  After warmup, it follows a cosine decay
    from ``base_lr`` down to ``eta_min``.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total number of training epochs (including warmup).
        eta_min: Minimum learning rate at the end of cosine decay.
        last_epoch: The index of last epoch (default: -1).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min: float = 1e-7,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup: scale from eta_min → base_lr
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * alpha
                for base_lr in self.base_lrs
            ]
        # Cosine decay phase
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]


# ========================================================================== #
#                       KNOWLEDGE DISTILLATION LOSS                           #
# ========================================================================== #

class DistillationLoss(nn.Module):
    """Combined hard-target + soft-target knowledge-distillation loss.

    ``L = alpha * KL(softmax(student/T), softmax(teacher/T)) * T^2
         + (1 - alpha) * CE(student, label)``

    Args:
        alpha: Blending weight for the soft-target KL term.
        temperature: Softmax temperature for distillation.
        base_criterion: Hard-target loss function (e.g., weighted CE).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 4.0,
        base_criterion: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.base_criterion = base_criterion or nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the blended distillation loss.

        Args:
            student_logits: Raw logits from the student model (B, C).
            teacher_logits: Raw logits from the teacher model (B, C).
            targets: Ground-truth class indices (B,).

        Returns:
            Scalar loss tensor.
        """
        # Hard-target loss
        hard_loss = self.base_criterion(student_logits, targets)

        # Soft-target KL-divergence loss (scaled by T^2 as per Hinton et al.)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = (
            F.kl_div(soft_student, soft_teacher, reduction="batchmean")
            * (self.temperature ** 2)
        )

        return self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss


# ========================================================================== #
#                           EARLY STOPPING                                    #
# ========================================================================== #

class EarlyStopping:
    """Tracks validation loss and signals when to stop training.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum decrease in loss to qualify as an improvement.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """Update state with the latest validation loss.

        Returns:
            ``True`` if training should be stopped.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def state_dict(self) -> Dict[str, Any]:
        """Serialise progress so patience survives a resume."""
        return {
            "best_loss": self.best_loss,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore progress saved by :meth:`state_dict`."""
        self.best_loss = state.get("best_loss")
        self.counter = int(state.get("counter", 0))
        self.should_stop = bool(state.get("should_stop", False))


# ========================================================================== #
#                       METRIC COMPUTATION HELPERS                            #
# ========================================================================== #

class AverageMeter:
    """Running mean tracker for a scalar metric."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy (as a fraction in [0, 1])."""
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


# ========================================================================== #
#                       TRAIN / VALIDATE LOOPS                                #
# ========================================================================== #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    gradient_clip_norm: float = 1.0,
    use_amp: bool = True,
    teacher: Optional[nn.Module] = None,
    distill_criterion: Optional[DistillationLoss] = None,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mix_prob: float = 0.0,
) -> Tuple[float, float]:
    """Run one training epoch.

    Args:
        model: Student model to train.
        loader: Training DataLoader.
        criterion: Hard-target loss function.
        optimizer: Optimizer instance.
        scaler: GradScaler for mixed precision.
        device: Target device.
        epoch: Current epoch index (0-based, for display).
        gradient_clip_norm: Max gradient norm for clipping.
        use_amp: Whether to use automatic mixed precision.
        teacher: (Optional) pretrained teacher model for KD.
        distill_criterion: (Optional) distillation loss module.
        mixup_alpha: mixup Beta parameter (<= 0 disables mixup).
        cutmix_alpha: CutMix Beta parameter (<= 0 disables CutMix).
        mix_prob: Probability of mixing a batch. 0 disables mixing, in which
            case the loss and accuracy are bit-identical to the unmixed path.

    Returns:
        (avg_loss, avg_accuracy) over the epoch. When mixing is active, accuracy
        is measured against the dominant label, since ``lam >= 0.5`` is enforced.
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"  Train Epoch {epoch + 1}", leave=False, unit="batch")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        # Batch-level mixing. With mix_prob=0 this returns the batch untouched
        # and lam=1.0, so the loss below reduces exactly to criterion(logits, y).
        images, targets_a, targets_b, lam = apply_batch_mixing(
            images, targets,
            mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, prob=mix_prob,
        )

        optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()

        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(images)

            if teacher is not None and distill_criterion is not None:
                # The teacher sees the same mixed input, so its soft targets
                # describe the image the student is actually being shown.
                with torch.no_grad():
                    teacher_logits = teacher(images)
                loss = distill_criterion(logits, teacher_logits, targets_a)
            else:
                loss = mixed_criterion(criterion, logits, targets_a, targets_b, lam)

        # Mixed-precision backward + step
        scaler.scale(loss).backward()
        if gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        acc = compute_accuracy(logits.detach(), targets)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """Evaluate model on a validation / test set.

    Args:
        model: Model to evaluate.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Target device.
        epoch: Current epoch index (for display).
        use_amp: Whether to use automatic mixed precision.

    Returns:
        (avg_loss, avg_accuracy) over the dataset.
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"  Val   Epoch {epoch + 1}", leave=False, unit="batch")
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        acc = compute_accuracy(logits, targets)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return loss_meter.avg, acc_meter.avg


# ========================================================================== #
#                         CHECKPOINTING                                       #
# ========================================================================== #

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_acc: float,
    config: dict[str, Any],
    early_stopping_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a training checkpoint to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_acc": best_val_acc,
            "early_stopping_state": early_stopping_state,
            "config": config,
        },
        path,
    )
    logger.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint from disk.

    Returns:
        The full checkpoint dictionary.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    logger.info("Checkpoint loaded ← %s (epoch %d)", path, ckpt["epoch"])
    return ckpt


# ========================================================================== #
#                            ARGUMENT PARSER                                  #
# ========================================================================== #

def parse_args() -> argparse.Namespace:
    """Build CLI argument parser with defaults mirroring config.py."""
    # Pull defaults from the config dataclasses
    tc = TrainingConfig()
    mc = ModelConfig()
    dc = DataConfig()

    p = argparse.ArgumentParser(
        description="Train a Hybrid CNN-Mamba histopathology classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Training hyper-parameters ---
    g = p.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=tc.epochs, help="Total training epochs")
    g.add_argument("--batch_size", type=int, default=tc.batch_size, help="Batch size")
    g.add_argument("--lr", type=float, default=tc.learning_rate, help="Peak learning rate")
    g.add_argument("--weight_decay", type=float, default=tc.weight_decay, help="AdamW weight decay")
    g.add_argument("--warmup_epochs", type=int, default=tc.warmup_epochs, help="Linear warmup epochs")
    g.add_argument("--gradient_clip_norm", type=float, default=tc.gradient_clip_norm, help="Max gradient norm (0 to disable)")
    g.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor for CE loss")
    g.add_argument("--mixed_precision", action="store_true", default=tc.mixed_precision, help="Enable AMP (FP16)")
    g.add_argument("--no_mixed_precision", dest="mixed_precision", action="store_false", help="Disable AMP")
    g.add_argument("--patience", type=int, default=tc.patience, help="Early stopping patience (epochs)")
    g.add_argument("--seed", type=int, default=tc.seed, help="Random seed")
    g.add_argument(
        "--device", type=str, default=None, choices=["cuda", "mps", "cpu"],
        help="Force a device. Default: auto-detect (cuda > mps > cpu)",
    )
    g.add_argument(
        "--num_threads", type=int, default=0,
        help="Intra-op CPU threads. 0 = auto (physical cores, leaving one free "
             "for the dataloader). Only relevant for CPU training, where the "
             "PyTorch default is often well below the core count",
    )
    g.add_argument(
        "--scan_chunk_size", type=int, default=32,
        help="Chunk length for the parallel selective scan (see models/ssm_block.py)",
    )
    g.add_argument(
        "--grad_checkpoint", action="store_true",
        help="Recompute Mamba activations in backward: ~30%% slower, ~2x less memory",
    )

    # --- Knowledge distillation ---
    g_kd = p.add_argument_group("Knowledge Distillation")
    g_kd.add_argument("--use_distillation", action="store_true", default=tc.use_distillation, help="Enable KD from teacher")
    g_kd.add_argument("--distill_alpha", type=float, default=tc.distill_alpha, help="KD blending weight (0=hard only, 1=soft only)")
    g_kd.add_argument("--distill_temperature", type=float, default=tc.distill_temperature, help="KD softmax temperature")
    g_kd.add_argument("--teacher_checkpoint", type=str, default=None, help="Path to pretrained teacher .pth file")

    # --- Model ---
    g_m = p.add_argument_group("Model")
    g_m.add_argument("--image_size", type=int, default=mc.image_size, help="Input image resolution")
    g_m.add_argument("--num_classes", type=int, default=mc.num_classes, help="Number of output classes")
    g_m.add_argument("--cnn_backbone", type=str, default=mc.cnn_backbone, help="CNN backbone name")
    g_m.add_argument(
        "--mamba_depth", type=int, default=mc.mamba_depth,
        help="Number of Vision Mamba blocks. Dominates step time: depth 2 is "
             "~2.2x faster than depth 4 on MPS",
    )
    g_m.add_argument("--mamba_dim", type=int, default=mc.mamba_dim, help="Mamba hidden dimension")
    g_m.add_argument(
        "--arch", type=str, default="hybrid", choices=list(ARCHITECTURES),
        help="Architecture. Baselines and single-pathway ablations run through "
             "this identical pipeline so comparisons are not confounded",
    )
    g_m.add_argument(
        "--fusion_mode", type=str, default=mc.fusion_mode, choices=["token", "pooled"],
        help="'token' cross-attends over spatial tokens; 'pooled' reproduces the "
             "degenerate length-1 attention, for ablation only",
    )

    # --- Data ---
    g_d = p.add_argument_group("Data")
    g_d.add_argument("--data_root", type=str, default=tc.data_root, help="Root directory for the dataset")
    g_d.add_argument("--dataset", type=str, default=dc.dataset, help="Dataset name")
    g_d.add_argument("--magnification", type=str, default=None, help="Magnification filter (None for LC25000)")
    g_d.add_argument("--stain_normalize", action="store_true", default=dc.stain_normalize, help="Apply stain normalisation")
    g_d.add_argument("--no_stain_normalize", dest="stain_normalize", action="store_false", help="Skip stain normalisation")
    g_d.add_argument("--train_ratio", type=float, default=dc.train_ratio, help="Train split ratio")
    g_d.add_argument("--val_ratio", type=float, default=dc.val_ratio, help="Validation split ratio")
    g_d.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")
    g_d.add_argument(
        "--use_groups", action="store_true", default=True,
        help="Split by recovered LC25000 source image (leakage-free)",
    )
    g_d.add_argument(
        "--no_use_groups", dest="use_groups", action="store_false",
        help="Split per file — leaks augmented siblings; for ablation only",
    )
    g_d.add_argument(
        "--groups_file", type=str, default=None,
        help="Path to grouping JSON (default: lc25000_groups.json at project root)",
    )
    g_d.add_argument(
        "--split_file", type=str, default=None,
        help="Explicit {'<class>/<file>': 'train'|'val'|'test'} JSON; overrides "
             "--use_groups and the ratios (patient-level folds, external tests)",
    )
    g_d.add_argument("--stain_method", type=str, default=dc.stain_method,
                     help="Stain normalisation method (only 'macenko' implemented)")

    # --- Augmentation (all defaults mirror DataConfig) ---
    g_a = p.add_argument_group("Augmentation")
    g_a.add_argument("--random_rotation", type=int, default=dc.random_rotation,
                     help="Max rotation in degrees (0 to disable)")
    g_a.add_argument("--random_flip", action="store_true", default=dc.random_flip,
                     help="Random horizontal/vertical flips")
    g_a.add_argument("--no_random_flip", dest="random_flip", action="store_false",
                     help="Disable random flips")
    g_a.add_argument("--color_jitter", type=float, default=dc.color_jitter,
                     help="ColorJitter strength, simulates stain variation (0 to disable)")
    g_a.add_argument("--elastic_deform", action="store_true", default=dc.elastic_deform,
                     help="Elastic deformation, models sectioning distortion (costs throughput)")
    g_a.add_argument("--no_elastic_deform", dest="elastic_deform", action="store_false",
                     help="Disable elastic deformation")
    g_a.add_argument("--elastic_alpha", type=float, default=dc.elastic_alpha,
                     help="Elastic displacement magnitude")
    g_a.add_argument("--elastic_sigma", type=float, default=dc.elastic_sigma,
                     help="Elastic displacement smoothness")
    g_a.add_argument("--mixup_alpha", type=float, default=dc.mixup_alpha,
                     help="mixup Beta parameter (needs --mix_prob > 0 to take effect)")
    g_a.add_argument("--cutmix_alpha", type=float, default=dc.cutmix_alpha,
                     help="CutMix Beta parameter (needs --mix_prob > 0 to take effect)")
    g_a.add_argument("--mix_prob", type=float, default=dc.mix_prob,
                     help="Probability of mixup/CutMix per batch. 0 disables mixing")

    # --- Paths ---
    g_p = p.add_argument_group("Paths")
    g_p.add_argument("--checkpoint_dir", type=str, default=tc.checkpoint_dir, help="Directory for saved checkpoints")
    g_p.add_argument("--log_dir", type=str, default=tc.log_dir, help="Directory for TensorBoard logs")
    g_p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from")

    return p.parse_args()


# ========================================================================== #
#                               MAIN                                          #
# ========================================================================== #

def main() -> None:
    """Full training pipeline: setup → data → model → train → save."""

    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Logging & reproducibility                                        #
    # ------------------------------------------------------------------ #
    _setup_logging(args.log_dir, verbose=True)
    seed_everything(args.seed)

    device = get_device(args.device)
    log_device_info(device)
    configure_cpu_threads(args.num_threads, device, args.num_workers)

    # ------------------------------------------------------------------ #
    # 2. Create DataLoaders                                               #
    # ------------------------------------------------------------------ #
    logger.info("Creating dataloaders …")
    train_loader, val_loader, test_loader, data_info = create_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        stain_normalize=args.stain_normalize,
        magnification=args.magnification,
        binary=False,
        seed=args.seed,
        pin_memory=device.type == "cuda",
        use_groups=args.use_groups,
        groups_file=args.groups_file,
        split_file=args.split_file,
        stain_method=args.stain_method,
        random_rotation=args.random_rotation,
        random_flip=args.random_flip,
        color_jitter=args.color_jitter,
        elastic_deform=args.elastic_deform,
        elastic_alpha=args.elastic_alpha,
        elastic_sigma=args.elastic_sigma,
    )

    num_classes: int = data_info["num_classes"]
    class_names: list[str] = data_info["class_names"]
    class_weights: torch.Tensor = data_info["class_weights"].to(device)

    logger.info("Dataset: %s | Magnification: %s", args.dataset, args.magnification)
    logger.info(
        "Splits — train: %d | val: %d | test: %d",
        data_info["train_size"],
        data_info["val_size"],
        data_info["test_size"],
    )
    logger.info("Classes (%d): %s", num_classes, class_names)
    logger.info("Class weights: %s", class_weights.cpu().tolist())

    # Make the split regime unmissable in the log — it determines whether the
    # resulting accuracy is reportable or just augmentation memorisation.
    if data_info["split_type"] == "explicit":
        logger.info(
            "Split: EXPLICIT from %s (train %d / val %d / test %d). Leakage "
            "control is whatever that file encodes, e.g. patient-level folds.",
            data_info["split_file"], data_info["train_size"],
            data_info["val_size"], data_info["test_size"],
        )
    elif data_info["split_type"] == "grouped":
        logger.info(
            "Split: GROUPED by source image — %d groups "
            "(train %d / val %d / test %d). No augmented-sibling leakage.",
            data_info["n_groups"], data_info["groups_train"],
            data_info["groups_val"], data_info["groups_test"],
        )
    else:
        logger.warning(
            "Split: PER-FILE. On LC25000 this leaks augmented siblings across "
            "train/test — accuracy will be inflated and is NOT reportable."
        )

    aug = data_info["augmentation"]
    logger.info(
        "Augmentation: rotation=%s flip=%s color_jitter=%s elastic=%s stain=%s",
        aug["random_rotation"], aug["random_flip"], aug["color_jitter"],
        aug["elastic_deform"], aug["stain_method"],
    )
    if args.mix_prob > 0:
        logger.info(
            "Batch mixing: p=%.2f (mixup_alpha=%.2f, cutmix_alpha=%.2f). "
            "Train accuracy is measured against the dominant label.",
            args.mix_prob, args.mixup_alpha, args.cutmix_alpha,
        )
    else:
        logger.info("Batch mixing: OFF (--mix_prob 0)")

    # ------------------------------------------------------------------ #
    # 3. Build model                                                      #
    # ------------------------------------------------------------------ #
    model_config = ModelConfig()
    # Override config fields that the user may have changed via CLI
    model_config.image_size = args.image_size
    model_config.num_classes = num_classes  # use actual count from dataset
    model_config.cnn_backbone = args.cnn_backbone
    model_config.mamba_depth = args.mamba_depth
    model_config.mamba_dim = args.mamba_dim
    model_config.scan_chunk_size = args.scan_chunk_size
    model_config.grad_checkpoint = args.grad_checkpoint
    model_config.fusion_mode = args.fusion_mode

    model = build_model(args.arch, model_config).to(device)

    logger.info("Architecture: %s", args.arch)
    if args.arch in ("hybrid", "hybrid_pooled"):
        logger.info("Fusion mode: %s", model_config.fusion_mode)
        if model_config.fusion_mode == "pooled":
            logger.warning(
                "Fusion mode 'pooled' attends over length-1 sequences, so the "
                "softmax is identically 1.0. This is the ablation baseline, not "
                "the proposed model."
            )
    param_info = model.count_parameters()
    logger.info("Model parameter counts: %s", param_info)

    # ------------------------------------------------------------------ #
    # 4. Optimizer (differential LR) & scheduler                          #
    # ------------------------------------------------------------------ #
    param_groups = model.get_parameter_groups(lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    scheduler = CosineAnnealingWithWarmup(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        eta_min=1e-7,
    )

    # ------------------------------------------------------------------ #
    # 5. Loss function                                                    #
    # ------------------------------------------------------------------ #
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    logger.info("Loss: CrossEntropyLoss (label_smoothing=%.2f, weighted)", args.label_smoothing)

    # ------------------------------------------------------------------ #
    # 6. (Optional) Knowledge distillation teacher                        #
    # ------------------------------------------------------------------ #
    teacher: Optional[nn.Module] = None
    distill_criterion: Optional[DistillationLoss] = None

    if args.use_distillation:
        if args.teacher_checkpoint is None:
            raise ValueError(
                "--teacher_checkpoint must be provided when --use_distillation is set."
            )
        logger.info("Loading teacher model from %s …", args.teacher_checkpoint)

        # Assume teacher has the same architecture; load weights only.
        teacher = HybridCNNMamba(config=model_config).to(device)
        teacher_ckpt = torch.load(
            args.teacher_checkpoint, map_location=device, weights_only=False
        )
        # Support both raw state_dict and wrapped checkpoint dicts
        state_dict = (
            teacher_ckpt["model_state_dict"]
            if "model_state_dict" in teacher_ckpt
            else teacher_ckpt
        )
        teacher.load_state_dict(state_dict)
        teacher.eval()
        # Freeze teacher — no gradients needed
        for p in teacher.parameters():
            p.requires_grad = False

        distill_criterion = DistillationLoss(
            alpha=args.distill_alpha,
            temperature=args.distill_temperature,
            base_criterion=criterion,
        )
        logger.info(
            "KD enabled — alpha=%.2f, temperature=%.1f",
            args.distill_alpha,
            args.distill_temperature,
        )

    # ------------------------------------------------------------------ #
    # 7. Mixed precision scaler                                           #
    # ------------------------------------------------------------------ #
    use_amp = amp_enabled(args.mixed_precision, device)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    logger.info(
        "Mixed precision (AMP): %s (device=%s)",
        "ON" if use_amp else "OFF",
        device.type,
    )

    # ------------------------------------------------------------------ #
    # 8. Resume from checkpoint (if requested)                            #
    # ------------------------------------------------------------------ #
    start_epoch = 0
    best_val_acc = 0.0

    resumed_early_stopping: Optional[Dict[str, Any]] = None

    if args.resume is not None:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        resumed_early_stopping = ckpt.get("early_stopping_state")
        logger.info("Resuming from epoch %d (best_val_acc=%.4f)", start_epoch, best_val_acc)

    # ------------------------------------------------------------------ #
    # 9. TensorBoard writer & early stopping                              #
    # ------------------------------------------------------------------ #
    writer = SummaryWriter(log_dir=args.log_dir)
    early_stopping = EarlyStopping(patience=args.patience)
    if resumed_early_stopping is not None:
        early_stopping.load_state_dict(resumed_early_stopping)
        logger.info(
            "Early stopping restored: best_loss=%s, %d/%d epochs without improvement",
            f"{early_stopping.best_loss:.4f}" if early_stopping.best_loss is not None else "n/a",
            early_stopping.counter, args.patience,
        )
    elif args.resume is not None:
        logger.warning(
            "Checkpoint has no early-stopping state (written before this was "
            "tracked); patience restarts from 0."
        )

    # ------------------------------------------------------------------ #
    # 10. Training loop                                                   #
    # ------------------------------------------------------------------ #
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    last_ckpt_path = os.path.join(args.checkpoint_dir, "last_model.pth")

    logger.info(
        "Starting training for %d epochs (from epoch %d) …",
        args.epochs,
        start_epoch + 1,
    )
    total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("═" * 60)
        logger.info("Epoch %d/%d  |  LR: %.2e", epoch + 1, args.epochs, current_lr)
        logger.info("═" * 60)

        # --- Train ---
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            gradient_clip_norm=args.gradient_clip_norm,
            use_amp=use_amp,
            teacher=teacher,
            distill_criterion=distill_criterion,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mix_prob,
        )

        # --- Validate ---
        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
        )

        # Step the scheduler *after* the epoch
        scheduler.step()

        epoch_time = time.time() - epoch_start
        logger.info(
            "  Train — loss: %.4f | acc: %.4f", train_loss, train_acc
        )
        logger.info(
            "  Val   — loss: %.4f | acc: %.4f", val_loss, val_acc
        )
        logger.info("  Time: %.1fs", epoch_time)

        # --- TensorBoard logging ---
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # --- Checkpointing ---
        ckpt_config = {
            "model_config": vars(model_config),
            "args": vars(args),
        }

        # Update the best-so-far BEFORE writing last_model.pth. Writing it first
        # persisted a best_val_acc one epoch stale, so resuming restored an
        # understated best and could then overwrite best_model.pth with a worse
        # checkpoint. That only ever showed up across a resume boundary, which is
        # exactly how long runs are executed.
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        # --- Early stopping ---
        # Stepped before checkpointing so its updated state is what gets saved;
        # otherwise patience silently restarted on every resume and early
        # stopping could never trigger in a multi-session run.
        should_stop = early_stopping.step(val_loss)

        # Always save the last checkpoint
        save_checkpoint(
            last_ckpt_path, model, optimizer, scheduler, scaler,
            epoch, best_val_acc, ckpt_config,
            early_stopping_state=early_stopping.state_dict(),
        )

        # Save best checkpoint (by validation accuracy)
        if is_best:
            save_checkpoint(
                best_ckpt_path, model, optimizer, scheduler, scaler,
                epoch, best_val_acc, ckpt_config,
                early_stopping_state=early_stopping.state_dict(),
            )
            logger.info("  ★ New best val accuracy: %.4f", best_val_acc)

        if should_stop:
            logger.info(
                "Early stopping triggered after %d epochs without improvement.",
                args.patience,
            )
            break

    total_time = time.time() - total_start
    logger.info("Training finished in %.1f s (%.1f min).", total_time, total_time / 60)
    logger.info("Best validation accuracy: %.4f", best_val_acc)

    # ------------------------------------------------------------------ #
    # 11. Final evaluation on test set using the best checkpoint          #
    # ------------------------------------------------------------------ #
    logger.info("Evaluating best model on test set …")
    load_checkpoint(best_ckpt_path, model)
    model.to(device)
    test_loss, test_acc = validate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        epoch=0,
        use_amp=use_amp,
    )
    logger.info("Test — loss: %.4f | acc: %.4f", test_loss, test_acc)

    writer.add_scalar("Loss/test", test_loss, 0)
    writer.add_scalar("Accuracy/test", test_acc, 0)
    writer.close()

    logger.info("All done. Checkpoints saved in %s", args.checkpoint_dir)
    logger.info("TensorBoard logs saved in %s", args.log_dir)


if __name__ == "__main__":
    main()
