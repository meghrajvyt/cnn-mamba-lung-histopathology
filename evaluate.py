#!/usr/bin/env python3
"""
Evaluation and Analysis Script — Hybrid CNN-Mamba Histopathology Classifier.

This script performs comprehensive evaluation of a trained HybridCNNMamba model:
  1. Loads a checkpoint and reconstructs the model.
  2. Computes test-set metrics: accuracy, precision, recall, F1, AUC-ROC (per-class & macro).
  3. Generates visualizations:
     - Confusion matrix heatmap (seaborn)
     - Per-class ROC curves + macro-average
     - Precision-Recall curves
     - Dual-pathway Grad-CAM++ XAI heatmaps (CNN, Mamba, combined)
  4. Benchmarks inference speed (with GPU warmup).
  5. Saves all metrics to JSON and prints a summary table.

Usage:
    python evaluate.py --checkpoint path/to/best_model.pth --data-root path/to/dataset

Dependencies:
    torch, torchvision, numpy, scikit-learn, matplotlib, seaborn, tqdm
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
# Use non-interactive backend so the script works on headless servers / CI
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# Ensure project root is on sys.path so local imports resolve correctly
# regardless of the working directory from which this script is invoked.
# ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import ModelConfig, TrainingConfig, DataConfig  # noqa: E402
from models.baselines import build_model  # noqa: E402
from models.hybrid_model import HybridCNNMamba  # noqa: E402
from models.xai import GradCAMPlusPlus, visualize_heatmap  # noqa: E402
from data.dataset import create_dataloaders  # noqa: E402

# ──────────────────────────────────────────────────────────────────────
# Logger setup
# ──────────────────────────────────────────────────────────────────────
logger = logging.getLogger("evaluate")


def setup_logging(log_dir: Optional[str] = None) -> None:
    """Configure root logger with console + optional file handler.

    Args:
        log_dir: If provided, also write logs to ``<log_dir>/evaluate.log``.
    """
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "evaluate.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)


# ======================================================================
#  1. Argument Parsing
# ======================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the Hybrid CNN-Mamba histopathology classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved model checkpoint (.pth).",
    )

    # Data
    parser.add_argument(
        "--data-root", "--data_root",
        dest="data_root",
        type=str,
        default="./data",
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--magnification",
        type=str,
        default=None,
        help="BreakHis magnification filter (e.g. 40X, 100X, 200X, 400X).",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        default=False,
        help="Use binary (benign/malignant) classification instead of 8-class.",
    )
    # Split reproduction. These default to whatever the checkpoint recorded, so
    # the test set here is byte-for-byte the one held out during training.
    # Passing them explicitly overrides the checkpoint.
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Split seed. Default: read from the checkpoint (else 42).",
    )
    parser.add_argument(
        "--train-ratio", "--train_ratio", dest="train_ratio",
        type=float, default=None,
        help="Train split ratio. Default: read from the checkpoint (else 0.7).",
    )
    parser.add_argument(
        "--val-ratio", "--val_ratio", dest="val_ratio",
        type=float, default=None,
        help="Val split ratio. Default: read from the checkpoint (else 0.15).",
    )
    parser.add_argument(
        "--stain-normalize", "--stain_normalize", dest="stain_normalize",
        action="store_true", default=None,
        help="Apply Macenko normalisation. Default: read from the checkpoint.",
    )
    parser.add_argument(
        "--no-stain-normalize", "--no_stain_normalize", dest="stain_normalize",
        action="store_false",
        help="Skip Macenko normalisation.",
    )
    parser.add_argument(
        "--groups-file", "--groups_file", dest="groups_file",
        type=str, default=None,
        help="Grouping JSON for the leakage-free split.",
    )
    parser.add_argument(
        "--split-file", "--split_file", dest="split_file",
        type=str, default=None,
        help="Explicit split JSON. Overrides the checkpoint's split; use a "
             "test-only file to evaluate on an external dataset.",
    )
    parser.add_argument(
        "--no-use-groups", "--no_use_groups", dest="use_groups",
        action="store_false", default=None,
        help="Evaluate on a per-file split (leaky; ablation only).",
    )
    parser.add_argument(
        "--num-workers", "--num_workers", dest="num_workers",
        type=int, default=4,
        help="DataLoader worker count.",
    )

    # Output
    parser.add_argument(
        "--output-dir", "--output_dir",
        dest="output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation outputs (plots, metrics JSON).",
    )

    # Hyper-parameters
    parser.add_argument(
        "--batch-size", "--batch_size",
        dest="batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num-xai-samples", "--num_xai_samples",
        dest="num_xai_samples",
        type=int,
        default=10,
        help="Number of test images to generate Grad-CAM++ heatmaps for.",
    )

    return parser.parse_args()


# ======================================================================
#  2. Model Loading
# ======================================================================

def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[HybridCNNMamba, ModelConfig, Dict[str, Any]]:
    """Load a HybridCNNMamba model from a training checkpoint.

    The checkpoint is expected to contain at minimum a ``model_state_dict``
    key.  Optionally it may contain ``config`` (a serialised ModelConfig)
    and arbitrary training metadata (epoch, best_metric, etc.) which are
    returned as-is.

    Args:
        checkpoint_path: Absolute or relative path to the ``.pth`` file.
        device: Device to map the model weights onto.

    Returns:
        Tuple of (model, config, metadata_dict).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint cannot be loaded.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ── Reconstruct config ──────────────────────────────────────────
    # train.py writes config as {"model_config": {...}, "args": {...}}. Older or
    # hand-made checkpoints may store a flat ModelConfig dict, or nothing.
    raw_config = checkpoint.get("config")
    train_args: Dict[str, Any] = {}

    if isinstance(raw_config, ModelConfig):
        config = raw_config
    elif isinstance(raw_config, dict) and "model_config" in raw_config:
        train_args = raw_config.get("args") or {}
        valid = {f.name for f in dataclasses.fields(ModelConfig)}
        fields = {k: v for k, v in raw_config["model_config"].items() if k in valid}
        config = ModelConfig(**fields)
    elif isinstance(raw_config, dict):
        valid = {f.name for f in dataclasses.fields(ModelConfig)}
        fields = {k: v for k, v in raw_config.items() if k in valid}
        config = ModelConfig(**fields)
    else:
        logger.warning(
            "No ModelConfig found in checkpoint — using default ModelConfig."
        )
        config = ModelConfig()

    # ── Build model and load weights ────────────────────────────────
    # Rebuild the architecture the checkpoint was trained with. Defaults to
    # "hybrid" so checkpoints written before --arch existed still load.
    arch = (train_args or {}).get("arch", "hybrid")
    config.fusion_mode = (train_args or {}).get("fusion_mode",
                                                getattr(config, "fusion_mode", "token"))
    logger.info("Rebuilding architecture: %s (fusion_mode=%s)", arch, config.fusion_mode)
    model = build_model(arch, config)

    state_key = "model_state_dict"
    if state_key not in checkpoint:
        # Fall back: the checkpoint file *is* a raw state_dict
        logger.info("Checkpoint has no '%s' key — treating file as a raw state dict.", state_key)
        model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint[state_key], strict=False)

    model = model.to(device)
    model.eval()

    # ── Collect metadata ────────────────────────────────────────────
    meta_keys = {"epoch", "best_metric", "best_epoch", "optimizer_state_dict",
                 "scheduler_state_dict", "training_config", "data_config"}
    metadata: Dict[str, Any] = {
        k: v for k, v in checkpoint.items()
        if k in meta_keys
    }
    # The training args are how we reproduce the exact same test split.
    metadata["train_args"] = train_args

    logger.info(
        "Model loaded successfully (epoch=%s, best_metric=%s).",
        metadata.get("epoch", "?"),
        metadata.get("best_metric", "?"),
    )

    return model, config, metadata


# ======================================================================
#  3. Model Summary — Parameter Counts
# ======================================================================

def print_model_summary(model: HybridCNNMamba) -> Dict[str, int]:
    """Print and return parameter counts per model component.

    Args:
        model: The loaded HybridCNNMamba model.

    Returns:
        Dict mapping component names to parameter counts.
    """
    param_counts = model.count_parameters()

    logger.info("=" * 60)
    logger.info("MODEL PARAMETER SUMMARY")
    logger.info("=" * 60)
    for name, count in param_counts.items():
        logger.info("  %-20s %12s", name, f"{count:,}")
    logger.info("=" * 60)

    return param_counts


# ======================================================================
#  4. Test-Set Inference  (collect predictions + probabilities)
# ======================================================================

@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model on the entire dataloader and collect outputs.

    Args:
        model: Trained model in eval mode.
        dataloader: Test DataLoader.
        device: Device to run inference on.
        num_classes: Number of target classes.

    Returns:
        Tuple of (all_labels, all_preds, all_probs) as numpy arrays.
        - all_labels: (N,) int ground-truth labels.
        - all_preds:  (N,) int predicted labels.
        - all_probs:  (N, C) float predicted probabilities.
    """
    model.eval()

    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)

        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_probs.append(probs)

    all_labels_arr = np.concatenate(all_labels)
    all_preds_arr = np.concatenate(all_preds)
    all_probs_arr = np.concatenate(all_probs)

    return all_labels_arr, all_preds_arr, all_probs_arr


# ======================================================================
#  5. Metrics Computation
# ======================================================================

def _expected_calibration_error(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predicted probabilities (of the predicted class) and compares
    the mean predicted confidence to the actual accuracy within each bin.

    Args:
        labels: Ground-truth labels (N,).
        probs: Predicted probabilities (N, C).
        n_bins: Number of equal-width bins.

    Returns:
        ECE as a float in [0, 1].
    """
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accuracies = (preds == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += (n_in_bin / len(labels)) * abs(avg_acc - avg_conf)
    return float(ece)


def _brier_score_multiclass(
    labels: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
) -> float:
    """Compute multi-class Brier score.

    Brier = (1/N) * Σ Σ (p_ij - y_ij)^2, where y is one-hot.

    Args:
        labels: Ground-truth labels (N,).
        probs: Predicted probabilities (N, C).
        num_classes: Number of classes.

    Returns:
        Brier score as a float (lower is better).
    """
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    num_classes: int,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics.

    Metrics computed:
      - Overall: accuracy, balanced accuracy, macro/weighted precision,
        recall, F1, AUC-ROC, MCC, Cohen's kappa, ECE, Brier score.
      - Per-class: precision, recall, F1, support, AUC-ROC.
      - Confusion matrix.

    Args:
        labels: Ground-truth label array (N,).
        preds: Predicted label array (N,).
        probs: Predicted probability array (N, C).
        class_names: List of human-readable class names.
        num_classes: Number of classes.

    Returns:
        Nested dict with all computed metrics.
    """
    metrics: Dict[str, Any] = {}

    # ── Overall metrics ─────────────────────────────────────────────
    metrics["accuracy"] = float(accuracy_score(labels, preds))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, preds))

    metrics["precision_macro"] = float(
        precision_score(labels, preds, average="macro", zero_division=0)
    )
    metrics["precision_weighted"] = float(
        precision_score(labels, preds, average="weighted", zero_division=0)
    )
    metrics["recall_macro"] = float(
        recall_score(labels, preds, average="macro", zero_division=0)
    )
    metrics["recall_weighted"] = float(
        recall_score(labels, preds, average="weighted", zero_division=0)
    )
    metrics["f1_macro"] = float(
        f1_score(labels, preds, average="macro", zero_division=0)
    )
    metrics["f1_weighted"] = float(
        f1_score(labels, preds, average="weighted", zero_division=0)
    )

    # ── Matthews Correlation Coefficient ─────────────────────────────
    metrics["mcc"] = float(matthews_corrcoef(labels, preds))

    # ── Cohen's Kappa ────────────────────────────────────────────────
    metrics["cohens_kappa"] = float(
        cohen_kappa_score(labels, preds, weights=None)
    )
    metrics["cohens_kappa_weighted"] = float(
        cohen_kappa_score(labels, preds, weights="quadratic")
    )

    # ── Calibration metrics ──────────────────────────────────────────
    metrics["ece"] = _expected_calibration_error(labels, probs)
    metrics["brier_score"] = _brier_score_multiclass(labels, probs, num_classes)

    # AUC-ROC (one-vs-rest) — requires probabilities for all classes
    try:
        if num_classes == 2:
            metrics["auc_roc"] = float(
                roc_auc_score(labels, probs[:, 1])
            )
        else:
            metrics["auc_roc"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            )
    except ValueError as exc:
        logger.warning("Could not compute AUC-ROC: %s", exc)
        metrics["auc_roc"] = None

    # ── Per-class metrics ───────────────────────────────────────────
    report = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0,
    )
    per_class: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        if name in report:
            entry: Dict[str, Any] = {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1-score": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            # Per-class AUC-ROC (one-vs-rest)
            try:
                binary_labels = (labels == i).astype(int)
                if binary_labels.sum() > 0 and binary_labels.sum() < len(labels):
                    entry["auc_roc"] = float(
                        roc_auc_score(binary_labels, probs[:, i])
                    )
                else:
                    entry["auc_roc"] = None
            except ValueError:
                entry["auc_roc"] = None
            per_class[name] = entry
    metrics["per_class"] = per_class

    # ── Confusion matrix ────────────────────────────────────────────
    cm = confusion_matrix(labels, preds)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


# ======================================================================
#  6. Visualizations
# ======================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """Generate and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (C, C).
        class_names: Class names for axis labels.
        save_path: Absolute path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def plot_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    num_classes: int,
    save_path: str,
) -> None:
    """Generate and save per-class ROC curves with macro-average.

    For binary classification a single ROC curve is drawn.  For
    multi-class, one curve per class (OVR) plus a macro-average curve.

    Args:
        labels: Ground-truth labels (N,).
        probs: Predicted probabilities (N, C).
        class_names: Human-readable class names.
        num_classes: Number of classes.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if num_classes == 2:
        # Binary case — single ROC curve
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    else:
        # One-vs-Rest per class
        all_fpr: List[np.ndarray] = []
        all_tpr: List[np.ndarray] = []
        mean_fpr = np.linspace(0, 1, 200)

        for i, name in enumerate(class_names):
            binary_labels = (labels == i).astype(int)
            # Skip classes not present in test set
            if binary_labels.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
            roc_auc_i = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, alpha=0.7,
                    label=f"{name} (AUC = {roc_auc_i:.3f})")
            all_tpr.append(np.interp(mean_fpr, fpr, tpr))
            all_tpr[-1][0] = 0.0

        # Macro-average ROC
        if all_tpr:
            mean_tpr = np.mean(all_tpr, axis=0)
            mean_tpr[-1] = 1.0
            macro_auc = auc(mean_fpr, mean_tpr)
            ax.plot(
                mean_fpr, mean_tpr, color="navy", lw=2.5, linestyle="--",
                label=f"Macro-avg (AUC = {macro_auc:.3f})",
            )

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved to %s", save_path)


def plot_precision_recall_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    num_classes: int,
    save_path: str,
) -> None:
    """Generate and save per-class Precision-Recall curves.

    Args:
        labels: Ground-truth labels (N,).
        probs: Predicted probabilities (N, C).
        class_names: Human-readable class names.
        num_classes: Number of classes.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, name in enumerate(class_names):
        binary_labels = (labels == i).astype(int)
        if binary_labels.sum() == 0:
            continue
        precision_vals, recall_vals, _ = precision_recall_curve(
            binary_labels, probs[:, i],
        )
        pr_auc = auc(recall_vals, precision_vals)
        ax.plot(
            recall_vals, precision_vals, lw=1.5, alpha=0.7,
            label=f"{name} (AP = {pr_auc:.3f})",
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Precision-Recall curves saved to %s", save_path)


# ======================================================================
#  7. XAI Heatmap Generation
# ======================================================================

def _tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW float tensor (ImageNet-normalised) to HWC uint8.

    Applies the inverse of the standard ImageNet normalisation
    (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) so the
    resulting image looks natural.

    Args:
        tensor: Image tensor (C, H, W) in [~-2, ~3] range.

    Returns:
        Numpy array (H, W, 3) in [0, 255] uint8.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
    img = img * std + mean  # De-normalise
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def generate_xai_heatmaps(
    model: HybridCNNMamba,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
    num_samples: int,
    save_dir: str,
) -> None:
    """Generate and save Grad-CAM++ heatmap visualisations.

    For each sample a 1×4 subplot grid is saved:
        [Original | CNN Grad-CAM++ | Mamba Grad-CAM++ | Combined]

    Args:
        model: Loaded HybridCNNMamba model.
        dataloader: Test DataLoader.
        class_names: Human-readable class names.
        device: Computation device.
        num_samples: Number of samples to visualise.
        save_dir: Directory to save heatmap images.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Instantiate Grad-CAM++
    gradcam = GradCAMPlusPlus(model)

    saved = 0
    model.eval()

    for images, labels in dataloader:
        for i in range(images.size(0)):
            if saved >= num_samples:
                break

            single_image = images[i].unsqueeze(0).to(device)
            true_label = int(labels[i].item())

            # Run Grad-CAM++ (forward + backward internally)
            try:
                heatmaps = gradcam(single_image, target_class=true_label)
            except Exception as exc:
                logger.warning(
                    "Grad-CAM++ failed for sample %d: %s", saved, exc
                )
                continue

            # Convert input tensor to displayable numpy image
            orig_img = _tensor_to_numpy_image(images[i])

            # Build 1×4 subplot
            titles = ["Original", "CNN Grad-CAM++", "Mamba Grad-CAM++", "Combined"]
            panels = [orig_img]

            for key in ("cnn", "mamba", "combined"):
                if key in heatmaps:
                    overlay = visualize_heatmap(orig_img, heatmaps[key], alpha=0.4)
                    panels.append(overlay)
                else:
                    # Pathway heatmap unavailable — show placeholder
                    placeholder = np.full_like(orig_img, 128)
                    panels.append(placeholder)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            for ax, panel, title in zip(axes, panels, titles):
                ax.imshow(panel)
                ax.set_title(title, fontsize=11)
                ax.axis("off")

            # Predict class for the subtitle
            with torch.no_grad():
                logits = model(single_image)
                pred_label = int(logits.argmax(dim=1).item())

            true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
            pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
            status = "✓" if true_label == pred_label else "✗"
            fig.suptitle(
                f"Sample {saved + 1}  |  True: {true_name}  |  Pred: {pred_name}  {status}",
                fontsize=13, fontweight="bold", y=1.02,
            )

            plt.tight_layout()
            fname = os.path.join(save_dir, f"xai_sample_{saved + 1:03d}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("XAI heatmap saved: %s", fname)

            saved += 1

        if saved >= num_samples:
            break

    # Clean up hooks to free memory
    gradcam.remove_hooks()

    logger.info("Generated %d / %d XAI heatmap panels.", saved, num_samples)


# ======================================================================
#  8. Inference Speed Benchmark
# ======================================================================

@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    num_warmup: int = 20,
    num_iterations: int = 100,
) -> Dict[str, float]:
    """Benchmark average inference time per image.

    Performs ``num_warmup`` forward passes to warm up CUDA kernels,
    then measures ``num_iterations`` passes and reports statistics.

    Args:
        model: Model in eval mode.
        device: Device to benchmark on.
        image_size: Spatial size of the dummy input.
        num_warmup: Number of warmup iterations (not timed).
        num_iterations: Number of timed iterations.

    Returns:
        Dict with ``mean_ms``, ``std_ms``, ``throughput_fps`` keys.
    """
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    # Warmup — let CUDA JIT / cuDNN autotuner settle
    for _ in range(num_warmup):
        _ = model(dummy)

    # Synchronise before timing (important for CUDA)
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings: List[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)  # ms

    timings_arr = np.array(timings)
    mean_ms = float(timings_arr.mean())
    std_ms = float(timings_arr.std())
    throughput = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    logger.info("=" * 60)
    logger.info("INFERENCE SPEED BENCHMARK (%s)", device)
    logger.info("=" * 60)
    logger.info("  Input size       : 1 × 3 × %d × %d", image_size, image_size)
    logger.info("  Warmup iters     : %d", num_warmup)
    logger.info("  Timed iters      : %d", num_iterations)
    logger.info("  Mean latency     : %.2f ms  (± %.2f ms)", mean_ms, std_ms)
    logger.info("  Throughput       : %.1f images/s", throughput)
    logger.info("=" * 60)

    return {"mean_ms": mean_ms, "std_ms": std_ms, "throughput_fps": throughput}


# ======================================================================
#  9. Summary Table
# ======================================================================

def print_summary_table(
    metrics: Dict[str, Any],
    class_names: List[str],
) -> None:
    """Print a nicely formatted summary table of evaluation metrics.

    Args:
        metrics: Metrics dict as returned by :func:`compute_metrics`.
        class_names: List of class names.
    """
    logger.info("")
    logger.info("=" * 72)
    logger.info("                    EVALUATION RESULTS SUMMARY")
    logger.info("=" * 72)

    # Overall
    logger.info("  Accuracy             : %.4f", metrics["accuracy"])
    logger.info("  Balanced Accuracy    : %.4f", metrics.get("balanced_accuracy", 0))
    logger.info("  Precision (macro)    : %.4f", metrics["precision_macro"])
    logger.info("  Precision (weighted) : %.4f", metrics["precision_weighted"])
    logger.info("  Recall    (macro)    : %.4f", metrics["recall_macro"])
    logger.info("  Recall    (weighted) : %.4f", metrics["recall_weighted"])
    logger.info("  F1-Score  (macro)    : %.4f", metrics["f1_macro"])
    logger.info("  F1-Score  (weighted) : %.4f", metrics["f1_weighted"])
    if metrics.get("auc_roc") is not None:
        logger.info("  AUC-ROC   (macro)    : %.4f", metrics["auc_roc"])
    else:
        logger.info("  AUC-ROC   (macro)    : N/A")
    logger.info("  MCC                  : %.4f", metrics.get("mcc", 0))
    logger.info("  Cohen's Kappa        : %.4f", metrics.get("cohens_kappa", 0))
    logger.info("  Cohen's Kappa (wt)   : %.4f", metrics.get("cohens_kappa_weighted", 0))
    logger.info("  ECE                  : %.4f", metrics.get("ece", 0))
    logger.info("  Brier Score          : %.4f", metrics.get("brier_score", 0))

    # Per-class
    logger.info("-" * 72)
    logger.info(
        "  %-16s %10s %10s %10s %10s %10s",
        "Class", "Precision", "Recall", "F1", "AUC", "Support",
    )
    logger.info("-" * 72)
    per_class = metrics.get("per_class", {})
    for name in class_names:
        if name in per_class:
            c = per_class[name]
            auc_str = f"{c['auc_roc']:.4f}" if c.get('auc_roc') is not None else "N/A"
            logger.info(
                "  %-16s %10.4f %10.4f %10.4f %10s %10d",
                name, c["precision"], c["recall"], c["f1-score"],
                auc_str, c["support"],
            )
    logger.info("=" * 72)


# ======================================================================
#  10. Main
# ======================================================================

def main() -> None:
    """Entry point for the evaluation script."""
    args = parse_args()

    # ── Output directory ────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=str(output_dir))

    logger.info("Evaluation started.")
    logger.info("Arguments: %s", vars(args))

    # ── Device selection ────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device.")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available — running on CPU.")

    # ── Load model ──────────────────────────────────────────────────
    model, config, meta = load_model(args.checkpoint, device)

    # ── Model summary ───────────────────────────────────────────────
    param_counts = print_model_summary(model)

    # ── Create dataloaders ──────────────────────────────────────────
    logger.info("Loading dataset from %s ...", args.data_root)

    # Reproduce the training split exactly. Silently defaulting these would
    # evaluate the model on images it was trained on.
    train_args = meta.get("train_args") or {}

    def _split_arg(name: str, fallback):
        cli = getattr(args, name, None)
        if cli is not None:
            return cli
        if name in train_args:
            return train_args[name]
        logger.warning(
            "Checkpoint records no '%s'; falling back to %r. If this differs "
            "from training, the test set will not match.", name, fallback,
        )
        return fallback

    seed = _split_arg("seed", 42)
    train_ratio = _split_arg("train_ratio", 0.7)
    val_ratio = _split_arg("val_ratio", 0.15)
    stain_normalize = _split_arg("stain_normalize", True)
    use_groups = _split_arg("use_groups", True)
    groups_file = args.groups_file if args.groups_file is not None else train_args.get("groups_file")
    # An explicit split on the CLI wins (external tests); otherwise reuse the
    # split file the model was trained with, if any.
    split_file = args.split_file if args.split_file is not None else train_args.get("split_file")

    _, _, test_loader, info = create_dataloaders(
        data_root=args.data_root,
        image_size=config.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        stain_normalize=stain_normalize,
        magnification=args.magnification,
        binary=args.binary,
        seed=seed,
        use_groups=use_groups,
        groups_file=groups_file,
        split_file=split_file,
    )

    logger.info(
        "Split reproduced: type=%s seed=%s ratios=%.2f/%.2f stain_normalize=%s",
        info["split_type"], seed, train_ratio, val_ratio, stain_normalize,
    )
    if info["split_type"] == "explicit":
        logger.info("Evaluating on explicit split %s (test %d images).",
                    split_file, info["test_size"])
    elif info["split_type"] != "grouped":
        logger.warning(
            "Evaluating on a PER-FILE split — augmented siblings of test images "
            "were seen during training. These metrics are not reportable."
        )

    num_classes = info["num_classes"]
    class_names: List[str] = info["class_names"]

    # Sanity check — ensure num_classes matches the model head
    if num_classes != config.num_classes:
        logger.warning(
            "Dataset has %d classes but model was built for %d classes. "
            "Metrics may be unreliable.",
            num_classes, config.num_classes,
        )

    if info["test_size"] == 0:
        logger.error("Test set is empty. Cannot evaluate. Exiting.")
        sys.exit(1)

    logger.info(
        "Dataset info: %d classes, test_size=%d, class_names=%s",
        num_classes, info["test_size"], class_names,
    )

    # ── Run inference ───────────────────────────────────────────────
    labels, preds, probs = run_inference(model, test_loader, device, num_classes)
    logger.info("Inference complete on %d test samples.", len(labels))

    # ── Compute metrics ─────────────────────────────────────────────
    metrics = compute_metrics(labels, preds, probs, class_names, num_classes)
    print_summary_table(metrics, class_names)

    # ── Visualizations ──────────────────────────────────────────────
    cm = np.array(metrics["confusion_matrix"])

    plot_confusion_matrix(
        cm, class_names,
        save_path=str(output_dir / "confusion_matrix.png"),
    )

    plot_roc_curves(
        labels, probs, class_names, num_classes,
        save_path=str(output_dir / "roc_curves.png"),
    )

    plot_precision_recall_curves(
        labels, probs, class_names, num_classes,
        save_path=str(output_dir / "precision_recall_curves.png"),
    )

    # ── XAI Heatmaps ───────────────────────────────────────────────
    if args.num_xai_samples > 0:
        logger.info(
            "Generating Grad-CAM++ heatmaps for %d samples...",
            args.num_xai_samples,
        )
        xai_dir = str(output_dir / "xai_heatmaps")
        generate_xai_heatmaps(
            model=model,
            dataloader=test_loader,
            class_names=class_names,
            device=device,
            num_samples=args.num_xai_samples,
            save_dir=xai_dir,
        )

    # ── Inference benchmark ─────────────────────────────────────────
    bench = benchmark_inference(
        model, device, image_size=config.image_size,
    )

    # ── Save everything to JSON ─────────────────────────────────────
    results = {
        "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": metrics["confusion_matrix"],
        "parameter_counts": param_counts,
        "inference_benchmark": bench,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "test_size": info["test_size"],
        "num_classes": num_classes,
        "class_names": class_names,
    }

    json_path = str(output_dir / "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, default=str)
    logger.info("Full results saved to %s", json_path)

    # Persist per-sample predictions. Aggregate metrics cannot support a paired
    # significance test: comparing two models needs to know *which* samples each
    # got right. McNemar's test in aggregate_results.py consumes this file, and
    # without it the statistical-validation gap cannot be closed after the fact.
    # Small (a few hundred KB) and it makes every reported number auditable.
    preds_path = output_dir / "predictions.npz"
    np.savez_compressed(
        preds_path,
        labels=labels.astype(np.int16),
        preds=preds.astype(np.int16),
        probs=probs.astype(np.float32),
        class_names=np.array(class_names, dtype=object),
        checkpoint=np.array(args.checkpoint, dtype=object),
    )
    logger.info("Per-sample predictions saved to %s", preds_path)

    logger.info("Evaluation complete. All outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
