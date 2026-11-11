#!/usr/bin/env python3
"""
aggregate_results.py — Combine multi-seed evaluation results for paper reporting.

Usage:
    python aggregate_results.py \
        --results_dirs results/seed_42 results/seed_123 results/seed_456 \
        --output_dir results/aggregated

Reads `evaluation_results.json` from each seed directory and produces:
  - aggregated_metrics.json: mean ± std for every metric across seeds
  - summary_table.txt: human-readable summary table
  - per_seed_comparison.csv: side-by-side seed comparison
  - combined_confusion_matrix.png: summed confusion matrix across seeds
  - combined_roc_curves.png: overlaid ROC curves from all seeds (if probs saved)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Optional plotting imports — graceful fallback if not available
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def load_seed_results(results_dirs: List[str]) -> List[Dict[str, Any]]:
    """Load evaluation_results.json from each seed directory.

    Args:
        results_dirs: List of directories, each containing an
            ``evaluation_results.json`` file.

    Returns:
        List of parsed JSON dicts.
    """
    all_results = []
    for d in results_dirs:
        json_path = os.path.join(d, "evaluation_results.json")
        if not os.path.isfile(json_path):
            print(f"WARNING: {json_path} not found, skipping.")
            continue
        with open(json_path, "r") as f:
            all_results.append(json.load(f))
        print(f"Loaded: {json_path}")
    return all_results


def aggregate_scalar_metrics(
    all_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute mean ± std for all scalar metrics across seeds.

    Args:
        all_results: List of evaluation result dicts from multiple seeds.

    Returns:
        Dict mapping metric name -> {"mean": float, "std": float, "values": list}.
    """
    # Scalar metric keys present in the top-level "metrics" dict
    scalar_keys = [
        "accuracy", "balanced_accuracy",
        "precision_macro", "precision_weighted",
        "recall_macro", "recall_weighted",
        "f1_macro", "f1_weighted",
        "auc_roc", "mcc",
        "cohens_kappa", "cohens_kappa_weighted",
        "ece", "brier_score",
    ]

    aggregated: Dict[str, Dict[str, Any]] = {}
    for key in scalar_keys:
        values = []
        for res in all_results:
            m = res.get("metrics", res)  # handle both nested and flat
            val = m.get(key)
            if val is not None:
                values.append(float(val))
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1) if len(values) > 1 else 0.0),
                "values": values,
            }

    return aggregated


def aggregate_per_class_metrics(
    all_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute mean ± std for per-class metrics across seeds.

    Args:
        all_results: List of evaluation result dicts.

    Returns:
        Nested dict: class_name -> metric_name -> {"mean", "std", "values"}.
    """
    per_class_keys = ["precision", "recall", "f1-score", "auc_roc"]

    # Collect class names from all results
    all_class_names = set()
    for res in all_results:
        m = res.get("metrics", res)
        for name in m.get("per_class", {}):
            all_class_names.add(name)

    aggregated: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for cls_name in sorted(all_class_names):
        aggregated[cls_name] = {}
        for key in per_class_keys:
            values = []
            for res in all_results:
                m = res.get("metrics", res)
                cls_data = m.get("per_class", {}).get(cls_name, {})
                val = cls_data.get(key)
                if val is not None:
                    values.append(float(val))
            if values:
                aggregated[cls_name][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1) if len(values) > 1 else 0.0),
                    "values": values,
                }

    return aggregated


def aggregate_confusion_matrices(
    all_results: List[Dict[str, Any]],
) -> Optional[np.ndarray]:
    """Sum confusion matrices across seeds.

    Args:
        all_results: List of evaluation result dicts.

    Returns:
        Summed confusion matrix as numpy array, or None.
    """
    matrices = []
    for res in all_results:
        cm = res.get("confusion_matrix")
        if cm is None:
            m = res.get("metrics", {})
            cm = m.get("confusion_matrix")
        if cm is not None:
            matrices.append(np.array(cm))

    if not matrices:
        return None
    return sum(matrices)


def aggregate_inference_benchmark(
    all_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Average inference benchmark stats across seeds.

    Args:
        all_results: List of evaluation result dicts.

    Returns:
        Dict with mean/std for latency and throughput.
    """
    latencies = []
    throughputs = []
    for res in all_results:
        bench = res.get("inference_benchmark", {})
        if "mean_ms" in bench:
            latencies.append(bench["mean_ms"])
        if "throughput_fps" in bench:
            throughputs.append(bench["throughput_fps"])

    result: Dict[str, Dict[str, float]] = {}
    if latencies:
        result["latency_ms"] = {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies, ddof=1) if len(latencies) > 1 else 0.0),
        }
    if throughputs:
        result["throughput_fps"] = {
            "mean": float(np.mean(throughputs)),
            "std": float(np.std(throughputs, ddof=1) if len(throughputs) > 1 else 0.0),
        }
    return result


def format_summary_table(
    scalar_agg: Dict[str, Dict[str, float]],
    per_class_agg: Dict[str, Dict[str, Dict[str, float]]],
    n_seeds: int,
) -> str:
    """Format a human-readable summary table.

    Args:
        scalar_agg: Aggregated scalar metrics.
        per_class_agg: Aggregated per-class metrics.
        n_seeds: Number of seeds.

    Returns:
        Formatted table string.
    """
    lines = []
    lines.append("=" * 72)
    lines.append(f"  AGGREGATED RESULTS (n={n_seeds} seeds)")
    lines.append("=" * 72)
    lines.append("")

    # Overall metrics
    lines.append("  Overall Metrics:")
    lines.append("  " + "-" * 50)
    metric_display = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "precision_macro": "Precision (macro)",
        "recall_macro": "Recall (macro)",
        "f1_macro": "F1-Score (macro)",
        "auc_roc": "AUC-ROC (macro)",
        "mcc": "MCC",
        "cohens_kappa": "Cohen's Kappa",
        "cohens_kappa_weighted": "Cohen's Kappa (weighted)",
        "ece": "ECE",
        "brier_score": "Brier Score",
    }
    for key, name in metric_display.items():
        if key in scalar_agg:
            m = scalar_agg[key]
            lines.append(f"  {name:28s}: {m['mean']:.4f} ± {m['std']:.4f}")

    lines.append("")
    lines.append("  Per-Class Metrics:")
    lines.append("  " + "-" * 68)
    lines.append(f"  {'Class':16s} {'Precision':>14s} {'Recall':>14s} {'F1':>14s} {'AUC':>14s}")
    lines.append("  " + "-" * 68)

    for cls_name, cls_metrics in per_class_agg.items():
        parts = [f"  {cls_name:16s}"]
        for key in ["precision", "recall", "f1-score", "auc_roc"]:
            if key in cls_metrics:
                m = cls_metrics[key]
                parts.append(f"{m['mean']:.4f}±{m['std']:.4f}")
            else:
                parts.append("     N/A     ")
        lines.append(" ".join(parts))

    lines.append("=" * 72)
    return "\n".join(lines)


def generate_latex_table(
    scalar_agg: Dict[str, Dict[str, float]],
    per_class_agg: Dict[str, Dict[str, Dict[str, float]]],
    n_seeds: int,
) -> str:
    """Generate a LaTeX table for the paper.

    Args:
        scalar_agg: Aggregated scalar metrics.
        per_class_agg: Aggregated per-class metrics.
        n_seeds: Number of seeds.

    Returns:
        LaTeX table string.
    """
    lines = []

    # Table 1: Overall metrics
    lines.append("% Table 1: Overall Classification Performance")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Classification performance of the Hybrid CNN-Mamba model on the LC25000 test set (mean $\pm$ std, $n={n_seeds}$ seeds).}}")
    lines.append(r"\label{tab:overall_results}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{Value} \\")
    lines.append(r"\midrule")

    metric_display = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "precision_macro": "Precision (macro)",
        "recall_macro": "Recall (macro)",
        "f1_macro": "F1-Score (macro)",
        "auc_roc": "AUC-ROC (macro)",
        "mcc": "MCC",
        "cohens_kappa": "Cohen's $\\kappa$",
        "ece": "ECE",
        "brier_score": "Brier Score",
    }
    for key, name in metric_display.items():
        if key in scalar_agg:
            m = scalar_agg[key]
            lines.append(rf"{name} & ${m['mean']:.4f} \pm {m['std']:.4f}$ \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # Table 2: Per-class metrics
    lines.append("% Table 2: Per-Class Performance")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Per-class performance breakdown (mean $\pm$ std, $n={n_seeds}$ seeds).}}")
    lines.append(r"\label{tab:per_class_results}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{AUC-ROC} \\")
    lines.append(r"\midrule")

    cls_display = {
        "lung_n": "Normal",
        "lung_aca": "Adenocarcinoma",
        "lung_scc": "Squamous Cell",
    }
    for cls_name, cls_metrics in per_class_agg.items():
        display_name = cls_display.get(cls_name, cls_name)
        parts = [display_name]
        for key in ["precision", "recall", "f1-score", "auc_roc"]:
            if key in cls_metrics:
                m = cls_metrics[key]
                parts.append(f"${m['mean']:.4f} \\pm {m['std']:.4f}$")
            else:
                parts.append("---")
        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def plot_combined_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    n_seeds: int,
) -> None:
    """Plot a combined (summed) confusion matrix.

    Args:
        cm: Summed confusion matrix array.
        class_names: Class names for labels.
        save_path: Path to save the figure.
        n_seeds: Number of seeds that were summed.
    """
    if not HAS_PLOTTING:
        print("Plotting not available (matplotlib/seaborn not installed).")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(
        f"Combined Confusion Matrix (summed over {n_seeds} seeds)",
        fontsize=14, fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined confusion matrix saved to {save_path}")


def write_csv_comparison(
    all_results: List[Dict[str, Any]],
    results_dirs: List[str],
    save_path: str,
) -> None:
    """Write a CSV with side-by-side per-seed metrics.

    Args:
        all_results: List of evaluation result dicts.
        results_dirs: Corresponding directory names.
        save_path: Path to save the CSV.
    """
    scalar_keys = [
        "accuracy", "balanced_accuracy",
        "precision_macro", "recall_macro",
        "f1_macro", "auc_roc",
        "mcc", "cohens_kappa", "ece", "brier_score",
    ]

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Metric"] + [os.path.basename(d) for d in results_dirs]
        writer.writerow(header)

        for key in scalar_keys:
            row = [key]
            for res in all_results:
                m = res.get("metrics", res)
                val = m.get(key)
                row.append(f"{val:.6f}" if val is not None else "N/A")
            writer.writerow(row)

    print(f"Per-seed comparison saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed evaluation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dirs", nargs="+", required=True,
        help="Directories containing evaluation_results.json from each seed.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/aggregated",
        help="Directory to save aggregated outputs.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all results
    all_results = load_seed_results(args.results_dirs)
    if len(all_results) < 2:
        print(f"WARNING: Only {len(all_results)} seed(s) loaded. "
              "Standard deviations will be 0.")
    if not all_results:
        print("ERROR: No results loaded.")
        sys.exit(1)

    n_seeds = len(all_results)

    # Aggregate metrics
    scalar_agg = aggregate_scalar_metrics(all_results)
    per_class_agg = aggregate_per_class_metrics(all_results)
    bench_agg = aggregate_inference_benchmark(all_results)

    # Summary table
    summary = format_summary_table(scalar_agg, per_class_agg, n_seeds)
    print("\n" + summary)

    summary_path = os.path.join(args.output_dir, "summary_table.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary table saved to {summary_path}")

    # LaTeX table
    latex = generate_latex_table(scalar_agg, per_class_agg, n_seeds)
    latex_path = os.path.join(args.output_dir, "paper_tables.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX tables saved to {latex_path}")

    # Aggregated JSON
    agg_json = {
        "n_seeds": n_seeds,
        "seed_dirs": args.results_dirs,
        "overall_metrics": scalar_agg,
        "per_class_metrics": per_class_agg,
        "inference_benchmark": bench_agg,
    }

    # Get class names and param counts from first result
    class_names = all_results[0].get("class_names", [])
    param_counts = all_results[0].get("parameter_counts", {})
    agg_json["class_names"] = class_names
    agg_json["parameter_counts"] = param_counts

    json_path = os.path.join(args.output_dir, "aggregated_metrics.json")
    with open(json_path, "w") as f:
        json.dump(agg_json, f, indent=2)
    print(f"Aggregated JSON saved to {json_path}")

    # CSV comparison
    csv_path = os.path.join(args.output_dir, "per_seed_comparison.csv")
    write_csv_comparison(all_results, args.results_dirs, csv_path)

    # Combined confusion matrix
    cm = aggregate_confusion_matrices(all_results)
    if cm is not None:
        cm_path = os.path.join(args.output_dir, "combined_confusion_matrix.png")
        plot_combined_confusion_matrix(cm, class_names, cm_path, n_seeds)

    print(f"\nAll aggregated outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
