#!/usr/bin/env python3
"""Quantitative explainability metrics for one trained checkpoint.

Two measurements, both on the checkpoint's own test split:

1. PATHWAY ATTRIBUTION (ResidualHybrid only, ``hybrid_res*``)
   The Mamba branch enters as f = f_cnn + delta. For every test image this
   records the relative magnitude ||delta|| / ||f_cnn||, then re-runs the model
   with delta forced to zero. The accuracy change, the number of flipped
   predictions (with McNemar's test) and the drop in predicted-class probability
   quantify how much the decision actually depends on the Mamba pathway, rather
   than asserting it from a heatmap.

2. SALIENCY FAITHFULNESS (every architecture)
   Deletion and insertion curves (Petsiuk et al., RISE, BMVC 2018) for
   Grad-CAM++ maps, compared with a random map:
     deletion   remove the most salient pixels first; the predicted-class
                probability should fall fast (lower AUC is better)
     insertion  start from a blurred image and restore the most salient pixels
                first; the probability should rise fast (higher AUC is better)
   For two-pathway models the CNN, Mamba and combined maps are scored
   separately. Each map is compared with the random baseline per image
   (paired Wilcoxon signed-rank test).

The image subset (stratified by class) and the random maps are seeded and
depend only on the test split, so every model is scored on the same images
against the same random baseline.

Deletion canvas. The default replaces deleted pixels with the blurred image,
the same canvas insertion starts from. The alternative, a flat mean colour, is
out of distribution for H&E: in a pilot on a cnn_only checkpoint (90 images;
the archived run trained on the defective Macenko images, so before any
result of the current study existed) Grad-CAM++ beat random on insertion but
lost to it on grey-canvas deletion,
the signature of the classifier reacting to the grey patches rather than to the
removed tissue. It is kept as a secondary analysis (--deletion_baseline mean).
With the blur canvas, deleting the top fraction f of map M leaves exactly the
image obtained by inserting the top 1 - f of the inverted map, so deletion
AUC(M) = insertion AUC(1 - M) and insertion - deletion compares a map with
its own inversion.

Usage:
    python3 tools/xai_metrics.py --checkpoint checkpoints/<run>/best_model.pth \
        [--data_root ./data_macenko] [--num_samples 300] [--steps 25]

Outputs (default results/<run>/xai_metrics/):
    xai_metrics.json            summary statistics
    xai_per_sample.npz          per-image AUCs, curves and attribution values
    faithfulness_curves.png     mean deletion / insertion curves
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compare_models import mcnemar  # noqa: E402
from data.dataset import create_dataloaders  # noqa: E402
from evaluate import load_model  # noqa: E402

logger = logging.getLogger("xai_metrics")

ToBCHW = Callable[[torch.Tensor], torch.Tensor]


# ======================================================================
#  Target layers
# ======================================================================

def _tokens_to_bchw(t: torch.Tensor, prefix: int = 0) -> torch.Tensor:
    """(B, L, D) token sequence on a square grid -> (B, D, H, W)."""
    t = t[:, prefix:]
    B, L, D = t.shape
    g = int(round(math.sqrt(L)))
    if g * g != L:
        raise ValueError(f"cannot reshape {L} tokens to a square grid")
    return t.transpose(1, 2).reshape(B, D, g, g)


def _identity(t: torch.Tensor) -> torch.Tensor:
    return t


def _nhwc_to_bchw(t: torch.Tensor) -> torch.Tensor:
    return t.permute(0, 3, 1, 2)


def target_layers(model: nn.Module, arch: str) -> Dict[str, Tuple[nn.Module, ToBCHW]]:
    """Grad-CAM++ target layer(s) per architecture, with a reshape to (B,C,H,W).

    Two-pathway models get one map per pathway. The choices follow common
    practice: the last spatial feature map before global pooling for CNNs, the
    last block's output tokens for the Mamba pathways, and the last block's
    first LayerNorm for the transformers (as in pytorch-grad-cam).
    """
    if arch in ("hybrid", "hybrid_pooled", "hybrid_res", "hybrid_res_xattn"):
        return {
            "cnn": (model.cnn_pathway.channel_projections[-1], _identity),
            "mamba": (model.mamba_pathway.blocks[-1], _tokens_to_bchw),
        }
    if arch == "cnn_only":
        return {"cnn": (model.pathway.channel_projections[-1], _identity)}
    if arch == "mamba_only":
        return {"mamba": (model.pathway.blocks[-1], _tokens_to_bchw)}
    if arch == "effnet_b0":
        return {"cnn": (model.backbone.bn2, _identity)}
    if arch == "vit_small":
        prefix = int(getattr(model.backbone, "num_prefix_tokens", 1))
        return {"vit": (model.backbone.blocks[-1].norm1,
                        lambda t: _tokens_to_bchw(t, prefix))}
    if arch == "swin_tiny":
        # timm >= 0.9 Swin keeps activations channels-last: (B, H, W, C).
        return {"swin": (model.backbone.layers[-1].blocks[-1].norm1, _nhwc_to_bchw)}
    raise ValueError(f"no Grad-CAM++ target defined for arch {arch!r}")


# ======================================================================
#  Batched Grad-CAM++
# ======================================================================

class BatchGradCAMPP:
    """Grad-CAM++ for a batch, on any number of target layers.

    Gradients are captured with tensor hooks rather than module backward hooks,
    which break on modules containing in-place ops (the CNN projections end in
    ReLU(inplace=True)). In eval mode samples do not interact, so backpropagating
    the sum of each sample's target logit yields per-sample gradients.

    The Grad-CAM++ weighting is identical to models/xai.py:
        alpha = g^2 / (2 g^2 + sum_ij A g^3),  w_k = sum_ij alpha relu(g)
        cam   = relu(sum_k w_k A_k), min-max normalised per image
    """

    def __init__(self, model: nn.Module, targets: Dict[str, Tuple[nn.Module, ToBCHW]]):
        self.model = model
        self.targets = targets
        self.acts: Dict[str, torch.Tensor] = {}
        self.grads: Dict[str, torch.Tensor] = {}
        self.handles = []
        for name, (module, _) in targets.items():
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            out = out[0] if isinstance(out, tuple) else out
            self.acts[name] = out
            if out.requires_grad:
                out.register_hook(lambda g: self.grads.__setitem__(name, g))
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()

    def __call__(self, x: torch.Tensor, cls: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return {name: (B, h, w) maps in [0, 1]} for target classes ``cls``."""
        self.acts.clear()
        self.grads.clear()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            logits = self.model(x)
            logits.gather(1, cls.view(-1, 1)).sum().backward()

        maps = {}
        for name, (_, to_bchw) in self.targets.items():
            if name not in self.grads:
                raise RuntimeError(f"no gradient reached target layer {name!r}")
            a = to_bchw(self.acts[name].detach()).float()
            g = to_bchw(self.grads[name].detach()).float()
            g2, g3 = g ** 2, g ** 3
            denom = 2 * g2 + (a * g3).sum(dim=(2, 3), keepdim=True) + 1e-8
            alpha = g2 / denom
            w = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
            cam = F.relu((w * a).sum(dim=1))                     # (B, h, w)
            maps[name] = _minmax(cam)
        self.model.zero_grad(set_to_none=True)
        return maps


def _minmax(cam: torch.Tensor) -> torch.Tensor:
    """Per-image min-max to [0, 1]. Relative tolerance, as in models/xai.py:
    Mamba-pathway gradients can be ~1e-7, so an absolute threshold would
    discard informative maps as constant."""
    flat = cam.flatten(1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    rng = hi - lo
    scale = torch.maximum(hi.abs(), lo.abs()).clamp_min(1e-30)
    ok = rng > torch.finfo(cam.dtype).eps * scale
    out = torch.where(ok, (flat - lo) / rng.clamp_min(1e-30), torch.zeros_like(flat))
    return out.view_as(cam)


def upsample(maps: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(maps.unsqueeze(1), size=size, mode="bilinear",
                         align_corners=False).squeeze(1)


# ======================================================================
#  Deletion / insertion
# ======================================================================

def gaussian_blur(x: torch.Tensor, ksize: int = 11, sigma: float = 5.0) -> torch.Tensor:
    """Depthwise Gaussian blur on (B, C, H, W); RISE's insertion canvas."""
    ax = torch.arange(ksize, device=x.device, dtype=x.dtype) - (ksize - 1) / 2
    k1 = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    k2 = (k1[:, None] * k1[None, :]).expand(x.shape[1], 1, ksize, ksize)
    return F.conv2d(F.pad(x, [ksize // 2] * 4, mode="reflect"), k2,
                    groups=x.shape[1])


@torch.no_grad()
def perturbation_curves(
    model: nn.Module, x: torch.Tensor, sal: torch.Tensor, cls: int,
    steps: int, del_base: torch.Tensor, ins_base: torch.Tensor,
    chunk: int, generator: torch.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deletion and insertion probability curves for one image and one map.

    Args:
        x: (C, H, W) normalised image.  sal: (H, W) saliency.
        del_base / ins_base: (C, H, W) replacement / starting canvases.

    Returns:
        (deletion, insertion) arrays of length steps + 1, the softmax
        probability of ``cls`` after removing / inserting a fraction k/steps
        of the pixels in saliency order.
    """
    C, H, W = x.shape
    n = H * W
    # A tiny seeded jitter breaks ties (ReLU zeros, bilinear plateaus) randomly
    # instead of by raster order, which would bias towards the top-left.
    jitter = torch.rand(n, generator=generator).to(sal.device) * 1e-6
    order = torch.argsort(sal.flatten().float() + jitter, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(n, device=order.device)
    counts = torch.round(torch.linspace(0, n, steps + 1, device=x.device)).long()
    masks = (rank.view(1, n) < counts.view(-1, 1)).view(-1, 1, H, W).to(x.dtype)

    deleted = x * (1 - masks) + del_base * masks
    inserted = ins_base * (1 - masks) + x * masks
    batch = torch.cat([deleted, inserted])
    probs = []
    for i in range(0, batch.shape[0], chunk):
        probs.append(F.softmax(model(batch[i:i + chunk]).float(), dim=1)[:, cls])
    probs = torch.cat(probs).cpu().numpy()
    return probs[:steps + 1], probs[steps + 1:]


def curve_auc(curves: np.ndarray) -> np.ndarray:
    """Trapezoidal area under (N, steps+1) curves on x in [0, 1]."""
    return ((curves[:, 1:] + curves[:, :-1]) / 2).mean(axis=1)


# ======================================================================
#  Statistics
# ======================================================================

def bootstrap_mean_ci(v: np.ndarray, n_boot: int = 5000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def summarise(v: np.ndarray) -> Dict[str, float]:
    lo, hi = bootstrap_mean_ci(v)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "ci95": [lo, hi], "median": float(np.median(v))}


# ======================================================================
#  Data
# ======================================================================

def build_test_set(args, train_args: Dict, image_size: int):
    """Reproduce the checkpoint's test split, exactly as evaluate.py does."""
    def pick(name, fallback):
        cli = getattr(args, name, None)
        return cli if cli is not None else train_args.get(name, fallback)

    data_root = args.data_root or train_args.get("data_root") or "./data"
    split_file = args.split_file or train_args.get("split_file")
    _, _, loader, info = create_dataloaders(
        data_root=data_root,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=pick("train_ratio", 0.7),
        val_ratio=pick("val_ratio", 0.15),
        stain_normalize=pick("stain_normalize", True),
        # Not pick(): --seed here seeds the image subset, not the split.
        seed=train_args.get("seed", 42),
        use_groups=pick("use_groups", True),
        groups_file=pick("groups_file", None),
        split_file=split_file,
    )
    logger.info("Test split: %s, %d images, data_root=%s",
                info["split_type"], info["test_size"], data_root)
    return loader, info, data_root


def stratified_subset(labels: np.ndarray, n_total: int, seed: int) -> np.ndarray:
    """Up to ceil(n_total / K) images per class, seeded; sorted positions."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per = math.ceil(n_total / len(classes))
    pick = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        pick.append(rng.choice(idx, size=min(per, len(idx)), replace=False))
    return np.sort(np.concatenate(pick))


# ======================================================================
#  1. Pathway attribution
# ======================================================================

@torch.no_grad()
def pathway_attribution(model: nn.Module, loader, device) -> Optional[Dict]:
    if not hasattr(model, "ablate_mamba"):
        return None
    labels, p_full, p_abl, ratio, cosine = [], [], [], [], []
    try:
        for x, y in loader:
            x = x.to(device)
            model.ablate_mamba = False
            p = F.softmax(model(x).float(), dim=1)
            c = model.last_contributions
            f, d = c["f_local"].float(), c["delta"].float()
            ratio.append((d.norm(dim=1) / f.norm(dim=1).clamp_min(1e-12)).cpu())
            cosine.append(F.cosine_similarity(d, f, dim=1).cpu())
            model.ablate_mamba = True
            pa = F.softmax(model(x).float(), dim=1)
            labels.append(y)
            p_full.append(p.cpu())
            p_abl.append(pa.cpu())
    finally:
        model.ablate_mamba = False

    y = torch.cat(labels).numpy()
    pf, pa = torch.cat(p_full).numpy(), torch.cat(p_abl).numpy()
    r, cos = torch.cat(ratio).numpy(), torch.cat(cosine).numpy()
    pred_f, pred_a = pf.argmax(1), pa.argmax(1)
    cf, ca = pred_f == y, pred_a == y
    # Probability the full model assigns to its own prediction, before/after.
    rows = np.arange(len(y))
    drop = pf[rows, pred_f] - pa[rows, pred_f]

    per_class = {}
    for k in np.unique(y):
        m = y == k
        per_class[int(k)] = {"n": int(m.sum()),
                             "ratio_mean": float(r[m].mean()),
                             "acc_full": float(cf[m].mean()),
                             "acc_ablated": float(ca[m].mean())}
    return {
        "summary": {
            "n": int(len(y)),
            "acc_full": float(cf.mean()),
            "acc_mamba_ablated": float(ca.mean()),
            "acc_change_when_ablated": float(ca.mean() - cf.mean()),
            "predictions_flipped": int((pred_f != pred_a).sum()),
            "mcnemar_full_vs_ablated": mcnemar(cf, ca),
            "delta_to_cnn_norm_ratio": summarise(r),
            "delta_cnn_cosine": summarise(cos),
            "pred_prob_drop_when_ablated": summarise(drop),
            "ratio_correct_vs_incorrect": {
                "correct": float(r[cf].mean()) if cf.any() else None,
                "incorrect": float(r[~cf].mean()) if (~cf).any() else None,
            },
            "per_class": per_class,
        },
        "arrays": {"attr_labels": y, "attr_prob_full": pf, "attr_prob_ablated": pa,
                   "attr_ratio": r, "attr_cosine": cos},
    }


# ======================================================================
#  2. Faithfulness
# ======================================================================

def faithfulness(model, arch, test_ds, positions, device, args) -> Dict:
    cam = BatchGradCAMPP(model, target_layers(model, arch))
    gen = torch.Generator().manual_seed(args.seed)
    rand_gen = torch.Generator().manual_seed(args.seed + 1)
    names: Optional[List[str]] = None
    curves: Dict[str, Dict[str, list]] = {}
    preds, labels = [], []
    t0 = time.time()

    try:
        for start in range(0, len(positions), args.cam_batch):
            pos = positions[start:start + args.cam_batch]
            xs, ys = zip(*(test_ds[int(p)] for p in pos))
            x = torch.stack(xs).to(device)
            y = torch.as_tensor(ys)
            with torch.no_grad():
                pred = model(x).argmax(1)
            maps = cam(x, pred)                               # explain the decision
            H, W = x.shape[2:]
            maps = {k: upsample(v, (H, W)) for k, v in maps.items()}
            if "cnn" in maps and "mamba" in maps:
                maps["combined"] = _minmax(0.5 * maps["cnn"] + 0.5 * maps["mamba"])
            if args.include_inverse:
                # Sanity check: a map that carries information about the model
                # must beat its own inversion (least salient pixels first).
                maps.update({f"{k}_inv": 1 - v for k, v in list(maps.items())})
            # Random baseline: a coarse seeded map, so its spatial granularity
            # is comparable to a CAM's rather than per-pixel noise. Generated
            # per image in a fixed order, hence identical across models.
            rnd = torch.rand(len(pos), args.random_grid, args.random_grid,
                             generator=rand_gen).to(device)
            maps["random"] = upsample(rnd, (H, W))

            if names is None:
                names = list(maps)
                curves = {k: {"deletion": [], "insertion": []} for k in names}
            del_canvas = (torch.zeros_like(x) if args.deletion_baseline == "mean"
                          else gaussian_blur(x))
            ins_canvas = gaussian_blur(x)
            for i in range(len(pos)):
                for k in names:
                    d, ins = perturbation_curves(
                        model, x[i], maps[k][i], int(pred[i]), args.steps,
                        del_canvas[i], ins_canvas[i], args.batch_size, gen)
                    curves[k]["deletion"].append(d)
                    curves[k]["insertion"].append(ins)
            preds.append(pred.cpu())
            labels.append(y)
            done = min(start + args.cam_batch, len(positions))
            logger.info("faithfulness %d/%d (%.0fs)", done, len(positions), time.time() - t0)
    finally:
        cam.remove()

    preds_np = torch.cat(preds).numpy()
    labels_np = torch.cat(labels).numpy()
    arrays = {"faith_positions": positions, "faith_labels": labels_np,
              "faith_preds": preds_np}
    summary: Dict[str, Dict] = {}
    auc = {}
    for k in names:
        dc = np.stack(curves[k]["deletion"])
        ic = np.stack(curves[k]["insertion"])
        arrays[f"{k}_deletion_curves"] = dc
        arrays[f"{k}_insertion_curves"] = ic
        auc[k] = {"deletion": curve_auc(dc), "insertion": curve_auc(ic)}
        auc[k]["ins_minus_del"] = auc[k]["insertion"] - auc[k]["deletion"]
        for m, v in auc[k].items():
            arrays[f"{k}_{m}_auc"] = v
    for k in names:
        summary[k] = {}
        for m in ("deletion", "insertion", "ins_minus_del"):
            s = summarise(auc[k][m])
            if k != "random":
                diff = auc[k][m] - auc["random"][m]
                s["minus_random"] = summarise(diff)
                s["wilcoxon_p_vs_random"] = wilcoxon_p(auc[k][m], auc["random"][m])
            summary[k][m] = s
    return {"summary": {"n": int(len(positions)),
                        "accuracy_on_subset": float((preds_np == labels_np).mean()),
                        "maps": summary},
            "arrays": arrays, "names": names, "curves": curves}


def plot_curves(curves: Dict, names: List[str], steps: int, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, steps + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, mode in zip(axes, ("deletion", "insertion")):
        for k in names:
            c = np.stack(curves[k][mode])
            a = curve_auc(c).mean()
            ax.plot(x, c.mean(0), ls="--" if k == "random" else "-",
                    label=f"{k} (AUC {a:.3f})")
        ax.set_xlabel(f"fraction of pixels {'removed' if mode == 'deletion' else 'restored'}")
        ax.set_ylabel("predicted-class probability")
        ax.set_title(mode)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ======================================================================
#  Main
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default=None,
                   help="Default: the data_root the checkpoint was trained with.")
    p.add_argument("--split_file", "--split-file", dest="split_file", default=None)
    p.add_argument("--output_dir", default=None,
                   help="Default: results/<run>/xai_metrics")
    p.add_argument("--num_samples", type=int, default=300,
                   help="Images for the faithfulness test (stratified by class).")
    p.add_argument("--steps", type=int, default=25,
                   help="Points per deletion/insertion curve, minus one.")
    p.add_argument("--deletion_baseline", choices=("blur", "mean"), default="blur",
                   help="'blur' replaces deleted pixels with a blurred copy "
                        "(primary); 'mean' with the normalisation mean, a flat "
                        "grey that is out of distribution for H&E (secondary).")
    p.add_argument("--random_grid", type=int, default=14)
    p.add_argument("--batch_size", type=int, default=64,
                   help="Forward batch for perturbed images and attribution.")
    p.add_argument("--cam_batch", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include_inverse", action="store_true",
                   help="Also score each map inverted (sanity check).")
    p.add_argument("--skip_attribution", action="store_true")
    p.add_argument("--skip_faithfulness", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def pick_device(name: Optional[str]) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    device = pick_device(args.device)
    ckpt = Path(args.checkpoint)
    run = ckpt.parent.name
    out = Path(args.output_dir) if args.output_dir else ROOT / "results" / run / "xai_metrics"
    out.mkdir(parents=True, exist_ok=True)

    model, config, meta = load_model(str(ckpt), device)
    model.eval()
    train_args = meta.get("train_args") or {}
    arch = train_args.get("arch", "hybrid")
    logger.info("run=%s arch=%s device=%s", run, arch, device)

    loader, info, data_root = build_test_set(args, train_args, config.image_size)
    result: Dict = {"run": run, "arch": arch, "checkpoint": str(ckpt),
                    "data_root": str(data_root), "split_type": info["split_type"],
                    "test_size": info["test_size"], "class_names": info["class_names"],
                    "settings": {k: getattr(args, k) for k in
                                 ("num_samples", "steps", "deletion_baseline",
                                  "random_grid", "seed", "include_inverse")}}
    arrays: Dict[str, np.ndarray] = {}

    if not args.skip_attribution:
        attr = pathway_attribution(model, loader, device)
        if attr is None:
            logger.info("pathway attribution: n/a for %s (no residual Mamba branch)", arch)
            result["pathway_attribution"] = None
        else:
            result["pathway_attribution"] = attr["summary"]
            arrays.update(attr["arrays"])
            s = attr["summary"]
            logger.info("pathway attribution: acc %.4f -> %.4f with Mamba ablated, "
                        "%d flips, McNemar p=%.3g, ||delta||/||f_cnn|| = %.4f",
                        s["acc_full"], s["acc_mamba_ablated"], s["predictions_flipped"],
                        s["mcnemar_full_vs_ablated"]["p_value"],
                        s["delta_to_cnn_norm_ratio"]["mean"])

    if not args.skip_faithfulness:
        test_ds = loader.dataset
        labels = np.array([test_ds.dataset.samples[i][1] for i in test_ds.indices])
        positions = stratified_subset(labels, args.num_samples, args.seed)
        faith = faithfulness(model, arch, test_ds, positions, device, args)
        result["faithfulness"] = faith["summary"]
        arrays.update(faith["arrays"])
        root = Path(data_root).resolve()
        result["faithfulness_images"] = [
            str(Path(test_ds.dataset.samples[test_ds.indices[int(p)]][0]).resolve()
                .relative_to(root)) for p in positions]
        plot_curves(faith["curves"], faith["names"], args.steps,
                    out / "faithfulness_curves.png", f"{run} ({arch})")
        for k, s in faith["summary"]["maps"].items():
            logger.info("%-8s deletion %.4f  insertion %.4f  (ins-del %.4f)%s", k,
                        s["deletion"]["mean"], s["insertion"]["mean"],
                        s["ins_minus_del"]["mean"],
                        "" if k == "random" else
                        f"  p_vs_random(ins-del)={s['ins_minus_del']['wilcoxon_p_vs_random']}")

    (out / "xai_metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out / "xai_per_sample.npz", **arrays)
    logger.info("written to %s", out)


if __name__ == "__main__":
    main()
