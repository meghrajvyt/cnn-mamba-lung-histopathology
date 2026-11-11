"""Cluster-level summary statistics (numpy only).

Used for per-image quantities that must be analysed with the patient (or
source group) as the unit, e.g. the difference in insertion − deletion AUC
between a saliency map and a random map on LungHist700.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def cluster_means(values: Sequence[float], clusters: Sequence) -> Dict:
    """Mean of `values` within each cluster. Returns ids, means and sizes."""
    v = np.asarray(values, float)
    c = np.asarray(clusters)
    if v.shape != c.shape:
        raise ValueError(f"{v.shape} values vs {c.shape} clusters")
    uniq, inv = np.unique(c, return_inverse=True)
    size = np.bincount(inv, minlength=len(uniq))
    means = np.bincount(inv, weights=v, minlength=len(uniq)) / size
    return {"ids": uniq, "means": means, "sizes": size}


def cluster_mean_test(values: Sequence[float], clusters: Sequence, n: int = 10000,
                      seed: int = 42) -> Dict[str, float]:
    """Test whether the mean of per-cluster means differs from zero.

    Each cluster contributes one number (the mean of its images), so clusters
    are weighted equally and images within a cluster are not treated as
    independent. CI: percentile bootstrap over clusters. p: two-sided
    sign-flip randomisation test on the per-cluster means (exchangeable signs
    under H0 of a distribution symmetric about zero).
    """
    cm = cluster_means(values, clusters)
    m = cm["means"]
    k = len(m)
    rng = np.random.default_rng(seed)
    boots = m[rng.integers(0, k, size=(n, k))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    t_obs = abs(m.mean())
    t_perm = np.abs((rng.choice([-1.0, 1.0], size=(n, k)) * m).mean(1))
    p = (1 + np.sum(t_perm >= t_obs - 1e-12)) / (n + 1)
    return {"n_clusters": int(k), "n_images": int(len(np.asarray(values))),
            "mean": float(m.mean()), "ci": [float(lo), float(hi)], "p": float(p),
            "frac_clusters_positive": float((m > 0).mean())}
