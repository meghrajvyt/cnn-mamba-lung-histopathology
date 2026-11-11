"""Tests for tools/cluster_stats.py on synthetic data with known answers."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from cluster_stats import cluster_mean_test, cluster_means  # noqa: E402


def test_cluster_means_weights_clusters_not_images():
    # Patient 1 has 4 images at +1, patient 2 has 1 image at -1.
    v = [1, 1, 1, 1, -1]
    c = [1, 1, 1, 1, 2]
    cm = cluster_means(v, c)
    assert list(cm["ids"]) == [1, 2]
    assert np.allclose(cm["means"], [1.0, -1.0])
    assert list(cm["sizes"]) == [4, 1]
    r = cluster_mean_test(v, c, n=2000)
    assert r["mean"] == pytest.approx(0.0)  # image mean would be 0.6
    assert r["n_clusters"] == 2 and r["n_images"] == 5


def test_clear_positive_effect_is_detected():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(40), 15)
    # Patient-level effect +0.1 with patient random effects and image noise.
    v = 0.1 + rng.normal(0, 0.05, 40)[clusters] + rng.normal(0, 0.05, len(clusters))
    r = cluster_mean_test(v, clusters, n=5000)
    assert r["p"] < 0.001
    assert r["ci"][0] > 0 and r["ci"][0] < r["mean"] < r["ci"][1]
    assert r["frac_clusters_positive"] > 0.9


def test_null_with_strong_clustering_is_not_overconfident():
    # Few clusters, large between-cluster variance, no true effect: an
    # image-level test would be badly anti-conservative; the cluster test
    # must not reject at 5% in most seeds.
    rejections = 0
    for s in range(40):
        rng = np.random.default_rng(100 + s)
        clusters = np.repeat(np.arange(8), 30)
        v = rng.normal(0, 1, 8)[clusters] + rng.normal(0, 0.1, len(clusters))
        rejections += cluster_mean_test(v, clusters, n=999, seed=s)["p"] < 0.05
    assert rejections <= 6  # nominal 2 of 40; generous bound


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cluster_means([1, 2, 3], [1, 2])
