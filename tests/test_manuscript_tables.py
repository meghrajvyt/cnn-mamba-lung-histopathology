"""Tests for tools/manuscript_tables.py."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import manuscript_tables as mt  # noqa: E402


def _pooled(acc):
    return {"pooled": {"accuracy": acc, "accuracy_ci_patient": [acc - 0.05, acc + 0.05],
                       "f1_macro": acc - 0.01, "superclass_accuracy": acc + 0.2}}


FAKE = {
    "T4": {"models": {"hybrid_res": _pooled(0.7308), "cnn_only": _pooled(0.7352)},
           "tests": [{"baseline": "cnn_only", "diff": -0.0043,
                      "cluster": {"ci": [-0.0473, 0.0387], "p": 0.899}, "p_holm": 0.899}]},
    "T5": {"models": {"hybrid_res": _pooled(0.4689), "cnn_only": _pooled(0.4863)},
           "tests": [{"baseline": "cnn_only", "diff": -0.0174,
                      "cluster": {"ci": [-0.0621, 0.0272], "p": 0.508}, "p_holm": 1.0}]},
    "T1": {"hybrid_res": {"42": {"n": 2261}, "1337": {"n": 2195}}},
    "T2": {},
    "summary": {"T1": {"hybrid_res": {k: {"mean": 0.97, "sd": 0.013, "n": 2,
                                          "min": 0.9562, "max": 0.981} for k in
                                      ("accuracy", "balanced_accuracy", "f1_macro", "auc_macro")}}},
}


def test_table2_values_and_minus_sign():
    t = mt.table2(FAKE)
    assert "73.08 (68.1 to 78.1)" in t
    assert "\u22121.74 (\u22126.21 to +2.72)" in t  # typographic minus
    assert "| 0.899 |" in t


def test_table1_test_sizes():
    assert mt.test_sizes(FAKE) == "2,195–2,261"


def test_table1_mean_and_range_not_sd():
    t = mt.table1(FAKE)
    assert "97.00 (95.62–98.10)" in t
    assert "±" not in t
    assert "Mean (minimum–maximum)" in t


def test_table1_single_seed_shows_one_value():
    d = json.loads(json.dumps(FAKE))
    d["summary"]["T1"]["hybrid_res"] = {k: {"mean": 0.9934, "sd": 0.0, "n": 1,
                                            "min": 0.9934, "max": 0.9934}
                                        for k in ("accuracy", "balanced_accuracy",
                                                  "f1_macro", "auc_macro")}
    t = mt.table1(d)
    assert "| 99.34 |" in t and "99.34 (" not in t


def test_inject_is_idempotent(tmp_path):
    js = tmp_path / "t.json"
    js.write_text(json.dumps(FAKE))
    md = tmp_path / "m.md"
    md.write_text("before\n<!-- TABLE2:BEGIN -->\nSTALE_PLACEHOLDER\n<!-- TABLE2:END -->\nafter\n")
    cmd = [sys.executable, str(ROOT / "tools/manuscript_tables.py"), "--tables", str(js),
           "--table", "2", "--inject", str(md)]
    subprocess.run(cmd, check=True, capture_output=True)
    first = md.read_text()
    subprocess.run(cmd, check=True, capture_output=True)
    assert md.read_text() == first
    assert "STALE_PLACEHOLDER" not in first and first.count("Table 2.") == 1
    assert first.startswith("before\n") and first.endswith("after\n")
