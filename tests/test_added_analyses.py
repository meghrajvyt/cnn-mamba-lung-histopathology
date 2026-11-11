"""Tables of the analyses added after the pre-submission review (tools/paper_tables.py
T10, T11; docs/added_analyses_protocol.md).

Each is checked on a copy of the results tree in which the new outputs are
links to the primary ones: the 7 x 7 re-scoring then has to reproduce the
primary faithfulness statistics exactly, and the learning-rate sensitivity
analysis, with EfficientNet-B0's "repeated" runs linked to its primary runs,
has to reproduce the primary paired tests. Slow (paper_tables.py runs twice).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import paper_tables as pt  # noqa: E402

NEEDED = [ROOT / "lc25000_image_groups.csv", ROOT / "logs/selected_hybrid.txt",
          ROOT / "logs/lh700_lr_selection.txt"]
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not all(p.exists() for p in NEEDED),
                                 reason="untracked inputs missing")]
PRIMARY_SENS = (  # the record format written by scripts/queue_review.sh
    "# hybrid_res lr=3e-3 best_val_acc=0.70 (pre-specified choice lr=1e-3 best_val_acc=0.75)\n"
    "sensitivity: hybrid_res lr=1e-3 (2026-10-20 01:05 UTC); lr=3e-3 is not better\n"
    "# cnn_only lr=3e-3 best_val_acc=0.7810218995505006 (pre-specified choice lr=1e-3 "
    "best_val_acc=0.7810218995505006)\n"
    "sensitivity: cnn_only lr=1e-3 (2026-10-20 01:09 UTC); lr=3e-3 is not better\n"
    "# effnet_b0 lr=3e-3 best_val_acc=0.80 (pre-specified choice lr=1e-3 best_val_acc=0.69)\n"
    "sensitivity: effnet_b0 lr=3e-3 (2026-10-20 01:12 UTC); its LungHist700 runs are repeated\n"
    "# vit_small lr=3e-3 best_val_acc=0.70 (pre-specified choice lr=1e-3 best_val_acc=0.77)\n"
    "sensitivity: vit_small lr=1e-3 (2026-10-20 01:20 UTC); lr=3e-3 is not better\n"
    "# swin_tiny lr=3e-3 best_val_acc=0.70 (pre-specified choice lr=1e-3 best_val_acc=0.79)\n"
    "sensitivity: swin_tiny lr=1e-3 (2026-10-20 01:30 UTC); lr=3e-3 is not better\n"
    "# vmamba_tiny lr=3e-3 best_val_acc=0.70 (pre-specified choice lr=1e-3 best_val_acc=0.82)\n"
    "sensitivity: vmamba_tiny lr=1e-3 (2026-10-20 07:48 UTC); lr=3e-3 is not better\n")


def link_tree(tmp: Path) -> Path:
    """tmp/results/<run>/<entry> -> the real entries, without the outputs of the
    added analyses (the tests put links to the primary ones in their place);
    tmp/logs holds copies of the records, never links, so that a test cannot
    write into the real ones; tmp/splits is a link."""
    res = tmp / "results"
    for d in (ROOT / "results").iterdir():
        if not d.is_dir() or d.name.startswith(pt.SENS_PREFIX):
            continue
        (res / d.name).mkdir(parents=True)
        for e in d.iterdir():
            if not e.name.endswith("_rg7"):
                os.symlink(e, res / d.name / e.name)
    (tmp / "logs").mkdir()
    for f in (ROOT / "logs").glob("*.txt"):
        if f.name != "lh700_lr_sensitivity.txt":
            shutil.copy(f, tmp / "logs" / f.name)
    os.symlink(ROOT / "splits", tmp / "splits")
    return res


def build(res: Path, out: Path) -> dict:
    r = subprocess.run([sys.executable, str(ROOT / "tools/paper_tables.py"), "--mag", "all",
                        "--results", str(res), "--splits", str(res.parent / "splits"),
                        "--out", str(out)], cwd=ROOT, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return json.loads((out / "tables.json").read_text())


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("added")
    res = link_tree(tmp)
    # A3: the 7 x 7 outputs are the 14 x 14 ones.
    for d in res.iterdir():
        for sub in ("xai_metrics", "xai_metrics_meandel"):
            if (d / sub).exists():
                os.symlink(ROOT / "results" / d.name / sub, d / f"{sub}_rg7")
    # A2: EfficientNet-B0 "wins" at 3e-3; its repeated runs are its primary runs.
    (tmp / "logs" / "lh700_lr_sensitivity.txt").write_text(PRIMARY_SENS)
    for f in range(5):
        for task in ("", "7c_"):
            src = ROOT / "results" / f"lh700_{task}patientall_f{f}_effnet_b0"
            os.symlink(src, res / f"{pt.SENS_PREFIX}{task}patientall_f{f}_effnet_b0")
    return build(res, tmp / "out")


def strip(tests):
    return [{k: t[k] for k in ("baseline", "diff", "cluster", "mcnemar", "p_holm")}
            for t in tests]


def test_sensitivity_reproduces_the_primary_tests_when_runs_are_identical(tables):
    t10 = tables["T10"]
    assert t10["changed"] == ["effnet_b0"] and t10["pending"] == []
    assert t10["rates"]["effnet_b0"]["sensitivity"] == "3e-3"
    assert t10["rates"]["cnn_only"]["sensitivity"] == "1e-3"  # a tie keeps the smaller rate
    for task, tag in (("3c", "T4"), ("7c", "T5")):
        assert t10[task]["complete"]
        assert strip(t10[task]["tests"]) == strip(tables[tag]["tests"])
        assert (t10[task]["models"]["effnet_b0"]["pooled"]
                == tables[tag]["models"]["effnet_b0"]["pooled"])


def test_rg7_reproduces_the_primary_faithfulness_when_outputs_are_identical(tables):
    t11, t8, t8c = tables["T11"], tables["T8"], tables["T8c"]
    for canvas in ("blur", "mean"):
        lc = t11["lc25000"][canvas]
        assert lc["complete"]
        assert lc["summary"] == t8[f"summary_{canvas}"]
        assert lc["runs"] == t8[canvas]
    for canvas, key in (("blur", "faithfulness"), ("mean", "faithfulness_mean")):
        lh = t11["lunghist700"][canvas]
        assert lh["complete"] and lh["maps"] == t8c[key]


def test_lr_rule_is_enforced(tmp_path):
    """A record that contradicts the protocol's rule is an error, not a table."""
    res = link_tree(tmp_path)
    bad = PRIMARY_SENS.replace("sensitivity: cnn_only lr=1e-3", "sensitivity: cnn_only lr=3e-3")
    (tmp_path / "logs" / "lh700_lr_sensitivity.txt").write_text(bad)
    b = pt.Builder(res, tmp_path / "splits", "hybrid_res", None)
    with pytest.raises(ValueError, match="cnn_only"):
        b.lr_records()
