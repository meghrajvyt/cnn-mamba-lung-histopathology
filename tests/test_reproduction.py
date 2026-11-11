"""Every paper table regenerates from the tracked per-image predictions and
explanation arrays, without a GPU, to the committed values.

Slow (about a minute); run with `-m slow`. Skipped when the inputs that are
not tracked are absent: the LC25000-clean grouping CSV (third-party, fetched
from the URL in tools/compare_groupings.py) and logs/selected_hybrid.txt and
logs/lh700_lr_selection.txt (copied from logs/aws/).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_json import compare  # noqa: E402

NEEDED = [ROOT / "lc25000_image_groups.csv", ROOT / "logs/selected_hybrid.txt",
          ROOT / "logs/lh700_lr_selection.txt"]
# Keys added after the committed snapshot may appear as "added"; nothing may
# change or disappear.
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not all(p.exists() for p in NEEDED),
                                 reason="untracked inputs missing")]


def run(script, out):
    r = subprocess.run([sys.executable, str(ROOT / "tools" / script), "--mag", "all",
                        "--out", str(out)], cwd=ROOT, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


@pytest.mark.parametrize("script,name", [("paper_tables.py", "tables.json"),
                                         ("supp_tables.py", "supp_tables.json")])
def test_tables_regenerate_exactly(tmp_path, script, name):
    run(script, tmp_path)
    old = json.loads((ROOT / "results/_paper" / name).read_text())
    new = json.loads((tmp_path / name).read_text())
    r = compare(old, new, tol=1e-12)
    assert r["n"] > 100
    assert r["diff"] == [], r["diff"][:10]
    assert r["missing"] == [], r["missing"][:10]
