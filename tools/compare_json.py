#!/usr/bin/env python3
"""Compare two results JSON files value by value.

Used to prove that the paper tables regenerate on any machine from the tracked
per-image predictions (tests/test_reproduction.py).

Usage:
    python3 tools/compare_json.py old.json new.json [--tol 1e-12]

Reports values that differ (numbers beyond the tolerance, anything else
unequal), keys missing from the new file, and keys only in the new file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, List, Tuple


def walk(a: Any, b: Any, path: str, tol: float, out: dict) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k not in b:
                out["missing"].append(f"{path}/{k}")
            else:
                walk(a[k], b[k], f"{path}/{k}", tol, out)
        for k in b:
            if k not in a:
                out["added"].append(f"{path}/{k}")
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["diff"].append((path, f"len {len(a)}", f"len {len(b)}"))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", tol, out)
        return
    out["n"] += 1
    if isinstance(a, bool) or isinstance(b, bool) or not (
            isinstance(a, (int, float)) and isinstance(b, (int, float))):
        if a != b:
            out["diff"].append((path, a, b))
        return
    if math.isnan(a) and math.isnan(b):
        return
    if abs(a - b) > tol * max(1.0, abs(a), abs(b)):
        out["diff"].append((path, a, b))


def compare(old: Any, new: Any, tol: float = 1e-12) -> dict:
    out = {"n": 0, "diff": [], "missing": [], "added": []}
    walk(old, new, "", tol, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--tol", type=float, default=1e-12)
    a = ap.parse_args()
    r = compare(json.load(open(a.old)), json.load(open(a.new)), a.tol)
    print(f"{r['n']} values compared, {len(r['diff'])} differences, "
          f"{len(r['missing'])} keys missing, {len(r['added'])} keys added")
    for p, x, y in r["diff"][:30]:
        print(f"  DIFF {p}: {x!r} -> {y!r}")
    for p in r["missing"][:30]:
        print(f"  MISSING {p}")
    for p in r["added"][:30]:
        print(f"  ADDED {p}")
    return 1 if r["diff"] or r["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
