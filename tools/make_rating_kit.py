#!/usr/bin/env python3
"""Build a blinded heatmap-rating kit for a pathologist.

For a seeded, class-stratified set of test images, renders one panel per
(image, model explanation): the image beside a Grad-CAM++ overlay, labelled
only with a random panel id and the class being explained. Model identities
are kept in a separate key file, so the rater is blind to the architecture.

Outputs (default results/_rating_kit/):
    panels/<panel_id>.png    image | overlay, 512 px each
    rating_form.csv          one row per panel, in randomised order, with
                             empty columns for the rater
    key.csv                  panel_id -> run, map, image, true/predicted class
    README.txt               instructions for the rater

All models must have been evaluated on the same test split (the seed-42
LC25000 runs share it), so every model is rated on the same images.

Usage:
    python3 tools/make_rating_kit.py --runs hybrid_res_d4_224px_s42:combined \
        cnn_only_d4_224px_s42:cnn swin_tiny_d4_224px_s42:swin --per_class 20
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate import load_model  # noqa: E402
from models.xai import visualize_heatmap  # noqa: E402
from xai_metrics import (BatchGradCAMPP, _minmax, build_test_set,  # noqa: E402
                         pick_device, stratified_subset, target_layers, upsample)

README = """\
Heatmap rating — instructions

Each panel shows a lung histology image (left) and the same image with a
heatmap overlay (right). Warm colours (red/yellow) mark the regions the
computer model relied on most when it assigned the class printed above the
panel ("explained class"). Several panels may show the same image with
different heatmaps; please rate each panel independently.

For every row of rating_form.csv, please fill in:

  relevance (1-5)   How well do the highlighted regions correspond to tissue
                    that is diagnostically relevant for the explained class?
                    1 = not at all (background, artefact, irrelevant tissue)
                    3 = partially
                    5 = closely matches the regions you would use
  artefact (Y/N)    Does the heatmap mainly highlight background, empty
                    space, edges or artefacts?
  comment           Optional.

You are not asked whether the class itself is correct. Rate the heatmap for
the class shown. The panel ids are random and do not identify the model.
"""


def denorm_display(path: str, size: int) -> np.ndarray:
    """Display copy of the original file (higher resolution than the input)."""
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True,
                    help="<run_dir_name>:<map>, map one of cnn/mamba/combined/"
                         "vit/swin (as named by tools/xai_metrics.py)")
    ap.add_argument("--checkpoints", default=str(ROOT / "checkpoints"))
    ap.add_argument("--per_class", type=int, default=20)
    ap.add_argument("--display", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "results" / "_rating_kit"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()
    # build_test_set reads these attributes
    args.data_root, args.split_file, args.batch_size = None, None, 16

    device = pick_device(args.device)
    out = Path(args.out)
    (out / "panels").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    panels, positions, ref_keys = [], None, None
    for spec in args.runs:
        run, mname = spec.split(":")
        model, config, meta = load_model(str(Path(args.checkpoints) / run / "best_model.pth"),
                                         device)
        model.eval()
        train_args = meta.get("train_args") or {}
        arch = train_args.get("arch", "hybrid")
        loader, info, data_root = build_test_set(args, train_args, config.image_size)
        ds = loader.dataset
        keys = [ds.dataset.samples[i][0] for i in ds.indices]
        if ref_keys is None:
            ref_keys = keys
            labels = np.array([ds.dataset.samples[i][1] for i in ds.indices])
            positions = stratified_subset(labels, args.per_class * len(info["class_names"]),
                                          args.seed)
        elif [Path(k).name for k in keys] != [Path(k).name for k in ref_keys]:
            raise SystemExit(f"{run}: test split differs from the first run's")
        classes = info["class_names"]

        cam = BatchGradCAMPP(model, target_layers(model, arch))
        try:
            for start in range(0, len(positions), 8):
                pos = positions[start:start + 8]
                xs, ys = zip(*(ds[int(p)] for p in pos))
                x = torch.stack(xs).to(device)
                with torch.no_grad():
                    pred = model(x).argmax(1)
                maps = cam(x, pred)
                maps = {k: upsample(v, x.shape[2:]) for k, v in maps.items()}
                if "cnn" in maps and "mamba" in maps:
                    maps["combined"] = _minmax(0.5 * maps["cnn"] + 0.5 * maps["mamba"])
                if mname not in maps:
                    raise SystemExit(f"{run}: no map {mname!r}; has {sorted(maps)}")
                for i, p in enumerate(pos):
                    path = ds.dataset.samples[ds.indices[int(p)]][0]
                    img = denorm_display(path, args.display)
                    heat = maps[mname][i].float().cpu().numpy()
                    over = visualize_heatmap(img, heat, alpha=0.4)
                    pid = f"P{rng.randrange(16 ** 6):06x}"
                    while any(q["panel_id"] == pid for q in panels):
                        pid = f"P{rng.randrange(16 ** 6):06x}"
                    canvas = Image.new("RGB", (2 * args.display + 30, args.display + 50), "white")
                    canvas.paste(Image.fromarray(img), (10, 40))
                    canvas.paste(Image.fromarray(over), (20 + args.display, 40))
                    ImageDraw.Draw(canvas).text(
                        (10, 12), f"{pid}    explained class: {classes[int(pred[i])]}",
                        fill="black")
                    canvas.save(out / "panels" / f"{pid}.png")
                    panels.append({
                        "panel_id": pid, "run": run, "map": mname,
                        "image": str(Path(path).resolve().relative_to(Path(data_root).resolve())),
                        "true_class": classes[int(ys[i])],
                        "explained_class": classes[int(pred[i])],
                        "correct": int(ys[i]) == int(pred[i]),
                    })
        finally:
            cam.remove()
        n = sum(q["run"] == run and q["map"] == mname for q in panels)
        print(f"{run}:{mname} -> {n} panels", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    order = panels[:]
    rng.shuffle(order)
    with open(out / "rating_form.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel_id", "panel_file", "explained_class", "relevance (1-5)",
                    "artefact (Y/N)", "comment"])
        for q in order:
            w.writerow([q["panel_id"], f"panels/{q['panel_id']}.png",
                        q["explained_class"], "", "", ""])
    with open(out / "key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(panels[0]))
        w.writeheader()
        w.writerows(panels)
    (out / "README.txt").write_text(README)
    print(f"{len(panels)} panels, form and key written to {out}")
    print("Send panels/, rating_form.csv and README.txt to the rater. "
          "Keep key.csv back until ratings are returned.")


if __name__ == "__main__":
    main()
