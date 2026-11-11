"""Recover LC25000's source images so splits can be made leakage-free.

WHY THIS EXISTS
---------------
LC25000 (Borkowski et al. 2019) expanded 250 original images per class to 5,000
using Augmentor: rotations up to +/-25 degrees at probability 1.0, plus
horizontal and vertical flips at probability 0.5. Every published result that
splits LC25000 per-file therefore trains and tests on augmented copies of the
same physical tissue. With 20 variants per original, the chance that all 20 land
in a 70% training split is 0.7^20, about 0.08%: essentially every original
appears on both sides of the boundary. Accuracies near 100% measure augmentation
memorisation, not diagnosis.

This tool recovers the grouping so that `StratifiedGroupKFold` can keep all
variants of an original together.

APPROACH
--------
Two stages, because neither alone is sufficient:

1. SHORTLIST with deep embeddings (flip-averaged EfficientNet features).
   Cheap and recalls most siblings, but cannot be used alone: the similarity
   graph percolates into one giant component before sibling capture is adequate
   (measured: recall 0.83 at 243 groups, then the largest cluster jumps from
   2.9% to 56% of the class between thresholds 0.90 and 0.88). ImageNet features
   encode "adenocarcinoma tissue", not "this specific field of view".

2. VERIFY with SIFT + RANSAC. Siblings are the same physical field under a
   similarity transform, so a geometrically consistent set of keypoint matches
   exists between them and not between unrelated fields. Measured on this
   dataset: siblings average 272 inliers, random pairs average 0.6 with a
   maximum of 7. A threshold of 15 gives a 0% false-positive rate.

Zero false positives is what makes connected components safe here: groups are
merged only on geometric evidence, so the transitive chaining that ruined the
embedding-only approach cannot occur.

Usage:
    python tools/group_lc25000.py --data-root ./data --out lc25000_groups.json
    python tools/group_lc25000.py --data-root ./data --class-dir lung_aca   # one class
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from PIL import Image

cv2.setNumThreads(0)  # we parallelise at the pair level instead

CLASSES = ("lung_aca", "lung_n", "lung_scc")
EXPECTED_GROUPS = 250
EXPECTED_SIZE = 20

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Stage 1: embedding shortlist
# ---------------------------------------------------------------------------

def _load_batch(paths, size):
    arrs = []
    for p in paths:
        im = Image.open(p)
        im.draft("RGB", (size * 2, size * 2))
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arrs.append(np.asarray(im, dtype=np.float32) / 255.0)
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
    return (x - MEAN) / STD


@torch.no_grad()
def embed(files, backbone="efficientnet_b0", size=224, batch=64) -> np.ndarray:
    """Flip-averaged, L2-normalised embeddings."""
    import timm

    device = get_device()
    model = timm.create_model(backbone, pretrained=True, num_classes=0)
    model = model.eval().to(device)
    out = []
    for i in range(0, len(files), batch):
        x = _load_batch(files[i : i + batch], size).to(device)
        acc = None
        for dims in ([], [3], [2], [2, 3]):  # the flip group
            f = model(torch.flip(x, dims=dims) if dims else x).float()
            acc = f if acc is None else acc + f
        acc = acc / 4.0
        out.append((acc / acc.norm(dim=1, keepdim=True).clamp_min(1e-8)).cpu().numpy())
        print(f"    embedding {min(i + batch, len(files))}/{len(files)}",
              end="\r", flush=True)
    print(" " * 50, end="\r")
    return np.concatenate(out)


def shortlist(E: np.ndarray, topk: int) -> set:
    """Candidate pairs: each image's top-k most similar neighbours."""
    sim = E @ E.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, topk, axis=1)[:, :topk]
    pairs = set()
    for i, row in enumerate(idx):
        for j in row:
            pairs.add((i, int(j)) if i < j else (int(j), i))
    return pairs


# ---------------------------------------------------------------------------
# Stage 2: SIFT geometric verification
# ---------------------------------------------------------------------------

def sift_descriptors(files, size, nfeatures):
    """Detect once per image, not once per pair. This is the main speedup.

    We also detect on the horizontally mirrored image. SIFT descriptors are
    rotation-invariant but *not* mirror-invariant, and a similarity transform
    cannot represent a reflection. LC25000 applies horizontal and vertical flips
    at probability 0.5 each, so roughly half of all sibling pairs differ by a
    reflection and would otherwise be unmatchable. One mirror variant suffices,
    because a vertical flip equals a horizontal flip composed with a 180-degree
    rotation, which SIFT's rotation invariance already covers.
    """
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kps, descs, kps_m, descs_m = [], [], [], []
    for n, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        for src_img, kl, dl in ((img, kps, descs), (cv2.flip(img, 1), kps_m, descs_m)):
            k, d = sift.detectAndCompute(src_img, None)
            kl.append(np.float32([p.pt for p in k]) if k
                      else np.zeros((0, 2), np.float32))
            dl.append(d if d is not None else np.zeros((0, 128), np.float32))
        if n % 100 == 0:
            print(f"    SIFT {n}/{len(files)}", end="\r", flush=True)
    print(" " * 50, end="\r")
    return (kps, descs), (kps_m, descs_m)


def _ransac_inliers(kp_a, d_a, kp_b, d_b):
    if len(d_a) < 8 or len(d_b) < 8:
        return 0
    raw = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d_a, d_b, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2)
            if m.distance < 0.75 * n.distance]  # Lowe's ratio test
    if len(good) < 8:
        return 0
    src = kp_a[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = kp_b[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    # Partial affine = rotation + uniform scale + translation (4 DOF). Kept
    # deliberately constrained rather than full affine, because the tight model
    # is what gives the 0% false-positive rate.
    _, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0,
        maxIters=3000, confidence=0.99,
    )
    return 0 if mask is None else int(mask.sum())


def verify(pair, plain, mirror, min_inliers):
    """Geometrically verify a pair, trying both orientations."""
    i, j = pair
    kps, descs = plain
    kps_m, descs_m = mirror
    n_in = _ransac_inliers(kps[i], descs[i], kps[j], descs[j])
    if n_in < min_inliers:
        # Same field but reflected: match against the mirrored variant.
        n_in = max(n_in, _ransac_inliers(kps[i], descs[i], kps_m[j], descs_m[j]))
    return (i, j, n_in) if n_in >= min_inliers else None


# ---------------------------------------------------------------------------

def connected_components(n: int, edges) -> np.ndarray:
    """Union-find over verified sibling edges."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, _ in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    roots = {}
    labels = np.empty(n, dtype=int)
    for i in range(n):
        r = find(i)
        labels[i] = roots.setdefault(r, len(roots))
    return labels


def _run_verifications(pairs, plain, mirror, min_inliers, workers, tag=""):
    edges = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, res in enumerate(ex.map(
            lambda p: verify(p, plain, mirror, min_inliers), pairs
        )):
            if res:
                edges.append(res)
            if n % 2000 == 0:
                print(f"    {tag}verifying {n}/{len(pairs)}  edges={len(edges)}",
                      end="\r", flush=True)
    print(" " * 70, end="\r")
    return edges


def merge_clusters(labels, E, plain, mirror, min_inliers, workers,
                   top_clusters=25, probes=6):
    """Second pass: reconnect clusters that are halves of one source group.

    A true group of 20 splits when its members differ by too much rotation for
    SIFT to match directly (siblings can be up to 50 degrees apart, since each is
    +/-25 from the original). Those halves still belong together.

    Because geometric verification has a 0% false-positive rate, merging two
    clusters on the strength of a single verified pair is safe. So for each
    cluster we probe its most embedding-similar neighbouring clusters, testing
    the most promising member pairs, and merge on any geometric hit.
    """
    n_before = len(np.unique(labels))
    members = {}
    for i, l in enumerate(labels):
        members.setdefault(int(l), []).append(i)
    ids = sorted(members)

    cent = np.stack([E[members[c]].mean(axis=0) for c in ids])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True).clip(1e-8)
    csim = cent @ cent.T
    np.fill_diagonal(csim, -np.inf)

    k = min(top_clusters, len(ids) - 1)
    nbr = np.argpartition(-csim, k, axis=1)[:, :k]

    # For each candidate cluster pair, probe the most similar member pairs.
    seen, pairs = set(), []
    for a_i, row in enumerate(nbr):
        for b_i in row:
            key = (min(a_i, int(b_i)), max(a_i, int(b_i)))
            if key in seen:
                continue
            seen.add(key)
            ma, mb = members[ids[key[0]]], members[ids[key[1]]]
            sub = E[ma] @ E[mb].T
            flat = np.argsort(-sub, axis=None)[:probes]
            for f in flat:
                r, c = divmod(int(f), sub.shape[1])
                i, j = ma[r], mb[c]
                pairs.append((i, j) if i < j else (j, i))
    pairs = sorted(set(pairs))
    print(f"  merge pass: {len(seen)} cluster pairs -> {len(pairs)} probe pairs")

    edges = _run_verifications(pairs, plain, mirror, min_inliers, workers, "merge ")
    print(f"  merge pass verified {len(edges)} bridging edges")

    # Re-run union-find on the union of both edge sets.
    same = [(members[ids[c]][0], m, 0)
            for c in range(len(ids)) for m in members[ids[c]][1:]]
    labels2 = connected_components(len(labels), same + edges)
    print(f"  clusters {n_before} -> {len(np.unique(labels2))}")
    return labels2


def _summarise(labels, note=""):
    _, counts = np.unique(labels, return_counts=True)
    exact = int((counts == EXPECTED_SIZE).sum())
    print(f"  -> {len(counts)} groups (target {EXPECTED_GROUPS}){note}")
    print(f"     size: min={counts.min()} median={int(np.median(counts))} "
          f"max={counts.max()} | exactly-{EXPECTED_SIZE}: {exact}")
    print(f"     singletons: {int((counts == 1).sum())}  "
          f"largest group = {counts.max() / counts.sum():.1%} of class")


def group_class(class_dir: pathlib.Path, topk: int, min_inliers: int,
                sift_size: int, nfeatures: int, workers: int,
                cache_dir: pathlib.Path | None = None,
                do_merge: bool = True) -> dict:
    files = sorted(class_dir.glob("*.jpeg"))
    print(f"\n=== {class_dir.name}: {len(files)} images ===")

    ecache = cache_dir / f"{class_dir.name}_emb.npy" if cache_dir else None
    scache = cache_dir / f"{class_dir.name}_sift.npz" if cache_dir else None

    if ecache and ecache.exists():
        E = np.load(ecache)
        print(f"  embeddings {E.shape} (cached)")
    else:
        t0 = time.perf_counter()
        E = embed(files)
        print(f"  embeddings {E.shape} in {time.perf_counter() - t0:.0f}s")
        if ecache:
            ecache.parent.mkdir(parents=True, exist_ok=True)
            np.save(ecache, E)

    if scache and scache.exists():
        z = np.load(scache, allow_pickle=True)
        plain = (list(z["kps"]), list(z["descs"]))
        mirror = (list(z["kps_m"]), list(z["descs_m"]))
        print(f"  SIFT descriptors (cached, mean "
              f"{np.mean([len(d) for d in plain[1]]):.0f} kp/image)")
    else:
        t0 = time.perf_counter()
        plain, mirror = sift_descriptors(files, sift_size, nfeatures)
        print(f"  SIFT descriptors (+mirror) in {time.perf_counter() - t0:.0f}s "
              f"(mean {np.mean([len(d) for d in plain[1]]):.0f} kp/image)")
        if scache:
            scache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(scache,
                     kps=np.array(plain[0], dtype=object),
                     descs=np.array(plain[1], dtype=object),
                     kps_m=np.array(mirror[0], dtype=object),
                     descs_m=np.array(mirror[1], dtype=object))

    pairs = sorted(shortlist(E, topk))
    print(f"  shortlisted {len(pairs)} candidate pairs (top-{topk} per image)")

    t0 = time.perf_counter()
    edges = _run_verifications(pairs, plain, mirror, min_inliers, workers)
    print(f"  verified {len(edges)} sibling edges in {time.perf_counter() - t0:.0f}s")

    labels = connected_components(len(files), edges)
    _summarise(labels, "  [pass 1]")

    if do_merge:
        labels = merge_clusters(labels, E, plain, mirror, min_inliers, workers)
        _summarise(labels, "  [after merge]")

    return {f.name: int(l) for f, l in zip(files, labels)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--class-dir", default="", help="single class, else all three")
    ap.add_argument("--out", default="lc25000_groups.json")
    ap.add_argument("--topk", type=int, default=60,
                    help="embedding candidates per image to geometrically verify")
    ap.add_argument("--min-inliers", type=int, default=15,
                    help="RANSAC inliers to accept a sibling pair (0%% FPR at 15)")
    ap.add_argument("--sift-size", type=int, default=384)
    ap.add_argument("--nfeatures", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-dir", default="", help="cache embeddings + SIFT descriptors")
    ap.add_argument("--no-merge", action="store_true",
                    help="skip the cluster-merge second pass")
    args = ap.parse_args()

    root = pathlib.Path(args.data_root)
    classes = [args.class_dir] if args.class_dir else list(CLASSES)
    cache = pathlib.Path(args.cache_dir) if args.cache_dir else None

    groups, offset = {}, 0
    for c in classes:
        d = root / c
        if not d.is_dir():
            print(f"skipping missing {d}")
            continue
        g = group_class(d, args.topk, args.min_inliers, args.sift_size,
                        args.nfeatures, args.workers, cache,
                        do_merge=not args.no_merge)
        # Offset labels so group ids are globally unique across classes.
        for name, lab in g.items():
            groups[f"{c}/{name}"] = lab + offset
        offset += max(g.values()) + 1

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(
        {"n_groups": len(set(groups.values())), "assignments": groups}, indent=1
    ))
    print(f"\nwrote {out} — {len(groups)} images in "
          f"{len(set(groups.values()))} groups")


if __name__ == "__main__":
    main()
