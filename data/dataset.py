"""
Histopathology Dataset — BreakHis and Camelyon16 data loading.

Supports:
  - BreakHis: Breast cancer histopathology (2-class benign/malignant or 8-class subtypes)
  - Camelyon16: Lymph node metastasis detection from WSI patches
  - Generic folder-based datasets

Reference:
  - BreakHis: Spanhol et al. (2016)
  - Camelyon16: Bejnordi et al. (2017)
"""

import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# Default location of the LC25000 source-image grouping produced by
# tools/group_lc25000.py. Resolved relative to the project root so it is found
# regardless of the working directory.
DEFAULT_GROUPS_FILE = Path(__file__).resolve().parent.parent / "lc25000_groups.json"

from data.transforms import (
    HistopathologyAugmentation,
    HistopathologyEvalTransform,
    MacenkoNormalize,
)


class HistopathologyDataset(Dataset):
    """
    Dataset for histopathology image classification.

    Expects a folder structure:
        root/
          class_0/
            img001.png
            img002.png
          class_1/
            img003.png
            ...

    For BreakHis, the standard structure is:
        root/
          benign/
            adenosis/ ...
            fibroadenoma/ ...
            phyllodes_tumor/ ...
            tubular_adenoma/ ...
          malignant/
            ductal_carcinoma/ ...
            lobular_carcinoma/ ...
            mucinous_carcinoma/ ...
            papillary_carcinoma/ ...

    Args:
        root: Root directory of the dataset.
        transform: Transform to apply to images.
        stain_normalize: Whether to apply Macenko stain normalization.
        magnification: Magnification filter (e.g., '200X') or None for all.
        binary: If True, use 2-class (benign/malignant). If False, use 8-class.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        stain_normalize: bool = True,
        magnification: Optional[str] = None,
        binary: bool = False,
    ):
        self.root = Path(root)
        self.transform = transform
        self.binary = binary
        self.stain_normalizer = MacenkoNormalize() if stain_normalize else None

        # Collect image paths and labels
        self.samples: List[Tuple[str, int]] = []
        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        self._load_dataset(magnification)

    def _load_dataset(self, magnification: Optional[str]):
        """Scan directory structure and build sample list."""
        valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

        if self.binary:
            # 2-class: benign vs malignant
            self.class_names = ["benign", "malignant"]
            self.class_to_idx = {"benign": 0, "malignant": 1}

            for class_name in self.class_names:
                class_dir = self.root / class_name
                if not class_dir.exists():
                    continue
                # Walk all subdirectories (subtypes)
                for img_path in sorted(class_dir.rglob("*")):
                    if img_path.suffix.lower() in valid_extensions:
                        if magnification and magnification not in str(img_path):
                            continue
                        self.samples.append((str(img_path), self.class_to_idx[class_name]))
        else:
            # Multi-class: scan all leaf directories
            class_dirs = sorted([
                d for d in self.root.rglob("*")
                if d.is_dir() and any(
                    f.suffix.lower() in valid_extensions for f in d.iterdir() if f.is_file()
                )
            ])

            if not class_dirs:
                # Flat structure: direct subdirectories are classes
                class_dirs = sorted([
                    d for d in self.root.iterdir() if d.is_dir()
                ])

            for idx, class_dir in enumerate(class_dirs):
                class_name = class_dir.name
                self.class_names.append(class_name)
                self.class_to_idx[class_name] = idx

                for img_path in sorted(class_dir.iterdir()):
                    if img_path.suffix.lower() in valid_extensions:
                        if magnification and magnification not in str(img_path):
                            continue
                        self.samples.append((str(img_path), idx))

        if len(self.samples) == 0:
            print(f"Warning: No samples found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply stain normalization
        if self.stain_normalizer is not None:
            img_np = np.array(image)
            img_np = self.stain_normalizer(img_np)
            image = Image.fromarray(img_np)

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_labels(self) -> List[int]:
        """Get all labels (for stratified splitting and weighted sampling)."""
        return [label for _, label in self.samples]

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced datasets."""
        labels = self.get_labels()
        class_counts = np.bincount(labels, minlength=len(self.class_names))
        class_counts = np.clip(class_counts, 1, None)  # Avoid division by zero
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(self.class_names)
        return torch.FloatTensor(weights)


# ────────────────────────────────────────────────────────────────────────
#                    LEAKAGE-FREE GROUPED SPLITTING
# ────────────────────────────────────────────────────────────────────────
#
# WHY THIS EXISTS
# ---------------
# LC25000 expands 250 original images per class to 5,000 using Augmentor
# (rotations, flips). Splitting per-file puts augmented siblings of the same
# physical tissue on both sides of the train/test boundary: with 20 variants per
# original, the chance all 20 land in a 70% train split is 0.7^20 ~= 0.08%. The
# resulting ~99% accuracy measures augmentation memorisation, not diagnosis.
#
# tools/group_lc25000.py recovers which files came from the same original
# (embedding shortlist + SIFT/RANSAC geometric verification). This module
# consumes that mapping so every variant of an original stays in one split.
#
# Note on method: the split below partitions *groups* per class rather than
# calling StratifiedGroupKFold. Because group ids are globally unique across
# classes, each group belongs to exactly one class, so partitioning groups
# within each class is already stratified — and unlike k-fold it gives exact
# control over the 70/15/15 ratios. StratifiedGroupKFold remains the right tool
# for repeated-run cross-validation, which is a separate change.


def load_group_assignments(groups_file: Optional[str] = None) -> Optional[Dict[str, int]]:
    """Load the source-image grouping produced by tools/group_lc25000.py.

    Args:
        groups_file: Path to the JSON file. ``None`` uses ``DEFAULT_GROUPS_FILE``.

    Returns:
        Mapping of ``"<class_dir>/<filename>"`` to integer group id, or ``None``
        if the file does not exist.
    """
    path = Path(groups_file) if groups_file else DEFAULT_GROUPS_FILE
    if not path.is_file():
        return None

    payload = json.loads(path.read_text())
    assignments = payload.get("assignments", payload)
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError(f"{path} contains no 'assignments' mapping.")
    return {str(k): int(v) for k, v in assignments.items()}


def load_split_file(split_file: str) -> Dict[str, str]:
    """Load an explicit ``{"<class_dir>/<filename>": "train"|"val"|"test"}`` map.

    Accepts either the bare mapping or ``{"assignments": {...}}``.
    """
    payload = json.loads(Path(split_file).read_text())
    mapping = payload.get("assignments", payload)
    bad = {v for v in mapping.values()} - {"train", "val", "test"}
    if bad:
        raise ValueError(f"{split_file}: unknown split names {sorted(bad)}")
    return {str(k): str(v) for k, v in mapping.items()}


def apply_split_file(
    samples: Sequence[Tuple[str, int]], split_map: Dict[str, str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index ``samples`` by an explicit split map; unmapped files are excluded."""
    out: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for i, (path, _) in enumerate(samples):
        s = split_map.get(_sample_group_key(path))
        if s is not None:
            out[s].append(i)
    if not any(out.values()):
        raise ValueError(
            "Split file matched none of the dataset's files. Keys must look like "
            f"'<class_dir>/<filename>', e.g. {_sample_group_key(samples[0][0])!r}."
        )
    return tuple(np.array(out[s], dtype=int) for s in ("train", "val", "test"))


def _sample_group_key(img_path: str) -> str:
    """Build the grouping key for a sample: ``"<parent_dir>/<filename>"``."""
    p = Path(img_path)
    return f"{p.parent.name}/{p.name}"


def grouped_split(
    samples: Sequence[Tuple[str, int]],
    assignments: Dict[str, int],
    num_classes: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Split sample indices so that no source-image group spans two splits.

    Groups are partitioned within each class, which keeps the class balance of
    the original data while guaranteeing zero sibling leakage.

    Args:
        samples: ``(path, label)`` pairs, indexed as in the dataset.
        assignments: Mapping from group key to group id.
        num_classes: Number of classes.
        train_ratio: Fraction of each class's *groups* used for training.
        val_ratio: Fraction of each class's *groups* used for validation.
        seed: Seed for the deterministic group shuffle.

    Returns:
        ``(train_idx, val_idx, test_idx, stats)``.

    Raises:
        KeyError: If any sample is missing from ``assignments``.
        ValueError: If a group spans more than one class.
    """
    missing = [p for p, _ in samples if _sample_group_key(p) not in assignments]
    if missing:
        raise KeyError(
            f"{len(missing)} sample(s) absent from the grouping file "
            f"(e.g. {_sample_group_key(missing[0])}). Regenerate it with "
            f"tools/group_lc25000.py, or pass use_groups=False to split per file."
        )

    # Bucket sample indices by group, and record each group's class.
    group_members: Dict[int, List[int]] = {}
    group_label: Dict[int, int] = {}
    for idx, (path, label) in enumerate(samples):
        gid = assignments[_sample_group_key(path)]
        group_members.setdefault(gid, []).append(idx)
        if group_label.setdefault(gid, label) != label:
            raise ValueError(
                f"Group {gid} spans classes {group_label[gid]} and {label}; "
                f"the grouping file is inconsistent with the directory layout."
            )

    rng = np.random.RandomState(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    per_class_groups: Dict[int, Dict[str, int]] = {}

    for cls in range(num_classes):
        gids = sorted(g for g, lab in group_label.items() if lab == cls)
        if not gids:
            continue
        rng.shuffle(gids)  # deterministic given `seed`

        n = len(gids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Guarantee non-empty val/test whenever the class has enough groups.
        n_train = min(n_train, max(1, n - 2)) if n >= 3 else n_train
        n_val = min(n_val, n - n_train - 1) if n - n_train >= 2 else n_val
        n_val = max(n_val, 0)

        splits = {
            "train": gids[:n_train],
            "val": gids[n_train : n_train + n_val],
            "test": gids[n_train + n_val :],
        }
        for gid in splits["train"]:
            train_idx.extend(group_members[gid])
        for gid in splits["val"]:
            val_idx.extend(group_members[gid])
        for gid in splits["test"]:
            test_idx.extend(group_members[gid])

        per_class_groups[cls] = {k: len(v) for k, v in splits.items()}

    # Hard guarantee: the whole point of this function.
    def _gids(idx_list: List[int]) -> set:
        return {assignments[_sample_group_key(samples[i][0])] for i in idx_list}

    g_tr, g_va, g_te = _gids(train_idx), _gids(val_idx), _gids(test_idx)
    overlap = (g_tr & g_va) | (g_tr & g_te) | (g_va & g_te)
    if overlap:
        raise AssertionError(f"Group leakage across splits: {sorted(overlap)[:5]}")

    stats = {
        "n_groups": len(group_members),
        "groups_train": len(g_tr),
        "groups_val": len(g_va),
        "groups_test": len(g_te),
        "per_class_groups": per_class_groups,
    }
    return (
        np.array(sorted(train_idx), dtype=int),
        np.array(sorted(val_idx), dtype=int),
        np.array(sorted(test_idx), dtype=int),
        stats,
    )


def create_dataloaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    stain_normalize: bool = True,
    magnification: Optional[str] = None,
    binary: bool = False,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
    use_groups: bool = True,
    groups_file: Optional[str] = None,
    split_file: Optional[str] = None,
    stain_method: str = "macenko",
    random_rotation: int = 90,
    random_flip: bool = True,
    color_jitter: float = 0.2,
    elastic_deform: bool = False,
    elastic_alpha: float = 50.0,
    elastic_sigma: float = 5.0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train/val/test DataLoaders with stratified splitting.

    Args:
        data_root: Path to dataset root.
        image_size: Target image size.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        stain_normalize: Whether to apply stain normalization.
        magnification: Magnification filter.
        binary: Whether to use binary classification.
        seed: Random seed.
        pin_memory: Pin host memory for faster host-to-device copies. Only
            meaningful for CUDA; defaults to ``torch.cuda.is_available()``.
        use_groups: Split by recovered source image rather than per file. This
            is what makes LC25000 results meaningful; see ``grouped_split``.
            Falls back to a per-file split (with a warning) if no grouping file
            is available.
        groups_file: Path to the grouping JSON. ``None`` uses
            ``DEFAULT_GROUPS_FILE`` (``lc25000_groups.json`` at project root).
        split_file: Optional JSON mapping ``"<class_dir>/<filename>"`` to
            ``"train"``, ``"val"`` or ``"test"``. Takes precedence over
            ``use_groups`` and the ratios. Used for patient-level folds where
            groups span classes (LungHist700) and for test-only external
            evaluation. Files absent from the mapping are excluded.
        stain_method: Stain normalisation method. Only ``"macenko"`` is
            implemented; anything else raises.
        random_rotation: Max rotation in degrees for training augmentation.
        random_flip: Apply random horizontal/vertical flips.
        color_jitter: ColorJitter strength (simulates stain variation).
        elastic_deform: Apply elastic deformation (models sectioning distortion).
        elastic_alpha: Elastic displacement magnitude.
        elastic_sigma: Elastic displacement smoothness.

    Raises:
        ValueError: If ``stain_method`` is not implemented.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, info_dict). ``info_dict``
        carries ``split_type`` (``"grouped"`` or ``"per_file"``) so the training
        log records which regime produced the numbers.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Fail loudly on an unimplemented method. Silently applying Macenko when the
    # config says "vahadane" would misreport the preprocessing in the paper.
    if stain_normalize and stain_method.lower() != "macenko":
        raise ValueError(
            f"stain_method={stain_method!r} is declared in DataConfig but not "
            f"implemented; only 'macenko' is available. Pass "
            f"stain_normalize=False to skip normalisation entirely."
        )
    # Create full dataset (no augmentation, just to get labels for splitting)
    full_dataset = HistopathologyDataset(
        root=data_root,
        transform=None,
        stain_normalize=False,
        magnification=magnification,
        binary=binary,
    )

    labels = full_dataset.get_labels()
    indices = list(range(len(full_dataset)))

    # ── Split: explicit file > grouped by source image > per file ──────
    split_map = load_split_file(split_file) if split_file else None
    assignments = (load_group_assignments(groups_file)
                   if (use_groups and split_map is None) else None)
    split_type = "per_file"
    group_stats: Dict = {}

    if split_map is not None:
        train_idx, val_idx, test_idx = apply_split_file(full_dataset.samples, split_map)
        split_type = "explicit"
        group_stats = {"split_file": str(split_file)}
        logger.info(
            "Explicit split from %s: train %d / val %d / test %d (excluded %d)",
            split_file, len(train_idx), len(val_idx), len(test_idx),
            len(full_dataset) - len(train_idx) - len(val_idx) - len(test_idx),
        )
    elif use_groups and assignments is None:
        logger.warning(
            "Grouped split requested but no grouping file was found at %s. "
            "Falling back to a PER-FILE split, which leaks augmented siblings "
            "across train/test on LC25000. Run tools/group_lc25000.py to fix.",
            groups_file or DEFAULT_GROUPS_FILE,
        )

    if split_map is not None:
        pass
    elif assignments is not None:
        train_idx, val_idx, test_idx, group_stats = grouped_split(
            samples=full_dataset.samples,
            assignments=assignments,
            num_classes=len(full_dataset.class_names),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        split_type = "grouped"
        logger.info(
            "Grouped split: %d source groups -> train %d / val %d / test %d groups",
            group_stats["n_groups"], group_stats["groups_train"],
            group_stats["groups_val"], group_stats["groups_test"],
        )
    else:
        # Per-file stratified split: train/temp → temp/val/test.
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed)
        train_idx, temp_idx = next(sss1.split(indices, labels))

        temp_labels = [labels[i] for i in temp_idx]
        relative_test = (1 - train_ratio - val_ratio) / (1 - train_ratio)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed)
        val_relative_idx, test_relative_idx = next(sss2.split(temp_idx, temp_labels))
        val_idx = temp_idx[val_relative_idx]
        test_idx = temp_idx[test_relative_idx]

    # Create subset datasets with appropriate transforms. The augmentation
    # settings are forwarded rather than left at the transform's own defaults,
    # so DataConfig actually governs the pipeline.
    train_transform = HistopathologyAugmentation(
        image_size=image_size,
        random_rotation=random_rotation,
        random_flip=random_flip,
        color_jitter=color_jitter,
        elastic_deform=elastic_deform,
        elastic_alpha=elastic_alpha,
        elastic_sigma=elastic_sigma,
    )
    eval_transform = HistopathologyEvalTransform(image_size=image_size)

    train_dataset = SubsetDataset(full_dataset, train_idx, train_transform, stain_normalize)
    val_dataset = SubsetDataset(full_dataset, val_idx, eval_transform, stain_normalize)
    test_dataset = SubsetDataset(full_dataset, test_idx, eval_transform, stain_normalize)

    # Weighted sampler for class imbalance. A test-only split file (external
    # evaluation) leaves the training set empty, and WeightedRandomSampler
    # rejects zero samples, so skip it there.
    train_labels = [labels[i] for i in train_idx]
    sampler = None
    if train_labels:
        class_counts = np.bincount(train_labels, minlength=len(full_dataset.class_names))
        class_counts = np.clip(class_counts, 1, None)
        sample_weights = 1.0 / class_counts[train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_labels), replacement=True,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    info = {
        "num_classes": len(full_dataset.class_names),
        "class_names": full_dataset.class_names,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "class_weights": full_dataset.get_class_weights(),
        "split_type": split_type,
        "augmentation": {
            "random_rotation": random_rotation,
            "random_flip": random_flip,
            "color_jitter": color_jitter,
            "elastic_deform": elastic_deform,
            "stain_normalize": stain_normalize,
            "stain_method": stain_method if stain_normalize else None,
        },
        **group_stats,
    }

    return train_loader, val_loader, test_loader, info


class SubsetDataset(Dataset):
    """Subset of a dataset with its own transform and optional stain normalization."""

    def __init__(self, dataset, indices, transform, stain_normalize):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.stain_normalizer = MacenkoNormalize() if stain_normalize else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, label = self.dataset.samples[real_idx]

        image = Image.open(img_path).convert("RGB")

        if self.stain_normalizer is not None:
            img_np = np.array(image)
            img_np = self.stain_normalizer(img_np)
            image = Image.fromarray(img_np)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
