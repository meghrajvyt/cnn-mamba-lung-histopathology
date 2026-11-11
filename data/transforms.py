"""
Custom transforms for histopathology image preprocessing.

Includes:
  - Macenko stain normalization
  - Histopathology-specific augmentations
  - Patch extraction utilities for WSI

References:
  - Macenko et al. (2009) — stain normalization
  - Krishna et al. (2022) — arXiv:2201.00957
  - Zingman et al. (2023) — arXiv:2303.17009
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class MacenkoNormalize:
    """Macenko et al. (2009) stain normalisation for H&E images.

    Follows the widely used reference implementation of the method (the
    numpy code distributed with staintools / torchstain):

    1. optical density OD = -ln((I + 1) / Io), with Io = 240;
    2. tissue pixels are those with OD >= beta in every channel (beta = 0.15);
    3. the stain plane is spanned by the top two eigenvectors of the tissue-OD
       covariance, signed so that projections are positive;
    4. the two stain vectors are the directions at the alpha and 100 - alpha
       percentiles of the projected angle (alpha = 1); haematoxylin is the one
       with the larger red OD, because it absorbs red more strongly than eosin;
    5. per-pixel concentrations by least squares, clipped at zero, rescaled
       so their 99th percentiles equal the reference maxima;
    6. recomposition with the reference stain matrix.

    The reference stain matrix and maxima are the standard values of that
    implementation (``HE_REF``, ``MAX_C_REF``).

    HISTORY. The implementation this replaces (in use until 2026-10-11)
    deviated from the method: it put the stain with the SMALLER red OD first
    and so swapped haematoxylin and eosin, rescaled concentrations to 1.0
    instead of the reference maxima, did not fix the eigenvector signs, and
    selected tissue with OD > 0.12 in ANY channel. Its output had cyan nuclei
    and washed-out tissue. Every result produced before that date, including
    ./data_macenko and ./data_lh700_macenko, used it.
    """

    # Columns: haematoxylin, eosin (unit-norm OD vectors in R, G, B).
    HE_REF = np.array([[0.5626, 0.2159],
                       [0.7201, 0.8012],
                       [0.4062, 0.5581]])
    MAX_C_REF = np.array([1.9705, 1.0308])

    def __init__(
        self,
        io: float = 240.0,
        alpha: float = 1.0,
        beta: float = 0.15,
        min_tissue_pixels: int = 100,
    ):
        self.io = io
        self.alpha = alpha
        self.beta = beta
        self.min_tissue_pixels = min_tissue_pixels
        # Images returned unnormalised because estimation failed (too little
        # tissue, degenerate stain plane). Counted so callers can report it.
        self.n_fallback = 0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Normalise one RGB uint8 image (H, W, 3).

        If the stain vectors cannot be estimated (e.g. almost no tissue), the
        image is returned unchanged and ``n_fallback`` is incremented.
        """
        out, ok = self.normalize_with_status(image)
        if not ok:
            self.n_fallback += 1
        return out

    def normalize_with_status(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Like ``__call__`` but returns ``(image, normalised_ok)``."""
        try:
            return self._normalize(image), True
        except (ValueError, np.linalg.LinAlgError):
            return image, False

    def stain_matrix(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate ``(HE, od)``: the (3, 2) stain matrix [H, E] and the
        per-pixel OD (N, 3) of ``image``."""
        od = -np.log((image.reshape(-1, 3).astype(np.float64) + 1.0) / self.io)
        tissue = od[np.all(od >= self.beta, axis=1)]
        if len(tissue) < self.min_tissue_pixels:
            raise ValueError(f"only {len(tissue)} tissue pixels")

        _, eigvecs = np.linalg.eigh(np.cov(tissue, rowvar=False))
        plane = eigvecs[:, [2, 1]]          # eigh is ascending: top two
        # Eigenvector signs are arbitrary. Make both point into positive OD so
        # the projected angles do not straddle +/-pi.
        plane = plane * np.where(plane[0] < 0, -1.0, 1.0)

        proj = tissue @ plane
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        lo, hi = np.percentile(phi, [self.alpha, 100.0 - self.alpha])
        v_lo = plane @ np.array([np.cos(lo), np.sin(lo)])
        v_hi = plane @ np.array([np.cos(hi), np.sin(hi)])
        he = (np.stack([v_lo, v_hi], axis=1) if v_lo[0] > v_hi[0]
              else np.stack([v_hi, v_lo], axis=1))
        he = he / np.linalg.norm(he, axis=0, keepdims=True)
        return he, od

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        he, od = self.stain_matrix(image)
        conc = np.linalg.lstsq(he, od.T, rcond=None)[0]          # (2, N)
        # Stain amounts are physically non-negative. Unconstrained least
        # squares gives negative eosin in near-black nuclear pixels, which
        # recompose as red speckles; staintools avoids this with a
        # non-negative lasso, approximated here by clipping at zero.
        conc = np.maximum(conc, 0.0)
        max_c = np.percentile(conc, 99, axis=1)
        if np.any(max_c <= 1e-6):
            raise ValueError(f"degenerate stain concentrations {max_c}")
        conc *= (self.MAX_C_REF / max_c)[:, None]
        out = self.io * np.exp(-self.HE_REF @ conc)              # (3, N)
        out = np.clip(out, 0, 255)
        return out.T.reshape(h, w, 3).astype(np.uint8)


class HistopathologyAugmentation:
    """
    Augmentation pipeline tailored for histopathology images.

    Includes rotation-invariant transforms (pathology slides have no
    canonical orientation) and stain-aware color augmentation.
    """

    def __init__(
        self,
        image_size: int = 224,
        random_rotation: int = 90,
        random_flip: bool = True,
        color_jitter: float = 0.2,
        elastic_deform: bool = False,
        elastic_alpha: float = 50.0,
        elastic_sigma: float = 5.0,
        normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.image_size = image_size

        transforms = []

        # Spatial augmentations
        transforms.append(T.Resize((image_size, image_size)))

        if random_flip:
            transforms.append(T.RandomHorizontalFlip(p=0.5))
            transforms.append(T.RandomVerticalFlip(p=0.5))

        if random_rotation > 0:
            transforms.append(T.RandomRotation(random_rotation))

        # Elastic deformation. Previously this argument was accepted and then
        # silently ignored, so `elastic_deform=True` did nothing at all.
        # Models tissue stretching/compression from sectioning and mounting,
        # which is a real source of morphological variation in histopathology.
        # Applied before ColorJitter so the displacement acts on unaltered
        # intensities, and before ToTensor since it takes PIL input.
        if elastic_deform:
            transforms.append(T.ElasticTransform(
                alpha=elastic_alpha, sigma=elastic_sigma,
            ))

        # Color augmentation (simulates stain variation)
        if color_jitter > 0:
            transforms.append(T.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter * 0.5,
            ))

        # Random affine for slight geometric variation
        transforms.append(T.RandomAffine(
            degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05),
        ))

        # To tensor and normalize
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(mean=normalize_mean, std=normalize_std))

        # Random erasing (simulates tissue artifacts)
        transforms.append(T.RandomErasing(p=0.1, scale=(0.02, 0.1)))

        self.transform = T.Compose(transforms)

    def __call__(self, image):
        return self.transform(image)


class HistopathologyEvalTransform:
    """Deterministic transform for validation/test (no augmentation)."""

    def __init__(
        self,
        image_size: int = 224,
        normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ])

    def __call__(self, image):
        return self.transform(image)


def get_inverse_normalize(
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> T.Normalize:
    """Get inverse normalization transform for visualization."""
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1.0 / s for s in std]
    return T.Normalize(mean=inv_mean, std=inv_std)


# ────────────────────────────────────────────────────────────────────────
#                     BATCH-LEVEL MIXING AUGMENTATION
# ────────────────────────────────────────────────────────────────────────
#
# mixup and CutMix operate on a *batch* and mix labels as well as pixels, so
# they cannot live in the per-sample torchvision pipeline above. They are
# applied in the training loop instead, which is why `mixup_alpha` and
# `cutmix_alpha` sat unused in DataConfig for so long.
#
# Both return soft targets as (targets_a, targets_b, lam); the caller blends the
# loss as `lam * L(pred, a) + (1 - lam) * L(pred, b)`. That formulation is
# equivalent to cross-entropy against the mixed label distribution while
# remaining compatible with class weighting and label smoothing.
#
# References:
#   mixup:  Zhang et al. (2018), arXiv:1710.09412
#   CutMix: Yun et al. (2019),   arXiv:1905.04899


def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Convex-combine a batch with a shuffled copy of itself.

    Args:
        images: Batch of images (B, C, H, W).
        targets: Class indices (B,).
        alpha: Beta distribution parameter. Larger means stronger mixing.
        generator: Optional RNG for reproducible permutations.

    Returns:
        ``(mixed_images, targets_a, targets_b, lam)``.
    """
    if alpha <= 0:
        return images, targets, targets, 1.0

    lam = float(np.random.beta(alpha, alpha))
    # Keep lam >= 0.5 so `targets_a` is always the dominant label. This makes
    # training accuracy against targets_a interpretable rather than arbitrary.
    lam = max(lam, 1.0 - lam)

    index = torch.randperm(images.size(0), device=images.device, generator=generator)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, targets, targets[index], lam


def cutmix_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Paste a random rectangle from a shuffled copy of the batch.

    ``lam`` is recomputed from the realised patch area, since clipping the box
    to the image bounds changes the true mixing ratio.

    Args:
        images: Batch of images (B, C, H, W).
        targets: Class indices (B,).
        alpha: Beta distribution parameter.
        generator: Optional RNG for reproducible permutations.

    Returns:
        ``(mixed_images, targets_a, targets_b, lam)``.
    """
    if alpha <= 0:
        return images, targets, targets, 1.0

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)  # dominant label is targets_a

    B, _, H, W = images.shape
    index = torch.randperm(B, device=images.device, generator=generator)

    # Box side scales with sqrt(1 - lam) so its area is the (1 - lam) fraction.
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    if cut_h < 1 or cut_w < 1:
        # Degenerate box: nothing would be pasted, so leave the batch alone
        # rather than returning a lam that misdescribes the images.
        return images, targets, targets, 1.0

    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, H)
    x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, W)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Actual, not nominal, mixing ratio after clipping.
    lam_actual = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))
    return mixed, targets, targets[index], lam_actual


def apply_batch_mixing(
    images: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    prob: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup or CutMix to a batch with probability ``prob``.

    When both alphas are positive, one of the two is chosen at random per batch.

    Args:
        images: Batch of images (B, C, H, W).
        targets: Class indices (B,).
        mixup_alpha: mixup Beta parameter; <= 0 disables mixup.
        cutmix_alpha: CutMix Beta parameter; <= 0 disables CutMix.
        prob: Probability of mixing a given batch. 0 disables mixing entirely.
        generator: Optional RNG for reproducible permutations.

    Returns:
        ``(images, targets_a, targets_b, lam)``. When no mixing is applied,
        ``targets_a is targets_b`` and ``lam == 1.0``, so the caller's blended
        loss reduces exactly to the unmixed loss.
    """
    available = [
        (fn, a) for fn, a in
        ((mixup_batch, mixup_alpha), (cutmix_batch, cutmix_alpha))
        if a > 0
    ]
    if prob <= 0 or not available or np.random.rand() >= prob:
        return images, targets, targets, 1.0

    fn, alpha = available[np.random.randint(len(available))]
    return fn(images, targets, alpha, generator)


def mixed_criterion(
    criterion: "torch.nn.Module",
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Blend a loss over the two label sets produced by mixing.

    Reduces to ``criterion(logits, targets_a)`` when ``lam == 1.0``, so this is
    safe to call unconditionally.
    """
    if lam >= 1.0:
        return criterion(logits, targets_a)
    return lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
