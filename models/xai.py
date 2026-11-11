"""
Explainable AI Module — Grad-CAM++ for the Hybrid CNN-Mamba Architecture.

Provides visual explanations by generating class-discriminative heatmaps
from both pathways:

1. **CNN Pathway**: Standard Grad-CAM++ on convolutional feature maps
2. **Mamba Pathway**: Novel adaptation — project Mamba's sequential hidden states
   back to 2D spatial maps, then apply gradient-based attribution

This addresses Gap 2 from the literature review:
  "No existing XAI framework has been designed for or validated on
   hybrid CNN-Mamba architectures."

References:
  - Grad-CAM: Selvaraju et al. (2020), arXiv:1610.02391
  - Grad-CAM++: Chattopadhay et al. (2018), arXiv:1710.11063
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for the hybrid CNN-Mamba architecture.

    Computes pixel-level importance weights using second-order gradients:

        α_k^c = Σ_ij [ ReLU(∂²y^c/∂A_ij^k) /
                        (2·∂²y^c/∂A_ij^k + Σ_ab A_ab^k · ∂³y^c/∂A_ij^k) ]

        L^c = ReLU(Σ_k α_k^c · A^k)

    This provides more precise localization than Grad-CAM by weighting
    individual pixel-level gradients rather than using global average pooling.

    Args:
        model: The HybridCNNMamba model instance.
        target_layers: Dict mapping pathway names to target layer modules.
    """

    def __init__(self, model: nn.Module, target_layers: Optional[Dict[str, nn.Module]] = None):
        self.model = model
        self.gradients: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List = []

        # Default target layers
        if target_layers is None:
            target_layers = self._get_default_target_layers()

        self.target_layers = target_layers

        # Register hooks
        self._register_hooks()

    def _get_default_target_layers(self) -> Dict[str, nn.Module]:
        """Get default target layers for each pathway."""
        layers = {}

        # CNN pathway: use the last channel projection (before final pooling)
        if hasattr(self.model, "cnn_pathway"):
            cnn = self.model.cnn_pathway
            if hasattr(cnn, "channel_projections") and len(cnn.channel_projections) > 0:
                layers["cnn"] = cnn.channel_projections[-1]

        # Mamba pathway: use the last Vision Mamba block's norm
        if hasattr(self.model, "mamba_pathway"):
            mamba = self.model.mamba_pathway
            if hasattr(mamba, "blocks") and len(mamba.blocks) > 0:
                layers["mamba"] = mamba.blocks[-1].norm2

        return layers

    def _register_hooks(self):
        """Register forward and backward hooks on target layers."""
        for name, layer in self.target_layers.items():
            # Forward hook: capture activations
            def fwd_hook(module, input, output, name=name):
                self.activations[name] = output.detach()

            # Backward hook: capture gradients
            def bwd_hook(module, grad_input, grad_output, name=name):
                self.gradients[name] = grad_output[0].detach()

            self.hooks.append(layer.register_forward_hook(fwd_hook))
            self.hooks.append(layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM++ heatmaps for both pathways.

        Args:
            input_tensor: Input image (B, C, H, W). B should be 1 for visualization.
            target_class: Target class index. If None, uses the predicted class.

        Returns:
            Dict mapping pathway names to heatmaps (H, W) normalized to [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)

        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)

        # Generate heatmaps for each pathway
        heatmaps = {}

        for name in self.target_layers:
            if name in self.activations and name in self.gradients:
                activation = self.activations[name]
                gradient = self.gradients[name]

                heatmap = self._compute_gradcampp(activation, gradient)
                heatmaps[name] = heatmap

        # Generate combined dual-scale heatmap
        if "cnn" in heatmaps and "mamba" in heatmaps:
            heatmaps["combined"] = self._combine_heatmaps(
                heatmaps["cnn"], heatmaps["mamba"],
                input_tensor.shape[2], input_tensor.shape[3],
            )

        return heatmaps

    def _compute_gradcampp(
        self, activation: torch.Tensor, gradient: torch.Tensor
    ) -> np.ndarray:
        """
        Compute Grad-CAM++ heatmap from activations and gradients.

        Grad-CAM++ uses second and third-order gradients for more precise
        weighting compared to standard Grad-CAM's global average pooling.

        Args:
            activation: Feature map activations (B, C, H, W) or (B, L, D).
            gradient: Gradients of target score w.r.t. activations.

        Returns:
            Heatmap as numpy array (H, W), normalized to [0, 1].
        """
        # Handle both 2D (CNN) and 1D (Mamba) feature formats
        if activation.dim() == 3:
            # Mamba features: (B, L, D) → reshape to 2D
            B, L, D = activation.shape
            grid_size = int(L ** 0.5)
            if grid_size * grid_size != L:
                # Non-square; pad or use as-is
                grid_size = int(np.ceil(L ** 0.5))
                pad_len = grid_size * grid_size - L
                if pad_len > 0:
                    activation = F.pad(activation, (0, 0, 0, pad_len))
                    gradient = F.pad(gradient, (0, 0, 0, pad_len))
            activation = activation.reshape(B, grid_size, grid_size, D).permute(0, 3, 1, 2)
            gradient = gradient.reshape(B, grid_size, grid_size, D).permute(0, 3, 1, 2)

        # Now activation and gradient are (B, C, H, W)
        # Grad-CAM++ weights
        grad_2 = gradient ** 2
        grad_3 = gradient ** 3

        # Spatial sum of activation * grad^3
        spatial_sum = (activation * grad_3).sum(dim=(2, 3), keepdim=True)

        # Denominator with stability term
        denom = 2 * grad_2 + spatial_sum + 1e-8

        # Alpha weights
        alpha = grad_2 / denom  # (B, C, H, W)
        alpha = alpha * F.relu(gradient)  # Only positive gradient contributions

        # Channel weights via spatial pooling of alpha
        weights = alpha.sum(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activation).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)  # ReLU to keep only positive contributions

        # Normalize to [0, 1].
        #
        # The degeneracy test has to be RELATIVE to the map's own magnitude.
        # A previous absolute threshold (range > 1e-8) silently zeroed the whole
        # Mamba heatmap: gradients reaching the Mamba pathway are ~1e-7 where the
        # CNN's are ~1e-3, so a perfectly informative Mamba CAM had a range of
        # ~5e-9 and was discarded as if it were constant. Grad-CAM output is
        # scale-free by construction — only a genuinely *constant* map should
        # collapse to zeros, and that is what this now tests.
        cam = cam.squeeze(0).squeeze(0)  # (H, W)
        cam_min, cam_max = cam.min(), cam.max()
        cam_range = cam_max - cam_min
        scale = torch.maximum(cam_max.abs(), cam_min.abs())
        tol = torch.finfo(cam.dtype).eps * torch.clamp(scale, min=1e-30)

        if torch.isfinite(cam_range) and cam_range > tol:
            cam = (cam - cam_min) / cam_range
        else:
            # Genuinely constant map: no position is more important than another.
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy()

    def _combine_heatmaps(
        self,
        cnn_heatmap: np.ndarray,
        mamba_heatmap: np.ndarray,
        target_h: int,
        target_w: int,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Combine CNN and Mamba heatmaps into a unified dual-scale visualization.

        Args:
            cnn_heatmap: CNN pathway heatmap (h1, w1).
            mamba_heatmap: Mamba pathway heatmap (h2, w2).
            target_h: Target height (original image height).
            target_w: Target width (original image width).
            alpha: Blending weight (0 = all Mamba, 1 = all CNN).

        Returns:
            Combined heatmap (target_h, target_w), normalized to [0, 1].
        """
        # Resize both heatmaps to target size using bilinear interpolation
        cnn_resized = self._resize_heatmap(cnn_heatmap, target_h, target_w)
        mamba_resized = self._resize_heatmap(mamba_heatmap, target_h, target_w)

        # Weighted combination
        combined = alpha * cnn_resized + (1 - alpha) * mamba_resized

        # Normalize. An absolute threshold is safe *here*, unlike in
        # _compute_gradcampp: both inputs are already normalised to [0, 1], so
        # this range is O(1) unless both pathway maps are constant. Do not copy
        # this pattern back into code whose scale is unbounded.
        c_min, c_max = combined.min(), combined.max()
        if c_max - c_min > 1e-8:
            combined = (combined - c_min) / (c_max - c_min)

        return combined

    @staticmethod
    def _resize_heatmap(heatmap: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resize heatmap using bilinear interpolation."""
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
        resized = F.interpolate(
            heatmap_tensor, size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        )
        return resized.squeeze().numpy()


def visualize_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay a heatmap on an image for visualization.

    Args:
        image: Original image (H, W, 3) in [0, 255] uint8 or [0, 1] float.
        heatmap: Heatmap (H, W) in [0, 1].
        alpha: Overlay transparency.
        colormap: Matplotlib colormap name.

    Returns:
        Overlaid visualization (H, W, 3) as uint8 in [0, 255].
    """
    import matplotlib

    # Ensure image is float in [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Resize heatmap to match image
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = GradCAMPlusPlus._resize_heatmap(
            heatmap, image.shape[0], image.shape[1]
        )

    # Apply colormap. matplotlib.cm.get_cmap was deprecated in 3.7 and REMOVED
    # in 3.9, so the old call raised AttributeError on any current matplotlib
    # (requirements.txt allows >=3.8). matplotlib.colormaps is the supported
    # registry from 3.5 onward; fall back for older installs.
    try:
        cmap = matplotlib.colormaps[colormap]
    except AttributeError:  # matplotlib < 3.5
        cmap = matplotlib.cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel

    # Overlay
    overlay = (1 - alpha) * image + alpha * heatmap_colored
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return overlay
