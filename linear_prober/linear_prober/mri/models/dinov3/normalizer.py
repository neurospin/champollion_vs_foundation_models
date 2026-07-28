"""
models/dinov3/normalizer.py

DINOv3 ImageNet normalization for 2D slices.
Identical to skeleton project — agnostic to slice spatial size.

Source: DINOv3 README.md, make_transform() for LVD-1689M weights:
  mean = (0.485, 0.456, 0.406)
  std  = (0.229, 0.224, 0.225)

Our IRM slices arrive as [N, 3, H, W] float32 ∈ [0,1]
(crops already clipped to [0,1] at masking stage → no affine mapping needed).
"""

from __future__ import annotations

import torch

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

_MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)
_STD = torch.tensor(IMAGENET_STD, dtype=torch.float32)


def normalize_imagenet(slices: torch.Tensor) -> torch.Tensor:
    """
    Standard ImageNet normalization: x_norm[c] = (x[c] - mean[c]) / std[c]

    Input:  [N, 3, H, W] float32 ∈ [0, 1]
    Output: [N, 3, H, W] float32, ImageNet-normalized
    """
    mean = _MEAN.to(slices.device).reshape(1, 3, 1, 1)
    std = _STD.to(slices.device).reshape(1, 3, 1, 1)
    return (slices - mean) / std
