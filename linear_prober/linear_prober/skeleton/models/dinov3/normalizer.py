"""
models/dinov3/normalizer.py

DINOv3-specific intensity normalization for sulcal skeleton slices.

DINOv3 was pretrained on web images (LVD-1689M dataset) using the standard
ImageNet evaluation transform:

  1. ToImage()                        — convert to tensor
  2. Resize((224, 224), antialias=True) — already done by preprocessor.py
  3. ToDtype(float32, scale=True)     — divide by 255: [0,255] → [0,1]
                                        already done: our slices are in [0,1]
  4. Normalize(mean, std)             — THIS is what normalizer.py does

Source: official DINOv3 repo (facebookresearch/dinov3), README.md
  make_transform() for LVD-1689M weights:
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

Input contract
==============
Slices arrive from slicers.py as [N_slices, 3, 224, 224] float32
with values in [v0, v1] after the affine intensity mapping:

  phi(x) = v0 + x * (v1 - v0)   (from normalizer grid search)

  For binary inputs {0.0, 1.0}:   values are exactly {v0, v1}
  For continuous inputs [0.0, 1.0]: values are in [v0, v1]

  With default full range (p0=0.0, p1=1.0) and MODEL_RANGE=(0.0, 1.0):
    v0=0.0, v1=1.0 → no change, slices already in [0,1]

The ImageNet normalization is then applied on top:
  x_norm[c] = (x[c] - mean[c]) / std[c]   for c in {0, 1, 2}

Since our slices are grayscale replicated across 3 channels (all 3 channels
are identical), the 3 mean/std values differ by channel — this is intentional
and consistent with the pretraining distribution. The model has learned to
process RGB images where channels carry different statistics.

Normalizer grid search
======================
MODEL_RANGE = (0.0, 1.0): the affine mapping phi(x) searches (p0, p1) over
[0,1]^2, mapping binary voxels {0,1} to {v0, v1} ⊆ [0.0, 1.0].
This is applied BEFORE the ImageNet normalization.

Regime detection (same logic as other models):
  Binary-preserving (upscale_pad, nearest_neighbors):
    preprocess_batch → phi(x) → normalize_imagenet → encoder
  Continuous (trilinear):
    phi(x) → preprocess_batch → normalize_imagenet → encoder

Two-step normalization in extract_features.py:
  1. normalize_affine(slice, v0, v1)    — affine mapping {0,1} → {v0,v1}
  2. normalize_imagenet(slice)          — ImageNet standardization

Both are also available as a single call:
  normalize(slice, v0, v1)              — affine then ImageNet

Exported constants
==================
  MODEL_RANGE       : (0.0, 1.0) — expected input range before ImageNet norm
  IMAGENET_MEAN     : (0.485, 0.456, 0.406)
  IMAGENET_STD      : (0.229, 0.224, 0.225)
"""

from __future__ import annotations

import torch

# =============================================================================
# Constants
# =============================================================================

# Expected input range for the affine mapping grid search.
# Slices are binary {0,1} in [0.0, 1.0] before the affine mapping.
# Must be consistent with config_probe_dinov3.yaml → model_normalization.range
MODEL_RANGE: tuple[float, float] = (0.0, 1.0)

# ImageNet normalization parameters — LVD-1689M weights only.
# Source: facebookresearch/dinov3 README.md, make_transform() for LVD-1689M.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Pre-built tensors — registered once, reshaped at call time for broadcasting.
# Shape will be [1, 3, 1, 1] when used on [N, 3, H, W] batches.
_MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)
_STD = torch.tensor(IMAGENET_STD, dtype=torch.float32)


# =============================================================================
# Step 1 — Affine intensity mapping (normalizer grid search)
# =============================================================================


def normalize_affine(
    slices: torch.Tensor,
    v0: float = 0.0,
    v1: float = 1.0,
) -> torch.Tensor:
    """
    Affine intensity mapping: 0.0 → v0,  1.0 → v1.

    Formula: v0 + slices * (v1 - v0)

    Default (v0=0.0, v1=1.0): identity — no change.

    For the normaliser search, v0 and v1 are derived from percentages
    (p0, p1) over MODEL_RANGE = (0.0, 1.0):
      v0 = 0.0 + p0 * (1.0 - 0.0) = p0
      v1 = 0.0 + p1 * (1.0 - 0.0) = p1

    Works for ALL input types:
      Binary {0.0, 1.0}   → maps exactly to {v0, v1}
      Continuous [0.0, 1.0] → maps linearly to [v0, v1]

    Input:  [N, 3, H, W] float32  (N = number of slices in the batch)
    Output: [N, 3, H, W] float32  in [v0, v1]
    """
    return v0 + slices * (v1 - v0)


# =============================================================================
# Step 2 — ImageNet normalization
# =============================================================================


def normalize_imagenet(slices: torch.Tensor) -> torch.Tensor:
    """
    Standard ImageNet normalization: x_norm[c] = (x[c] - mean[c]) / std[c]

    Applied channel-wise on 3-channel slices.
    Mean and std are moved to the same device as `slices` on first use.

    Input:  [N, 3, H, W] float32  in [v0, v1] ⊆ [0.0, 1.0]
    Output: [N, 3, H, W] float32  in approximately [-2.1, 2.6]
            (range depends on v0, v1; full range [0,1] → [-2.12, 2.64])

    Note: with full range (v0=0, v1=1) and identical channels (grayscale):
      channel 0: (x - 0.485) / 0.229
      channel 1: (x - 0.406) / 0.225   ← different mean/std per channel
      channel 2: (x - 0.406) / 0.225   ← intentional: matches RGB pretraining
    """
    mean = _MEAN.to(slices.device).reshape(1, 3, 1, 1)
    std = _STD.to(slices.device).reshape(1, 3, 1, 1)
    return (slices - mean) / std


# =============================================================================
# Combined normalization (affine + ImageNet)
# =============================================================================


def normalize(
    slices: torch.Tensor,
    v0: float = 0.0,
    v1: float = 1.0,
) -> torch.Tensor:
    """
    Full normalization pipeline: affine mapping then ImageNet standardization.

    Equivalent to:
      normalize_imagenet(normalize_affine(slices, v0, v1))

    Used by extract_features.py for standard extraction (non-grid-search).
    For the normalizer grid search, normalize_affine and normalize_imagenet
    are called separately to allow regime-dependent ordering
    (see extract_mean_pool_for_mapping in extract_features.py).

    Input:  [N, 3, H, W] float32  in [0.0, 1.0]
    Output: [N, 3, H, W] float32  ImageNet-normalized
    """
    return normalize_imagenet(normalize_affine(slices, v0, v1))
