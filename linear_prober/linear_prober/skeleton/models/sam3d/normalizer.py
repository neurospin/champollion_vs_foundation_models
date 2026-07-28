"""
models/sam3d/normalizer.py

SAM-Med3D-specific intensity normalisation for sulcal skeleton volumes.

SAM-Med3D (vit_b_ori) was pretrained on medical volumes normalised with a
fixed pipeline that maps intensities to a clinical [0, 255] grey-level scale
and then standardises with ImageNet-like statistics (channel 0 only,
since the model operates on single-channel 3D volumes):

  pixel_mean = 123.675   (= ImageNet RGB mean, channel R, reused for grayscale)
  pixel_std  = 58.395    (= ImageNet RGB std,  channel R, reused for grayscale)

Full normalisation pipeline applied by this module:
  step 1 — optimal mapping  : x ∈ {0.0, 1.0} → {v0, v1}  with v0, v1 ∈ [0, 1]
                               formula: x = v0 + volume * (v1 - v0)
  step 2 — rescale to [0,255]: x = x * 255.0
  step 3 — fixed standardisation:
               x = (x - pixel_mean) / pixel_std
             = (x - 123.675) / 58.395

Resulting output range for default mapping (v0=0.0, v1=1.0):
  background voxel (value 0): (  0 - 123.675) / 58.395 = -2.116
  active voxel     (value 1): (255 - 123.675) / 58.395 = +2.247

MODEL_RANGE = (0.0, 1.0) — exported constant read by the normaliser search.
  p0/p1 percentages are expressed in [0, 1] BEFORE the fixed normalisation,
  which gives them the same intuitive meaning as for 3DINO:
    p0 = 0% → background voxels receive the minimum input value (0.0)
    p1 = 100% → active voxels receive the maximum input value (1.0)

Note: this module is applied AFTER preprocess_batch and BEFORE the GPU forward
  pass for binary-preserving preprocessings (upscale_pad, nearest_neighbors,
 ). For continuous preprocessings (trilinear),
  the normaliser search applies normalize() BEFORE preprocess_batch.
"""

from __future__ import annotations

import torch

# Exported constant — read by the normaliser search
# Expressed in input space BEFORE the fixed SAM-Med3D normalisation.
# Must be consistent with config_probe_sam3d.yaml → model_normalization.range.
MODEL_RANGE: tuple[float, float] = (0.0, 1.0)

# Fixed SAM-Med3D normalisation constants — channel 0 (grayscale)
# Source: build_sam3D.py → pixel_mean=[123.675, 116.28, 103.53]
#                           pixel_std =[58.395,  57.12,  57.375]
_PIXEL_MEAN: float = 123.675
_PIXEL_STD: float = 58.395


def normalize(
    volume: torch.Tensor,
    v0: float = 0.0,
    v1: float = 1.0,
) -> torch.Tensor:
    """
    Apply the full SAM-Med3D normalisation pipeline with optional optimal mapping.

    Pipeline:
      1. Optimal mapping : 0.0 → v0,  1.0 → v1   (linear, formula: v0 + x*(v1-v0))
      2. Rescale         : x = x * 255.0
      3. Standardise     : x = (x - 123.675) / 58.395

    Default behaviour (v0=0.0, v1=1.0):
      Reproduces the original SAM-Med3D preprocessing pipeline exactly:
        background voxels (0.0) → (  0 - 123.675) / 58.395 = -2.116
        active voxels     (1.0) → (255 - 123.675) / 58.395 = +2.247

    With optimal mapping (v0, v1 from resolve_mapping):
      Example p0=0.1, p1=0.9 → v0=0.1, v1=0.9:
        background: (0.1*255 - 123.675) / 58.395 = -1.679
        active:     (0.9*255 - 123.675) / 58.395 = +1.809

    Works for ALL preprocessings:
      Binary inputs  {0.0, 1.0}  → maps exactly to standardised {v0_norm, v1_norm}
      Continuous inputs [0.0, 1.0] → maps linearly through the full pipeline

    Input:  [B, 1, T, T, T] float32
    Output: [B, 1, T, T, T] float32  (standardised, range ~[-2.1, +2.2] by default)
    """
    # Step 1 — optimal mapping: {0,1} → {v0,v1}
    x = v0 + volume * (v1 - v0)
    # Step 2 — rescale to [0, 255] grey-level range
    x = x * 255.0
    # Step 3 — fixed SAM-Med3D standardisation
    x = (x - _PIXEL_MEAN) / _PIXEL_STD
    return x
