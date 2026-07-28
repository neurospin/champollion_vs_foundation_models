"""
models/vista3d/normalizer.py

Intensity normalization for VISTA3D inputs.

SCIENTIFIC CONTEXT
==================
VISTA3D was pretrained on CT scans normalized via ScaleIntensityRanged:
  a_min=-963.82 HU, a_max=1053.68 HU → b_min=0.0, b_max=1.0, clip=True

Our binary skeleton volumes {0, 1} are already in [0.0, 1.0] — they fall
naturally into VISTA3D's expected input range without any rescaling.

VISTA3D applies NO internal normalization module (unlike SAM-Med3D which
applies a fixed mean/std standardization). Our normalize() is therefore
a pure affine mapping identical to 3DINO and SAM-Med3D:
  φ(x) = v0 + x * (v1 - v0)   →   0 ↦ v0,  1 ↦ v1

Default mapping (v0=0.0, v1=1.0) is the identity transform — background
stays at 0.0, foreground stays at 1.0, matching VISTA3D's expected range.

MODEL_RANGE = (0.0, 1.0)
"""

from __future__ import annotations

import torch

# VISTA3D expects inputs in [0.0, 1.0] — same as SAM-Med3D
MODEL_RANGE = (0.0, 1.0)


def normalize(x: torch.Tensor, v0: float, v1: float) -> torch.Tensor:
    """
    Apply affine intensity mapping: φ(x) = v0 + x * (v1 - v0).

    Maps binary volumes {0.0, 1.0} → {v0, v1} with v0, v1 ∈ [0.0, 1.0].
    Default mapping (v0=0.0, v1=1.0) is identity — no change applied.

    Args:
        x  : float32 tensor, values in {0.0, 1.0}
        v0 : value assigned to background voxels (label=0)
        v1 : value assigned to foreground voxels (label=1)

    Returns:
        float32 tensor in [v0, v1]
    """
    return v0 + x * (v1 - v0)
