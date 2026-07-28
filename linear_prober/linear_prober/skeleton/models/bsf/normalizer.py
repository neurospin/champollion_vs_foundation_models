"""
models/bsf/normalizer.py

Intensity normalization for BrainSegFounder inputs.

SCIENTIFIC CONTEXT
==================
BrainSegFounder (SSLHead) was pretrained on UK Biobank MRI data using
standard MONAI transforms. The SwinViT encoder expects inputs normalized
to [0.0, 1.0] — consistent with typical MRI preprocessing pipelines.

Our binary skeleton volumes {0, 1} are already in [0.0, 1.0].
BrainSegFounder applies NO internal normalization module after the input.
Our normalize() is therefore a pure affine mapping:
  φ(x) = v0 + x * (v1 - v0)   →   0 ↦ v0,  1 ↦ v1

Default mapping (v0=0.0, v1=1.0) is identity — background stays at 0.0,
foreground stays at 1.0, matching BSF's expected input range.

MODEL_RANGE = (0.0, 1.0)
"""

from __future__ import annotations

import torch

MODEL_RANGE = (0.0, 1.0)


def normalize(x: torch.Tensor, v0: float, v1: float) -> torch.Tensor:
    """
    Apply affine intensity mapping: φ(x) = v0 + x * (v1 - v0).

    Maps binary volumes {0.0, 1.0} → {v0, v1} with v0, v1 ∈ [0.0, 1.0].
    Default mapping (v0=0.0, v1=1.0) is identity.

    Args:
        x  : float32 tensor, values in {0.0, 1.0}
        v0 : value assigned to background voxels (label=0)
        v1 : value assigned to foreground voxels (label=1)

    Returns:
        float32 tensor in [v0, v1]
    """
    return v0 + x * (v1 - v0)
