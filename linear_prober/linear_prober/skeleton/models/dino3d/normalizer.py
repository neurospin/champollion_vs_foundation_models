"""
models/dino3d/normalizer.py

3DINO-specific intensity normalization for sulcal skeleton volumes.

3DINO was pretrained on MRI data normalized to [-1, 1] via:
  ScaleIntensityRangePercentilesd(lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True)

In that pipeline, MRI background (dark regions) is clipped to -1,
confirmed by: CropForegroundSwapSliceDims(select_fn=lambda x: x > -1)
i.e. foreground = everything strictly above -1.

For binary sulcal skeletons {0, 1}:
  0 (background voxel) → -1   (consistent with MRI background = dark = -1)
  1 (skeleton voxel)   → +1   (consistent with MRI bright structure)

Default formula: v0 + volume * (v1 - v0)
  with v0=-1.0, v1=+1.0  →  volume * 2 - 1  (identical to original hardcoded behaviour)

MODEL_RANGE: exported constant — used by the normaliser search to
  express (p0, p1) as percentages of the model's expected input range.
  Must match config_probe_dino3d.yaml → model_normalization.range.

Note: this is applied AFTER preprocess_batch and BEFORE the GPU forward pass
  for binary-preserving preprocessings (upscale_pad, nearest_neighbors,
 ). For continuous preprocessings (trilinear),
  the normaliser search applies normalize() BEFORE preprocess_batch.
"""

from __future__ import annotations

import torch

# Exported constant — read by the normaliser search
# Must be consistent with config_probe_dino3d.yaml → model_normalization.range
MODEL_RANGE: tuple[float, float] = (-1.0, 1.0)


def normalize(
    volume: torch.Tensor,
    v0: float = -1.0,
    v1: float = 1.0,
) -> torch.Tensor:
    """
    Linear map: 0.0 → v0,  1.0 → v1.

    Formula: v0 + volume * (v1 - v0)

    Default behaviour (v0=-1.0, v1=1.0):
      Maps {0.0, 1.0} → {-1.0, +1.0}  (identical to original `volume * 2 - 1`)

    For the normaliser search, v0 and v1 are derived from percentages
    (p0, p1) over MODEL_RANGE = (-1.0, +1.0):
      v0 = alpha + p0 * (beta - alpha)
      v1 = alpha + p1 * (beta - alpha)

    Works for ALL preprocessings:
      - Binary inputs  {0.0, 1.0}  → maps exactly to {v0, v1}
      - Continuous inputs [0.0, 1.0] → maps linearly to [v0, v1]

    Input:  [B, 1, T, T, T] float32
    Output: [B, 1, T, T, T] float32
    """
    return v0 + volume * (v1 - v0)
