"""
models/dinov3/slicers.py

3D → 2D slicing module for the DINOv3 zero-shot probing pipeline.

Converts a preprocessed 3D volume [1, 224, 224, 224] into a set of 2D
slices [N_slices, 3, 224, 224] ready to be fed into DINOv3.

Slicing is always tri-axial: performed independently along axes D, H, W.
The output is a dict with keys "D", "H", "W", one tensor per axis.

Two slicing modes
=================

  "2d" — one slice at a time, replicated across 3 channels:
    Each of the 224 planes along an axis is extracted as [224, 224],
    then replicated to [3, 224, 224] (no memory copy — expand).
    Output per axis: [224, 3, 224, 224]
    Total slices: 224 × 3 axes = 672

  "25d" — groups of 3 consecutive slices as RGB channels:
    Slices 0 and 223 are never used (border exclusion).
    Usable slices: 1 → 222 inclusive = 222 slices per axis.
    Non-overlapping groups of 3: (1,2,3), (4,5,6), ..., (220,221,222)
    → 74 groups per axis.
    Each group stacks 3 consecutive planes → [3, 224, 224].
    Output per axis: [74, 3, 224, 224]
    Total slices: 74 × 3 axes = 222

Design notes
============
  - Operates on a single volume [1, 224, 224, 224], NOT on a batch.
    Batching of slices for the DINOv3 forward pass is handled in
    extract_features.py (slice_batch_size controls GPU memory).

  - "2d" uses expand() instead of repeat() — no memory copy.
    The [224, 3, 224, 224] tensor shares storage with the source volume.
    extract_features.py calls .contiguous() before passing to the model.

  - "25d" uses torch.stack() — produces a new contiguous tensor.

  - No normalisation, no model forward, no aggregation here.
    This module is purely geometric.

  - Target volume size is hardcoded to 224 (= preprocessor target_shape).
    If target_shape changes, update TARGET_SIZE accordingly.

Exported constants (used by extract_features.py and aggregator.py)
===================================================================
  SLICER_MODES        : set of valid mode strings
  N_SLICES_PER_AXIS   : dict mapping mode → number of slices per axis
  AXES                : ordered tuple of axis names ("D", "H", "W")
  TARGET_SIZE         : expected spatial size of the input volume (224)

Usage
=====
  from linear_prober.skeleton.models.dinov3.slicers import get_slices, SLICER_MODES, N_SLICES_PER_AXIS

  # volume: [1, 224, 224, 224] float32, already preprocessed + normalised
  slices = get_slices(volume, slicer_mode="2d")
  # slices["D"]: [224, 3, 224, 224]
  # slices["H"]: [224, 3, 224, 224]
  # slices["W"]: [224, 3, 224, 224]

  slices = get_slices(volume, slicer_mode="25d")
  # slices["D"]: [74, 3, 224, 224]
  # slices["H"]: [74, 3, 224, 224]
  # slices["W"]: [74, 3, 224, 224]
"""

from __future__ import annotations

import torch

# =============================================================================
# Constants
# =============================================================================

TARGET_SIZE: int = 224

# 2.5D: exclude border slices 0 and 223 → usable range [1, 222]
# 222 usable slices / 3 per group = 74 non-overlapping groups
_25D_SLICE_START: int = 1
_25D_SLICE_END: int = 223  # exclusive upper bound
_25D_GROUP_SIZE: int = 3
_25D_N_GROUPS: int = 74  # (223 - 1) // 3 = 74

AXES: tuple[str, ...] = ("D", "H", "W")

SLICER_MODES: set[str] = {"2d", "25d"}

N_SLICES_PER_AXIS: dict[str, int] = {
    "2d": TARGET_SIZE,  # 224 slices per axis
    "25d": _25D_N_GROUPS,  # 74 groups per axis
}


# =============================================================================
# Internal helpers — axis extraction
# =============================================================================


def _extract_planes_along_axis(
    vol: torch.Tensor,
    axis: str,
) -> torch.Tensor:
    """
    Extract all 224 planes along the given spatial axis.

    Input:  vol [1, D, H, W]  with D = H = W = 224
    Output: [224, 224, 224]   planes stacked along dim 0

    axis="D" → vol[0, :, :, :]   each plane is [H, W] = [224, 224]
    axis="H" → vol[0, :, :, :]ᵀ  each plane is [D, W] = [224, 224]
    axis="W" → vol[0, :, :, :]ᵀ  each plane is [D, H] = [224, 224]

    Implementation uses torch.unbind along the appropriate dimension,
    which avoids copies — each element shares storage with vol.
    """
    v = vol[0]  # [D, H, W]

    if axis == "D":
        # planes: v[0], v[1], ..., v[223]  — each [H, W]
        planes = torch.stack(torch.unbind(v, dim=0), dim=0)  # [224, H, W]
    elif axis == "H":
        # planes: v[:, 0, :], ..., v[:, 223, :]  — each [D, W]
        planes = torch.stack(torch.unbind(v, dim=1), dim=0)  # [224, D, W]
    elif axis == "W":
        # planes: v[:, :, 0], ..., v[:, :, 223]  — each [D, H]
        planes = torch.stack(torch.unbind(v, dim=2), dim=0)  # [224, D, H]
    else:
        raise ValueError(f"Unknown axis '{axis}'. Expected one of {AXES}.")

    return planes  # [224, 224, 224]


# =============================================================================
# Mode 2D
# =============================================================================


def _slice_axis_2d(vol: torch.Tensor, axis: str) -> torch.Tensor:
    """
    Extract 224 slices along `axis`, each replicated across 3 channels.

    Input:  vol [1, 224, 224, 224]
    Output: [224, 3, 224, 224]

    Each plane [224, 224] is unsqueezed to [1, 1, 224, 224],
    then expanded to [1, 3, 224, 224] — no memory copy.
    All planes are then concatenated along dim 0.

    Note: expand() returns a view; contiguous() must be called before
    passing to the DINOv3 model (done in extract_features.py).
    """
    planes = _extract_planes_along_axis(vol, axis)  # [224, 224, 224]

    # unsqueeze channel dim: [224, 1, 224, 224]
    planes = planes.unsqueeze(1)

    # replicate across 3 channels (no copy): [224, 3, 224, 224]
    slices = planes.expand(-1, 3, -1, -1)

    return slices  # [224, 3, 224, 224]


def slice_2d(vol: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    2D slicing: 224 slices per axis, each replicated to 3 channels.

    Input:  vol [1, 224, 224, 224]
    Output: {"D": [224, 3, 224, 224],
             "H": [224, 3, 224, 224],
             "W": [224, 3, 224, 224]}
    """
    return {axis: _slice_axis_2d(vol, axis) for axis in AXES}


# =============================================================================
# Mode 2.5D
# =============================================================================


def _slice_axis_25d(vol: torch.Tensor, axis: str) -> torch.Tensor:
    """
    Extract 74 groups of 3 consecutive slices along `axis`.

    Input:  vol [1, 224, 224, 224]
    Output: [74, 3, 224, 224]

    Usable slices: indices 1 to 222 inclusive (borders 0 and 223 excluded).
    Groups: (1,2,3), (4,5,6), ..., (220,221,222) → 74 non-overlapping groups.
    Each group stacks 3 planes → [3, 224, 224].
    All groups stacked → [74, 3, 224, 224].

    Implementation:
      1. Extract all 224 planes → [224, 224, 224]
      2. Slice usable range [1:223] → [222, 224, 224]
      3. Reshape to [74, 3, 224, 224]
         (74 groups × 3 consecutive planes × 224 × 224)
    """
    planes = _extract_planes_along_axis(vol, axis)  # [224, 224, 224]

    # Keep only usable range: slices 1..222 inclusive
    usable = planes[_25D_SLICE_START:_25D_SLICE_END]  # [222, 224, 224]

    # Sanity check
    assert usable.shape[0] == _25D_N_GROUPS * _25D_GROUP_SIZE, (
        f"Expected {_25D_N_GROUPS * _25D_GROUP_SIZE} usable slices, "
        f"got {usable.shape[0]}"
    )

    # Reshape to [74, 3, 224, 224]
    H, W = usable.shape[1], usable.shape[2]
    groups = usable.reshape(_25D_N_GROUPS, _25D_GROUP_SIZE, H, W)

    return groups  # [74, 3, 224, 224]


def slice_25d(vol: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    2.5D slicing: 74 groups of 3 consecutive slices per axis.

    Input:  vol [1, 224, 224, 224]
    Output: {"D": [74, 3, 224, 224],
             "H": [74, 3, 224, 224],
             "W": [74, 3, 224, 224]}
    """
    return {axis: _slice_axis_25d(vol, axis) for axis in AXES}


# =============================================================================
# Public dispatcher
# =============================================================================


def get_slices(
    volume: torch.Tensor,
    slicer_mode: str,
) -> dict[str, torch.Tensor]:
    """
    Dispatch 3D → 2D slicing for a single preprocessed volume.

    Arguments:
      volume      : [1, 224, 224, 224] float32, already preprocessed
                    and normalised (output of preprocessor.py + normalizer.py)
      slicer_mode : "2d"  → 224 slices per axis, replicated to 3 channels
                    "25d" → 74 groups of 3 consecutive slices per axis

    Returns:
      dict with keys "D", "H", "W", one tensor per axis:
        "2d"  → each value [224, 3, 224, 224]
        "25d" → each value [ 74, 3, 224, 224]

    Raises:
      ValueError : if slicer_mode is unknown or volume shape is wrong
    """
    # --- Input validation ---
    if slicer_mode not in SLICER_MODES:
        raise ValueError(
            f"Unknown slicer_mode '{slicer_mode}'. "
            f"Expected one of: {sorted(SLICER_MODES)}"
        )

    if volume.ndim != 4:
        raise ValueError(
            f"Expected volume of shape [1, D, H, W], got {tuple(volume.shape)}"
        )

    if volume.shape[0] != 1:
        raise ValueError(
            f"get_slices operates on a single volume (batch dim must be 1), "
            f"got shape {tuple(volume.shape)}. "
            f"Loop over the batch in extract_features.py."
        )

    spatial = tuple(volume.shape[1:])
    expected = (TARGET_SIZE, TARGET_SIZE, TARGET_SIZE)
    if spatial != expected:
        raise ValueError(
            f"Expected spatial shape {expected}, got {spatial}. "
            f"Run preprocessor.preprocess_batch with target_shape={list(expected)} first."
        )

    # --- Dispatch ---
    if slicer_mode == "2d":
        return slice_2d(volume)
    else:
        return slice_25d(volume)
