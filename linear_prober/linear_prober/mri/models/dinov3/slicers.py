"""
models/dinov3/slicers.py

3D → 2D slicing for the DINOv3 zero-shot probing pipeline on MRI crops.

Input volume: [1, 112, 112, 112] float32
  (center-crop 128→112 from window128 NIfTI, applied in extract_features.py)

Two slicing modes
=================

  "2d" — one slice at a time, replicated across 3 channels:
    Each of the 112 planes along an axis → [3, 112, 112] (expand, no copy).
    Output per axis: [112, 3, 112, 112]
    Total slices: 112 × 3 axes = 336

  "25d" — groups of 3 consecutive slices as RGB channels:
    Exclude slice 0 (border). Usable: 1 → 111 inclusive = 111 slices.
    Non-overlapping groups of 3: (1,2,3), (4,5,6), ..., (109,110,111) → 37 groups.
    Output per axis: [37, 3, 112, 112]
    Total slices: 37 × 3 axes = 111

Exported constants
==================
  TARGET_SIZE         : 112
  SLICER_MODES        : {"2d", "25d"}
  N_SLICES_PER_AXIS   : {"2d": 112, "25d": 37}
  AXES                : ("D", "H", "W")
"""

from __future__ import annotations

import torch

# =============================================================================
# Constants
# =============================================================================

TARGET_SIZE: int = 112

# 2.5D: exclude slice 0 → usable range [1, 111] = 111 slices
# 111 // 3 = 37 non-overlapping groups
_25D_SLICE_START: int = 1
_25D_SLICE_END: int = 112  # exclusive upper bound → indices 1..111
_25D_GROUP_SIZE: int = 3
_25D_N_GROUPS: int = 37  # 111 // 3 = 37

AXES: tuple[str, ...] = ("D", "H", "W")

SLICER_MODES: set[str] = {"2d", "25d"}

N_SLICES_PER_AXIS: dict[str, int] = {
    "2d": TARGET_SIZE,  # 112 slices per axis
    "25d": _25D_N_GROUPS,  # 37 groups per axis
}


# =============================================================================
# Internal helpers
# =============================================================================


def _extract_planes_along_axis(vol: torch.Tensor, axis: str) -> torch.Tensor:
    """
    Extract all 112 planes along the given spatial axis.

    Input:  vol [1, D, H, W]  with D = H = W = 112
    Output: [112, 112, 112]
    """
    v = vol[0]  # [D, H, W]

    if axis == "D":
        return torch.stack(torch.unbind(v, dim=0), dim=0)  # [112, H, W]
    elif axis == "H":
        return torch.stack(torch.unbind(v, dim=1), dim=0)  # [112, D, W]
    elif axis == "W":
        return torch.stack(torch.unbind(v, dim=2), dim=0)  # [112, D, H]
    else:
        raise ValueError(f"Unknown axis '{axis}'. Expected one of {AXES}.")


# =============================================================================
# Mode 2D
# =============================================================================


def _slice_axis_2d(vol: torch.Tensor, axis: str) -> torch.Tensor:
    """
    Extract 112 slices along `axis`, each replicated across 3 channels.

    Input:  vol [1, 112, 112, 112]
    Output: [112, 3, 112, 112]

    Uses expand() — no memory copy. Caller must call .contiguous() before
    passing to DINOv3.
    """
    planes = _extract_planes_along_axis(vol, axis)  # [112, 112, 112]
    planes = planes.unsqueeze(1)  # [112, 1, 112, 112]
    return planes.expand(-1, 3, -1, -1)  # [112, 3, 112, 112]


def slice_2d(vol: torch.Tensor) -> dict[str, torch.Tensor]:
    return {axis: _slice_axis_2d(vol, axis) for axis in AXES}


# =============================================================================
# Mode 2.5D
# =============================================================================


def _slice_axis_25d(vol: torch.Tensor, axis: str) -> torch.Tensor:
    """
    Extract 37 groups of 3 consecutive slices along `axis`.

    Input:  vol [1, 112, 112, 112]
    Output: [37, 3, 112, 112]

    Usable slices: 1..111 inclusive (slice 0 excluded).
    Groups: (1,2,3), (4,5,6), ..., (109,110,111) → 37 groups.
    """
    planes = _extract_planes_along_axis(vol, axis)  # [112, 112, 112]
    usable = planes[_25D_SLICE_START:_25D_SLICE_END]  # [111, 112, 112]

    assert usable.shape[0] == _25D_N_GROUPS * _25D_GROUP_SIZE, (
        f"Expected {_25D_N_GROUPS * _25D_GROUP_SIZE} usable slices, "
        f"got {usable.shape[0]}"
    )

    H, W = usable.shape[1], usable.shape[2]
    return usable.reshape(_25D_N_GROUPS, _25D_GROUP_SIZE, H, W)  # [37, 3, 112, 112]


def slice_25d(vol: torch.Tensor) -> dict[str, torch.Tensor]:
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
      volume      : [1, 112, 112, 112] float32
      slicer_mode : "2d"  → 112 slices per axis, replicated to 3 channels
                    "25d" →  37 groups of 3 consecutive slices per axis

    Returns:
      dict {"D", "H", "W"}:
        "2d"  → each value [112, 3, 112, 112]
        "25d" → each value [ 37, 3, 112, 112]
    """
    if slicer_mode not in SLICER_MODES:
        raise ValueError(
            f"Unknown slicer_mode '{slicer_mode}'. Expected one of: {sorted(SLICER_MODES)}"
        )
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError(f"Expected [1, D, H, W], got {tuple(volume.shape)}")
    spatial = tuple(volume.shape[1:])
    expected = (TARGET_SIZE, TARGET_SIZE, TARGET_SIZE)
    if spatial != expected:
        raise ValueError(f"Expected spatial shape {expected}, got {spatial}.")

    if slicer_mode == "2d":
        return slice_2d(volume)
    else:
        return slice_25d(volume)
