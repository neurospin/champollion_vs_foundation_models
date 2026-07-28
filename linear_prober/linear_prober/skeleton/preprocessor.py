"""Geometric preprocessing for binary sulcal-skeleton grids.

This is modality-specific to the skeleton inputs: it resizes a native binary
volume to the frozen encoder's cubic ``target_shape``. It is agnostic to the
model and to intensity — value mapping ``[0,1] -> [v0,v1]`` lives in each
model's ``normalizer``.

Public function: :func:`preprocess_batch`.

Preprocessings (all computed on-the-fly from native volumes):
    ``upscale_pad``       — isotropic scale + centered zero-pad (default)
    ``nearest_neighbors`` — direct resize to target_shape, nearest interpolation
    ``trilinear``         — direct resize to target_shape, trilinear interpolation

Design notes (``upscale_pad``):
  Isotropic scale preserves aspect ratio; centered padding keeps the sulcal
  content near the volume centre, consistent with 3DINO's pretraining
  distribution. A ViT positional encoding is spatial, so systematically
  corner-padding a skeleton would give it wrong positional embeddings.

Design notes (``nearest_neighbors`` / ``trilinear``):
  Direct resize with no isotropic scale or padding — aspect ratio is not
  preserved. Nearest keeps binary {0,1} exactly; trilinear yields continuous
  values ∈ [0,1].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# The available geometric preprocessings.
ALL_PREPROCESSINGS = {"upscale_pad", "nearest_neighbors", "trilinear"}


# =============================================================================
# Internal helpers
# =============================================================================


def _upscale_pad(volumes: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Isotropic scale + centered zero-pad to a cubic ``target_shape``.

    Input  ``[B, 1, D, H, W]`` -> Output ``[B, 1, T, T, T]``.

    Example (FIP, target=112): ``[B,1,39,45,44]`` scaled by 112/45 to
    ``[B,1,97,112,110]`` then centre-padded to ``[B,1,112,112,112]``.
    """
    T = target_shape[0]
    _, _, D, H, W = volumes.shape

    scale = T / max(D, H, W)
    new_d = min(int(round(D * scale)), T)
    new_h = min(int(round(H * scale)), T)
    new_w = min(int(round(W * scale)), T)

    scaled = F.interpolate(volumes, size=(new_d, new_h, new_w), mode="nearest-exact")

    pad_d = T - new_d
    pad_d_b = pad_d // 2
    pad_d_a = pad_d - pad_d_b
    pad_h = T - new_h
    pad_h_b = pad_h // 2
    pad_h_a = pad_h - pad_h_b
    pad_w = T - new_w
    pad_w_b = pad_w // 2
    pad_w_a = pad_w - pad_w_b

    # F.pad order for [B, C, D, H, W] is last dim first.
    padded = F.pad(
        scaled,
        (pad_w_b, pad_w_a, pad_h_b, pad_h_a, pad_d_b, pad_d_a),
        mode="constant",
        value=0,
    )

    assert tuple(padded.shape[2:]) == (
        T,
        T,
        T,
    ), f"Padding error: expected {(T, T, T)}, got {tuple(padded.shape[2:])}"
    return padded


def _direct_resize(
    volumes: torch.Tensor, target_shape: tuple, mode: str
) -> torch.Tensor:
    """Direct resize to cubic ``target_shape`` — no isotropic scale, no padding.

    ``mode="nearest"`` preserves binary {0,1}; ``mode="trilinear"`` produces
    continuous values ∈ [0,1]. Aspect ratio is not preserved.
    """
    T = target_shape[0]
    kwargs = {"align_corners": False} if mode == "trilinear" else {}
    return F.interpolate(volumes, size=(T, T, T), mode=mode, **kwargs)


# =============================================================================
# Public dispatcher
# =============================================================================


def preprocess_batch(
    volumes: torch.Tensor,
    target_shape: tuple,
    preprocessing: str = "upscale_pad",
) -> torch.Tensor:
    """Dispatch geometric preprocessing for a batch of skeleton volumes.

    Args:
        volumes: ``[B, 1, D, H, W]`` float32.
        target_shape: cubic ``(T, T, T)`` target dimensions.
        preprocessing: one of :data:`ALL_PREPROCESSINGS`.

    Returns:
        ``[B, 1, T, T, T]`` float32.
    """
    if volumes.ndim != 5:
        raise ValueError(f"Expected [B, 1, D, H, W], got {tuple(volumes.shape)}")
    if len(target_shape) != 3:
        raise ValueError(f"target_shape must have 3 dimensions, got {target_shape}")
    if len(set(target_shape)) != 1:
        raise ValueError(f"target_shape must be cubic (T, T, T), got {target_shape}.")

    if preprocessing == "upscale_pad":
        return _upscale_pad(volumes, target_shape)
    elif preprocessing == "nearest_neighbors":
        return _direct_resize(volumes, target_shape, mode="nearest")
    elif preprocessing == "trilinear":
        return _direct_resize(volumes, target_shape, mode="trilinear")
    else:
        raise ValueError(
            f"Unknown preprocessing '{preprocessing}'. "
            f"Expected one of: {sorted(ALL_PREPROCESSINGS)}"
        )
