# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def squeeze_trailing_singleton_channel(volume_np: np.ndarray) -> np.ndarray:
    """
    Remove a trailing parasite singleton channel if present.

    Accepted inputs:
      - [D, H, W]
      - [D, H, W, 1]

    Returned:
      - [D, H, W]

    Important:
      No axis permutation is applied.
    """
    if volume_np.ndim == 4:
        if volume_np.shape[-1] != 1:
            raise ValueError(
                "Expected trailing singleton channel for 4D volume, "
                f"got shape {volume_np.shape}"
            )
        volume_np = volume_np[..., 0]

    if volume_np.ndim != 3:
        raise ValueError(
            f"Expected volume shape [D,H,W] or [D,H,W,1], got {volume_np.shape}"
        )

    return volume_np


def upscale_pad_sulcal_volume(
    volumes: torch.Tensor,
    target: int,
) -> torch.Tensor:
    """
    Process-2-compatible upscale + centered padding.

    Input:
      volumes: [B, 1, D, H, W], float32, binary {0,1}

    Output:
      volumes: [B, 1, target, target, target]

    Geometry:
      - no axis permutation
      - isotropic scale: target / max(D, H, W)
      - nearest-exact interpolation
      - centered zero-padding
    """
    if volumes.ndim != 5:
        raise ValueError(f"Expected [B,1,D,H,W], got {tuple(volumes.shape)}")

    if volumes.shape[1] != 1:
        raise ValueError(f"Expected channel dimension = 1, got {tuple(volumes.shape)}")

    T = int(target)
    _, _, D, H, W = volumes.shape

    scale = T / max(D, H, W)

    new_d = min(int(round(D * scale)), T)
    new_h = min(int(round(H * scale)), T)
    new_w = min(int(round(W * scale)), T)

    scaled = F.interpolate(
        volumes,
        size=(new_d, new_h, new_w),
        mode="nearest-exact",
    )

    pad_d = T - new_d
    pad_h = T - new_h
    pad_w = T - new_w

    pad_d_before = pad_d // 2
    pad_d_after = pad_d - pad_d_before

    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before

    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before

    padded = F.pad(
        scaled,
        (
            pad_w_before,
            pad_w_after,
            pad_h_before,
            pad_h_after,
            pad_d_before,
            pad_d_after,
        ),
        mode="constant",
        value=0,
    )

    if tuple(padded.shape[2:]) != (T, T, T):
        raise RuntimeError(
            f"Padding error: expected {(T, T, T)}, " f"got {tuple(padded.shape[2:])}"
        )

    return padded


def preprocess_sulcal_volume(
    volume_np: np.ndarray,
    target: int = 112,
    binarize_nonzero: bool = True,
) -> torch.Tensor:
    """
    Preprocess one raw sulcal skeleton volume for 3DINO SSL.

    Input:
      volume_np:
        - [D, H, W]
        - [D, H, W, 1]

    Output:
      torch.Tensor [1, target, target, target], float32, binary {0,1}

    Steps:
      1. remove trailing singleton channel if present
      2. binarize with x != 0
      3. convert to [B,1,D,H,W]
      4. isotropic upscale + centered zero-padding
      5. return [1,target,target,target]

    Important:
      No axis permutation is applied.
    """
    volume_np = squeeze_trailing_singleton_channel(volume_np)

    if binarize_nonzero:
        volume_np = (volume_np != 0).astype(np.float32)
    else:
        volume_np = volume_np.astype(np.float32)

    x = torch.from_numpy(volume_np).unsqueeze(0).unsqueeze(0)

    x = upscale_pad_sulcal_volume(
        volumes=x,
        target=target,
    )

    x = x.squeeze(0)

    # Safety: force binary after nearest-exact interpolation.
    if binarize_nonzero:
        x = (x > 0).float()

    if tuple(x.shape) != (1, target, target, target):
        raise RuntimeError(
            f"Expected output shape {(1, target, target, target)}, "
            f"got {tuple(x.shape)}"
        )

    return x
