# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

from __future__ import annotations

from numbers import Integral
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

Shape3D = Tuple[int, int, int]
Padding3D = Tuple[
    Tuple[int, int],
    Tuple[int, int],
    Tuple[int, int],
]


def _require_positive_int(value: Any, name: str) -> int:
    """
    Validate one strictly positive integer.

    Booleans are rejected explicitly because bool is a subclass of int.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"{name} must be an integer, got " f"{value!r} ({type(value).__name__})."
        )

    value = int(value)

    if value <= 0:
        raise ValueError(f"{name} must be strictly positive, got {value}.")

    return value


def normalize_target_shape(
    target: Any,
    name: str = "target_shape",
) -> Shape3D:
    """
    Validate and normalize an explicit 3D spatial shape.

    Accepted examples:
        [32, 112, 96]
        (32, 112, 96)
        OmegaConf ListConfig([32, 112, 96])

    A scalar is deliberately rejected. The dedicated anisotropic pipeline
    requires all three spatial dimensions to be explicit.

    Args:
        target:
            Sequence containing exactly three positive integers.

        name:
            Human-readable field name used in error messages.

    Returns:
        A Python tuple:
            (target_d, target_h, target_w)

    Raises:
        TypeError:
            If target is not an iterable sequence of integers.

        ValueError:
            If target does not contain exactly three strictly positive values.
    """
    if isinstance(target, (str, bytes)):
        raise TypeError(
            f"{name} must be a sequence of three integers, " f"got {target!r}."
        )

    try:
        dimensions = tuple(target)
    except TypeError as error:
        raise TypeError(
            f"{name} must be an explicit sequence of three integers, "
            f"got {target!r} ({type(target).__name__})."
        ) from error

    if len(dimensions) != 3:
        raise ValueError(
            f"{name} must contain exactly three dimensions, "
            f"got {len(dimensions)} values: {dimensions!r}."
        )

    return tuple(
        _require_positive_int(
            dimension,
            f"{name}[{axis}]",
        )
        for axis, dimension in enumerate(dimensions)
    )


def squeeze_trailing_singleton_channel(
    volume_np: np.ndarray,
) -> np.ndarray:
    """
    Remove an optional trailing singleton channel.

    Accepted inputs:
        [D, H, W]
        [D, H, W, 1]

    Returned shape:
        [D, H, W]

    Important:
        No transpose, permutation or axis reordering is performed.
    """
    if not isinstance(volume_np, np.ndarray):
        raise TypeError(
            "volume_np must be a NumPy array, " f"got {type(volume_np).__name__}."
        )

    if volume_np.ndim == 4:
        if volume_np.shape[-1] != 1:
            raise ValueError(
                "Expected a trailing singleton channel for a 4D volume "
                "[D,H,W,1], "
                f"got shape {tuple(volume_np.shape)}."
            )

        # Remove only the final singleton channel.
        # The D, H and W axes retain their original order.
        volume_np = volume_np[..., 0]

    if volume_np.ndim != 3:
        raise ValueError(
            "Expected volume shape [D,H,W] or [D,H,W,1], "
            f"got {tuple(volume_np.shape)}."
        )

    if any(int(dimension) <= 0 for dimension in volume_np.shape):
        raise ValueError(
            "All source spatial dimensions must be strictly positive, "
            f"got {tuple(volume_np.shape)}."
        )

    return volume_np


def compute_isotropic_resize_shape(
    source_shape: Any,
    target_shape: Any,
) -> Shape3D:
    """
    Compute the largest isotropic resize that fits inside a 3D target shape.

    The same scale factor is applied to all three source axes:

        scale = min(
            target_d / source_d,
            target_h / source_h,
            target_w / source_w,
        )

    Then:

        new_d = round(source_d * scale)
        new_h = round(source_h * scale)
        new_w = round(source_w * scale)

    Example:
        source = (18, 73, 57)
        target = (32, 112, 96)

        scale = min(
            32 / 18,
            112 / 73,
            96 / 57,
        )

        resize shape = (28, 112, 87)

    Args:
        source_shape:
            Source shape in [D,H,W] order.

        target_shape:
            Target shape in [D,H,W] order.

    Returns:
        Resized spatial shape in [D,H,W] order.
    """
    source_d, source_h, source_w = normalize_target_shape(
        source_shape,
        name="source_shape",
    )
    target_d, target_h, target_w = normalize_target_shape(
        target_shape,
        name="target_shape",
    )

    scale = min(
        target_d / source_d,
        target_h / source_h,
        target_w / source_w,
    )

    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(
            f"Invalid isotropic scale computed from "
            f"source_shape={(source_d, source_h, source_w)} and "
            f"target_shape={(target_d, target_h, target_w)}: {scale}."
        )

    resized_shape = (
        max(1, int(round(source_d * scale))),
        max(1, int(round(source_h * scale))),
        max(1, int(round(source_w * scale))),
    )

    for axis, (resized_dimension, target_dimension) in enumerate(
        zip(
            resized_shape,
            (target_d, target_h, target_w),
        )
    ):
        if resized_dimension > target_dimension:
            raise RuntimeError(
                "The computed isotropic resize does not fit inside the target: "
                f"resized_shape[{axis}]={resized_dimension} exceeds "
                f"target_shape[{axis}]={target_dimension}. "
                f"source_shape={(source_d, source_h, source_w)}, "
                f"target_shape={(target_d, target_h, target_w)}, "
                f"scale={scale}."
            )

    return resized_shape


def compute_centered_padding(
    resized_shape: Any,
    target_shape: Any,
) -> Padding3D:
    """
    Compute centered zero-padding from a resized shape to a target shape.

    For odd total padding, the additional voxel is placed after the volume.

    Example:
        resized = (28, 112, 87)
        target  = (32, 112, 96)

        D: 4 voxels -> (2, 2)
        H: 0 voxels -> (0, 0)
        W: 9 voxels -> (4, 5)

    Args:
        resized_shape:
            Spatial shape after interpolation, in [D,H,W] order.

        target_shape:
            Final spatial shape, in [D,H,W] order.

    Returns:
        Padding represented as:

            (
                (pad_d_before, pad_d_after),
                (pad_h_before, pad_h_after),
                (pad_w_before, pad_w_after),
            )
    """
    resized_shape = normalize_target_shape(
        resized_shape,
        name="resized_shape",
    )
    target_shape = normalize_target_shape(
        target_shape,
        name="target_shape",
    )

    axis_paddings = []

    for axis, (resized_dimension, target_dimension) in enumerate(
        zip(resized_shape, target_shape)
    ):
        if resized_dimension > target_dimension:
            raise ValueError(
                f"resized_shape[{axis}]={resized_dimension} exceeds "
                f"target_shape[{axis}]={target_dimension}."
            )

        total_padding = target_dimension - resized_dimension

        padding_before = total_padding // 2
        padding_after = total_padding - padding_before

        axis_paddings.append((padding_before, padding_after))

    return tuple(axis_paddings)


def _assert_binary_tensor(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Verify that a tensor contains only exact values 0 and 1.
    """
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty.")

    is_binary = torch.logical_or(
        tensor == 0,
        tensor == 1,
    ).all()

    if not bool(is_binary.item()):
        unique_values = torch.unique(tensor.detach().cpu())

        raise RuntimeError(
            f"{name} is expected to be binary {{0,1}}, "
            f"but contains values such as "
            f"{unique_values[:10].tolist()}."
        )


def upscale_pad_sulcal_volume_anisotropic(
    volumes: torch.Tensor,
    target_shape: Any,
    enforce_binary: bool = False,
) -> torch.Tensor:
    """
    Apply isotropic resize and centered padding to a batch of 3D volumes.

    Input:
        volumes:
            Tensor [B,1,D,H,W].

        target_shape:
            Explicit target spatial shape:
                (target_d, target_h, target_w)

        enforce_binary:
            If True:
                - validate that the input is binary;
                - force the final output to exact values {0,1};
                - validate that the final output is binary.

    Output:
        Tensor:
            [B,1,target_d,target_h,target_w]

    Geometry:
        - no axis permutation;
        - one common isotropic scale factor;
        - nearest-exact interpolation;
        - centered zero-padding;
        - extra padding voxel placed after the volume on odd axes.
    """
    if not isinstance(volumes, torch.Tensor):
        raise TypeError(
            "volumes must be a torch.Tensor, " f"got {type(volumes).__name__}."
        )

    if volumes.ndim != 5:
        raise ValueError(
            "Expected volumes with shape [B,1,D,H,W], " f"got {tuple(volumes.shape)}."
        )

    batch_size, channels, source_d, source_h, source_w = volumes.shape

    if batch_size <= 0:
        raise ValueError(
            f"Batch dimension must be strictly positive, got {batch_size}."
        )

    if channels != 1:
        raise ValueError(
            "Expected exactly one input channel in [B,1,D,H,W], "
            f"got shape {tuple(volumes.shape)}."
        )

    source_shape = (
        int(source_d),
        int(source_h),
        int(source_w),
    )

    if any(dimension <= 0 for dimension in source_shape):
        raise ValueError(
            "All source spatial dimensions must be strictly positive, "
            f"got {source_shape}."
        )

    target_shape = normalize_target_shape(
        target_shape,
        name="target_shape",
    )

    if enforce_binary:
        _assert_binary_tensor(
            volumes,
            name="Input sulcal volumes",
        )

    # F.interpolate requires a floating-point tensor.
    if not volumes.is_floating_point():
        volumes = volumes.float()

    # The preprocessing contract uses float32 independently of the raw input
    # dtype. Mixed precision is applied later by the collate function.
    volumes = volumes.to(dtype=torch.float32)

    resized_shape = compute_isotropic_resize_shape(
        source_shape=source_shape,
        target_shape=target_shape,
    )

    resized = F.interpolate(
        volumes,
        size=resized_shape,
        mode="nearest-exact",
    )

    if tuple(resized.shape) != (
        batch_size,
        1,
        *resized_shape,
    ):
        raise RuntimeError(
            "Unexpected shape after nearest-exact interpolation: "
            f"expected {(batch_size, 1, *resized_shape)}, "
            f"got {tuple(resized.shape)}."
        )

    (
        (pad_d_before, pad_d_after),
        (pad_h_before, pad_h_after),
        (pad_w_before, pad_w_after),
    ) = compute_centered_padding(
        resized_shape=resized_shape,
        target_shape=target_shape,
    )

    # torch.nn.functional.pad expects padding in reverse spatial order:
    #
    #   W before, W after,
    #   H before, H after,
    #   D before, D after.
    #
    # This does not permute the volume axes.
    padded = F.pad(
        resized,
        (
            pad_w_before,
            pad_w_after,
            pad_h_before,
            pad_h_after,
            pad_d_before,
            pad_d_after,
        ),
        mode="constant",
        value=0.0,
    )

    expected_output_shape = (
        batch_size,
        1,
        *target_shape,
    )

    if tuple(padded.shape) != expected_output_shape:
        raise RuntimeError(
            "Unexpected shape after centered padding: "
            f"expected {expected_output_shape}, "
            f"got {tuple(padded.shape)}."
        )

    if enforce_binary:
        # nearest-exact interpolation and zero-padding already preserve
        # binary values. This threshold is a final explicit safety guarantee.
        padded = (padded > 0).to(dtype=torch.float32)

        _assert_binary_tensor(
            padded,
            name="Preprocessed sulcal volumes",
        )

    if padded.dtype != torch.float32:
        raise RuntimeError(
            "The anisotropic preprocessing output must be float32, "
            f"got {padded.dtype}."
        )

    return padded


def preprocess_sulcal_volume_anisotropic(
    volume_np: np.ndarray,
    target_shape: Any = (32, 112, 96),
    binarize_nonzero: bool = True,
) -> torch.Tensor:
    """
    Preprocess one raw sulcal skeleton for anisotropic 3DINO adaptation.

    Accepted input:
        [D,H,W]
        [D,H,W,1]

    Output:
        [1,target_d,target_h,target_w]

    Default output:
        [1,32,112,96]

    Processing steps:
        1. remove an optional trailing singleton channel;
        2. preserve the original D,H,W axis order;
        3. binarize with x != 0 when requested;
        4. convert to float32;
        5. add batch and channel dimensions;
        6. apply isotropic nearest-exact resizing;
        7. apply centered zero-padding;
        8. remove the temporary batch dimension;
        9. validate the final shape, dtype and binarity.
    """
    target_shape = normalize_target_shape(
        target_shape,
        name="target_shape",
    )

    volume_np = squeeze_trailing_singleton_channel(volume_np)

    original_shape = tuple(volume_np.shape)

    if binarize_nonzero:
        # This operation constructs a new writable C-contiguous float32 array.
        volume_np = np.asarray(
            volume_np != 0,
            dtype=np.float32,
            order="C",
        )
    else:
        # copy=True prevents torch.from_numpy warnings when the source comes
        # from a read-only NumPy memmap.
        volume_np = np.array(
            volume_np,
            dtype=np.float32,
            copy=True,
            order="C",
        )

    if tuple(volume_np.shape) != original_shape:
        raise RuntimeError(
            "The preprocessing unexpectedly changed or permuted the source "
            "spatial axes before resizing: "
            f"expected {original_shape}, got {tuple(volume_np.shape)}."
        )

    x = torch.from_numpy(volume_np).unsqueeze(0).unsqueeze(0)

    expected_input_shape = (
        1,
        1,
        *original_shape,
    )

    if tuple(x.shape) != expected_input_shape:
        raise RuntimeError(
            "Unexpected tensor shape before anisotropic preprocessing: "
            f"expected {expected_input_shape}, got {tuple(x.shape)}."
        )

    x = upscale_pad_sulcal_volume_anisotropic(
        volumes=x,
        target_shape=target_shape,
        enforce_binary=binarize_nonzero,
    )

    # Remove only the temporary batch dimension.
    x = x.squeeze(0)

    expected_output_shape = (
        1,
        *target_shape,
    )

    if tuple(x.shape) != expected_output_shape:
        raise RuntimeError(
            "Unexpected final anisotropic preprocessing shape: "
            f"expected {expected_output_shape}, got {tuple(x.shape)}."
        )

    if x.dtype != torch.float32:
        raise RuntimeError("Expected float32 preprocessing output, " f"got {x.dtype}.")

    if binarize_nonzero:
        _assert_binary_tensor(
            x,
            name="Final anisotropic sulcal volume",
        )

    return x
