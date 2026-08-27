# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

from __future__ import annotations

import random
from numbers import Integral
from typing import Any, Tuple

import numpy as np
import torch

GridShape3D = Tuple[int, int, int]


def _require_positive_int(
    value: Any,
    name: str,
) -> int:
    """
    Validate and normalize one strictly positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"{name} must be an integer, got " f"{value!r} ({type(value).__name__})."
        )

    value = int(value)

    if value <= 0:
        raise ValueError(f"{name} must be strictly positive, got {value}.")

    return value


def _normalize_grid_shape(
    input_size: Any,
) -> GridShape3D:
    """
    Validate an explicit anisotropic patch-grid shape.

    Accepted examples:
        (2, 7, 6)
        [2, 7, 6]

    A scalar is deliberately rejected because the dedicated anisotropic
    pipeline requires an explicit three-dimensional patch grid.

    Returns:
        (grid_d, grid_h, grid_w)
    """
    if isinstance(input_size, (str, bytes)):
        raise TypeError(
            "input_size must be an explicit sequence of three integers, "
            f"got {input_size!r}."
        )

    try:
        dimensions = tuple(input_size)
    except TypeError as error:
        raise TypeError(
            "input_size must be an explicit sequence of three integers, "
            f"got {input_size!r} ({type(input_size).__name__})."
        ) from error

    if len(dimensions) != 3:
        raise ValueError(
            "input_size must contain exactly three dimensions "
            "(grid_d, grid_h, grid_w), "
            f"got {len(dimensions)} values: {dimensions!r}."
        )

    return tuple(
        _require_positive_int(
            dimension,
            f"input_size[{axis}]",
        )
        for axis, dimension in enumerate(dimensions)
    )


class MaskingGenerator3dAnisotropic:
    """
    Generate 3D patch masks for anisotropic global crops.

    Grid convention:
        input_size = (grid_d, grid_h, grid_w)

    Default A. Cingulate geometry:
        input_size = (2, 7, 6)
        patch_size = 16

    Corresponding volume:
        [1, 32, 112, 96]

    Flattening convention:
        C-order flattening is used everywhere.

        Therefore:
            W varies first,
            then H,
            then D.

        The linear index of patch (d,h,w) is:

            index = d * grid_h * grid_w
                  + h * grid_w
                  + w

    Masking modes:
        1. volume is None:
            Uniform masking over all patch positions.

        2. volume is provided:
            Masking is restricted to patches containing at least one active
            voxel.

            The ratio:

                num_masking_patches / num_patches

            is applied to the number of active patches.

    Empty-view behavior:
        If the supplied volume contains no active patch, the returned mask
        remains entirely false.

        No fallback to uniform background masking is performed.
    """

    def __init__(
        self,
        input_size=(2, 7, 6),
        patch_size: int = 16,
    ) -> None:
        """
        Args:
            input_size:
                Explicit patch-grid shape:

                    (grid_d, grid_h, grid_w)

            patch_size:
                Cubic patch size in voxels.

                For 3DINO:
                    patch_size = 16
        """
        (
            self.grid_d,
            self.grid_h,
            self.grid_w,
        ) = _normalize_grid_shape(input_size)

        self.patch_size = _require_positive_int(
            patch_size,
            "patch_size",
        )

        self.num_patches = self.grid_d * self.grid_h * self.grid_w

        self.expected_volume_shape = (
            1,
            self.grid_d * self.patch_size,
            self.grid_h * self.patch_size,
            self.grid_w * self.patch_size,
        )

    def __repr__(self) -> str:
        return (
            "MaskingGenerator3dAnisotropic("
            f"grid_d={self.grid_d}, "
            f"grid_h={self.grid_h}, "
            f"grid_w={self.grid_w}, "
            f"patch_size={self.patch_size}, "
            f"num_patches={self.num_patches}"
            ")"
        )

    def get_shape(self) -> GridShape3D:
        """
        Return the patch-grid shape in explicit D,H,W order.
        """
        return (
            self.grid_d,
            self.grid_h,
            self.grid_w,
        )

    def get_expected_volume_shape(self) -> Tuple[int, int, int, int]:
        """
        Return the expected channel-first global volume shape.
        """
        return self.expected_volume_shape

    def patch_coordinates_to_index(
        self,
        d: int,
        h: int,
        w: int,
    ) -> int:
        """
        Convert one patch coordinate (d,h,w) to its C-order linear index.

        Formula:
            index = d * grid_h * grid_w
                  + h * grid_w
                  + w
        """
        coordinates = (
            ("d", d, self.grid_d),
            ("h", h, self.grid_h),
            ("w", w, self.grid_w),
        )

        normalized = []

        for name, value, upper_bound in coordinates:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(
                    f"{name} must be an integer, got "
                    f"{value!r} ({type(value).__name__})."
                )

            value = int(value)

            if value < 0 or value >= upper_bound:
                raise IndexError(
                    f"Patch coordinate {name}={value} is outside "
                    f"[0,{upper_bound - 1}]."
                )

            normalized.append(value)

        d, h, w = normalized

        return d * self.grid_h * self.grid_w + h * self.grid_w + w

    def _validate_num_masking_patches(
        self,
        num_masking_patches: Any,
    ) -> int:
        """
        Validate the requested number of masks.
        """
        if isinstance(num_masking_patches, bool) or not isinstance(
            num_masking_patches, Integral
        ):
            raise TypeError(
                "num_masking_patches must be an integer, "
                f"got {num_masking_patches!r} "
                f"({type(num_masking_patches).__name__})."
            )

        num_masking_patches = int(num_masking_patches)

        if num_masking_patches < 0:
            raise ValueError(
                "num_masking_patches must be non-negative, "
                f"got {num_masking_patches}."
            )

        if num_masking_patches > self.num_patches:
            raise ValueError(
                "num_masking_patches cannot exceed the total number of "
                f"patches: requested {num_masking_patches}, "
                f"available {self.num_patches}."
            )

        return num_masking_patches

    def _volume_to_numpy(
        self,
        volume,
    ) -> np.ndarray:
        """
        Validate a channel-first volume and convert it to float32 NumPy.

        Expected shape:
            [1, grid_d*patch_size, grid_h*patch_size, grid_w*patch_size]

        For the default geometry:
            [1,32,112,96]
        """
        if isinstance(volume, torch.Tensor):
            actual_shape = tuple(volume.shape)

            if actual_shape != self.expected_volume_shape:
                raise ValueError(
                    "Invalid volume shape for anisotropic masking: "
                    f"expected {self.expected_volume_shape}, "
                    f"got {actual_shape}."
                )

            volume_np = (
                volume.detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
                .numpy()
            )

        else:
            volume_np = np.asarray(
                volume,
                dtype=np.float32,
            )

            actual_shape = tuple(volume_np.shape)

            if actual_shape != self.expected_volume_shape:
                raise ValueError(
                    "Invalid volume shape for anisotropic masking: "
                    f"expected {self.expected_volume_shape}, "
                    f"got {actual_shape}."
                )

        if not np.isfinite(volume_np).all():
            raise ValueError("The masking volume contains NaN or infinite values.")

        return volume_np

    def compute_patch_density(
        self,
        volume,
    ) -> np.ndarray:
        """
        Compute the number of active voxels in every patch.

        Input:
            volume:
                [1,D,H,W]

        Default reshape:
            [32,112,96]
                ↓
            [2,16,7,16,6,16]

        Summed patch axes:
            axis=(1,3,5)

        Output:
            [2,7,6]
        """
        volume_np = self._volume_to_numpy(volume)

        spatial_volume = volume_np[0]

        patch_density = spatial_volume.reshape(
            self.grid_d,
            self.patch_size,
            self.grid_h,
            self.patch_size,
            self.grid_w,
            self.patch_size,
        ).sum(axis=(1, 3, 5))

        expected_density_shape = self.get_shape()

        if tuple(patch_density.shape) != expected_density_shape:
            raise RuntimeError(
                "Unexpected patch-density shape: "
                f"expected {expected_density_shape}, "
                f"got {tuple(patch_density.shape)}."
            )

        return patch_density

    def get_active_patch_indices(
        self,
        volume,
    ) -> np.ndarray:
        """
        Return the C-order linear indices of all active patches.

        A patch is active when it contains at least one non-zero voxel.

        Flattening order:
            W first, then H, then D.
        """
        patch_density = self.compute_patch_density(volume)

        patch_density_flat = patch_density.reshape(
            -1,
            order="C",
        )

        if patch_density_flat.shape != (self.num_patches,):
            raise RuntimeError(
                "Unexpected flattened patch-density shape: "
                f"expected {(self.num_patches,)}, "
                f"got {patch_density_flat.shape}."
            )

        active_patch_indices = np.flatnonzero(patch_density_flat > 0)

        return active_patch_indices.astype(
            np.int64,
            copy=False,
        )

    def _mask_uniform(
        self,
        mask: np.ndarray,
        n_masked: int,
    ) -> None:
        """
        Apply uniform random masking over every patch position.
        """
        mask_indices = random.sample(
            range(self.num_patches),
            k=n_masked,
        )

        mask.ravel(order="C")[mask_indices] = True

    def _mask_active_only(
        self,
        mask: np.ndarray,
        n_masked: int,
        volume,
    ) -> None:
        """
        Mask only patches containing active sulcal voxels.

        The originally requested masking ratio is recovered from:

            ratio = n_masked / num_patches

        and applied to the number of active patches:

            n_to_mask = int(n_active * ratio)
        """
        active_patch_indices = self.get_active_patch_indices(volume)

        n_active = int(active_patch_indices.size)

        if n_active == 0:
            # No active sulcal patch exists in this view.
            # Keep the mask entirely empty.
            #
            # iBOT must not generate reconstruction targets from
            # pure-background patches.
            return

        ratio = n_masked / self.num_patches

        n_to_mask = int(n_active * ratio)

        if n_to_mask == 0:
            return

        if n_to_mask > n_active:
            raise RuntimeError(
                "Internal masking error: requested more active patches than "
                f"available: n_to_mask={n_to_mask}, n_active={n_active}."
            )

        mask_indices = np.random.choice(
            active_patch_indices,
            size=n_to_mask,
            replace=False,
        )

        mask.ravel(order="C")[mask_indices] = True

    def __call__(
        self,
        num_masking_patches: int = 0,
        volume=None,
    ) -> np.ndarray:
        """
        Generate one boolean anisotropic patch mask.

        Args:
            num_masking_patches:
                Number of patches requested by the collate function.

                In active-only mode, this value is converted back to a ratio
                and applied to the number of active patches.

            volume:
                Optional channel-first volume.

                If provided:
                    active-only masking is used.

                If omitted:
                    uniform masking is used.

        Returns:
            Boolean NumPy array:

                [grid_d,grid_h,grid_w]

            Default:
                [2,7,6]
        """
        num_masking_patches = self._validate_num_masking_patches(num_masking_patches)

        mask = np.zeros(
            shape=self.get_shape(),
            dtype=np.bool_,
            order="C",
        )

        if num_masking_patches == 0:
            return mask

        if volume is None:
            self._mask_uniform(
                mask=mask,
                n_masked=num_masking_patches,
            )
        else:
            self._mask_active_only(
                mask=mask,
                n_masked=num_masking_patches,
                volume=volume,
            )

        if tuple(mask.shape) != self.get_shape():
            raise RuntimeError(
                "Unexpected generated mask shape: "
                f"expected {self.get_shape()}, "
                f"got {tuple(mask.shape)}."
            )

        if mask.dtype != np.bool_:
            raise RuntimeError(
                "The generated mask must have boolean dtype, " f"got {mask.dtype}."
            )

        return mask
