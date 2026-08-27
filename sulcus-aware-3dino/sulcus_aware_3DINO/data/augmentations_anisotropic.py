# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

from __future__ import annotations

import logging
import math
from numbers import Integral, Real
from typing import Any, Sequence, Tuple

import torch
from monai.transforms import (
    RandAffine,
    RandCropByLabelClasses,
)

from .sulcal_preprocessing_anisotropic import (
    normalize_target_shape,
)

logger = logging.getLogger("dinov2")


Shape3D = Tuple[int, int, int]


def _require_positive_int(
    value: Any,
    name: str,
) -> int:
    """
    Validate one strictly positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"{name} must be an integer, got " f"{value!r} ({type(value).__name__})."
        )

    value = int(value)

    if value <= 0:
        raise ValueError(f"{name} must be strictly positive, got {value}.")

    return value


def _require_probability(
    value: Any,
    name: str,
) -> float:
    """
    Validate one probability in [0,1].
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{name} must be a real number, got " f"{value!r} ({type(value).__name__})."
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0,1], got {value}.")

    return value


def _normalize_affine_range(
    value: Sequence,
    name: str,
):
    """
    Validate a three-axis MONAI affine range.

    Each axis may be represented by:
        - one scalar;
        - a pair (minimum, maximum).

    Examples:
        (1, 1, 1)
        (0.314159, 0.314159, 0.314159)
        ((-1, 1), (-1, 1), (-1, 1))
    """
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be a sequence of three ranges, " f"got {value!r}."
        )

    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError(
            f"{name} must be a sequence of three ranges, " f"got {value!r}."
        ) from error

    if len(values) != 3:
        raise ValueError(
            f"{name} must contain exactly three axis ranges, "
            f"got {len(values)} values: {values!r}."
        )

    normalized = []

    for axis, axis_value in enumerate(values):
        axis_name = f"{name}[{axis}]"

        if not isinstance(axis_value, bool) and isinstance(axis_value, Real):
            axis_value = float(axis_value)

            if not math.isfinite(axis_value):
                raise ValueError(f"{axis_name} must be finite, got {axis_value}.")

            normalized.append(axis_value)
            continue

        if isinstance(axis_value, (str, bytes)):
            raise TypeError(
                f"{axis_name} must be a real number or a pair of real "
                f"numbers, got {axis_value!r}."
            )

        try:
            pair = tuple(axis_value)
        except TypeError as error:
            raise TypeError(
                f"{axis_name} must be a real number or a pair of real "
                f"numbers, got {axis_value!r}."
            ) from error

        if len(pair) != 2:
            raise ValueError(
                f"{axis_name} must contain exactly two values, " f"got {pair!r}."
            )

        normalized_pair = []

        for bound_index, bound in enumerate(pair):
            if isinstance(bound, bool) or not isinstance(bound, Real):
                raise TypeError(
                    f"{axis_name}[{bound_index}] must be a real number, "
                    f"got {bound!r}."
                )

            bound = float(bound)

            if not math.isfinite(bound):
                raise ValueError(
                    f"{axis_name}[{bound_index}] must be finite, " f"got {bound}."
                )

            normalized_pair.append(bound)

        normalized.append(tuple(normalized_pair))

    return tuple(normalized)


def _to_plain_tensor(
    tensor,
) -> torch.Tensor:
    """
    Convert a MONAI MetaTensor or tensor-like result to a plain Tensor.
    """
    if hasattr(tensor, "as_tensor") and callable(tensor.as_tensor):
        tensor = tensor.as_tensor()

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    return tensor


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
            f"{name} must contain only binary values {{0,1}}, "
            f"got values such as {unique_values[:10].tolist()}."
        )


class DataAugmentationDINO3dSulcalAnisotropic:
    """
    Dedicated DINO/iBOT augmentation for anisotropic binary sulcal volumes.

    Input contract:
        torch.Tensor [1,32,112,96]
        float32
        binary {0,1}

    Global views:
        - two independent affine transformations;
        - full anisotropic spatial shape;
        - output [1,32,112,96].

    Local views:
        - foreground-centered crop;
        - cubic crop shape [1,32,32,32];
        - independent local affine transformation;
        - retry when the transformed crop becomes empty.

    Excluded augmentations:
        - flips;
        - 90-degree rotations;
        - intensity normalization;
        - intensity noise;
        - anisotropic resizing.

    Axis convention:
        Tensor spatial axes are preserved in their existing order:

            [channel, D, H, W]

        No transpose or permutation is applied.
    """

    def __init__(
        self,
        global_crops_shape=(32, 112, 96),
        local_crops_size: int = 32,
        local_crops_number: int = 8,
        affine_prob_global: float = 1.0,
        translate_range_global=(1, 1, 1),
        rotate_range_global=(
            0.314159,
            0.314159,
            0.314159,
        ),
        affine_prob_local: float = 1.0,
        translate_range_local=(1, 1, 1),
        rotate_range_local=(
            0.314159,
            0.314159,
            0.314159,
        ),
        max_local_crop_retries: int = 3,
    ) -> None:
        self.global_crops_shape = normalize_target_shape(
            global_crops_shape,
            name="global_crops_shape",
        )

        self.local_crops_size = _require_positive_int(
            local_crops_size,
            "local_crops_size",
        )

        self.local_crops_number = _require_positive_int(
            local_crops_number,
            "local_crops_number",
        )

        self.max_local_crop_retries = _require_positive_int(
            max_local_crop_retries,
            "max_local_crop_retries",
        )

        self.affine_prob_global = _require_probability(
            affine_prob_global,
            "affine_prob_global",
        )

        self.affine_prob_local = _require_probability(
            affine_prob_local,
            "affine_prob_local",
        )

        self.translate_range_global = _normalize_affine_range(
            translate_range_global,
            "translate_range_global",
        )

        self.rotate_range_global = _normalize_affine_range(
            rotate_range_global,
            "rotate_range_global",
        )

        self.translate_range_local = _normalize_affine_range(
            translate_range_local,
            "translate_range_local",
        )

        self.rotate_range_local = _normalize_affine_range(
            rotate_range_local,
            "rotate_range_local",
        )

        for axis, global_dimension in enumerate(self.global_crops_shape):
            if self.local_crops_size > global_dimension:
                raise ValueError(
                    f"local_crops_size={self.local_crops_size} exceeds "
                    f"global_crops_shape[{axis}]={global_dimension}."
                )

        self.local_crops_shape: Shape3D = (
            self.local_crops_size,
            self.local_crops_size,
            self.local_crops_size,
        )

        self.expected_input_shape = (
            1,
            *self.global_crops_shape,
        )

        self.expected_global_crop_shape = (
            1,
            *self.global_crops_shape,
        )

        self.expected_local_crop_shape = (
            1,
            *self.local_crops_shape,
        )

        logger.info("##################################################")
        logger.info("Using anisotropic sulcal 3D augmentation")
        logger.info("Global crop shape       : %s", self.global_crops_shape)
        logger.info("Local crop shape        : %s", self.local_crops_shape)
        logger.info("Local crops number      : %d", self.local_crops_number)
        logger.info("Global affine prob      : %s", self.affine_prob_global)
        logger.info(
            "Global translation      : %s",
            self.translate_range_global,
        )
        logger.info(
            "Global rotation         : %s",
            self.rotate_range_global,
        )
        logger.info("Local affine prob       : %s", self.affine_prob_local)
        logger.info(
            "Local translation       : %s",
            self.translate_range_local,
        )
        logger.info(
            "Local rotation          : %s",
            self.rotate_range_local,
        )
        logger.info(
            "Local crop retries      : %d",
            self.max_local_crop_retries,
        )
        logger.info("Interpolation           : nearest")
        logger.info("Padding                 : zeros")
        logger.info("Flips                    : disabled")
        logger.info("90-degree rotations     : disabled")
        logger.info("Axis permutation        : none")
        logger.info("##################################################")

        # The output shape is explicit even though it is equal to the input
        # shape. This prevents the affine transform from becoming an implicit
        # source of spatial geometry.
        self.geometric_augmentation_global = RandAffine(
            prob=self.affine_prob_global,
            spatial_size=self.global_crops_shape,
            translate_range=self.translate_range_global,
            rotate_range=self.rotate_range_global,
            mode="nearest",
            padding_mode="zeros",
        )

        # ratios=[0,1] requests centers from the foreground class only.
        self.foreground_crop = RandCropByLabelClasses(
            spatial_size=self.local_crops_shape,
            ratios=[0, 1],
            num_classes=2,
            num_samples=1,
            allow_smaller=False,
        )

        self.geometric_augmentation_local = RandAffine(
            prob=self.affine_prob_local,
            spatial_size=self.local_crops_shape,
            translate_range=self.translate_range_local,
            rotate_range=self.rotate_range_local,
            mode="nearest",
            padding_mode="zeros",
        )

    def _validate_input(
        self,
        image,
    ) -> torch.Tensor:
        """
        Validate the deterministic preprocessing output.
        """
        image = _to_plain_tensor(image)

        if tuple(image.shape) != self.expected_input_shape:
            raise ValueError(
                "Expected anisotropic preprocessing output with shape "
                f"{self.expected_input_shape}, got {tuple(image.shape)}."
            )

        if image.dtype != torch.float32:
            raise TypeError(
                "Expected anisotropic preprocessing output with dtype "
                f"torch.float32, got {image.dtype}."
            )

        _assert_binary_tensor(
            image,
            name="Input anisotropic sulcal volume",
        )

        return image

    def _finalize_crop(
        self,
        crop,
        expected_shape,
        name: str,
    ) -> torch.Tensor:
        """
        Convert, binarize and validate one augmented crop.
        """
        crop = _to_plain_tensor(crop)

        if tuple(crop.shape) != tuple(expected_shape):
            raise RuntimeError(
                f"{name} has an invalid shape: "
                f"expected {tuple(expected_shape)}, "
                f"got {tuple(crop.shape)}."
            )

        crop = crop.to(dtype=torch.float32)

        # Nearest interpolation should preserve the labels. The threshold is
        # retained as an explicit final safety guarantee.
        crop = (crop > 0).to(dtype=torch.float32)

        _assert_binary_tensor(
            crop,
            name=name,
        )

        return crop

    def _augment_global(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply one affine transformation to the complete anisotropic volume.
        """
        if tuple(image.shape) != self.expected_input_shape:
            raise RuntimeError(
                "Invalid input shape before global augmentation: "
                f"expected {self.expected_input_shape}, "
                f"got {tuple(image.shape)}."
            )

        crop = self.geometric_augmentation_global(image)

        return self._finalize_crop(
            crop=crop,
            expected_shape=self.expected_global_crop_shape,
            name="Global anisotropic crop",
        )

    def _augment_local(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract one foreground-centered local crop and apply a local affine.

        If the transformed crop becomes empty, the full crop-and-affine
        operation is retried.

        After all retries, the last crop is returned even if it is empty. This
        preserves the behavior of the historical implementation.
        """
        last_crop = None

        for _ in range(self.max_local_crop_retries):
            crops = self.foreground_crop(
                img=image,
                label=image,
            )

            if not isinstance(crops, (list, tuple)):
                raise TypeError(
                    "RandCropByLabelClasses must return a crop sequence, "
                    f"got {type(crops).__name__}."
                )

            if len(crops) != 1:
                raise RuntimeError(
                    "Expected exactly one foreground-centered crop from "
                    "RandCropByLabelClasses, "
                    f"got {len(crops)}."
                )

            crop = crops[0]

            if tuple(crop.shape) != self.expected_local_crop_shape:
                raise RuntimeError(
                    "Invalid local shape before affine augmentation: "
                    f"expected {self.expected_local_crop_shape}, "
                    f"got {tuple(crop.shape)}."
                )

            crop = self.geometric_augmentation_local(crop)

            crop = self._finalize_crop(
                crop=crop,
                expected_shape=self.expected_local_crop_shape,
                name="Local anisotropic crop",
            )

            last_crop = crop

            if int(torch.count_nonzero(crop).item()) > 0:
                return crop

        if last_crop is None:
            raise RuntimeError("Local augmentation completed without producing a crop.")

        # Deliberate historical fallback: return the final crop after all
        # retries, even if the affine transform removed all foreground.
        return last_crop

    def __call__(
        self,
        image,
    ):
        """
        Produce DINO/iBOT multi-crop views.

        Accepted input:
            torch.Tensor:
                [1,32,112,96]

            dictionary:
                {"image": torch.Tensor [1,32,112,96]}

        Returns:
            (
                {
                    "global_crops": [global_1, global_2],
                    "global_crops_teacher": [global_1, global_2],
                    "local_crops": [local_1, ..., local_8],
                    "offsets": (),
                },
                None,
            )
        """
        if isinstance(image, dict):
            if "image" not in image:
                raise KeyError("Input dictionary must contain an 'image' key.")

            image = image["image"]

        image = self._validate_input(image)

        # Two independent calls produce two independently sampled global
        # affine transformations.
        global_crop_1 = self._augment_global(image)
        global_crop_2 = self._augment_global(image)

        local_crops = [
            self._augment_local(image) for _ in range(self.local_crops_number)
        ]

        if len(local_crops) != self.local_crops_number:
            raise RuntimeError(
                "Unexpected number of local crops: "
                f"expected {self.local_crops_number}, "
                f"got {len(local_crops)}."
            )

        output = {
            "global_crops": [
                global_crop_1,
                global_crop_2,
            ],
            "global_crops_teacher": [
                global_crop_1,
                global_crop_2,
            ],
            "local_crops": local_crops,
            "offsets": (),
        }

        return output, None
