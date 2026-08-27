# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)
#
# Portions of this file are derived from DINOv2 (Meta AI Research),
# licensed under the Apache License 2.0.

from __future__ import annotations

import random
from numbers import Real
from typing import Any, Dict, List, Sequence, Tuple

import torch


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

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0,1], got {value}.")

    return value


def _normalize_mask_ratio_tuple(
    mask_ratio_tuple: Sequence,
) -> Tuple[float, float]:
    """
    Validate the minimum and maximum masking ratios.
    """
    if isinstance(mask_ratio_tuple, (str, bytes)):
        raise TypeError(
            "mask_ratio_tuple must contain two real values, "
            f"got {mask_ratio_tuple!r}."
        )

    try:
        values = tuple(mask_ratio_tuple)
    except TypeError as error:
        raise TypeError(
            "mask_ratio_tuple must contain two real values, "
            f"got {mask_ratio_tuple!r}."
        ) from error

    if len(values) != 2:
        raise ValueError(
            "mask_ratio_tuple must contain exactly two values "
            "(minimum, maximum), "
            f"got {values!r}."
        )

    minimum = _require_probability(
        values[0],
        "mask_ratio_tuple[0]",
    )
    maximum = _require_probability(
        values[1],
        "mask_ratio_tuple[1]",
    )

    if minimum > maximum:
        raise ValueError(
            "mask_ratio_tuple minimum cannot exceed its maximum: "
            f"got ({minimum}, {maximum})."
        )

    return minimum, maximum


def _extract_output_dict(
    sample,
    sample_index: int,
) -> Dict:
    """
    Extract the augmentation dictionary from one dataset sample.

    Expected dataset output:
        (output_dict, None)
    """
    if not isinstance(sample, (tuple, list)):
        raise TypeError(
            f"samples_list[{sample_index}] must be a tuple or list "
            f"(output_dict, target), got {type(sample).__name__}."
        )

    if len(sample) != 2:
        raise ValueError(
            f"samples_list[{sample_index}] must contain exactly two "
            f"elements (output_dict, target), got {len(sample)}."
        )

    output_dict = sample[0]

    if not isinstance(output_dict, dict):
        raise TypeError(
            f"samples_list[{sample_index}][0] must be a dictionary, "
            f"got {type(output_dict).__name__}."
        )

    required_keys = {
        "global_crops",
        "global_crops_teacher",
        "local_crops",
        "offsets",
    }

    missing_keys = required_keys.difference(output_dict.keys())

    if missing_keys:
        raise KeyError(
            f"samples_list[{sample_index}] is missing augmentation keys: "
            f"{sorted(missing_keys)}."
        )

    return output_dict


def _validate_crop_tensor(
    crop,
    *,
    expected_shape: Tuple[int, ...],
    name: str,
) -> torch.Tensor:
    """
    Validate one crop before batch stacking.
    """
    if not isinstance(crop, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, " f"got {type(crop).__name__}."
        )

    if tuple(crop.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} has an invalid shape: "
            f"expected {tuple(expected_shape)}, "
            f"got {tuple(crop.shape)}."
        )

    if crop.ndim != 4:
        raise ValueError(
            f"{name} must have shape [C,D,H,W], " f"got {tuple(crop.shape)}."
        )

    if crop.shape[0] != 1:
        raise ValueError(
            f"{name} must contain one channel, " f"got shape {tuple(crop.shape)}."
        )

    return crop


def collate_data_and_cast_anisotropic(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    mask_generator,
    use_density_masking: bool = False,
):
    """
    Collate anisotropic DINO/iBOT multi-crop samples and generate patch masks.

    Expected sample format:
        (
            {
                "global_crops": [global_1, global_2],
                "global_crops_teacher": [global_1, global_2],
                "local_crops": [local_1, ..., local_8],
                "offsets": (),
            },
            None,
        )

    Expected shapes for a per-GPU subject batch size b:
        collated_global_crops:
            [2b,1,32,112,96]

        collated_local_crops:
            [8b,1,32,32,32]

        masks before flattening:
            [2b,2,7,6]

        collated_masks after flattening:
            [2b,84]

    The mask generator is the sole source of truth for the number of global
    patch tokens:

        N = mask_generator.num_patches

    For the A. Cingulate anisotropic pipeline:
        N = 2 * 7 * 6 = 84

    Important active-only invariant:
        Masks are generated directly in global-crop order and are never
        shuffled afterward.

        Therefore:

            collated_masks[i]

        always corresponds spatially to:

            collated_global_crops[i]

    Args:
        samples_list:
            Batch of augmented dataset outputs.

        mask_ratio_tuple:
            Minimum and maximum masking ratios.

        mask_probability:
            Fraction of global views selected for masking.

        dtype:
            Dtype used for the collated image tensors, typically torch.half.

        mask_generator:
            MaskingGenerator3dAnisotropic instance.

        use_density_masking:
            If True, pass each corresponding global volume to the generator
            and restrict masking to active sulcal patches.

            If False, sample uniformly over all patch positions.

    Returns:
        Dictionary containing:
            collated_global_crops
            collated_local_crops
            collated_masks
            mask_indices_list
            masks_weight
            upperbound
            n_masked_patches
    """
    if not isinstance(samples_list, (list, tuple)):
        raise TypeError(
            "samples_list must be a list or tuple, "
            f"got {type(samples_list).__name__}."
        )

    if len(samples_list) == 0:
        raise ValueError("samples_list must contain at least one sample.")

    if mask_generator is None:
        raise ValueError(
            "mask_generator must be provided to " "collate_data_and_cast_anisotropic."
        )

    if not hasattr(mask_generator, "num_patches"):
        raise TypeError("mask_generator must expose a num_patches attribute.")

    if not hasattr(mask_generator, "get_shape"):
        raise TypeError("mask_generator must expose a get_shape() method.")

    if not hasattr(mask_generator, "get_expected_volume_shape"):
        raise TypeError(
            "mask_generator must expose a " "get_expected_volume_shape() method."
        )

    if not callable(mask_generator):
        raise TypeError("mask_generator must be callable.")

    mask_probability = _require_probability(
        mask_probability,
        "mask_probability",
    )

    mask_ratio_min, mask_ratio_max = _normalize_mask_ratio_tuple(mask_ratio_tuple)

    if not isinstance(use_density_masking, bool):
        raise TypeError(
            "use_density_masking must be a boolean, " f"got {use_density_masking!r}."
        )

    output_dicts = [
        _extract_output_dict(
            sample,
            sample_index,
        )
        for sample_index, sample in enumerate(samples_list)
    ]

    first_output = output_dicts[0]

    n_global_crops = len(first_output["global_crops"])
    n_teacher_global_crops = len(first_output["global_crops_teacher"])
    n_local_crops = len(first_output["local_crops"])

    # The current DINO teacher logic splits the global batch into two views.
    if n_global_crops != 2:
        raise ValueError(
            "The anisotropic DINO pipeline requires exactly two global "
            f"crops per subject, got {n_global_crops}."
        )

    if n_teacher_global_crops != 2:
        raise ValueError(
            "The anisotropic DINO pipeline requires exactly two teacher "
            f"global crops per subject, got {n_teacher_global_crops}."
        )

    if n_local_crops <= 0:
        raise ValueError(
            "The anisotropic DINO pipeline requires at least one local crop."
        )

    expected_global_shape = tuple(
        int(value) for value in mask_generator.get_expected_volume_shape()
    )

    if len(expected_global_shape) != 4:
        raise ValueError(
            "mask_generator.get_expected_volume_shape() must return "
            "[C,D,H,W], "
            f"got {expected_global_shape}."
        )

    if expected_global_shape[0] != 1:
        raise ValueError(
            "The expected global volume must contain one channel, "
            f"got {expected_global_shape}."
        )

    # Infer the dedicated local crop shape from the first transformed sample,
    # then require every sample to use exactly that same shape.
    first_local_crop = first_output["local_crops"][0]

    if not isinstance(first_local_crop, torch.Tensor):
        raise TypeError(
            "The first local crop must be a torch.Tensor, "
            f"got {type(first_local_crop).__name__}."
        )

    expected_local_shape = tuple(first_local_crop.shape)

    if len(expected_local_shape) != 4:
        raise ValueError(
            "Local crops must have shape [C,D,H,W], " f"got {expected_local_shape}."
        )

    if expected_local_shape[0] != 1:
        raise ValueError(
            "Local crops must contain one channel, " f"got {expected_local_shape}."
        )

    for sample_index, output_dict in enumerate(output_dicts):
        if len(output_dict["global_crops"]) != n_global_crops:
            raise ValueError(
                f"Sample {sample_index} contains "
                f"{len(output_dict['global_crops'])} global crops, "
                f"expected {n_global_crops}."
            )

        if len(output_dict["global_crops_teacher"]) != n_teacher_global_crops:
            raise ValueError(
                f"Sample {sample_index} contains "
                f"{len(output_dict['global_crops_teacher'])} teacher global "
                f"crops, expected {n_teacher_global_crops}."
            )

        if len(output_dict["local_crops"]) != n_local_crops:
            raise ValueError(
                f"Sample {sample_index} contains "
                f"{len(output_dict['local_crops'])} local crops, "
                f"expected {n_local_crops}."
            )

        for crop_index, crop in enumerate(output_dict["global_crops"]):
            _validate_crop_tensor(
                crop,
                expected_shape=expected_global_shape,
                name=(
                    f"samples_list[{sample_index}]" f"['global_crops'][{crop_index}]"
                ),
            )

        for crop_index, crop in enumerate(output_dict["global_crops_teacher"]):
            _validate_crop_tensor(
                crop,
                expected_shape=expected_global_shape,
                name=(
                    f"samples_list[{sample_index}]"
                    f"['global_crops_teacher'][{crop_index}]"
                ),
            )

        for crop_index, crop in enumerate(output_dict["local_crops"]):
            _validate_crop_tensor(
                crop,
                expected_shape=expected_local_shape,
                name=(f"samples_list[{sample_index}]" f"['local_crops'][{crop_index}]"),
            )

    batch_size = len(samples_list)

    # Crop-major order is deliberately preserved:
    #
    #   global view 0 for all subjects,
    #   then global view 1 for all subjects.
    #
    # This is required by the teacher's later .chunk(2) operation.
    collated_global_crops = torch.stack(
        [
            output_dicts[sample_index]["global_crops"][crop_index]
            for crop_index in range(n_global_crops)
            for sample_index in range(batch_size)
        ]
    )

    # Same ordering for local crops:
    #
    #   local view 0 for all subjects,
    #   local view 1 for all subjects,
    #   ...
    collated_local_crops = torch.stack(
        [
            output_dicts[sample_index]["local_crops"][crop_index]
            for crop_index in range(n_local_crops)
            for sample_index in range(batch_size)
        ]
    )

    expected_collated_global_shape = (
        n_global_crops * batch_size,
        *expected_global_shape,
    )

    expected_collated_local_shape = (
        n_local_crops * batch_size,
        *expected_local_shape,
    )

    if tuple(collated_global_crops.shape) != expected_collated_global_shape:
        raise RuntimeError(
            "Unexpected collated global-crop shape: "
            f"expected {expected_collated_global_shape}, "
            f"got {tuple(collated_global_crops.shape)}."
        )

    if tuple(collated_local_crops.shape) != expected_collated_local_shape:
        raise RuntimeError(
            "Unexpected collated local-crop shape: "
            f"expected {expected_collated_local_shape}, "
            f"got {tuple(collated_local_crops.shape)}."
        )

    # Number of global crop instances in the collated batch.
    B = int(collated_global_crops.shape[0])

    # The mask generator is the unique source of truth.
    N = int(mask_generator.num_patches)

    if N <= 0:
        raise ValueError(f"mask_generator.num_patches must be positive, got {N}.")

    mask_grid_shape = tuple(int(value) for value in mask_generator.get_shape())

    if len(mask_grid_shape) != 3:
        raise ValueError(
            "mask_generator.get_shape() must return "
            "(grid_d,grid_h,grid_w), "
            f"got {mask_grid_shape}."
        )

    grid_numel = mask_grid_shape[0] * mask_grid_shape[1] * mask_grid_shape[2]

    if grid_numel != N:
        raise RuntimeError(
            "Mask-generator geometry is inconsistent: "
            f"get_shape()={mask_grid_shape} contains {grid_numel} patches, "
            f"but num_patches={N}."
        )

    # Historical behavior: use floor when selecting the number of masked
    # global views.
    n_samples_masked = int(B * mask_probability)

    if n_samples_masked < 0 or n_samples_masked > B:
        raise RuntimeError(
            "Invalid number of selected masked views: " f"{n_samples_masked} for B={B}."
        )

    # Choose crop rows directly. The masks will remain in this exact order;
    # there is no post-generation mask shuffle.
    masked_crop_indices = set(
        random.sample(
            range(B),
            k=n_samples_masked,
        )
    )

    # As in the historical implementation, masked views are assigned
    # stratified ratio intervals between the configured minimum and maximum.
    ratio_boundaries = torch.linspace(
        mask_ratio_min,
        mask_ratio_max,
        steps=n_samples_masked + 1,
        dtype=torch.float64,
    )

    upperbound = 0
    masks_list: List[torch.Tensor] = []
    masked_counter = 0

    for crop_row in range(B):
        if crop_row in masked_crop_indices:
            ratio_low = float(ratio_boundaries[masked_counter].item())
            ratio_high = float(ratio_boundaries[masked_counter + 1].item())

            sampled_ratio = random.uniform(
                ratio_low,
                ratio_high,
            )

            n_to_mask = int(N * sampled_ratio)

            if use_density_masking:
                raw_mask = mask_generator(
                    n_to_mask,
                    volume=collated_global_crops[crop_row],
                )
            else:
                raw_mask = mask_generator(
                    n_to_mask,
                )

            # Upper bound retained for iBOT target-buffer allocation.
            upperbound += int(N * ratio_high)

            masked_counter += 1

        else:
            raw_mask = mask_generator(0)

        mask = torch.as_tensor(
            raw_mask,
            dtype=torch.bool,
        )

        if tuple(mask.shape) != mask_grid_shape:
            raise RuntimeError(
                f"Mask row {crop_row} has an invalid grid shape: "
                f"expected {mask_grid_shape}, "
                f"got {tuple(mask.shape)}."
            )

        if mask.numel() != N:
            raise RuntimeError(
                f"Mask row {crop_row} contains {mask.numel()} values, "
                f"expected N={N}."
            )

        masks_list.append(mask)

    if masked_counter != n_samples_masked:
        raise RuntimeError(
            "Internal masked-view accounting mismatch: "
            f"processed {masked_counter}, "
            f"expected {n_samples_masked}."
        )

    if len(masks_list) != B:
        raise RuntimeError(f"Generated {len(masks_list)} masks for B={B} global crops.")

    # Before flattening:
    #   [2b,2,7,6]
    collated_masks_grid = torch.stack(
        masks_list,
        dim=0,
    )

    expected_masks_grid_shape = (
        B,
        *mask_grid_shape,
    )

    if tuple(collated_masks_grid.shape) != expected_masks_grid_shape:
        raise RuntimeError(
            "Unexpected pre-flatten mask shape: "
            f"expected {expected_masks_grid_shape}, "
            f"got {tuple(collated_masks_grid.shape)}."
        )

    # After flattening:
    #   [2b,84]
    #
    # torch.flatten follows the same C-style spatial order as the generator:
    # W varies fastest, followed by H, then D.
    collated_masks = collated_masks_grid.flatten(start_dim=1)

    expected_collated_masks_shape = (
        B,
        N,
    )

    if tuple(collated_masks.shape) != expected_collated_masks_shape:
        raise RuntimeError(
            "Unexpected flattened mask shape: "
            f"expected {expected_collated_masks_shape}, "
            f"got {tuple(collated_masks.shape)}."
        )

    # Linear indices into the fully flattened [B,N] token array.
    mask_indices_list = torch.nonzero(
        collated_masks.flatten(),
        as_tuple=False,
    ).flatten()

    masks_per_crop = collated_masks.sum(dim=-1)

    masks_weight = (
        (1.0 / masks_per_crop.clamp(min=1).to(torch.float32))
        .unsqueeze(-1)
        .expand_as(collated_masks)[collated_masks]
    )

    n_masked_patches = int(mask_indices_list.numel())

    if masks_weight.numel() != n_masked_patches:
        raise RuntimeError(
            "masks_weight and mask_indices_list have inconsistent lengths: "
            f"{masks_weight.numel()} versus {n_masked_patches}."
        )

    if upperbound < n_masked_patches:
        raise RuntimeError(
            "iBOT upperbound is smaller than the actual number of masked "
            f"patches: upperbound={upperbound}, "
            f"n_masked_patches={n_masked_patches}."
        )

    if n_masked_patches > 0:
        maximum_mask_index = int(mask_indices_list.max().item())

        minimum_mask_index = int(mask_indices_list.min().item())

        if minimum_mask_index < 0:
            raise RuntimeError(f"Negative mask index detected: {minimum_mask_index}.")

        if maximum_mask_index >= B * N:
            raise RuntimeError(
                "Mask index exceeds the flattened global-token array: "
                f"maximum={maximum_mask_index}, "
                f"valid upper limit={B * N - 1}."
            )

    return {
        "collated_global_crops": collated_global_crops.to(dtype=dtype),
        "collated_local_crops": collated_local_crops.to(dtype=dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": int(upperbound),
        "n_masked_patches": torch.full(
            size=(1,),
            fill_value=n_masked_patches,
            dtype=torch.long,
        ),
    }
