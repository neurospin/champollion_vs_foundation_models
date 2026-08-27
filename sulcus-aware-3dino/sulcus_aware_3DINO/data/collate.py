# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)
#
# Portions of this file are derived from DINOv2 (Meta AI Research),
# licensed under the Apache License 2.0.

"""Collate for sulcal SSL: batch the multi-crop views and build iBOT patch masks.

Independent implementation written for 3DINO's training loop. The output-dict
keys and tensor layout are the *interface* consumed by
``SSLMetaArch.forward_backward`` and must match it exactly:

- crop-major crop ordering (the loop chunks the batch by ``n_global_crops``),
- ``upperbound`` sizes the teacher/student patch buffers,
- ``mask_indices_list`` gathers the masked patch tokens,
- ``masks_weight`` weights the iBOT loss per masked patch.

The *masking policy* is specific to this pipeline: the masked crops are chosen
up front (never shuffled afterwards, so a volume-dependent mask stays aligned
with its own crop) and, in sulcal mode, masking is restricted to active patches.
"""

import random

import torch


def _as_plain_tensor(x):
    """Drop a MONAI MetaTensor wrapper if present.

    The fused xFormers index/select kernels inside 3DINO's attention reject
    MONAI MetaTensors, so the collated crops leave this function as plain
    ``torch.Tensor``s.
    """
    return x.as_tensor() if hasattr(x, "as_tensor") else x


def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    use_density_masking=False,
):
    """Collate a batch of augmented samples and generate iBOT patch masks.

    Args:
        samples_list:
            List of ``(output_dict, None)`` tuples from
            ``DataAugmentationDINO3d_sulcal``.
        mask_ratio_tuple:
            ``(min_ratio, max_ratio)`` fraction of patches to mask.
        mask_probability:
            Fraction of global crops in the batch that receive a non-empty mask.
        dtype:
            Target dtype for the collated crops (typically ``torch.half``).
        n_tokens:
            Number of patch tokens per global crop (e.g. ``7**3 = 343``).
        mask_generator:
            A ``MaskingGenerator3d`` instance.
        use_density_masking:
            If True, pass each crop volume to the generator so masking is
            restricted to active sulcal patches; otherwise mask uniformly.

    Returns:
        The dict consumed by ``SSLMetaArch.forward_backward``.
    """
    if n_tokens is None:
        raise ValueError("n_tokens must be provided to collate_data_and_cast.")
    if mask_generator is None:
        raise ValueError("mask_generator must be provided to collate_data_and_cast.")

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    # Crop-major layout: all crop-0 across the batch, then all crop-1, ...
    # forward_backward relies on this ordering when it chunks by n_global_crops.
    collated_global_crops = torch.stack(
        [
            sample[0]["global_crops"][crop_idx]
            for crop_idx in range(n_global_crops)
            for sample in samples_list
        ]
    )
    collated_local_crops = torch.stack(
        [
            sample[0]["local_crops"][crop_idx]
            for crop_idx in range(n_local_crops)
            for sample in samples_list
        ]
    )
    collated_global_crops = _as_plain_tensor(collated_global_crops)
    collated_local_crops = _as_plain_tensor(collated_local_crops)

    n_crops = collated_global_crops.shape[0]
    n_masked_crops = int(n_crops * mask_probability)

    # Pick the masked crops BEFORE generating any mask, so each volume-dependent
    # mask stays paired with its own crop (no post-hoc shuffle).
    masked_crop_ids = set(random.sample(range(n_crops), k=n_masked_crops))

    # Linear ramp of mask ratios spread across the masked crops (iBOT schedule).
    ratio_ramp = torch.linspace(*mask_ratio_tuple, n_masked_crops + 1)

    masks = []
    upperbound = 0
    ramp_pos = 0
    for crop_id in range(n_crops):
        if crop_id in masked_crop_ids:
            ratio_lo = ratio_ramp[ramp_pos]
            ratio_hi = ratio_ramp[ramp_pos + 1]
            n_to_mask = int(n_tokens * random.uniform(ratio_lo, ratio_hi))
            if use_density_masking:
                patch_mask = mask_generator(
                    n_to_mask, volume=collated_global_crops[crop_id]
                )
            else:
                patch_mask = mask_generator(n_to_mask)
            upperbound += int(n_tokens * ratio_hi)
            ramp_pos += 1
        else:
            patch_mask = mask_generator(0)
        masks.append(torch.BoolTensor(patch_mask))

    collated_masks = torch.stack(masks).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    # Per-masked-patch loss weight = 1 / (# masked patches in that crop), gathered
    # at the masked positions. clamp(min=1) guards crops with no masked patch.
    masked_per_crop = collated_masks.sum(dim=-1).clamp(min=1.0)
    weight_per_crop = (1.0 / masked_per_crop).unsqueeze(-1).expand_as(collated_masks)
    masks_weight = weight_per_crop[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full(
            (1,), fill_value=mask_indices_list.shape[0], dtype=torch.long
        ),
    }
