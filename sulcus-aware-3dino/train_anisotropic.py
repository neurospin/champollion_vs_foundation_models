# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)
#
# Portions of this file are derived from DINOv2 (Meta AI Research),
# licensed under the Apache License 2.0.

"""Training entry point for the anisotropic sulcal SSL pipeline.

Own training loop + diagnostics wiring the anisotropic sulcal data pipeline,
the standalone SSLMetaArchAnisotropic meta-architecture and the anisotropic
setup. Reuses only upstream 3DINO building blocks imported as ``dinov2.*``.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from functools import partial
from typing import Dict, Iterable, Tuple

import torch
from fvcore.common.checkpoint import PeriodicCheckpointer

import dinov2.distributed as distributed
from sulcus_aware_3DINO.data.augmentations_anisotropic import (
    DataAugmentationDINO3dSulcalAnisotropic,
)
from sulcus_aware_3DINO.data.collate_anisotropic import (
    collate_data_and_cast_anisotropic,
)
from sulcus_aware_3DINO.data.loaders_anisotropic import (
    SamplerType,
    make_data_loader,
    make_sulcal_npy_dataset_anisotropic_3d,
)
from sulcus_aware_3DINO.data.masking_non_empty_anisotropic import (
    MaskingGenerator3dAnisotropic,
)
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from sulcus_aware_3DINO.training.meta_arch_anisotropic import SSLMetaArchAnisotropic
from sulcus_aware_3DINO.training.setup_anisotropic import setup_3d_anisotropic
from dinov2.utils.utils import CosineScheduler

torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger("dinov2")


# =============================================================================
# Command-line interface
# =============================================================================


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        "3DINO anisotropic sulcal SSL training",
        add_help=add_help,
    )

    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="Path to the anisotropic experiment configuration file.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not attempt to resume from the checkpoint directory.",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load the current training checkpoint and export the teacher.",
    )

    parser.add_argument(
        "--eval",
        type=str,
        default="",
        help="Reserved evaluation label for CLI compatibility.",
    )

    parser.add_argument(
        "opts",
        help=(
            "Configuration overrides written as path=value, for example "
            "train.batch_size_per_gpu=2."
        ),
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Directory used for logs and checkpoints.",
    )

    parser.add_argument(
        "--local-rank",
        default=0,
        type=int,
        help="Local process rank supplied by the launcher.",
    )

    parser.add_argument(
        "--cache-dir",
        default=None,
        type=str,
        help=(
            "Accepted for CLI compatibility but unused: the anisotropic NPY "
            "pipeline does not use a MONAI cache."
        ),
    )

    return parser


# =============================================================================
# Optimizer and schedules
# =============================================================================


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(
        params_groups,
        betas=(
            float(cfg.optim.adamw_beta1),
            float(cfg.optim.adamw_beta2),
        ),
    )


def build_schedulers(cfg):
    """
    Build schedules over the complete configured training duration.

    train.stop_after_iterations deliberately does not shorten the schedules.
    A smoke test therefore follows the beginning of the full-run schedules.
    """
    official_epoch_length = int(cfg.train.OFFICIAL_EPOCH_LENGTH)

    configured_total_iters = int(cfg.optim.epochs) * official_epoch_length

    if official_epoch_length <= 0:
        raise ValueError(
            "train.OFFICIAL_EPOCH_LENGTH must be strictly positive, got "
            f"{official_epoch_length}."
        )

    if configured_total_iters <= 0:
        raise ValueError(
            "The configured training duration must be positive, got "
            f"optim.epochs={cfg.optim.epochs} and "
            f"OFFICIAL_EPOCH_LENGTH={official_epoch_length}."
        )

    lr = dict(
        base_value=float(cfg.optim.lr),
        final_value=float(cfg.optim.min_lr),
        total_iters=configured_total_iters,
        warmup_iters=(int(cfg.optim.warmup_epochs) * official_epoch_length),
        start_warmup_value=0.0,
    )

    wd = dict(
        base_value=float(cfg.optim.weight_decay),
        final_value=float(cfg.optim.weight_decay_end),
        total_iters=configured_total_iters,
    )

    momentum = dict(
        base_value=float(cfg.teacher.momentum_teacher),
        final_value=float(cfg.teacher.final_momentum_teacher),
        total_iters=configured_total_iters,
    )

    teacher_temperature = dict(
        base_value=float(cfg.teacher.teacher_temp),
        final_value=float(cfg.teacher.teacher_temp),
        total_iters=configured_total_iters,
        warmup_iters=(
            int(cfg.teacher.warmup_teacher_temp_epochs) * official_epoch_length
        ),
        start_warmup_value=float(cfg.teacher.warmup_teacher_temp),
    )

    lr_schedule = CosineScheduler(**lr)

    wd_schedule = CosineScheduler(**wd)

    momentum_schedule = CosineScheduler(**momentum)

    teacher_temp_schedule = CosineScheduler(**teacher_temperature)

    last_layer_lr_schedule = CosineScheduler(**lr)

    freeze_iterations = int(cfg.optim.freeze_last_layer_epochs) * official_epoch_length

    last_layer_lr_schedule.schedule[:freeze_iterations] = 0

    logger.info(
        "Schedulers ready | configured_total_iters=%d | "
        "warmup_iters=%d | freeze_last_layer_iters=%d",
        configured_total_iters,
        int(cfg.optim.warmup_epochs) * official_epoch_length,
        freeze_iterations,
    )

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(
    optimizer,
    lr: float,
    wd: float,
    last_layer_lr: float,
) -> None:
    for param_group in optimizer.param_groups:
        is_last_layer = bool(param_group["is_last_layer"])

        lr_multiplier = float(param_group["lr_multiplier"])

        wd_multiplier = float(param_group["wd_multiplier"])

        param_group["weight_decay"] = wd * wd_multiplier

        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


# =============================================================================
# Teacher export
# =============================================================================


def do_test(
    cfg,
    model,
    iteration,
) -> None:
    """
    Export the current FSDP teacher checkpoint for downstream probing.
    """
    teacher_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        evaluation_dir = os.path.join(
            cfg.train.output_dir,
            "eval",
            str(iteration),
        )

        os.makedirs(
            evaluation_dir,
            exist_ok=True,
        )

        teacher_checkpoint_path = os.path.join(
            evaluation_dir,
            "teacher_checkpoint.pth",
        )

        torch.save(
            {
                "teacher": teacher_state_dict,
            },
            teacher_checkpoint_path,
        )

        logger.info(
            "Teacher checkpoint exported: %s",
            teacher_checkpoint_path,
        )


# =============================================================================
# Dedicated anisotropic data pipeline
# =============================================================================


def _build_sulcal_augmentation(
    cfg,
):
    """
    Build stochastic anisotropic DINO/iBOT augmentations.
    """
    geometry = cfg.runtime_geometry

    global_crop_shape = tuple(int(value) for value in geometry.global_crop_shape)

    local_crop_shape = tuple(int(value) for value in geometry.local_crop_shape)

    if len(set(local_crop_shape)) != 1:
        raise ValueError(
            "The first anisotropic implementation requires cubic local crops, "
            f"got runtime_geometry.local_crop_shape={local_crop_shape}."
        )

    return DataAugmentationDINO3dSulcalAnisotropic(
        global_crops_shape=global_crop_shape,
        local_crops_size=local_crop_shape[0],
        local_crops_number=int(cfg.crops.local_crops_number),
        affine_prob_global=float(cfg.train.affine_prob_global),
        translate_range_global=tuple(cfg.train.translate_range_global),
        rotate_range_global=tuple(cfg.train.rotate_range_global),
        affine_prob_local=float(cfg.train.affine_prob_local),
        translate_range_local=tuple(cfg.train.translate_range_local),
        rotate_range_local=tuple(cfg.train.rotate_range_local),
        max_local_crop_retries=int(cfg.train.max_local_crop_retries),
    )


def _build_ssl_dataset(
    cfg,
):
    """
    Build the dedicated anisotropic NPY dataset.
    """
    dataset_format = str(cfg.train.dataset_format)

    if dataset_format != "npy_array":
        raise ValueError(
            "train_anisotropic.py supports only "
            "train.dataset_format='npy_array', got "
            f"{dataset_format!r}."
        )

    logger.info("##################################################")
    logger.info("Building dedicated anisotropic SSL dataset")
    logger.info(
        "Dataset format       : %s",
        dataset_format,
    )
    logger.info(
        "Dataset path         : %s",
        cfg.train.dataset_path,
    )
    logger.info(
        "Global crop shape    : %s",
        tuple(cfg.runtime_geometry.global_crop_shape),
    )
    logger.info("JSON pathway         : disabled")
    logger.info("MONAI cache          : disabled")
    logger.info("Axis permutation     : none")
    logger.info("##################################################")

    return make_sulcal_npy_dataset_anisotropic_3d(
        cfg=cfg,
        transform=_build_sulcal_augmentation(cfg),
    )


def _build_mask_generator(
    cfg,
):
    """
    Build the anisotropic active-only patch-mask generator.
    """
    geometry = cfg.runtime_geometry

    global_patch_grid = tuple(int(value) for value in geometry.global_patch_grid)

    patch_size = int(cfg.student.patch_size)

    mask_generator = MaskingGenerator3dAnisotropic(
        input_size=global_patch_grid,
        patch_size=patch_size,
    )

    expected_tokens = int(geometry.n_global_patch_tokens)

    if int(mask_generator.num_patches) != expected_tokens:
        raise RuntimeError(
            "Mask-generator token count does not match runtime geometry: "
            f"generator={mask_generator.num_patches}, "
            f"runtime_geometry={expected_tokens}."
        )

    expected_global_shape = (
        1,
        *tuple(int(value) for value in geometry.global_crop_shape),
    )

    if mask_generator.get_expected_volume_shape() != expected_global_shape:
        raise RuntimeError(
            "Mask-generator volume shape does not match runtime geometry: "
            f"generator={mask_generator.get_expected_volume_shape()}, "
            f"runtime_geometry={expected_global_shape}."
        )

    logger.info("##################################################")
    logger.info("Anisotropic masking geometry")
    logger.info(
        "Global crop shape     : %s",
        expected_global_shape,
    )
    logger.info(
        "Patch size            : %d",
        patch_size,
    )
    logger.info(
        "Global patch grid     : %s",
        global_patch_grid,
    )
    logger.info(
        "Global patch tokens   : %d",
        expected_tokens,
    )
    logger.info(
        "Masking strategy     : %s",
        (
            "non-empty sulcal patches only"
            if bool(cfg.train.use_non_empty_masking)
            else "uniform over all patches"
        ),
    )
    logger.info("##################################################")

    return mask_generator


# =============================================================================
# First-batch validation
# =============================================================================


def _tensor_is_finite(
    tensor: torch.Tensor,
) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _validate_and_log_first_batch(
    cfg,
    data: Dict[str, object],
    mask_generator: MaskingGenerator3dAnisotropic,
) -> None:
    """
    Validate the first complete collated anisotropic batch.
    """
    required_keys = {
        "collated_global_crops",
        "collated_local_crops",
        "collated_masks",
        "mask_indices_list",
        "masks_weight",
        "upperbound",
        "n_masked_patches",
    }

    missing_keys = required_keys.difference(data.keys())

    if missing_keys:
        raise KeyError(
            "First anisotropic batch is missing keys: " f"{sorted(missing_keys)}."
        )

    global_crops = data["collated_global_crops"]

    local_crops = data["collated_local_crops"]

    masks = data["collated_masks"]

    mask_indices = data["mask_indices_list"]

    masks_weight = data["masks_weight"]

    n_masked_tensor = data["n_masked_patches"]

    upperbound = int(data["upperbound"])

    for name, tensor in (
        (
            "collated_global_crops",
            global_crops,
        ),
        (
            "collated_local_crops",
            local_crops,
        ),
        (
            "collated_masks",
            masks,
        ),
        (
            "mask_indices_list",
            mask_indices,
        ),
        (
            "masks_weight",
            masks_weight,
        ),
        (
            "n_masked_patches",
            n_masked_tensor,
        ),
    ):
        if not isinstance(
            tensor,
            torch.Tensor,
        ):
            raise TypeError(
                f"First-batch field {name!r} must be a tensor, "
                f"got {type(tensor).__name__}."
            )

    batch_size = int(cfg.train.batch_size_per_gpu)

    n_global_crops = 2

    n_local_crops = int(cfg.crops.local_crops_number)

    global_crop_shape = tuple(
        int(value) for value in cfg.runtime_geometry.global_crop_shape
    )

    local_crop_shape = tuple(
        int(value) for value in cfg.runtime_geometry.local_crop_shape
    )

    n_global_tokens = int(cfg.runtime_geometry.n_global_patch_tokens)

    expected_global_shape = (
        batch_size * n_global_crops,
        1,
        *global_crop_shape,
    )

    expected_local_shape = (
        batch_size * n_local_crops,
        1,
        *local_crop_shape,
    )

    expected_mask_shape = (
        batch_size * n_global_crops,
        n_global_tokens,
    )

    if tuple(global_crops.shape) != expected_global_shape:
        raise RuntimeError(
            "Unexpected first-batch global crop shape: "
            f"expected {expected_global_shape}, "
            f"got {tuple(global_crops.shape)}."
        )

    if tuple(local_crops.shape) != expected_local_shape:
        raise RuntimeError(
            "Unexpected first-batch local crop shape: "
            f"expected {expected_local_shape}, "
            f"got {tuple(local_crops.shape)}."
        )

    if tuple(masks.shape) != expected_mask_shape:
        raise RuntimeError(
            "Unexpected first-batch mask shape: "
            f"expected {expected_mask_shape}, "
            f"got {tuple(masks.shape)}."
        )

    if masks.dtype != torch.bool:
        raise RuntimeError("First-batch masks must be bool, " f"got {masks.dtype}.")

    if not _tensor_is_finite(global_crops):
        raise FloatingPointError("Global crops contain NaN or infinity.")

    if not _tensor_is_finite(local_crops):
        raise FloatingPointError("Local crops contain NaN or infinity.")

    if not _tensor_is_finite(masks_weight):
        raise FloatingPointError("masks_weight contains NaN or infinity.")

    expected_mask_indices = torch.nonzero(
        masks.flatten(),
        as_tuple=False,
    ).flatten()

    if not torch.equal(
        mask_indices.cpu(),
        expected_mask_indices.cpu(),
    ):
        raise RuntimeError(
            "mask_indices_list does not match " "collated_masks.flatten()."
        )

    n_masked = int(mask_indices.numel())

    if n_masked_tensor.numel() != 1:
        raise RuntimeError("n_masked_patches must contain exactly one value.")

    if int(n_masked_tensor.item()) != n_masked:
        raise RuntimeError(
            "n_masked_patches does not match mask_indices_list: "
            f"tensor={int(n_masked_tensor.item())}, "
            f"indices={n_masked}."
        )

    if masks_weight.numel() != n_masked:
        raise RuntimeError(
            "masks_weight length does not match mask_indices_list: "
            f"weights={masks_weight.numel()}, "
            f"indices={n_masked}."
        )

    if upperbound < n_masked:
        raise RuntimeError(
            f"upperbound={upperbound} is smaller than " f"n_masked={n_masked}."
        )

    if n_masked > 0:
        maximum_index = int(mask_indices.max().item())

        flattened_capacity = expected_mask_shape[0] * n_global_tokens

        if maximum_index >= flattened_capacity:
            raise RuntimeError(
                "mask_indices_list contains an out-of-range index: "
                f"maximum={maximum_index}, "
                f"capacity={flattened_capacity}."
            )

    if mask_generator.num_patches != n_global_tokens:
        raise RuntimeError("First-batch mask generator and runtime token count differ.")

    logger.info("============================================================")
    logger.info("FIRST ANISOTROPIC BATCH VALIDATION")
    logger.info(
        "Global crops           : %s",
        tuple(global_crops.shape),
    )
    logger.info(
        "Local crops            : %s",
        tuple(local_crops.shape),
    )
    logger.info(
        "Masks                  : %s",
        tuple(masks.shape),
    )
    logger.info(
        "Mask indices           : %s",
        tuple(mask_indices.shape),
    )
    logger.info(
        "Number masked patches  : %d",
        n_masked,
    )
    logger.info(
        "Upperbound             : %d",
        upperbound,
    )
    logger.info(
        "Masks weight           : %s",
        tuple(masks_weight.shape),
    )
    logger.info(
        "Global crop dtype      : %s",
        global_crops.dtype,
    )
    logger.info(
        "Local crop dtype       : %s",
        local_crops.dtype,
    )
    logger.info("First batch validation : PASS")
    logger.info("============================================================")


# =============================================================================
# First-backward validation
# =============================================================================


def _safe_gradient_l2_norm(
    gradient: torch.Tensor,
) -> float:
    """
    Compute a stable L2 norm without float32 sum-of-squares overflow.
    """
    detached = gradient.detach().float()

    if detached.numel() == 0:
        return 0.0

    maximum_absolute_value = float(detached.abs().max().item())

    if not math.isfinite(maximum_absolute_value):
        return maximum_absolute_value

    if maximum_absolute_value == 0.0:
        return 0.0

    scaled = detached / maximum_absolute_value

    scaled_squared_sum = torch.sum(
        scaled * scaled,
        dtype=torch.float64,
    )

    return maximum_absolute_value * float(torch.sqrt(scaled_squared_sum).item())


def _sum_gradient_norm(
    parameters: Iterable[torch.nn.Parameter],
):
    """
    Return:
        gradient_norm or None,
        parameter_count,
        gradient_count
    """
    total_norm = 0.0
    parameter_count = 0
    gradient_count = 0

    for parameter in parameters:
        parameter_count += 1

        gradient = parameter.grad

        if gradient is None:
            continue

        gradient_count += 1

        gradient_norm = _safe_gradient_l2_norm(gradient)

        if not math.isfinite(gradient_norm):
            return (
                gradient_norm,
                parameter_count,
                gradient_count,
            )

        total_norm = math.hypot(
            total_norm,
            gradient_norm,
        )

    if gradient_count == 0:
        return (
            None,
            parameter_count,
            gradient_count,
        )

    return (
        total_norm,
        parameter_count,
        gradient_count,
    )


def _find_patch_embed_module(
    backbone,
):
    for module in backbone.modules():
        if module.__class__.__name__ == "PatchEmbed3d":
            return module

    return None


def _find_first_transformer_block(
    backbone,
):
    for module in backbone.modules():
        if (
            hasattr(module, "attn")
            and hasattr(module, "norm1")
            and hasattr(module, "norm2")
        ):
            return module

    return None


def _find_nonfinite_gradient_names(
    module: torch.nn.Module,
    maximum_names: int = 10,
):
    names = []

    for name, parameter in module.named_parameters():
        gradient = parameter.grad

        if gradient is None:
            continue

        if not bool(torch.isfinite(gradient).all().item()):
            names.append(name)

            if len(names) >= maximum_names:
                break

    return names


def _validate_teacher_has_no_gradients(
    model: SSLMetaArchAnisotropic,
) -> None:
    teacher_gradients = [
        name
        for name, parameter in model.teacher.named_parameters()
        if parameter.grad is not None
    ]

    if teacher_gradients:
        raise RuntimeError(
            "Teacher gradients were detected although the teacher must be "
            "EMA-only. First entries: "
            f"{teacher_gradients[:10]}."
        )


def _log_backward_diagnostics(
    *,
    loss_values: Dict[str, float],
    patch_gradient_norm,
    patch_gradient_count: int,
    patch_parameter_count: int,
    block_gradient_norm,
    block_gradient_count: int,
    block_parameter_count: int,
    total_backbone_gradient_norm,
    total_backbone_gradient_count: int,
    status: str,
) -> None:
    logger.info("============================================================")
    logger.info("FIRST ANISOTROPIC BACKWARD VALIDATION")

    for name, value in sorted(loss_values.items()):
        logger.info(
            "Loss %-28s : %.8f",
            name,
            value,
        )

    logger.info(
        "Patch embedding grad norm : %s | grads=%d/%d",
        (
            f"{patch_gradient_norm:.8f}"
            if patch_gradient_norm is not None
            else "not individually exposed by FSDP"
        ),
        patch_gradient_count,
        patch_parameter_count,
    )

    logger.info(
        "First block grad norm     : %s | grads=%d/%d",
        (
            f"{block_gradient_norm:.8f}"
            if block_gradient_norm is not None
            else "not individually exposed by FSDP"
        ),
        block_gradient_count,
        block_parameter_count,
    )

    logger.info(
        "Total backbone grad norm  : %s | gradient tensors=%d",
        (
            f"{total_backbone_gradient_norm:.8f}"
            if total_backbone_gradient_norm is not None
            else "none"
        ),
        total_backbone_gradient_count,
    )

    logger.info("Teacher gradients          : none")

    logger.info(
        "First backward validation  : %s",
        status,
    )

    logger.info("============================================================")


def _validate_and_log_first_backward(
    model: SSLMetaArchAnisotropic,
    loss_dict: Dict[str, torch.Tensor],
    allow_amp_overflow: bool = False,
) -> bool:
    """
    Validate losses, student gradients and teacher gradient isolation.

    Returns True for finite, non-zero student gradients.

    Returns False only for a recoverable AMP gradient overflow when
    allow_amp_overflow=True. The caller must still invoke scaler.step() and
    scaler.update() so that the optimizer step is skipped and the scale drops.

    Non-finite losses, teacher gradients, zero gradients and structural errors
    remain fatal.
    """
    if not loss_dict:
        raise RuntimeError("The first backward returned an empty loss dictionary.")

    loss_values: Dict[
        str,
        float,
    ] = {}

    for name, loss in loss_dict.items():
        if not isinstance(
            loss,
            torch.Tensor,
        ):
            raise TypeError(
                f"Loss {name!r} must be a tensor, got " f"{type(loss).__name__}."
            )

        if loss.numel() != 1:
            raise RuntimeError(
                f"Loss {name!r} must be scalar, " f"got shape {tuple(loss.shape)}."
            )

        value = float(loss.detach().item())

        if not math.isfinite(value):
            raise FloatingPointError(f"Loss {name!r} is not finite: " f"{value}.")

        loss_values[name] = value

    # This invariant remains strict even during an AMP overflow.
    _validate_teacher_has_no_gradients(model)

    backbone = model.student.backbone

    patch_embed = _find_patch_embed_module(backbone)

    first_block = _find_first_transformer_block(backbone)

    if patch_embed is None:
        raise RuntimeError(
            "Unable to locate PatchEmbed3d inside " "the FSDP student backbone."
        )

    if first_block is None:
        raise RuntimeError(
            "Unable to locate a transformer block inside " "the student backbone."
        )

    (
        patch_gradient_norm,
        patch_parameter_count,
        patch_gradient_count,
    ) = _sum_gradient_norm(patch_embed.parameters())

    (
        block_gradient_norm,
        block_parameter_count,
        block_gradient_count,
    ) = _sum_gradient_norm(first_block.parameters())

    (
        total_backbone_gradient_norm,
        _,
        total_backbone_gradient_count,
    ) = _sum_gradient_norm(backbone.parameters())

    if total_backbone_gradient_norm is None:
        raise RuntimeError(
            "No student-backbone gradient was found " "after the first backward."
        )

    nonfinite_gradient_names = _find_nonfinite_gradient_names(model.student)

    gradients_are_finite = (
        math.isfinite(total_backbone_gradient_norm) and not nonfinite_gradient_names
    )

    if not gradients_are_finite:
        _log_backward_diagnostics(
            loss_values=loss_values,
            patch_gradient_norm=(patch_gradient_norm),
            patch_gradient_count=(patch_gradient_count),
            patch_parameter_count=(patch_parameter_count),
            block_gradient_norm=(block_gradient_norm),
            block_gradient_count=(block_gradient_count),
            block_parameter_count=(block_parameter_count),
            total_backbone_gradient_norm=(total_backbone_gradient_norm),
            total_backbone_gradient_count=(total_backbone_gradient_count),
            status=("AMP OVERFLOW — " "RETRY ON A LATER ITERATION"),
        )

        if allow_amp_overflow:
            logger.warning(
                "AMP gradient overflow detected. GradScaler will skip the "
                "optimizer step and reduce its scale. Non-finite student "
                "gradients (first %d): %s",
                len(nonfinite_gradient_names),
                nonfinite_gradient_names,
            )

            return False

        raise FloatingPointError(
            "Student gradients are not finite. "
            "Non-finite gradient parameters: "
            f"{nonfinite_gradient_names}"
        )

    if total_backbone_gradient_norm <= 0:
        raise RuntimeError("The total student-backbone gradient norm is zero.")

    if patch_gradient_norm is not None:
        if not math.isfinite(patch_gradient_norm) or patch_gradient_norm <= 0:
            raise RuntimeError(
                "The patch-embedding gradient norm is zero "
                "or non-finite: "
                f"{patch_gradient_norm}."
            )
    else:
        logger.warning(
            "Patch-embedding child gradients are not individually exposed by "
            "the current FSDP wrapper; total backbone gradients are valid."
        )

    if block_gradient_norm is not None:
        if not math.isfinite(block_gradient_norm) or block_gradient_norm <= 0:
            raise RuntimeError(
                "The first transformer-block gradient norm is zero or "
                f"non-finite: {block_gradient_norm}."
            )
    else:
        logger.warning(
            "Transformer-block child gradients are not individually exposed "
            "by the current FSDP wrapper; total backbone gradients are valid."
        )

    _log_backward_diagnostics(
        loss_values=loss_values,
        patch_gradient_norm=(patch_gradient_norm),
        patch_gradient_count=(patch_gradient_count),
        patch_parameter_count=(patch_parameter_count),
        block_gradient_norm=(block_gradient_norm),
        block_gradient_count=(block_gradient_count),
        block_parameter_count=(block_parameter_count),
        total_backbone_gradient_norm=(total_backbone_gradient_norm),
        total_backbone_gradient_count=(total_backbone_gradient_count),
        status="PASS",
    )

    return True


# =============================================================================
# Training-loop helpers
# =============================================================================


def _resolve_max_iterations(
    cfg,
) -> Tuple[int, int]:
    official_epoch_length = int(cfg.train.OFFICIAL_EPOCH_LENGTH)

    configured_max_iter = int(cfg.optim.epochs) * official_epoch_length

    if configured_max_iter <= 0:
        raise ValueError(
            "configured_max_iter must be positive, " f"got {configured_max_iter}."
        )

    stop_after_iterations = int(cfg.train.stop_after_iterations)

    if stop_after_iterations > 0:
        max_iter = min(
            configured_max_iter,
            stop_after_iterations,
        )
    else:
        max_iter = configured_max_iter

    if max_iter <= 0:
        raise ValueError("Resolved max_iter must be positive, " f"got {max_iter}.")

    return (
        configured_max_iter,
        max_iter,
    )


# =============================================================================
# Training loop
# =============================================================================


def do_train(
    cfg,
    model,
    resume: bool = False,
):
    """
    Run the single-GPU anisotropic SSL training loop.

    AMP overflows are recoverable. Teacher EMA occurs only after a real
    student optimizer step.
    """
    model.train()

    if int(distributed.get_global_size()) != 1:
        raise ValueError("train_anisotropic.py supports exactly one GPU/process.")

    inputs_dtype = torch.half

    fp16_scaler = model.fp16_scaler

    optimizer = build_optimizer(
        cfg,
        model.get_params_groups(),
    )

    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    checkpointer = FSDPCheckpointer(
        model,
        cfg.train.output_dir,
        optimizer=optimizer,
        save_to_disk=True,
    )

    start_iter = (
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS,
            resume=resume,
        ).get(
            "iteration",
            -1,
        )
        + 1
    )

    (
        configured_max_iter,
        max_iter,
    ) = _resolve_max_iterations(cfg)

    logger.info(
        "Training duration | configured_max_iter=%d | "
        "stop_after_iterations=%d | effective_max_iter=%d | "
        "start_iter=%d",
        configured_max_iter,
        int(cfg.train.stop_after_iterations),
        max_iter,
        start_iter,
    )

    if start_iter >= max_iter:
        logger.info(
            "No training iteration required: " "start_iter=%d >= max_iter=%d.",
            start_iter,
            max_iter,
        )

        return {}

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=int(cfg.train.saveckp_period_iterations),
        max_iter=max_iter,
        max_to_keep=int(cfg.train.max_to_keep),
    )

    mask_generator = _build_mask_generator(cfg)

    collate_fn = partial(
        collate_data_and_cast_anisotropic,
        mask_ratio_tuple=tuple(cfg.ibot.mask_ratio_min_max),
        mask_probability=float(cfg.ibot.mask_sample_probability),
        dtype=inputs_dtype,
        mask_generator=mask_generator,
        use_density_masking=bool(cfg.train.use_non_empty_masking),
    )

    dataset = _build_ssl_dataset(cfg)

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=int(cfg.train.batch_size_per_gpu),
        num_workers=int(cfg.train.num_workers),
        shuffle=True,
        seed=start_iter,
        sampler_type=(SamplerType.SHARDED_INFINITE),
        sampler_advance=0,
        drop_last=True,
        persistent_workers=(int(cfg.train.num_workers) > 0),
        collate_fn=collate_fn,
    )

    iteration = start_iter

    validate_first_batch = bool(cfg.train.validate_first_batch)

    # Separate states: a first AMP overflow must not repeat shape validation.
    first_batch_shapes_validated = False
    first_backward_validated = False

    successful_optimizer_steps = 0
    skipped_optimizer_steps = 0
    ema_updates = 0

    logger.info(
        "Starting anisotropic training from iteration %d",
        start_iter,
    )

    metric_logger = MetricLogger(
        delimiter="  ",
        output_file=os.path.join(
            cfg.train.output_dir,
            "training_metrics.json",
        ),
    )

    header = "Anisotropic training"

    for data in metric_logger.log_every(
        data_loader,
        5,
        header,
        max_iter,
        start_iter,
    ):
        if iteration >= max_iter:
            break

        if validate_first_batch and not first_batch_shapes_validated:
            _validate_and_log_first_batch(
                cfg,
                data,
                mask_generator,
            )

            first_batch_shapes_validated = True

        current_batch_size = data["collated_global_crops"].shape[0] / 2

        lr = float(lr_schedule[iteration])

        wd = float(wd_schedule[iteration])

        momentum = float(momentum_schedule[iteration])

        teacher_temp = float(teacher_temp_schedule[iteration])

        last_layer_lr = float(last_layer_lr_schedule[iteration])

        apply_optim_scheduler(
            optimizer,
            lr,
            wd,
            last_layer_lr,
        )

        optimizer.zero_grad(set_to_none=True)

        loss_dict = model.forward_backward(
            data,
            teacher_temp=(teacher_temp),
        )

        # Loss NaN/Inf remains fatal. Only gradient overflow is recoverable.
        loss_dict_reduced = {
            name: float(loss.detach().item()) for name, loss in loss_dict.items()
        }

        if not loss_dict_reduced:
            raise RuntimeError("The model returned an empty loss dictionary.")

        if not all(math.isfinite(value) for value in loss_dict_reduced.values()):
            raise FloatingPointError(
                "NaN or infinite loss detected before optimizer step: "
                f"{loss_dict_reduced}."
            )

        gradients_unscaled = False
        backward_gradients_finite = True

        if validate_first_batch and not first_backward_validated:
            if fp16_scaler is not None:
                fp16_scaler.unscale_(optimizer)

                gradients_unscaled = True

            backward_gradients_finite = _validate_and_log_first_backward(
                model,
                loss_dict,
                allow_amp_overflow=(fp16_scaler is not None),
            )

        optimizer_step_skipped = False
        amp_scale_after = None

        if fp16_scaler is not None:
            amp_scale_before = float(fp16_scaler.get_scale())

            # Clip only finite, already-unscaled gradients.
            if float(cfg.optim.clip_grad) > 0 and backward_gradients_finite:
                if not gradients_unscaled:
                    fp16_scaler.unscale_(optimizer)

                    gradients_unscaled = True

                for submodel in model.student.values():
                    submodel.clip_grad_norm_(float(cfg.optim.clip_grad))

            # On overflow, step() is skipped and update() reduces the scale.
            fp16_scaler.step(optimizer)

            fp16_scaler.update()

            amp_scale_after = float(fp16_scaler.get_scale())

            optimizer_step_skipped = amp_scale_after < amp_scale_before

            if optimizer_step_skipped:
                skipped_optimizer_steps += 1

                logger.warning(
                    "AMP overflow at iteration %d: optimizer step skipped | "
                    "scale %.1f -> %.1f",
                    iteration,
                    amp_scale_before,
                    amp_scale_after,
                )

            else:
                successful_optimizer_steps += 1

                logger.info(
                    "AMP optimizer step completed at iteration %d | " "scale=%.1f",
                    iteration,
                    amp_scale_after,
                )

        else:
            if float(cfg.optim.clip_grad) > 0:
                for submodel in model.student.values():
                    submodel.clip_grad_norm_(float(cfg.optim.clip_grad))

            optimizer.step()

            successful_optimizer_steps += 1

        # EMA follows only a real student update.
        if not optimizer_step_skipped:
            model.update_teacher(momentum)

            ema_updates += 1

            if (
                validate_first_batch
                and not first_backward_validated
                and backward_gradients_finite
            ):
                first_backward_validated = True

                logger.info(
                    "First finite backward, optimizer step and teacher EMA "
                    "validated at iteration %d.",
                    iteration,
                )

        else:
            logger.warning(
                "Teacher EMA skipped at iteration %d because the student "
                "optimizer step was skipped.",
                iteration,
            )

        losses_reduced = sum(loss_dict_reduced.values())

        metric_logger.update(lr=lr)

        metric_logger.update(wd=wd)

        metric_logger.update(mom=momentum)

        metric_logger.update(last_layer_lr=last_layer_lr)

        metric_logger.update(current_batch_size=(current_batch_size))

        metric_logger.update(
            total_loss=losses_reduced,
            **loss_dict_reduced,
        )

        if amp_scale_after is not None:
            metric_logger.update(amp_scale=(amp_scale_after))

            metric_logger.update(optimizer_step_skipped=float(optimizer_step_skipped))

        if int(cfg.evaluation.eval_period_iterations) > 0 and (
            (iteration + 1) % int(cfg.evaluation.eval_period_iterations) == 0
        ):
            do_test(
                cfg,
                model,
                f"training_{iteration + 1}",
            )

            torch.cuda.synchronize()

        periodic_checkpointer.step(iteration)

        iteration += 1

    metric_logger.synchronize_between_processes()

    # Final smoke-test guarantees.
    if validate_first_batch and not first_batch_shapes_validated:
        raise RuntimeError(
            "The smoke test ended without validating an anisotropic batch."
        )

    if validate_first_batch and not first_backward_validated:
        raise RuntimeError(
            "No finite student backward followed by a successful optimizer "
            f"step and teacher EMA was observed during "
            f"{max_iter - start_iter} smoke-test iterations. "
            "Increase stop_after_iterations or inspect the AMP scale and "
            "non-finite gradient diagnostics."
        )

    if successful_optimizer_steps <= 0:
        raise RuntimeError("Training completed without any successful optimizer step.")

    if ema_updates != successful_optimizer_steps:
        raise RuntimeError(
            "Teacher EMA accounting mismatch: "
            f"ema_updates={ema_updates}, "
            f"successful_optimizer_steps={successful_optimizer_steps}."
        )

    logger.info(
        "Anisotropic training stopped at iteration %d/%d | "
        "successful optimizer steps=%d | skipped AMP steps=%d | "
        "teacher EMA updates=%d",
        iteration,
        max_iter,
        successful_optimizer_steps,
        skipped_optimizer_steps,
        ema_updates,
    )

    return {name: meter.global_avg for name, meter in metric_logger.meters.items()}


# =============================================================================
# Main entry point
# =============================================================================


def main(
    args,
):
    cfg = setup_3d_anisotropic(args)

    if int(distributed.get_global_size()) != 1:
        raise ValueError("The anisotropic pipeline supports one H100 only.")

    logger.info(
        "Using dedicated anisotropic training entry point | "
        "global_shape=%s | global_grid=%s | global_tokens=%d",
        tuple(cfg.runtime_geometry.global_crop_shape),
        tuple(cfg.runtime_geometry.global_patch_grid),
        int(cfg.runtime_geometry.n_global_patch_tokens),
    )

    model = SSLMetaArchAnisotropic(cfg).to(torch.device("cuda"))

    model.prepare_for_distributed_training()

    logger.info(
        "Model:\n%s",
        model,
    )

    if args.eval_only:
        iteration = (
            FSDPCheckpointer(
                model,
                save_dir=(cfg.train.output_dir),
            )
            .resume_or_load(
                cfg.MODEL.WEIGHTS,
                resume=(not args.no_resume),
            )
            .get(
                "iteration",
                -1,
            )
            + 1
        )

        return do_test(
            cfg,
            model,
            f"manual_{iteration}",
        )

    return do_train(
        cfg,
        model,
        resume=(not args.no_resume),
    )


if __name__ == "__main__":
    parsed_args = get_args_parser(add_help=True).parse_args()

    main(parsed_args)
