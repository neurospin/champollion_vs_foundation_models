# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)
#
# Portions of this file are derived from DINOv2 (Meta AI Research),
# licensed under the Apache License 2.0.

"""Training entry point for isotropic sulcal SSL over frozen upstream 3DINO.

Own training loop + diagnostics wiring the sulcal data pipeline, the
SulcusAwareSSLMetaArch subclass and setup. The optimizer/scheduler/eval
helpers (build_optimizer, build_schedulers, apply_optim_scheduler, do_test)
are imported unchanged from upstream ``dinov2.train.train3d``.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from functools import partial
from typing import Dict, Iterable

import torch
from fvcore.common.checkpoint import PeriodicCheckpointer

import dinov2.distributed as distributed
from sulcus_aware_3DINO.data import (
    DataAugmentationDINO3d_sulcal,
    MaskingGenerator3d,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    make_sulcal_npy_dataset_3d,
)
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from sulcus_aware_3DINO.training.meta_arch import SulcusAwareSSLMetaArch
from sulcus_aware_3DINO.training.setup import setup_3d

# Optimizer/scheduler/eval helpers are functionally identical to upstream
# 3DINO (black-only formatting diffs), so they are imported, not vendored.
from dinov2.train.train3d import (
    apply_optim_scheduler,
    build_optimizer,
    build_schedulers,
    do_test,
)

torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser(
        "3DINO sulcal SSL training",
        add_help=add_help,
    )

    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory.",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="save teacher checkpoint only",
    )

    parser.add_argument(
        "--eval",
        type=str,
        default="",
        help="Eval type to perform",
    )

    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    parser.add_argument(
        "--local-rank",
        default=0,
        type=int,
        help="Variable for distributed computing.",
    )

    parser.add_argument(
        "--cache-dir",
        default=None,
        type=str,
        help="Path to cache directory for MONAI dataset.",
    )

    return parser


def _cfg_get(node, key, default):
    """
    Safe getter for OmegaConf nodes.

    This keeps train.py backward-compatible with older YAML configs where
    new keys such as train.dataset_format do not exist yet.
    """
    try:
        if key in node:
            return node[key]
    except Exception:
        pass

    return default


def _build_sulcal_augmentation(cfg):
    """
    Build the stochastic DINO/iBOT sulcal augmentation.

    It is applied after the npy_array dataset's deterministic preprocessing
    and accepts a direct tensor [1, D, H, W].
    """
    return DataAugmentationDINO3d_sulcal(
        local_crops_number=(cfg.crops.local_crops_number),
        global_crops_size=(cfg.crops.global_crops_size),
        local_crops_size=(cfg.crops.local_crops_size),
        affine_prob_global=(cfg.train.affine_prob_global),
        translate_range_global=tuple(cfg.train.translate_range_global),
        rotate_range_global=tuple(cfg.train.rotate_range_global),
        affine_prob_local=(cfg.train.affine_prob_local),
        translate_range_local=tuple(cfg.train.translate_range_local),
        rotate_range_local=tuple(cfg.train.rotate_range_local),
    )


def _build_sulcal_transform_from_array(cfg):
    """
    Sulcal SSL pipeline for one large raw .npy array.

    The deterministic preprocessing is handled by SulcalNpyArrayDataset:
      - index along first dimension
      - remove trailing singleton channel if present
      - binarize with x != 0
      - isotropic upscale + centered padding to target size
      - return tensor [1,T,T,T]

    Therefore the transform here is only the stochastic DINO/iBOT augmentation.
    """
    return _build_sulcal_augmentation(cfg)


def _build_ssl_dataset(cfg):
    """
    Build the SSL dataset from cfg.train.dataset_format.

    Only ``npy_array`` is supported: one large .npy array with shape
    [N,D,H,W] or [N,D,H,W,1], read with numpy memmap and preprocessed at
    runtime (index, squeeze, binarize, isotropic upscale + centered pad).
    """
    dataset_format = str(
        _cfg_get(
            cfg.train,
            "dataset_format",
            "npy_array",
        )
    )

    logger.info("###################################")
    logger.info("Building SSL dataset")
    logger.info(
        "dataset_format: %s",
        dataset_format,
    )
    logger.info(
        "dataset_path  : %s",
        cfg.train.dataset_path,
    )
    logger.info("###################################")

    if dataset_format == "npy_array":
        data_transform = _build_sulcal_transform_from_array(cfg)

        target_size = int(
            _cfg_get(
                cfg.train,
                "preprocess_target_size",
                cfg.crops.global_crops_size,
            )
        )

        global_crops_size = int(cfg.crops.global_crops_size)

        if target_size != global_crops_size:
            raise ValueError(
                "In dataset_format='npy_array', "
                "train.preprocess_target_size must match "
                "crops.global_crops_size. "
                f"Got preprocess_target_size={target_size}, "
                f"global_crops_size={global_crops_size}."
            )

        mmap_mode = str(
            _cfg_get(
                cfg.train,
                "npy_mmap_mode",
                "r",
            )
        )

        input_layout = str(
            _cfg_get(
                cfg.train,
                "npy_input_layout",
                "auto",
            )
        )

        binarize_nonzero = bool(
            _cfg_get(
                cfg.train,
                "binarize_nonzero",
                True,
            )
        )

        logger.info("Data pipeline: single raw NPY array")
        logger.info("Runtime preprocessing: enabled")
        logger.info("MONAI LoadImaged: disabled")
        logger.info("MONAI cache: disabled")
        logger.info(
            "Preprocess target size: %d",
            target_size,
        )
        logger.info(
            "NPY mmap mode         : %s",
            mmap_mode,
        )
        logger.info(
            "NPY input layout      : %s",
            input_layout,
        )
        logger.info(
            "Binarize nonzero      : %s",
            binarize_nonzero,
        )
        logger.info("Expected raw shape    : [N,D,H,W] or [N,D,H,W,1]")
        logger.info("Trailing channel      : squeezed if present")
        logger.info("Axis permutation      : none")

        dataset = make_sulcal_npy_dataset_3d(
            dataset_path=(cfg.train.dataset_path),
            target_size=target_size,
            transform=data_transform,
            mmap_mode=mmap_mode,
            input_layout=input_layout,
            binarize_nonzero=(binarize_nonzero),
        )

        return dataset

    raise ValueError(
        "Unknown train.dataset_format. "
        "Expected 'npy_array', "
        f"got: {dataset_format}"
    )


# =============================================================================
# Numerical-safety helpers
# =============================================================================


def _safe_gradient_l2_norm(
    gradient: torch.Tensor,
) -> float:
    """
    Compute a stable L2 norm without overflowing the float32 sum of squares.
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


def _get_student_backbone(model):
    student = model.student

    if hasattr(
        student,
        "backbone",
    ):
        return student.backbone

    try:
        return student["backbone"]
    except Exception as error:
        raise RuntimeError("Unable to locate the student backbone.") from error


def _find_patch_embed_module(backbone):
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
    model,
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
    logger.info("FIRST ISOTROPIC BACKWARD VALIDATION")

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
    model,
    loss_dict: Dict[str, torch.Tensor],
    allow_amp_overflow: bool = False,
) -> bool:
    """
    Validate finite scalar losses, student gradients and teacher isolation.

    Returns True when the student gradients are finite and non-zero.

    Returns False only for a recoverable AMP overflow when
    allow_amp_overflow=True. The caller must still invoke scaler.step() and
    scaler.update() so that GradScaler skips the optimizer step and lowers
    its scale.
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

    _validate_teacher_has_no_gradients(model)

    backbone = _get_student_backbone(model)

    patch_embed = _find_patch_embed_module(backbone)

    first_block = _find_first_transformer_block(backbone)

    if patch_embed is None:
        raise RuntimeError(
            "Unable to locate PatchEmbed3d inside " "the student backbone."
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
        raise RuntimeError("No student-backbone gradient was found after backward.")

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
                "The patch-embedding gradient norm is zero or non-finite: "
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


def _amp_found_inf_after_unscale(
    fp16_scaler,
    optimizer,
):
    """
    Read GradScaler's overflow flag after unscale_.

    Returns:
        True  -> at least one non-finite gradient was detected
        False -> all gradients inspected by GradScaler were finite
        None  -> the scaler implementation does not expose the state
    """
    per_optimizer_states = getattr(
        fp16_scaler,
        "_per_optimizer_states",
        None,
    )

    if per_optimizer_states is None:
        return None

    optimizer_state = per_optimizer_states.get(id(optimizer))

    if not optimizer_state:
        return None

    found_inf_per_device = optimizer_state.get("found_inf_per_device")

    if not found_inf_per_device:
        return None

    found_inf = sum(float(value.item()) for value in found_inf_per_device.values())

    return found_inf > 0.0


def _all_student_gradients_are_finite(
    model,
) -> bool:
    """
    Fallback used only when the scaler does not expose its found-inf state.
    """
    for parameter in model.student.parameters():
        gradient = parameter.grad

        if gradient is None:
            continue

        if not bool(torch.isfinite(gradient).all().item()):
            return False

    return True


def do_train(
    cfg,
    model,
    resume=False,
):
    model.train()

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

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH

    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=(cfg.train.saveckp_period_iterations),
        max_iter=max_iter,
        max_to_keep=(cfg.train.max_to_keep),
    )

    img_size = int(cfg.crops.global_crops_size)

    patch_size = int(cfg.student.patch_size)

    if img_size % patch_size != 0:
        raise ValueError(
            "crops.global_crops_size must be divisible by "
            "student.patch_size. "
            f"Got global_crops_size={img_size}, "
            f"patch_size={patch_size}."
        )

    grid_size = img_size // patch_size

    n_tokens = grid_size**3

    mask_generator = MaskingGenerator3d(
        input_size=(
            grid_size,
            grid_size,
            grid_size,
        ),
        patch_size=patch_size,
    )

    logger.info(
        "Global crop size: %d",
        img_size,
    )

    logger.info(
        "Patch size: %d",
        patch_size,
    )

    logger.info(
        "Patch grid size: %d^3",
        grid_size,
    )

    logger.info(
        "Number of iBOT patch tokens: %d",
        n_tokens,
    )

    logger.info(
        "Masking strategy: %s",
        (
            "non-empty (sulcal active patches only)"
            if cfg.train.use_non_empty_masking
            else "uniform (all patches)"
        ),
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=(cfg.ibot.mask_ratio_min_max),
        mask_probability=(cfg.ibot.mask_sample_probability),
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
        use_density_masking=(cfg.train.use_non_empty_masking),
    )

    dataset = _build_ssl_dataset(cfg)

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=(cfg.train.batch_size_per_gpu),
        num_workers=(cfg.train.num_workers),
        shuffle=True,
        seed=start_iter,
        sampler_type=(SamplerType.SHARDED_INFINITE),
        sampler_advance=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    iteration = start_iter

    first_backward_validated = False

    successful_optimizer_steps = 0
    skipped_optimizer_steps = 0
    ema_updates = 0

    logger.info(
        "Starting training from iteration %d",
        start_iter,
    )

    metrics_file = os.path.join(
        cfg.train.output_dir,
        "training_metrics.json",
    )

    metric_logger = MetricLogger(
        delimiter="  ",
        output_file=metrics_file,
    )

    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        5,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2

        if iteration > max_iter:
            return

        lr = lr_schedule[iteration]

        wd = wd_schedule[iteration]

        mom = momentum_schedule[iteration]

        teacher_temp = teacher_temp_schedule[iteration]

        last_layer_lr = last_layer_lr_schedule[iteration]

        apply_optim_scheduler(
            optimizer,
            lr,
            wd,
            last_layer_lr,
        )

        optimizer.zero_grad(set_to_none=True)

        loss_dict = model.forward_backward(
            data,
            teacher_temp=teacher_temp,
        )

        # Non-finite losses are fatal and must be detected before any
        # optimizer or teacher-EMA update.
        local_loss_values = {
            name: float(loss.detach().item()) for name, loss in loss_dict.items()
        }

        if not local_loss_values:
            raise RuntimeError("The model returned an empty loss dictionary.")

        nonfinite_losses = {
            name: value
            for name, value in local_loss_values.items()
            if not math.isfinite(value)
        }

        if nonfinite_losses:
            raise FloatingPointError(
                "NaN or infinite loss detected before optimizer step: "
                f"{nonfinite_losses}. Full loss dictionary: "
                f"{local_loss_values}."
            )

        gradients_unscaled = False
        backward_gradients_finite = True
        optimizer_step_skipped = False
        amp_scale_after = None

        if fp16_scaler is not None:
            amp_scale_before = float(fp16_scaler.get_scale())

            # Explicit unscaling is required before gradient diagnostics and
            # clipping. It also populates GradScaler's found-inf state.
            if cfg.optim.clip_grad or not first_backward_validated:
                fp16_scaler.unscale_(optimizer)

                gradients_unscaled = True

            if not first_backward_validated:
                backward_gradients_finite = _validate_and_log_first_backward(
                    model,
                    loss_dict,
                    allow_amp_overflow=True,
                )

            elif gradients_unscaled:
                amp_found_inf = _amp_found_inf_after_unscale(
                    fp16_scaler,
                    optimizer,
                )

                if amp_found_inf is None:
                    backward_gradients_finite = _all_student_gradients_are_finite(model)
                else:
                    backward_gradients_finite = not amp_found_inf

            # Never clip gradients that are already known to contain inf/NaN.
            if cfg.optim.clip_grad and backward_gradients_finite:
                if not gradients_unscaled:
                    fp16_scaler.unscale_(optimizer)

                    gradients_unscaled = True

                for submodel in model.student.values():
                    submodel.clip_grad_norm_(cfg.optim.clip_grad)

            # GradScaler skips optimizer.step() on overflow and lowers scale.
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

        else:
            if not first_backward_validated:
                backward_gradients_finite = _validate_and_log_first_backward(
                    model,
                    loss_dict,
                    allow_amp_overflow=False,
                )
            else:
                backward_gradients_finite = _all_student_gradients_are_finite(model)

                if not backward_gradients_finite:
                    nonfinite_gradient_names = _find_nonfinite_gradient_names(
                        model.student
                    )

                    raise FloatingPointError(
                        "Non-finite student gradients detected without AMP. "
                        "First affected parameters: "
                        f"{nonfinite_gradient_names}."
                    )

            if cfg.optim.clip_grad and backward_gradients_finite:
                for submodel in model.student.values():
                    submodel.clip_grad_norm_(cfg.optim.clip_grad)

            optimizer.step()

            successful_optimizer_steps += 1

        # The EMA teacher is updated only after a real student optimizer step.
        if not optimizer_step_skipped:
            model.update_teacher(mom)

            ema_updates += 1

            if not first_backward_validated and backward_gradients_finite:
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

        if distributed.get_global_size() > 1:
            for value in loss_dict.values():
                torch.distributed.all_reduce(value)

        loss_dict_reduced = {
            name: (value.item() / distributed.get_global_size())
            for name, value in loss_dict.items()
        }

        # Re-check reduced losses because a non-finite value from another
        # process must not enter metrics or checkpoints unnoticed.
        nonfinite_reduced_losses = {
            name: value
            for name, value in loss_dict_reduced.items()
            if not math.isfinite(value)
        }

        if nonfinite_reduced_losses:
            raise FloatingPointError(
                "NaN or infinite reduced loss detected: "
                f"{nonfinite_reduced_losses}. "
                "Full reduced loss dictionary: "
                f"{loss_dict_reduced}."
            )

        losses_reduced = sum(loss_dict_reduced.values())

        metric_logger.update(lr=lr)

        metric_logger.update(wd=wd)

        metric_logger.update(mom=mom)

        metric_logger.update(last_layer_lr=last_layer_lr)

        metric_logger.update(current_batch_size=(current_batch_size))

        metric_logger.update(
            total_loss=losses_reduced,
            **loss_dict_reduced,
        )

        if amp_scale_after is not None:
            metric_logger.update(amp_scale=(amp_scale_after))

            metric_logger.update(optimizer_step_skipped=float(optimizer_step_skipped))

        if cfg.evaluation.eval_period_iterations > 0 and (
            (iteration + 1) % cfg.evaluation.eval_period_iterations == 0
        ):
            do_test(
                cfg,
                model,
                f"training_{iteration}",
            )

            torch.cuda.synchronize()

        periodic_checkpointer.step(iteration)

        iteration += 1

    metric_logger.synchronize_between_processes()

    logger.info(
        "Training optimizer/EMA summary | "
        "successful optimizer steps=%d | "
        "skipped AMP steps=%d | "
        "teacher EMA updates=%d",
        successful_optimizer_steps,
        skipped_optimizer_steps,
        ema_updates,
    )

    if ema_updates != successful_optimizer_steps:
        raise RuntimeError(
            "Teacher EMA accounting mismatch: "
            f"ema_updates={ema_updates}, "
            f"successful_optimizer_steps="
            f"{successful_optimizer_steps}."
        )

    return {name: meter.global_avg for name, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup_3d(args)

    model = SulcusAwareSSLMetaArch(cfg).to(torch.device("cuda"))

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
    args = get_args_parser(add_help=True).parse_args()

    main(args)
