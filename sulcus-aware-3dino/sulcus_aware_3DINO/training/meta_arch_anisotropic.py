# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)
#
# Portions of this file are derived from DINOv2 (Meta AI Research),
# licensed under the Apache License 2.0.

"""Sulcus-aware SSL meta-architecture for the anisotropic pipeline.

A standalone re-implementation of the DINO/iBOT self-supervised training
step (algorithm of origin: DINOv2, Apache-2.0) adapted to non-cubic
(anisotropic) crop/patch geometry. It builds its backbone via
``build_model_from_anisotropic_cfg`` and reuses only upstream 3DINO
building blocks (FSDP, DINOHead, losses) imported as ``dinov2.*``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn

import dinov2.distributed as distributed
from dinov2.fsdp import (
    ShardedGradScaler,
    get_fsdp_modules,
    get_fsdp_wrapper,
    reshard_fsdp_model,
)
from dinov2.layers import DINOHead
from dinov2.loss import DINOLoss, KoLeoLoss, iBOTPatchLoss
from sulcus_aware_3DINO.models.build_anisotropic import (
    build_model_from_anisotropic_cfg,
)
from dinov2.models.vision_transformer import BlockChunk
from sulcus_aware_3DINO.training.param_groups import (
    fuse_params_groups,
    get_params_groups_with_decay,
)
from dinov2.utils.utils import has_batchnorms

try:
    from xformers.ops import fmha
except ImportError as error:
    raise AssertionError("xFormers is required for training") from error


logger = logging.getLogger("dinov2")

PUBLIC_BACKBONE_PREFIX = "backbone."
PUBLIC_REFERENCE_SIZE = 112
PUBLIC_PATCH_SIZE = 16
PUBLIC_NUM_PATCHES = 343
PUBLIC_EMBED_DIM = 1024
PUBLIC_POS_EMBED_SHAPE = (1, 344, 1024)


# =============================================================================
# Public checkpoint loading
# =============================================================================


def extract_public_backbone_state_dict(
    checkpoint: Mapping[str, Any],
    expected_pos_embed_shape: Tuple[int, int, int] = (PUBLIC_POS_EMBED_SHAPE),
) -> Dict[str, torch.Tensor]:
    """
    Extract only ``teacher['backbone.*']`` and remove that exact prefix.

    No positional-embedding interpolation is performed here. The public
    positional embedding must remain on its native 7x7x7 grid.
    """
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Checkpoint must be a mapping, got " f"{type(checkpoint).__name__}."
        )

    if "teacher" not in checkpoint:
        raise KeyError("Public checkpoint must contain a top-level 'teacher' key.")

    teacher_state = checkpoint["teacher"]

    if not isinstance(teacher_state, Mapping):
        raise TypeError(
            "checkpoint['teacher'] must be a state-dict mapping, got "
            f"{type(teacher_state).__name__}."
        )

    backbone_state: Dict[str, torch.Tensor] = {}

    for key, value in teacher_state.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Checkpoint key must be str, got " f"{type(key).__name__}."
            )

        if not key.startswith(PUBLIC_BACKBONE_PREFIX):
            continue

        stripped_key = key[len(PUBLIC_BACKBONE_PREFIX) :]

        if not stripped_key:
            raise ValueError("Empty key after removing the 'backbone.' prefix.")

        if stripped_key in backbone_state:
            raise ValueError(f"Duplicate extracted backbone key: " f"{stripped_key!r}.")

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Checkpoint value for {key!r} must be a tensor, "
                f"got {type(value).__name__}."
            )

        backbone_state[stripped_key] = value

    if not backbone_state:
        raise ValueError(
            "No keys starting with 'backbone.' were found under "
            "checkpoint['teacher']."
        )

    if "pos_embed" not in backbone_state:
        raise KeyError(
            "Extracted backbone does not contain 'pos_embed' "
            "(source key 'backbone.pos_embed')."
        )

    actual_shape = tuple(backbone_state["pos_embed"].shape)

    expected_shape = tuple(expected_pos_embed_shape)

    if actual_shape != expected_shape:
        raise ValueError(
            "Unexpected public positional-embedding shape: "
            f"expected {expected_shape}, got {actual_shape}. "
            "The checkpoint positional embedding must not be resized."
        )

    return backbone_state


def load_public_backbone_checkpoint_strict(
    checkpoint_path: str | Path,
    student_backbone: nn.Module,
    teacher_backbone: nn.Module,
) -> None:
    """
    Load the public backbone strictly into both student and teacher.
    """
    checkpoint_path = Path(checkpoint_path).expanduser()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Public 3DINO checkpoint not found: {checkpoint_path}")

    logger.info(
        "Loading public 3DINO checkpoint: %s",
        checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = extract_public_backbone_state_dict(checkpoint)

    # strict=True raises on every missing, unexpected or
    # shape-incompatible key.
    student_backbone.load_state_dict(
        state_dict,
        strict=True,
    )

    teacher_backbone.load_state_dict(
        state_dict,
        strict=True,
    )

    logger.info(
        "Strict public-backbone load succeeded | " "keys=%d | pos_embed=%s",
        len(state_dict),
        tuple(state_dict["pos_embed"].shape),
    )


# =============================================================================
# Validation helpers
# =============================================================================


def log_trainable_parameters(
    module: nn.Module,
    prefix: str,
) -> None:
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )

    frozen = sum(
        parameter.numel()
        for parameter in module.parameters()
        if not parameter.requires_grad
    )

    total = trainable + frozen

    logger.info(
        "[%s] trainable: %s",
        prefix,
        f"{trainable:,}",
    )

    logger.info(
        "[%s] frozen   : %s",
        prefix,
        f"{frozen:,}",
    )

    logger.info(
        "[%s] total    : %s",
        prefix,
        f"{total:,}",
    )

    logger.info(
        "[%s] trainable ratio: %.4f%%",
        prefix,
        100.0 * trainable / max(total, 1),
    )


def _validate_supported_scope(cfg) -> None:
    """
    Validate the intentionally restricted first implementation.
    """
    if int(distributed.get_global_size()) != 1:
        raise ValueError("SSLMetaArchAnisotropic supports exactly one GPU/process.")

    if not bool(cfg.peft.enable) or str(cfg.peft.method) != "full_finetune":
        raise ValueError(
            "The anisotropic pipeline supports full fine-tuning only: "
            "peft.enable=true and peft.method='full_finetune'."
        )

    if str(cfg.student.full_pretrained_weights or "").strip():
        raise ValueError(
            "student.full_pretrained_weights is not supported. "
            "Use only student.pretrained_weights for the public backbone."
        )

    if not str(cfg.student.pretrained_weights or "").strip():
        raise ValueError(
            "student.pretrained_weights must point to the public " "3DINO checkpoint."
        )

    if str(cfg.student.arch) != "vit_large_3d":
        raise ValueError(
            "Only student.arch='vit_large_3d' is supported, got "
            f"{cfg.student.arch!r}."
        )

    if int(cfg.runtime_geometry.backbone_reference_size) != PUBLIC_REFERENCE_SIZE:
        raise ValueError(
            "backbone_reference_size must remain 112 " "for the public backbone."
        )

    if int(cfg.student.patch_size) != PUBLIC_PATCH_SIZE:
        raise ValueError("student.patch_size must remain 16.")

    if float(cfg.dino.loss_weight) <= 0:
        raise ValueError("DINO must be enabled with dino.loss_weight > 0.")

    if float(cfg.ibot.loss_weight) <= 0:
        raise ValueError("iBOT must be enabled with ibot.loss_weight > 0.")


def _validate_public_backbone(
    backbone: nn.Module,
    embed_dim: int,
    name: str,
) -> None:
    """
    Verify that the backbone retains the public 112-cubic geometry.
    """
    expected_img_size = (
        112,
        112,
        112,
    )

    expected_patch_size = (
        16,
        16,
        16,
    )

    checks = {
        "patch_embed.img_size": (
            tuple(backbone.patch_embed.img_size),
            expected_img_size,
        ),
        "patch_embed.patch_size": (
            tuple(backbone.patch_embed.patch_size),
            expected_patch_size,
        ),
        "patch_embed.num_patches": (
            int(backbone.patch_embed.num_patches),
            PUBLIC_NUM_PATCHES,
        ),
        "pos_embed.shape": (
            tuple(backbone.pos_embed.shape),
            PUBLIC_POS_EMBED_SHAPE,
        ),
        "embed_dim": (
            int(embed_dim),
            PUBLIC_EMBED_DIM,
        ),
    }

    for field, (actual, expected) in checks.items():
        if actual != expected:
            raise RuntimeError(
                f"{name} {field} mismatch: " f"expected {expected}, got {actual}."
            )

    logger.info(
        "[%s] public geometry validated | " "img_size=%s | patches=%d | pos=%s",
        name,
        expected_img_size,
        PUBLIC_NUM_PATCHES,
        PUBLIC_POS_EMBED_SHAPE,
    )


def _validate_gradient_policy(
    student: nn.Module,
    teacher: nn.Module,
) -> None:
    """
    Validate full-fine-tuning student and EMA-only teacher policies.
    """
    frozen_student = [
        name
        for name, parameter in student.named_parameters()
        if not parameter.requires_grad
    ]

    trainable_teacher = [
        name
        for name, parameter in teacher.named_parameters()
        if parameter.requires_grad
    ]

    if frozen_student:
        raise RuntimeError(
            "Full fine-tuning requires every student parameter "
            "to be trainable. "
            f"Frozen parameters: {frozen_student[:10]}"
        )

    if trainable_teacher:
        raise RuntimeError(
            "Teacher parameters must all be frozen. "
            f"Trainable parameters: {trainable_teacher[:10]}"
        )

    student_keys = set(student.state_dict().keys())

    teacher_keys = set(teacher.state_dict().keys())

    if student_keys != teacher_keys:
        raise RuntimeError(
            "Student/teacher architecture mismatch. "
            f"Only student: "
            f"{sorted(student_keys - teacher_keys)[:10]} | "
            f"Only teacher: "
            f"{sorted(teacher_keys - student_keys)[:10]}"
        )


# =============================================================================
# Dedicated anisotropic SSL meta-architecture
# =============================================================================


class SSLMetaArchAnisotropic(nn.Module):
    """
    Dedicated one-H100 anisotropic 3DINO SSL meta-architecture.

    Supported:
        - full fine-tuning;
        - public backbone checkpoint;
        - DINO;
        - iBOT;
        - KoLeo;
        - existing FSDP wrappers.

    Rejected:
        - LoRA;
        - LoRA plus last block;
        - additional blocks;
        - full_pretrained_weights;
        - multi-GPU execution.
    """

    def __init__(self, cfg):
        super().__init__()

        _validate_supported_scope(cfg)

        self.cfg = cfg

        self.fp16_scaler = (
            ShardedGradScaler() if cfg.compute_precision.grad_scaler else None
        )

        self.need_to_synchronize_fsdp_streams = True
        self._first_forward_validated = False

        self.global_crop_shape = tuple(
            int(value) for value in cfg.runtime_geometry.global_crop_shape
        )

        self.global_patch_grid = tuple(
            int(value) for value in cfg.runtime_geometry.global_patch_grid
        )

        self.global_patch_tokens = int(cfg.runtime_geometry.n_global_patch_tokens)

        self.local_crop_shape = tuple(
            int(value) for value in cfg.runtime_geometry.local_crop_shape
        )

        self.local_patch_grid = tuple(
            int(value) for value in cfg.runtime_geometry.local_patch_grid
        )

        self.local_patch_tokens = int(cfg.runtime_geometry.n_local_patch_tokens)

        (
            student_backbone,
            teacher_backbone,
            embed_dim,
        ) = build_model_from_anisotropic_cfg(cfg)

        _validate_public_backbone(
            student_backbone,
            embed_dim,
            "Student",
        )

        _validate_public_backbone(
            teacher_backbone,
            embed_dim,
            "Teacher",
        )

        load_public_backbone_checkpoint_strict(
            cfg.student.pretrained_weights,
            student_backbone,
            teacher_backbone,
        )

        for parameter in student_backbone.parameters():
            parameter.requires_grad = True

        for parameter in teacher_backbone.parameters():
            parameter.requires_grad = False

        self.embed_dim = int(embed_dim)

        self.do_dino = float(cfg.dino.loss_weight) > 0

        self.do_koleo = float(cfg.dino.koleo_loss_weight) > 0

        self.do_ibot = float(cfg.ibot.loss_weight) > 0

        self.ibot_separate_head = bool(cfg.ibot.separate_head)

        student_model_dict: Dict[str, nn.Module] = {
            "backbone": student_backbone,
        }

        teacher_model_dict: Dict[str, nn.Module] = {
            "backbone": teacher_backbone,
        }

        # ---------------------------------------------------------------------
        # DINO and KoLeo
        # ---------------------------------------------------------------------

        self.dino_loss_weight = float(cfg.dino.loss_weight)

        self.dino_out_dim = int(cfg.dino.head_n_prototypes)

        self.dino_loss = DINOLoss(self.dino_out_dim)

        if self.do_koleo:
            self.koleo_loss = KoLeoLoss()

        dino_head = partial(
            DINOHead,
            in_dim=self.embed_dim,
            out_dim=int(cfg.dino.head_n_prototypes),
            hidden_dim=int(cfg.dino.head_hidden_dim),
            bottleneck_dim=int(cfg.dino.head_bottleneck_dim),
            nlayers=int(cfg.dino.head_nlayers),
        )

        student_model_dict["dino_head"] = dino_head()

        teacher_model_dict["dino_head"] = dino_head()

        # ---------------------------------------------------------------------
        # iBOT
        # ---------------------------------------------------------------------

        if max(cfg.ibot.mask_ratio_min_max) <= 0:
            raise ValueError("ibot.mask_ratio_min_max must have " "a positive maximum.")

        if float(cfg.ibot.mask_sample_probability) <= 0:
            raise ValueError("ibot.mask_sample_probability must be positive.")

        self.ibot_loss_weight = float(cfg.ibot.loss_weight)

        self.ibot_out_dim = (
            int(cfg.ibot.head_n_prototypes)
            if self.ibot_separate_head
            else int(cfg.dino.head_n_prototypes)
        )

        self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)

        if self.ibot_separate_head:
            ibot_head = partial(
                DINOHead,
                in_dim=self.embed_dim,
                out_dim=int(cfg.ibot.head_n_prototypes),
                hidden_dim=int(cfg.ibot.head_hidden_dim),
                bottleneck_dim=int(cfg.ibot.head_bottleneck_dim),
                nlayers=int(cfg.ibot.head_nlayers),
            )

            student_model_dict["ibot_head"] = ibot_head()

            teacher_model_dict["ibot_head"] = ibot_head()

        self.student = nn.ModuleDict(student_model_dict)

        self.teacher = nn.ModuleDict(teacher_model_dict)

        # Student backbone and heads are trainable.
        # Teacher is EMA-only.
        for parameter in self.student.parameters():
            parameter.requires_grad = True

        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

        _validate_gradient_policy(
            self.student,
            self.teacher,
        )

        log_trainable_parameters(
            self.student.backbone,
            "Student backbone",
        )

        log_trainable_parameters(
            self.student.dino_head,
            "Student DINO head",
        )

        if self.ibot_separate_head:
            log_trainable_parameters(
                self.student.ibot_head,
                "Student iBOT head",
            )

        logger.info(
            "Anisotropic SSL model ready | "
            "public pos_embed=%s | "
            "global grid=%s (%d tokens) | "
            "local grid=%s (%d tokens)",
            PUBLIC_POS_EMBED_SHAPE,
            self.global_patch_grid,
            self.global_patch_tokens,
            self.local_patch_grid,
            self.local_patch_tokens,
        )

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(
        self,
        loss,
    ):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def _validate_first_inputs(
        self,
        global_crops,
        local_crops,
        masks,
        mask_indices_list,
        masks_weight,
        upperbound,
        n_local_crops,
    ) -> None:
        """
        Validate the first collated anisotropic batch.
        """
        expected_global = (
            1,
            *self.global_crop_shape,
        )

        expected_local = (
            1,
            *self.local_crop_shape,
        )

        if global_crops.ndim != 5 or tuple(global_crops.shape[1:]) != expected_global:
            raise RuntimeError(
                f"Expected global crops [2b,{expected_global}], "
                f"got {tuple(global_crops.shape)}."
            )

        if local_crops.ndim != 5 or tuple(local_crops.shape[1:]) != expected_local:
            raise RuntimeError(
                f"Expected local crops [8b,{expected_local}], "
                f"got {tuple(local_crops.shape)}."
            )

        if global_crops.shape[0] % 2 != 0:
            raise RuntimeError("Global crop rows must be divisible by two.")

        subject_batch_size = global_crops.shape[0] // 2

        expected_local_rows = subject_batch_size * n_local_crops

        if local_crops.shape[0] != expected_local_rows:
            raise RuntimeError(
                "Local crop row count mismatch: "
                f"expected {expected_local_rows}, "
                f"got {local_crops.shape[0]}."
            )

        expected_masks = (
            global_crops.shape[0],
            self.global_patch_tokens,
        )

        if tuple(masks.shape) != expected_masks:
            raise RuntimeError(
                f"Expected masks {expected_masks}, " f"got {tuple(masks.shape)}."
            )

        if masks.dtype != torch.bool:
            raise RuntimeError(f"Masks must be bool, got {masks.dtype}.")

        n_masked = int(mask_indices_list.numel())

        if masks_weight.numel() != n_masked:
            raise RuntimeError(
                "masks_weight and mask_indices_list lengths differ: "
                f"{masks_weight.numel()} versus {n_masked}."
            )

        if int(upperbound) < n_masked:
            raise RuntimeError(
                f"upperbound={upperbound} is smaller than " f"n_masked={n_masked}."
            )

        if n_masked > 0 and int(mask_indices_list.max()) >= masks.numel():
            raise RuntimeError("mask_indices_list contains an out-of-range index.")

        logger.info(
            "First anisotropic inputs validated | "
            "global=%s | local=%s | masks=%s | n_masked=%d",
            tuple(global_crops.shape),
            tuple(local_crops.shape),
            tuple(masks.shape),
            n_masked,
        )

    def _validate_backbone_output(
        self,
        output,
        name,
        expected_rows,
        expected_patch_tokens,
    ) -> None:
        """
        Validate CLS, patch-token and full-sequence shapes.
        """
        expected = {
            "x_norm_clstoken": (
                expected_rows,
                self.embed_dim,
            ),
            "x_norm_patchtokens": (
                expected_rows,
                expected_patch_tokens,
                self.embed_dim,
            ),
            "x_prenorm": (
                expected_rows,
                expected_patch_tokens + 1,
                self.embed_dim,
            ),
        }

        for key, expected_shape in expected.items():
            if key not in output:
                raise RuntimeError(f"{name} output is missing {key!r}.")

            actual_shape = tuple(output[key].shape)

            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"{name} {key} shape mismatch: "
                    f"expected {expected_shape}, "
                    f"got {actual_shape}."
                )

    def forward_backward(
        self,
        images,
        teacher_temp,
    ):
        """
        Execute teacher forward, student forward, SSL losses and backward.
        """
        n_global_crops = 2

        n_local_crops = int(self.cfg.crops.local_crops_number)

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)

        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        if hasattr(
            global_crops,
            "as_tensor",
        ):
            global_crops = global_crops.as_tensor()

        if hasattr(
            local_crops,
            "as_tensor",
        ):
            local_crops = local_crops.as_tensor()

        masks = images["collated_masks"].cuda(non_blocking=True)

        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)

        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)

        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_masked_patches = int(mask_indices_list.shape[0])

        upperbound = int(images["upperbound"])

        if not self._first_forward_validated:
            self._validate_first_inputs(
                global_crops,
                local_crops,
                masks,
                mask_indices_list,
                masks_weight,
                upperbound,
                n_local_crops,
            )

        n_local_loss_terms = max(
            n_local_crops * n_global_crops,
            1,
        )

        n_global_loss_terms = (n_global_crops - 1) * n_global_crops

        do_ibot = self.do_ibot and n_masked_patches > 0

        ibot_loss_scale = 1.0 / n_global_crops

        teacher_output_for_validation = None

        @torch.no_grad()
        def get_teacher_output():
            nonlocal teacher_output_for_validation

            teacher_output = self.teacher.backbone(
                global_crops,
                is_training=True,
            )

            teacher_output_for_validation = teacher_output

            teacher_cls_tokens = teacher_output["x_norm_clstoken"]

            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops)

            teacher_cls_tokens = torch.cat(
                (
                    teacher_cls_tokens[1],
                    teacher_cls_tokens[0],
                )
            )

            teacher_patch_tokens = teacher_output["x_norm_patchtokens"]

            token_dim = teacher_patch_tokens.shape[-1]

            n_cls_tokens = teacher_cls_tokens.shape[0]

            masked_teacher_after_head = None
            masked_teacher_targets = None

            if do_ibot and not self.ibot_separate_head:
                buffer = teacher_patch_tokens.new_zeros(
                    upperbound + n_cls_tokens,
                    token_dim,
                )

                buffer[:n_cls_tokens].copy_(teacher_cls_tokens)

                torch.index_select(
                    teacher_patch_tokens.flatten(
                        0,
                        1,
                    ),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )

                tokens_after_head = self.teacher.dino_head(buffer)

                teacher_cls_after_head = tokens_after_head[:n_cls_tokens]

                masked_teacher_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]

            elif do_ibot and self.ibot_separate_head:
                buffer = teacher_patch_tokens.new_zeros(
                    upperbound,
                    token_dim,
                )

                torch.index_select(
                    teacher_patch_tokens.flatten(
                        0,
                        1,
                    ),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer[:n_masked_patches],
                )

                teacher_cls_after_head = self.teacher.dino_head(teacher_cls_tokens)

                masked_teacher_after_head = self.teacher.ibot_head(buffer)[
                    :n_masked_patches
                ]

            else:
                teacher_cls_after_head = self.teacher.dino_head(teacher_cls_tokens)

            if self.cfg.train.centering == "centering":
                teacher_dino_targets = self.dino_loss.softmax_center_teacher(
                    teacher_cls_after_head,
                    teacher_temp=teacher_temp,
                ).view(
                    n_global_crops,
                    -1,
                    *teacher_cls_after_head.shape[1:],
                )

                self.dino_loss.update_center(teacher_cls_after_head)

                if do_ibot:
                    masked_teacher_for_centering = masked_teacher_after_head.unsqueeze(
                        0
                    )

                    masked_teacher_targets = (
                        self.ibot_patch_loss.softmax_center_teacher(
                            masked_teacher_for_centering[
                                :,
                                :n_masked_patches,
                            ],
                            teacher_temp=teacher_temp,
                        ).squeeze(0)
                    )

                    self.ibot_patch_loss.update_center(
                        masked_teacher_for_centering[
                            :,
                            :n_masked_patches,
                        ]
                    )

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_targets = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_after_head,
                    teacher_temp=teacher_temp,
                ).view(
                    n_global_crops,
                    -1,
                    *teacher_cls_after_head.shape[1:],
                )

                if do_ibot:
                    masked_teacher_targets = (
                        self.ibot_patch_loss.sinkhorn_knopp_teacher(
                            masked_teacher_after_head,
                            teacher_temp=teacher_temp,
                            n_masked_patches_tensor=(n_masked_patches_tensor),
                        )
                    )

            else:
                raise NotImplementedError(
                    "Unsupported centering mode: " f"{self.cfg.train.centering!r}."
                )

            return (
                teacher_dino_targets,
                masked_teacher_targets,
            )

        (
            teacher_dino_targets,
            masked_teacher_targets,
        ) = get_teacher_output()

        reshard_fsdp_model(self.teacher)

        (
            student_global_output,
            student_local_output,
        ) = self.student.backbone(
            [
                global_crops,
                local_crops,
            ],
            masks=[
                masks,
                None,
            ],
            is_training=True,
        )

        if not self._first_forward_validated:
            self._validate_backbone_output(
                teacher_output_for_validation,
                "Teacher global",
                global_crops.shape[0],
                self.global_patch_tokens,
            )

            self._validate_backbone_output(
                student_global_output,
                "Student global",
                global_crops.shape[0],
                self.global_patch_tokens,
            )

            self._validate_backbone_output(
                student_local_output,
                "Student local",
                local_crops.shape[0],
                self.local_patch_tokens,
            )

            logger.info(
                "First token shapes validated | "
                "global=%d patches + CLS | "
                "local=%d patches + CLS",
                self.global_patch_tokens,
                self.local_patch_tokens,
            )

            self._first_forward_validated = True

        student_local_cls = student_local_output["x_norm_clstoken"]

        student_global_cls = student_global_output["x_norm_clstoken"]

        head_inputs = [
            student_local_cls.unsqueeze(0),
            student_global_cls.unsqueeze(0),
        ]

        student_masked_after_head = None

        if do_ibot:
            student_patch_tokens = student_global_output["x_norm_patchtokens"]

            token_dim = student_patch_tokens.shape[-1]

            buffer = student_patch_tokens.new_zeros(
                upperbound,
                token_dim,
            )

            buffer[:n_masked_patches].copy_(
                torch.index_select(
                    student_patch_tokens.flatten(
                        0,
                        1,
                    ),
                    dim=0,
                    index=mask_indices_list,
                )
            )

            if self.ibot_separate_head:
                student_masked_after_head = self.student.ibot_head(buffer)[
                    :n_masked_patches
                ]
            else:
                head_inputs.append(buffer.unsqueeze(0))

        (
            attention_bias,
            concatenated_inputs,
        ) = fmha.BlockDiagonalMask.from_tensor_list(head_inputs)

        outputs = attention_bias.split(self.student.dino_head(concatenated_inputs))

        student_local_after_head = outputs.pop(0).squeeze(0)

        student_global_after_head = outputs.pop(0).squeeze(0)

        if do_ibot and not self.ibot_separate_head:
            student_masked_after_head = outputs.pop(0).squeeze(0)[:n_masked_patches]

        if outputs:
            raise RuntimeError(
                "Unexpected extra DINO-head outputs: " f"{len(outputs)}."
            )

        loss_dict = {}
        loss_accumulator = 0.0

        dino_local_loss = self.dino_loss(
            student_output_list=(student_local_after_head.chunk(n_local_crops)),
            teacher_out_softmaxed_centered_list=(teacher_dino_targets),
        ) / (n_global_loss_terms + n_local_loss_terms)

        loss_dict["dino_local_crops_loss"] = dino_local_loss

        loss_accumulator += self.dino_loss_weight * dino_local_loss

        loss_scales = 2

        dino_global_loss = (
            self.dino_loss(
                student_output_list=[student_global_after_head],
                teacher_out_softmaxed_centered_list=[
                    teacher_dino_targets.flatten(
                        0,
                        1,
                    )
                ],
            )
            * loss_scales
            / (n_global_loss_terms + n_local_loss_terms)
        )

        loss_dict["dino_global_crops_loss"] = dino_global_loss

        loss_accumulator += self.dino_loss_weight * dino_global_loss

        if self.do_koleo:
            koleo_loss = float(self.cfg.dino.koleo_loss_weight) * sum(
                self.koleo_loss(chunk)
                for chunk in student_global_cls.chunk(n_global_crops)
            )

            loss_dict["koleo_loss"] = koleo_loss / loss_scales

            loss_accumulator += koleo_loss

        if do_ibot:
            ibot_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_masked_after_head,
                    masked_teacher_targets,
                    student_masks_flat=masks,
                    n_masked_patches=(n_masked_patches),
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            loss_dict["ibot_loss"] = ibot_loss / 2

            loss_accumulator += self.ibot_loss_weight * ibot_loss

        if not isinstance(
            loss_accumulator,
            torch.Tensor,
        ):
            raise RuntimeError("No differentiable SSL loss was accumulated.")

        if not bool(torch.isfinite(loss_accumulator).item()):
            raise FloatingPointError("Anisotropic SSL loss is NaN or infinite.")

        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(
        self,
    ):
        """
        Synchronize FSDP CUDA streams once after the first backward.
        """
        if not self.need_to_synchronize_fsdp_streams:
            return

        torch.cuda.synchronize()

        if hasattr(
            self.student.backbone,
            "_streams",
        ):
            streams = self.student.backbone._streams

            self.teacher.backbone._streams = streams

            self.student.dino_head._streams = streams

            self.teacher.dino_head._streams = streams

            if self.ibot_separate_head:
                self.student.ibot_head._streams = streams

                self.teacher.ibot_head._streams = streams

            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(
        self,
        momentum,
    ):
        """
        Update every teacher submodule using EMA of student parameters.
        """
        student_params = []
        teacher_params = []

        with torch.no_grad():
            for key in self.student.keys():
                student_modules = get_fsdp_modules(self.student[key])

                teacher_modules = get_fsdp_modules(self.teacher[key])

                if len(student_modules) != len(teacher_modules):
                    raise RuntimeError(
                        "FSDP module-count mismatch for "
                        f"{key!r}: "
                        f"{len(student_modules)} versus "
                        f"{len(teacher_modules)}."
                    )

                for (
                    student_module,
                    teacher_module,
                ) in zip(
                    student_modules,
                    teacher_modules,
                ):
                    student_params += student_module.params

                    teacher_params += teacher_module.params

            if len(student_params) != len(teacher_params):
                raise RuntimeError("Student/teacher EMA parameter-count mismatch.")

            torch._foreach_mul_(
                teacher_params,
                momentum,
            )

            torch._foreach_add_(
                teacher_params,
                student_params,
                alpha=1 - momentum,
            )

    def train(
        self,
        mode=True,
    ):
        """
        Put the student in training mode and keep the teacher in eval mode.
        """
        super().train(mode)

        self.teacher.eval()

        return self

    def get_maybe_fused_params_for_submodel(
        self,
        module,
    ):
        logger.info("Building layer-wise-decayed parameter groups.")
        decayed_groups = get_params_groups_with_decay(
            model=module,
            lr_decay_rate=(self.cfg.optim.layerwise_decay),
            patch_embed_lr_mult=(self.cfg.optim.patch_embed_lr_mult),
        )
        fused = fuse_params_groups(decayed_groups)
        # Enable the fused-optimizer fast path on every resulting group.
        return [{**group, "foreach": True} for group in fused]

    def get_params_groups(
        self,
    ):
        """
        Return optimizer groups for all student submodules.
        """
        all_params_groups = []

        for module in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(module)

        return all_params_groups

    def prepare_for_distributed_training(
        self,
    ):
        """
        Initialize teacher heads from the student and apply the repository's
        existing FSDP wrappers.

        The distributed world size remains one.
        """
        if int(distributed.get_global_size()) != 1:
            raise ValueError("Anisotropic training supports one GPU only.")

        if has_batchnorms(self.student):
            raise NotImplementedError("BatchNorm modules are not supported.")

        logger.info("FSDP -- preparing single-GPU anisotropic model")

        for key in self.student.keys():
            self.teacher[key].load_state_dict(
                self.student[key].state_dict(),
                strict=True,
            )

            student_cfg = self.cfg.compute_precision.student[key]

            self.student[key] = get_fsdp_wrapper(
                student_cfg,
                modules_to_wrap={
                    BlockChunk,
                },
            )(self.student[key])

            teacher_cfg = self.cfg.compute_precision.teacher[key]

            self.teacher[key] = get_fsdp_wrapper(
                teacher_cfg,
                modules_to_wrap={
                    BlockChunk,
                },
            )(self.teacher[key])

        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

        self.teacher.eval()

        logger.info("FSDP preparation complete")
