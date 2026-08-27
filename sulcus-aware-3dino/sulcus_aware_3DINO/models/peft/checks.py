# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

"""PEFT policy verification and trainable-parameter logging helpers."""

import logging

from torch import nn

from .lora import LoRAQKVLinear

logger = logging.getLogger("dinov2")


def log_trainable_parameters(module: nn.Module, prefix: str) -> None:
    """Log trainable/frozen parameter counts for sanity checking PEFT policies."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    total = trainable + frozen
    logger.info(f"[{prefix}] trainable parameters: {trainable:,}")
    logger.info(f"[{prefix}] frozen parameters   : {frozen:,}")
    logger.info(f"[{prefix}] total parameters    : {total:,}")
    logger.info(
        f"[{prefix}] trainable ratio      : {100.0 * trainable / max(total, 1):.4f}%"
    )


def verify_peft_policy(
    student_backbone: nn.Module,
    teacher_backbone: nn.Module,
    method_name: str,
    skip_v1: bool = False,
) -> None:
    """
    Verify the PEFT freeze policy after parameter setup.

    Checks performed:
      V1 — W0 is frozen in the student backbone:
           All attn.qkv.linear.weight must have requires_grad=False.
           After LoRA injection, W0 lives under qkv.linear.weight.
           If it is trainable, the entire pretrained backbone is being
           fine-tuned unintentionally (catastrophic forgetting risk).
           Skipped when skip_v1=True, e.g. for lora_last_block where
           W0 of the last block is intentionally unfrozen.

      V2 — Teacher is fully frozen:
           Every parameter in teacher_backbone must have requires_grad=False.
           The teacher is updated only via EMA — gradients must never flow.

      V3 — Student/teacher architecture symmetry:
           Both backbones must expose the same set of parameter keys.
           Asymmetry (e.g. additional_blocks added to one only) would
           make EMA update incoherent (keys mismatch in update_teacher()).

      V4 — Student has at least one trainable parameter:
           If no parameter is trainable after PEFT setup, the training loop
           runs silently without learning anything.

    Args:
        student_backbone: ViT student after PEFT injection and freeze.
        teacher_backbone: ViT teacher after PEFT injection and freeze.
        method_name:      PEFT method string, used only for log messages.
        skip_v1:          If True, skip the W0-frozen check (V1). Use for
                          methods that intentionally unfreeze W0 in some
                          blocks, e.g. lora_last_block.

    Raises:
        AssertionError on any violation.
    """
    errors = []

    # V1: W0 must stay frozen in the student.
    if skip_v1:
        logger.info(
            f"[verify_peft_policy:{method_name}] V1 SKIPPED — "
            f"W0 intentionally unfrozen in selected blocks (skip_v1=True)."
        )
    else:
        w0_trainable = []
        for name, module in student_backbone.named_modules():
            if isinstance(module, LoRAQKVLinear):
                w = module.linear.weight
                if w.requires_grad:
                    w0_trainable.append(f"{name}.linear.weight")

        if w0_trainable:
            errors.append(
                f"[V1] W0 IS TRAINABLE in student backbone — catastrophic forgetting risk!\n"
                f"     Trainable W0 params: {w0_trainable[:5]}"
            )
        else:
            logger.info(
                f"[verify_peft_policy:{method_name}] V1 OK — W0 frozen in all LoRAQKVLinear layers."
            )

    # V2: teacher must be fully frozen.
    teacher_trainable = [
        name for name, p in teacher_backbone.named_parameters() if p.requires_grad
    ]
    if teacher_trainable:
        errors.append(
            f"[V2] TEACHER HAS TRAINABLE PARAMETERS — EMA will not be the only update!\n"
            f"     Trainable teacher params (first 5): {teacher_trainable[:5]}"
        )
    else:
        logger.info(f"[verify_peft_policy:{method_name}] V2 OK — teacher fully frozen.")

    # V3: student/teacher architecture symmetry.
    student_keys = set(student_backbone.state_dict().keys())
    teacher_keys = set(teacher_backbone.state_dict().keys())
    only_in_student = student_keys - teacher_keys
    only_in_teacher = teacher_keys - student_keys

    if only_in_student or only_in_teacher:
        errors.append(
            f"[V3] STUDENT/TEACHER ARCHITECTURE MISMATCH — EMA update_teacher() will fail!\n"
            f"     Keys only in student (first 5): {list(only_in_student)[:5]}\n"
            f"     Keys only in teacher (first 5): {list(only_in_teacher)[:5]}"
        )
    else:
        logger.info(
            f"[verify_peft_policy:{method_name}] V3 OK — student/teacher architectures are symmetric "
            f"({len(student_keys)} keys each)."
        )

    # V4: the student must have at least one trainable parameter.
    student_trainable_count = sum(
        p.numel() for p in student_backbone.parameters() if p.requires_grad
    )
    if student_trainable_count == 0:
        errors.append(
            "[V4] NO TRAINABLE PARAMETERS in student backbone — training will silently do nothing!"
        )
    else:
        logger.info(
            f"[verify_peft_policy:{method_name}] V4 OK — "
            f"{student_trainable_count:,} trainable parameters in student backbone."
        )

    if errors:
        error_msg = (
            f"\n{'='*60}\n"
            f"[verify_peft_policy] POLICY ERRORS DETECTED for method='{method_name}':\n"
            + "\n".join(f"  {e}" for e in errors)
            + f"\n{'='*60}"
        )
        raise AssertionError(error_msg)

    logger.info(
        f"[verify_peft_policy:{method_name}] All checks passed (V1-V4). "
        f"PEFT policy is correct."
    )
