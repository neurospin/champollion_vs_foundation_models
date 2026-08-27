# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)
#
# Portions of this file are derived from DINOv2 (Meta AI Research),
# licensed under the Apache License 2.0.

"""LoRA-aware optimizer parameter groups.

Thin extension over 3DINO's ``get_params_groups_with_decay``: it delegates the
whole layer-wise-decay grouping to the frozen upstream implementation, then
overrides the multipliers for LoRA parameters. ``fuse_params_groups`` is the
unmodified upstream helper, re-exported for callers.
"""

import logging

from dinov2.utils.param_groups import fuse_params_groups
from dinov2.utils.param_groups import (
    get_params_groups_with_decay as _upstream_get_params_groups_with_decay,
)

logger = logging.getLogger("dinov2")

__all__ = ["get_params_groups_with_decay", "fuse_params_groups"]


def get_params_groups_with_decay(*args, **kwargs):
    """Upstream 3DINO param groups, with a LoRA-specific override.

    Delegates layer-wise-decay grouping to upstream, then, for every parameter
    whose name contains ``lora_``, forces:

    - ``wd_multiplier = 0.0``
        With weight_decay annealed from 0.04 to 0.4, applying WD to lora_A /
        lora_B would regularize them toward zero and undo the adaptation. LoRA
        parameters must remain free to grow without penalty.
    - ``lr_multiplier = 1.0``
        layerwise_decay=0.9 over 24 blocks gives lr x 0.9^24 ~= 0.08 at block 0,
        so LoRA params in early blocks would receive a near-zero LR. A flat
        multiplier gives every LoRA param the same base LR regardless of depth.

    Captures ``lora_A_{q,k,v}`` / ``lora_B_{q,k,v}`` (and legacy ``lora_A`` /
    ``lora_B`` if present). The override is applied to the returned groups, so it
    is the last word on the multipliers, exactly as in the fused implementation.
    """
    all_param_groups = _upstream_get_params_groups_with_decay(*args, **kwargs)

    for d in all_param_groups:
        if "lora_" in d["name"]:
            d["wd_multiplier"] = 0.0
            d["lr_multiplier"] = 1.0

    # Post-construction sanity check: no LoRA parameter should slip through with
    # wrong multipliers. Catches future refactors that might break the override.
    lora_wd_violations = [
        d["name"]
        for d in all_param_groups
        if "lora_" in d["name"] and d["wd_multiplier"] > 0.0
    ]
    lora_lr_violations = [
        d["name"]
        for d in all_param_groups
        if "lora_" in d["name"] and abs(d["lr_multiplier"] - 1.0) > 1e-6
    ]
    if lora_wd_violations:
        logger.warning(
            "[param_groups] WARNING: LoRA params with wd_multiplier > 0 detected "
            f"(first 5): {lora_wd_violations[:5]}\n"
            "LoRA parameters should never have weight decay applied."
        )
    if lora_lr_violations:
        logger.warning(
            "[param_groups] WARNING: LoRA params with lr_multiplier != 1.0 detected "
            f"(first 5): {lora_lr_violations[:5]}\n"
            "LoRA parameters should use base LR (lr_multiplier=1.0) to avoid "
            "near-zero LR in early blocks due to layerwise decay."
        )
    if not lora_wd_violations and not lora_lr_violations:
        lora_count = sum(1 for d in all_param_groups if "lora_" in d["name"])
        if lora_count > 0:
            logger.info(
                f"[param_groups] OK -- {lora_count} LoRA param groups: "
                "wd_multiplier=0.0, lr_multiplier=1.0 confirmed."
            )

    return all_param_groups
