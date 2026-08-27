# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

"""Model-side extensions: PEFT adapters and the anisotropic model builder."""

from .peft import (
    LoRAQKVLinear,
    apply_lora_to_vit_qkv,
    create_additional_blocks,
    freeze_for_additional_blocks,
    verify_pretrained_loaded,
)

__all__ = [
    "LoRAQKVLinear",
    "apply_lora_to_vit_qkv",
    "create_additional_blocks",
    "freeze_for_additional_blocks",
    "verify_pretrained_loaded",
]
