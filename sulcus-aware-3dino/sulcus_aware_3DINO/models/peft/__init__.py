# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

"""Parameter-efficient fine-tuning (PEFT) adapters and policy checks."""

from .additional_blocks import create_additional_blocks, freeze_for_additional_blocks
from .checks import log_trainable_parameters, verify_peft_policy
from .lora import LoRAQKVLinear, apply_lora_to_vit_qkv, verify_pretrained_loaded

__all__ = [
    "LoRAQKVLinear",
    "apply_lora_to_vit_qkv",
    "verify_pretrained_loaded",
    "create_additional_blocks",
    "freeze_for_additional_blocks",
    "log_trainable_parameters",
    "verify_peft_policy",
]
