# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("dinov2")


class LoRAQKVLinear(nn.Module):
    """
    LoRA wrapper for a fused qkv projection: nn.Linear(dim, 3 * dim).

    The original pretrained layer W0 is kept intact and frozen (externally).
    Three independent LoRA branches are added for Q, K, and V separately:

        base_out  = W0(x)                             # [..., 3 * dim]
        delta_q   = scaling * B_q(A_q(x_drop))       # [..., dim]
        delta_k   = scaling * B_k(A_k(x_drop))       # [..., dim]
        delta_v   = scaling * B_v(A_v(x_drop))       # [..., dim]
        output    = base_out + cat(delta_q, delta_k, delta_v)

    where x_drop = lora_dropout(x) is computed once and shared across all
    three branches to preserve coherence between Q, K, V on the same token.

    Parameter names (visible in logs and param_groups):
        attn.qkv.lora_A_q, attn.qkv.lora_B_q
        attn.qkv.lora_A_k, attn.qkv.lora_B_k
        attn.qkv.lora_A_v, attn.qkv.lora_B_v

    The filter "lora_" in name in param_groups.py captures all six without
    any modification.
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(linear, nn.Linear):
            raise TypeError(
                f"LoRAQKVLinear expects nn.Linear, got {type(linear).__name__}. "
                "Make sure pretrained weights are loaded before LoRA injection."
            )
        if r <= 0:
            raise ValueError(f"LoRA rank r must be > 0, got {r}")

        out_features = linear.out_features
        if out_features % 3 != 0:
            raise ValueError(
                f"LoRAQKVLinear expects out_features divisible by 3 (fused qkv), "
                f"got {out_features}. This layer may not be a qkv projection."
            )

        self.linear = linear  # W0 encapsulated by reference, not copied
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # separates amplitude (alpha) from rank (r)

        in_features = linear.in_features
        self.in_features = in_features
        self.out_features = out_features
        self.qkv_dim = out_features // 3  # dim of each Q, K, V projection = embed_dim

        # ── Q branch ──────────────────────────────────────────────────────────
        self.lora_A_q = nn.Parameter(torch.empty(r, in_features))
        self.lora_B_q = nn.Parameter(torch.zeros(self.qkv_dim, r))

        # ── K branch ──────────────────────────────────────────────────────────
        self.lora_A_k = nn.Parameter(torch.empty(r, in_features))
        self.lora_B_k = nn.Parameter(torch.zeros(self.qkv_dim, r))

        # ── V branch ──────────────────────────────────────────────────────────
        self.lora_A_v = nn.Parameter(torch.empty(r, in_features))
        self.lora_B_v = nn.Parameter(torch.zeros(self.qkv_dim, r))

        # Initialization:
        #   A_* ~ N(0, 0.02) — breaks symmetry, enables gradient flow
        #   B_* = 0          — zero-impact at step 0 (W0 output unchanged)
        nn.init.normal_(self.lora_A_q, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_A_k, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_A_v, mean=0.0, std=0.02)
        # lora_B_* already zero via torch.zeros — no explicit init needed

        # Single dropout applied on x before all three branches.
        # A shared mask preserves Q/K/V coherence on the same input token.
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Pretrained path (W0, frozen externally) ───────────────────────────
        base_out = self.linear(x)  # [..., 3 * qkv_dim]

        # ── LoRA path ─────────────────────────────────────────────────────────
        # Apply dropout once — shared mask across Q, K, V branches.
        x_drop = self.lora_dropout(x)

        # Each branch: x_drop → low-rank projection (r) → full-rank projection (qkv_dim)
        # F.linear(input, weight) computes input @ weight.T
        delta_q = F.linear(
            F.linear(x_drop, self.lora_A_q), self.lora_B_q
        )  # [..., qkv_dim]
        delta_k = F.linear(
            F.linear(x_drop, self.lora_A_k), self.lora_B_k
        )  # [..., qkv_dim]
        delta_v = F.linear(
            F.linear(x_drop, self.lora_A_v), self.lora_B_v
        )  # [..., qkv_dim]

        # Concatenate along last dim to match the fused qkv layout
        delta_qkv = torch.cat([delta_q, delta_k, delta_v], dim=-1)  # [..., 3 * qkv_dim]

        return base_out + self.scaling * delta_qkv

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"qkv_dim={self.qkv_dim}, r={self.r}, "
            f"alpha={self.alpha}, scaling={self.scaling:.4f}"
        )


def verify_pretrained_loaded(model: nn.Module) -> None:
    """
    Assert that pretrained weights W0 are loaded into the model before LoRA injection.

    Checks the first real qkv Linear in the model:
      - The key must exist (architecture is built correctly)
      - The weight norm must be non-trivial (weights are not random initialization)

    This must be called BEFORE apply_lora_to_vit_qkv(). After injection, the
    key 'attn.qkv.weight' no longer exists — it becomes 'attn.qkv.linear.weight'.

    Raises:
        AssertionError if no qkv Linear is found or if weights look uninitialized.
    """
    target_param = None
    target_name = None

    for name, module in model.named_modules():
        if hasattr(module, "attn") and hasattr(module.attn, "qkv"):
            qkv = module.attn.qkv
            if isinstance(qkv, nn.Linear):
                # Found the first real qkv projection before injection
                target_param = qkv.weight
                target_name = name + ".attn.qkv.weight"
                break

    assert target_param is not None, (
        "[LoRA] verify_pretrained_loaded: no nn.Linear qkv found in model. "
        "Either the model has no attention blocks or LoRA was already injected."
    )

    mean_abs = target_param.detach().abs().mean().item()
    assert mean_abs > 1e-4, (
        f"[LoRA] verify_pretrained_loaded: '{target_name}' has suspiciously low "
        f"mean abs weight ({mean_abs:.6f}). Pretrained weights may not be loaded. "
        "Ensure load_state_dict() is called before apply_lora_to_vit_qkv()."
    )

    logger.info(
        f"[LoRA] verify_pretrained_loaded: OK — '{target_name}' "
        f"mean_abs={mean_abs:.6f} (pretrained weights confirmed)"
    )


def apply_lora_to_vit_qkv(
    model: nn.Module,
    r: int,
    alpha: float,
    dropout: float = 0.0,
    expected_replaced: int = 24,
) -> int:
    """
    Replace attn.qkv with LoRAQKVLinear in all transformer blocks of the ViT.

    Uses model.modules() to traverse recursively, which handles BlockChunk
    nesting (block_chunks=4 in config) without special-case logic.

    IMPORTANT: Call verify_pretrained_loaded(model) before this function to
    confirm that W0 is already loaded. After injection, the key 'attn.qkv.weight'
    becomes 'attn.qkv.linear.weight' and cannot be loaded from the pretrained
    checkpoint anymore.

    Guards:
      - Skips nn.Identity modules (should not happen in practice)
      - Skips already-injected LoRAQKVLinear (prevents double injection)
      - Skips non-Linear qkv (defensive, logs a warning)

    Args:
        model:            ViT backbone (student or teacher).
        r:                LoRA rank.
        alpha:            LoRA scaling factor (scaling = alpha / r).
        dropout:          Dropout probability on input x before LoRA branches.
        expected_replaced: Expected number of replaced qkv projections.
                           For ViT-L with 24 blocks: 24.
                           Triggers a warning if the count differs.

    Returns:
        replaced: Number of qkv projections actually replaced.
    """
    replaced = 0
    skipped_already_injected = 0
    skipped_other = 0

    for module in model.modules():
        if not hasattr(module, "attn"):
            continue
        if not hasattr(module.attn, "qkv"):
            continue

        qkv = module.attn.qkv

        if isinstance(qkv, LoRAQKVLinear):
            # Already injected — skip silently but count for diagnostics
            skipped_already_injected += 1
            continue

        if isinstance(qkv, nn.Identity):
            # Defensive: Identity used as placeholder, should not occur in practice
            skipped_other += 1
            continue

        if not isinstance(qkv, nn.Linear):
            logger.warning(
                f"[LoRA] apply_lora_to_vit_qkv: unexpected qkv type {type(qkv).__name__}, skipping."
            )
            skipped_other += 1
            continue

        module.attn.qkv = LoRAQKVLinear(qkv, r=r, alpha=alpha, dropout=dropout)
        replaced += 1

    # Diagnostic summary
    logger.info(
        f"[LoRA] apply_lora_to_vit_qkv: replaced={replaced}, "
        f"already_injected={skipped_already_injected}, other_skipped={skipped_other}"
    )
    logger.info(
        f"[LoRA] LoRAQKVLinear config: r={r}, alpha={alpha}, scaling={alpha/r:.4f}"
    )

    if replaced != expected_replaced:
        logger.warning(
            f"[LoRA] WARNING: expected {expected_replaced} replacements, got {replaced}. "
            f"Check model architecture (ViT-L should have 24 blocks). "
            f"If using additional_blocks, set expected_replaced accordingly."
        )
    else:
        logger.info(f"[LoRA] All {replaced} qkv projections replaced as expected.")

    return replaced


__all__ = [
    "LoRAQKVLinear",
    "verify_pretrained_loaded",
    "apply_lora_to_vit_qkv",
]
