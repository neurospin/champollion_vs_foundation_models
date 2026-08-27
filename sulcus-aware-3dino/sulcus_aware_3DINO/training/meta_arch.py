# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)
#
# Portions of this file are derived from DINOv2 (Meta AI Research),
# licensed under the Apache License 2.0.

"""Sulcus-aware SSL meta-architecture.

A thin subclass of 3DINO's ``SSLMetaArch``. The DINO/iBOT heads, losses and the
entire training math are inherited unchanged from the frozen upstream; this
subclass adds only the sulcal-pipeline extensions:

- weight loading tailored to sulcal SSL (teacher weights loaded into BOTH student
  and teacher, with 3D positional-encoding interpolation);
- PEFT injection (LoRA / LoRA+last-block / additional blocks / full fine-tune),
  applied in-place after the upstream architecture is built;
- a LoRA-aware optimizer parameter grouping (via training.param_groups);
- an iBOT guard for crops that contain no masked (active) patch.

No upstream training body is copied. ``__init__`` blanks the two weight paths
(and, defensively, ``peft.enable``) so the frozen upstream ``__init__`` builds the
architecture *without* loading, then this subclass owns weight loading and PEFT
(in-place surgery on the same modules, so the pre-/post-ModuleDict ordering is
equivalent). ``forward_backward``
delegates to ``super()`` after toggling ``do_ibot``. The MONAI MetaTensor ->
plain-tensor conversion lives in the collate, so upstream ``forward_backward``
receives plain tensors unchanged.
"""

import copy
import logging

import torch

from dinov2.train.ssl_meta_arch import SSLMetaArch, interpolate_pos_encoding
from sulcus_aware_3DINO.models.peft import (
    apply_lora_to_vit_qkv,
    create_additional_blocks,
    freeze_for_additional_blocks,
    log_trainable_parameters,
    verify_peft_policy,
    verify_pretrained_loaded,
)
from sulcus_aware_3DINO.training.param_groups import (
    fuse_params_groups,
    get_params_groups_with_decay,
)

logger = logging.getLogger("dinov2")


class SulcusAwareSSLMetaArch(SSLMetaArch):
    def __init__(self, cfg):
        # Build the upstream architecture (backbones, heads, losses, ModuleDict)
        # WITHOUT loading any weights and WITHOUT PEFT: this subclass owns weight
        # loading + PEFT below. LoRA injection and load_state_dict are in-place
        # module surgery, so applying them after the ModuleDict is built is
        # equivalent to the historical pre-ModuleDict ordering.
        #
        # The frozen upstream SSLMetaArch (3DINO/dinov2) has no PEFT path, so
        # blanking peft.enable is inert there; we blank it defensively so the
        # subclass stays single-injection even if ``dinov2`` were ever resolved to
        # a PEFT-carrying build. The two weight paths are blanked so the upstream
        # __init__ builds the architecture without loading W0 / the full SSL ckpt.
        cfg_noload = copy.deepcopy(cfg)
        cfg_noload.student.pretrained_weights = ""
        cfg_noload.student.full_pretrained_weights = ""
        if cfg_noload.get("peft", {}).get("enable", False):
            cfg_noload.peft.enable = False
        super().__init__(cfg_noload)
        self.cfg = cfg

        if cfg.student.pretrained_weights:
            self._load_pretrained_w0(cfg)
        if cfg.get("peft", {}).get("enable", False):
            self._apply_peft(cfg)
        if cfg.student.full_pretrained_weights:
            self._load_full_ssl_checkpoint(cfg)

        # Re-freeze the teacher after any in-place PEFT surgery (super() already
        # froze it once, and already logged that both networks are built).
        for p in self.teacher.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------ #
    # Weight loading + PEFT (own extensions)
    # ------------------------------------------------------------------ #
    def _load_pretrained_w0(self, cfg):
        """Load W0 (a previous teacher backbone) into both student and teacher."""
        student_backbone = self.student["backbone"]
        teacher_backbone = self.teacher["backbone"]

        chkpt = torch.load(cfg.student.pretrained_weights, map_location="cpu")
        logger.info(
            f"Loading pretrained W0 backbone (teacher weights) from "
            f"{cfg.student.pretrained_weights}"
        )

        state_dict = chkpt["teacher"]
        interpolate_pos_encoding(
            state_dict, cfg.crops.global_crops_size, cfg.student.patch_size
        )
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        missing, unexpected = student_backbone.load_state_dict(state_dict, strict=False)
        logger.info(f"[Pretrained] Student missing keys  : {missing[:5]}")
        logger.info(f"[Pretrained] Student unexpected keys: {unexpected[:5]}")

        missing_t, unexpected_t = teacher_backbone.load_state_dict(
            state_dict, strict=False
        )
        logger.info(f"[Pretrained] Teacher missing keys  : {missing_t[:5]}")
        logger.info(f"[Pretrained] Teacher unexpected keys: {unexpected_t[:5]}")

    def _apply_peft(self, cfg):
        """Inject the configured PEFT method in-place and set the freeze policy."""
        student_backbone = self.student["backbone"]
        teacher_backbone = self.teacher["backbone"]
        method = cfg.peft.get("method", "lora")

        if method == "lora":
            # V0: ensure W0 is loaded before injection.
            verify_pretrained_loaded(student_backbone)
            verify_pretrained_loaded(teacher_backbone)

            n_student = apply_lora_to_vit_qkv(
                student_backbone,
                r=cfg.peft.r,
                alpha=cfg.peft.alpha,
                dropout=cfg.peft.dropout,
            )
            n_teacher = apply_lora_to_vit_qkv(
                teacher_backbone,
                r=cfg.peft.r,
                alpha=cfg.peft.alpha,
                dropout=cfg.peft.dropout,
            )
            logger.info(
                f"[LoRA] Student: {n_student} blocks patched, Teacher: {n_teacher} blocks patched"
            )

            # Freeze everything except lora_A_* / lora_B_*.
            for name, p in student_backbone.named_parameters():
                p.requires_grad = "lora_" in name
            for p in teacher_backbone.parameters():
                p.requires_grad = False

            logger.info("[LoRA] Trainable scope: LoRA parameters only.")
            log_trainable_parameters(student_backbone, "LoRA student backbone")

            verify_peft_policy(student_backbone, teacher_backbone, method_name="lora")

        elif method == "lora_last_block":
            # V0: ensure W0 is loaded before injection.
            verify_pretrained_loaded(student_backbone)
            verify_pretrained_loaded(teacher_backbone)

            n_student = apply_lora_to_vit_qkv(
                student_backbone,
                r=cfg.peft.r,
                alpha=cfg.peft.alpha,
                dropout=cfg.peft.dropout,
            )
            n_teacher = apply_lora_to_vit_qkv(
                teacher_backbone,
                r=cfg.peft.r,
                alpha=cfg.peft.alpha,
                dropout=cfg.peft.dropout,
            )
            logger.info(
                f"[LoRA+LastBlock] Student: {n_student} blocks patched, Teacher: {n_teacher} blocks patched"
            )

            # Freeze all, then unfreeze LoRA + the last real transformer block.
            for p in student_backbone.parameters():
                p.requires_grad = False
            for name, p in student_backbone.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True

            last_real_block = None
            for chunk in reversed(student_backbone.blocks):
                for blk in reversed(chunk):
                    if hasattr(blk, "attn"):
                        last_real_block = blk
                        break
                if last_real_block is not None:
                    break
            assert (
                last_real_block is not None
            ), "[LoRA+LastBlock] No real transformer block found."
            for p in last_real_block.parameters():
                p.requires_grad = True

            for p in teacher_backbone.parameters():
                p.requires_grad = False

            logger.info(
                "[LoRA+LastBlock] Trainable scope: LoRA parameters + last transformer block."
            )
            log_trainable_parameters(
                student_backbone, "LoRA+LastBlock student backbone"
            )

            # V1 skipped: the last transformer block is intentionally unfrozen,
            # so its qkv.linear.weight has requires_grad=True (expected).
            # V2, V3, V4 remain active.
            verify_peft_policy(
                student_backbone,
                teacher_backbone,
                method_name="lora_last_block",
                skip_v1=True,
            )

        elif method == "additional_blocks":
            create_additional_blocks(student_backbone, n_blocks=cfg.peft.n_blocks)
            create_additional_blocks(teacher_backbone, n_blocks=cfg.peft.n_blocks)
            logger.info(
                f"[AdditionalBlocks] {cfg.peft.n_blocks} blocks added to student and teacher"
            )

            freeze_for_additional_blocks(student_backbone)
            for p in teacher_backbone.parameters():
                p.requires_grad = False

            log_trainable_parameters(
                student_backbone, "AdditionalBlocks student backbone"
            )

            # V1 not applicable (no LoRA); V2-V4 active.
            verify_peft_policy(
                student_backbone, teacher_backbone, method_name="additional_blocks"
            )

        elif method == "full_finetune":
            for p in student_backbone.parameters():
                p.requires_grad = True
            for p in teacher_backbone.parameters():
                p.requires_grad = False

            logger.info("[FullFinetune] Trainable scope: full student backbone.")
            log_trainable_parameters(student_backbone, "FullFinetune student backbone")

            # V1 not applicable (no LoRA); V2-V4 active.
            verify_peft_policy(
                student_backbone, teacher_backbone, method_name="full_finetune"
            )

        else:
            raise ValueError(f"Unsupported PEFT method: {method}")

    def _load_full_ssl_checkpoint(self, cfg):
        """Optionally resume a full SSL checkpoint (backbone + DINO/iBOT heads).

        IMPORTANT: set ``full_pretrained_weights: ""`` in configs for new runs.
        Loading pretrained MRI heads into a sulcal domain is harmful — the heads
        encode MRI prototype geometry incompatible with sulcal representations.
        """
        chkpt = torch.load(cfg.student.full_pretrained_weights, map_location="cpu")
        logger.info(
            f"Resuming full SSL checkpoint (backbone + heads) from "
            f"{cfg.student.full_pretrained_weights}"
        )
        state_dict = chkpt["teacher"]
        interpolate_pos_encoding(
            state_dict, cfg.crops.global_crops_size, cfg.student.patch_size
        )
        missing, unexpected = self.student.load_state_dict(state_dict, strict=False)

        # LoRA params and additional-block params do not exist in the checkpoint,
        # so they are expected to be missing.
        n_blocks_extra = getattr(
            self.student["backbone"], "_using_additional_blocks", False
        )
        non_peft_missing = [
            k
            for k in missing
            if "lora_" not in k
            and "qkv.linear" not in k
            and (not n_blocks_extra or not k.startswith("backbone.blocks.4."))
        ]
        if non_peft_missing:
            raise RuntimeError(
                f"[full_pretrained] Unexpected missing non-PEFT keys: {non_peft_missing}\n"
                f"This may indicate a checkpoint/architecture mismatch."
            )

        logger.info(
            f"[full_pretrained] loaded — {len(missing)} missing (PEFT keys, expected), "
            f"{len(unexpected)} unexpected"
        )

    # ------------------------------------------------------------------ #
    # Overrides that plug the extensions into the upstream training loop
    # ------------------------------------------------------------------ #
    def forward_backward(self, images, teacher_temp):
        """Disable iBOT for a step with no masked patch, else delegate to upstream.

        Reproduces the fork's ``do_ibot = self.do_ibot and n_masked_patches > 0``:
        a sulcal crop can have zero active patches, in which case iBOT must not
        build reconstruction targets. Toggling ``self.do_ibot`` for the call makes
        the frozen upstream ``forward_backward`` take exactly the non-iBOT path.
        """
        if self.do_ibot and images["mask_indices_list"].shape[0] == 0:
            saved = self.do_ibot
            self.do_ibot = False
            try:
                return super().forward_backward(images, teacher_temp)
            finally:
                self.do_ibot = saved
        return super().forward_backward(images, teacher_temp)

    def get_maybe_fused_params_for_submodel(self, m):
        """Same fused param grouping as upstream, but with LoRA-aware multipliers.

        The upstream method resolves ``get_params_groups_with_decay`` from the
        upstream module namespace; overriding here binds the LoRA-aware version
        (from ``training.param_groups``), which must run BEFORE fuse_params_groups.
        """
        logger.info("Building LoRA-aware, layer-wise-decayed parameter groups.")
        decayed_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused = fuse_params_groups(decayed_groups)
        # Enable the fused-optimizer fast path on every resulting group.
        return [{**group, "foreach": True} for group in fused]

    def fsdp_synchronize_streams(self):
        """Share FSDP CUDA streams across submodels, guarded for the PEFT case.

        The ``hasattr(..., "_streams")`` guard avoids an AttributeError when the
        backbone has not been FSDP-wrapped with a ``_streams`` attribute.
        """
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            if hasattr(self.student.backbone, "_streams"):
                self.student.dino_head._streams = self.teacher.dino_head._streams = (
                    self.student.backbone._streams
                ) = self.teacher.backbone._streams
                self.need_to_synchronize_fsdp_streams = False
