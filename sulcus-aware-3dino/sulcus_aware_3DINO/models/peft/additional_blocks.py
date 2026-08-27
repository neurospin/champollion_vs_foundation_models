# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

from functools import partial

import torch.nn as nn
from torch.nn.init import trunc_normal_

from dinov2.layers import MemEffAttention, Mlp
from dinov2.layers import NestedTensorBlock as Block
from dinov2.models.vision_transformer import BlockChunk


def create_additional_blocks(model: nn.Module, n_blocks: int) -> int:
    """
    Append newly initialized Transformer blocks after the pretrained ViT backbone.

    The additional blocks are designed to match the original backbone blocks:
      - same embed_dim
      - same number of heads
      - same MLP ratio
      - same qkv/proj/ffn bias settings
      - same LayerNorm configuration
      - same activation function
      - no stochastic depth on newly initialized blocks
      - LayerScale initialized from the pretrained backbone LayerScale value

    Important:
      init_values must be a scalar LayerScale value, not the embedding dimension.
      Using ref_block.ls1.gamma.shape[0] would incorrectly set init_values to
      embed_dim, e.g. 1024 for ViT-L, which can strongly destabilize training.

    Args:
        model:
            ViT backbone, either student or teacher.
        n_blocks:
            Number of additional blocks to append.

    Returns:
        Number of blocks effectively added.
    """
    if n_blocks <= 0:
        print("[AdditionalBlocks] No additional block added because n_blocks <= 0")
        return 0

    norm_layer = partial(nn.LayerNorm, eps=1e-6)

    # Find the first real Transformer block.
    # BlockChunk may contain leading Identity modules to preserve block indices.
    ref_block = None
    for chunk in model.blocks:
        for blk in chunk:
            if hasattr(blk, "attn"):
                ref_block = blk
                break
        if ref_block is not None:
            break

    assert ref_block is not None, "No Transformer block found in model.blocks"

    embed_dim = model.embed_dim
    num_heads = model.num_heads
    mlp_ratio = ref_block.mlp.fc1.out_features / embed_dim
    qkv_bias = ref_block.attn.qkv.bias is not None
    proj_bias = ref_block.attn.proj.bias is not None
    ffn_bias = ref_block.mlp.fc1.bias is not None

    # Recover the actual scalar LayerScale value from the pretrained block.
    # This avoids the previous bug where init_values was set to gamma.shape[0],
    # i.e. embed_dim, which could be 1024 for ViT-L.
    if hasattr(ref_block, "ls1") and hasattr(ref_block.ls1, "gamma"):
        gamma = ref_block.ls1.gamma.detach()
        init_values = gamma.mean().item()
        gamma_std = gamma.std().item()
        gamma_min = gamma.min().item()
        gamma_max = gamma.max().item()
    else:
        init_values = None
        gamma_std = None
        gamma_min = None
        gamma_max = None

    print("[AdditionalBlocks] Reference block hyperparameters:")
    print(f"[AdditionalBlocks]   embed_dim  = {embed_dim}")
    print(f"[AdditionalBlocks]   num_heads  = {num_heads}")
    print(f"[AdditionalBlocks]   mlp_ratio  = {mlp_ratio}")
    print(f"[AdditionalBlocks]   qkv_bias   = {qkv_bias}")
    print(f"[AdditionalBlocks]   proj_bias  = {proj_bias}")
    print(f"[AdditionalBlocks]   ffn_bias   = {ffn_bias}")

    if init_values is not None:
        print("[AdditionalBlocks] Reference LayerScale statistics:")
        print(f"[AdditionalBlocks]   gamma mean = {init_values:.8e}")
        print(f"[AdditionalBlocks]   gamma std  = {gamma_std:.8e}")
        print(f"[AdditionalBlocks]   gamma min  = {gamma_min:.8e}")
        print(f"[AdditionalBlocks]   gamma max  = {gamma_max:.8e}")
        print(f"[AdditionalBlocks]   using init_values = {init_values:.8e}")
    else:
        print("[AdditionalBlocks] No LayerScale gamma found in reference block.")
        print(
            "[AdditionalBlocks] Additional blocks will be created without LayerScale."
        )

    new_blocks = [
        Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop_path=0.0,
            norm_layer=norm_layer,
            act_layer=nn.GELU,
            attn_class=MemEffAttention,
            ffn_layer=Mlp,
            init_values=init_values,
        )
        for _ in range(n_blocks)
    ]

    # Standard ViT/timm-style initialization for newly added Linear layers.
    for blk in new_blocks:
        for m in blk.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # The new BlockChunk contains leading Identity modules so that newly added
    # blocks keep their correct global block indices in parameter names.
    # Example with ViT-L/24 blocks + 2 new blocks:
    #   blocks.4.24...
    #   blocks.4.25...
    # It is appended after the original chunks. model.n_blocks is updated so
    # get_intermediate_layers can correctly account for the full depth.

    old_n_blocks = model.n_blocks

    new_chunk = BlockChunk([nn.Identity()] * old_n_blocks + new_blocks)
    model.blocks.append(new_chunk)

    model.n_blocks += n_blocks
    model._using_additional_blocks = True

    print("[AdditionalBlocks] Blocks appended successfully:")
    print(f"[AdditionalBlocks]   added blocks = {n_blocks}")
    print(f"[AdditionalBlocks]   old depth    = {old_n_blocks}")
    print(f"[AdditionalBlocks]   new depth    = {model.n_blocks}")

    return n_blocks


def freeze_for_additional_blocks(student_backbone: nn.Module) -> None:
    """
    Freeze the pretrained backbone and train only the newly appended blocks
    and the final LayerNorm.

    Rationale for unfreezing student_backbone.norm:
      The final LayerNorm is applied on the output of the additional blocks
      before the DINO/iBOT heads. If it remains frozen, it is calibrated for
      pretrained backbone activations and cannot adapt to the distribution
      produced by the newly initialized additional blocks. Even though
      gradients flow through the additional blocks correctly, the representations
      passed to the heads never improve because the norm is mismatched.
      Unfreezing norm together with the additional blocks fixes this.

    Args:
        student_backbone:
            ViT student backbone after create_additional_blocks().
    """
    # Step 1 : freeze everything
    for p in student_backbone.parameters():
        p.requires_grad = False

    # Step 2 : unfreeze the last BlockChunk (contains the additional blocks)
    for p in student_backbone.blocks[-1].parameters():
        p.requires_grad = True

    # Step 3 : unfreeze the final LayerNorm  ← FIX B6
    # Without this, the norm remains calibrated for pretrained MRI activations
    # and cannot adapt to the distribution of the additional blocks' outputs.
    for p in student_backbone.norm.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in student_backbone.parameters() if p.requires_grad)
    frozen = sum(
        p.numel() for p in student_backbone.parameters() if not p.requires_grad
    )

    print("[AdditionalBlocks] Freeze policy applied:")
    print(f"[AdditionalBlocks]   frozen parameters    = {frozen:,}")
    print(f"[AdditionalBlocks]   trainable parameters = {trainable:,}")
    print(
        "[AdditionalBlocks]   trainable scope      = last BlockChunk + final LayerNorm"
    )
