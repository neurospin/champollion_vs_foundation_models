#!/usr/bin/env python3
"""
models/sam3d/extract_features.py

Zero-shot feature extraction for SAM-Med3D (vit_b_ori) on MRI crops (OFC, FIP, SC).

Architecture — from build_sam3D_vit_b_ori + image_encoder3D.py:
  img_size=128, patch_size=16, embed_dim=768 (ViT), out_chans=384 (neck)
  depth=12, num_heads=12
  N_patches = (128 // 16)^3 = 8^3 = 512

Preprocessing (native — from train.py):
  Input : window128 NIfTI crop, float32 ∈ [0,1] (clip q99 applied at masking stage)
  tio.ZNormalization(masking_method=lambda x: x > 0):
    mean, std computed on non-zero voxels only
    x = (x - mean) / std  applied to the entire volume
  This matches exactly what SAM-Med3D saw during training.
  No resize needed — window128 is already at the correct 128³ input size.

Feature dims:
  mean_pool              → [N, 384]    GAP on neck output [B,384,8,8,8]
  mean_pool_multi_layers → [N, 1152]   cat(neck_mean[384], vit_mean[768])
  flatten                → [N, 196608] neck output flattened [384×512]

Checkpoint format (sam_med3d_turbo.pth):
  {"model_state_dict": {weights}}
  Full Sam3D loaded, then only image_encoder extracted.

mean_pool_multi_layers decomposed forward (verified from image_encoder3D.py):
  patch_embed → (+pos_embed) → blocks[0..11] → vit_tokens
  → permute → neck → neck_out
  vit_mean  = mean(vit_tokens.flatten(1,3), dim=1)  [B,768]
  neck_mean = mean(neck_out, dims=(2,3,4))           [B,384]
  output    = cat([neck_mean, vit_mean], dim=-1)     [B,1152]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# =============================================================================
# Architecture constants
# =============================================================================

_IMG_SIZE = 128
_PATCH_SIZE = 16
_EMBED_DIM = 768  # ViT embedding dim
_NECK_DIM = 384  # neck output channels
_N_PATCHES = (_IMG_SIZE // _PATCH_SIZE) ** 3  # 512

FEATURE_DIM: Dict[str, int] = {
    "mean_pool": _NECK_DIM,  # 384
    "mean_pool_multi_layers": _NECK_DIM + _EMBED_DIM,  # 1152
    "flatten": _NECK_DIM * _N_PATCHES,  # 196608
}

SUPPORTED_MODES = list(FEATURE_DIM.keys())


# =============================================================================
# Repository path
# =============================================================================


def _add_repo_to_path(repo_path: str | Path) -> None:
    """Add SAM-Med3D repo root to sys.path so 'segment_anything.*' imports work."""
    p = str(repo_path)
    if p not in sys.path:
        sys.path.insert(0, p)


# =============================================================================
# Model builder
# =============================================================================


def _build_model(
    checkpoint_path: str | Path, repo_path: str | Path, device: str
) -> nn.Module:
    """
    Instantiate and freeze SAM-Med3D image encoder from checkpoint.

    Loads full Sam3D from checkpoint["model_state_dict"], extracts image_encoder only.
    prompt_encoder and mask_decoder are discarded after loading.

    Returns: frozen ImageEncoderViT3D on `device`, eval mode.
    """
    _add_repo_to_path(repo_path)
    from segment_anything.build_sam3D import sam_model_registry3D

    print(f"[SAM3D] Loading checkpoint: {checkpoint_path}")
    chkpt = torch.load(str(checkpoint_path), map_location="cpu")

    if "model_state_dict" not in chkpt:
        raise ValueError(
            f"Checkpoint missing 'model_state_dict' key. Found: {list(chkpt.keys())}"
        )

    model = sam_model_registry3D["vit_b_ori"]()
    model.load_state_dict(chkpt["model_state_dict"])

    encoder = model.image_encoder
    del model

    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[SAM3D] Image encoder ready — {n_params:,} parameters (frozen)")
    return encoder


# =============================================================================
# Preprocessing — native (from train.py: tio.ZNormalization masking non-zero)
# =============================================================================


def _preprocess_volume(vol_128: np.ndarray) -> np.ndarray:
    """
    Preprocess one MRI crop for SAM-Med3D input.

    Args:
        vol_128: [128, 128, 128] float32 ∈ [0,1]  (NIfTI crop, clip q99)

    Returns:
        [128, 128, 128] float32  (z-normalized)

    Native SAM-Med3D preprocessing (train.py):
      tio.ZNormalization(masking_method=lambda x: x > 0)
      → mean, std computed on non-zero voxels only
      → (x - mean) / std applied to the full volume

    No resize needed: window128 is already at 128³.
    """
    nonzero = vol_128[vol_128 > 0]

    if len(nonzero) == 0:
        return np.zeros_like(vol_128)

    mean = float(nonzero.mean())
    std = float(nonzero.std())

    if std < 1e-8:
        return np.zeros_like(vol_128)

    return ((vol_128 - mean) / std).astype(np.float32)


def _load_and_preprocess(
    crop_dir: Path,
    roi_dirname: str,
    subject_id: str,
) -> np.ndarray:
    """
    Load NIfTI window128 crop and apply native SAM-Med3D preprocessing.

    Returns: [128, 128, 128] float32 (z-normalized)
    """
    fname = f"nobias_{subject_id}_MNI09c_1mm__{roi_dirname}__crop_window128.nii.gz"
    fpath = crop_dir / roi_dirname / "window128" / fname

    if not fpath.exists():
        raise FileNotFoundError(f"NIfTI crop not found: {fpath}")

    vol_128 = nib.load(str(fpath)).get_fdata(dtype=np.float32)

    if vol_128.shape != (128, 128, 128):
        raise ValueError(
            f"Expected (128,128,128) for {subject_id}, got {vol_128.shape}"
        )

    return _preprocess_volume(vol_128)


# =============================================================================
# Forward pass — feature aggregation
# =============================================================================


@torch.no_grad()
def _forward(encoder: nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Forward pass + feature aggregation.

    Args:
        x    : [B, 1, 128, 128, 128] float32 (z-normalized) on device
        mode : "mean_pool" | "mean_pool_multi_layers" | "flatten"

    mean_pool:
        encoder(x) → [B,384,8,8,8] → mean(dims=(2,3,4)) → [B,384]

    mean_pool_multi_layers:
        Decomposed forward (verified against image_encoder3D.py source):
          patch_embed → (+pos_embed) → blocks → vit_tokens[B,8,8,8,768]
          → permute → neck → neck_out[B,384,8,8,8]
        vit_mean  = mean(flatten(vit_tokens,1,3), dim=1)   [B,768]
        neck_mean = mean(neck_out, dims=(2,3,4))           [B,384]
        output    = cat([neck_mean, vit_mean], dim=-1)     [B,1152]

    flatten:
        encoder(x) → [B,384,8,8,8] → flatten(1) → [B,196608]
    """
    if mode == "mean_pool":
        out = encoder(x)  # [B,384,8,8,8]
        return out.mean(dim=(2, 3, 4))  # [B,384]

    if mode == "flatten":
        out = encoder(x)  # [B,384,8,8,8]
        return out.flatten(1)  # [B,196608]

    if mode == "mean_pool_multi_layers":
        # Step 1: patch embedding
        h = encoder.patch_embed(x)  # [B,8,8,8,768]

        # Step 2: positional embedding
        if encoder.pos_embed is not None:
            h = h + encoder.pos_embed

        # Step 3: transformer blocks
        for blk in encoder.blocks:
            h = blk(h)
        # h: [B,8,8,8,768]

        # Step 4: ViT-space mean (before neck)
        vit_tokens = h.flatten(1, 3)  # [B,512,768]
        vit_mean = vit_tokens.mean(dim=1)  # [B,768]

        # Step 5: neck
        neck_out = encoder.neck(h.permute(0, 4, 1, 2, 3))  # [B,384,8,8,8]
        neck_mean = neck_out.mean(dim=(2, 3, 4))  # [B,384]

        return torch.cat([neck_mean, vit_mean], dim=-1)  # [B,1152]

    raise ValueError(f"Unknown mode '{mode}'. Supported: {SUPPORTED_MODES}")


# =============================================================================
# Main extraction function
# =============================================================================


@torch.no_grad()
def extract_features(
    checkpoint_path: str | Path,
    repo_path: str | Path,
    master_table_path: str | Path,
    crop_dir: str | Path,
    roi_dirname: str,
    mode: str,
    device: str = "cuda",
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Extract SAM-Med3D features for all subjects in master_table.

    Reads NIfTI crops from:
      crop_dir / roi_dirname / window128 / nobias_{SID}_MNI09c_1mm__{ROI}__crop_window128.nii.gz

    Preprocessing per subject (native SAM-Med3D, from train.py):
      tio.ZNormalization(masking_method=lambda x: x > 0)
      → z-normalize using mean/std of non-zero voxels only

    Args:
        checkpoint_path   : path to sam_med3d_turbo.pth
        repo_path         : path to SAM-Med3D repo root (for segment_anything imports)
        master_table_path : master_table_{roi}.csv produced by prepare_hcp_irm_data.py
        crop_dir          : root of crop_mni09c_1mm/  (parent of OFC/, FIP/, Central/)
        roi_dirname       : "OFC" | "FIP" | "Central"
        mode              : "mean_pool" | "mean_pool_multi_layers" | "flatten"
        device            : "cuda" | "cpu"
        batch_size        : subjects per GPU forward pass

    Returns dict with keys:
        features       : [N, D]  float32
        subjects       : [N]     str
        labels         : [N]     int64  (classification) or [N, 6] float32 (regression)
        folds          : [N]     int64
        splits         : [N]     str
        volume_indices : [N]     int64
    """
    if mode not in FEATURE_DIM:
        raise ValueError(f"Unknown mode '{mode}'. Supported: {SUPPORTED_MODES}")

    crop_dir = Path(crop_dir)
    master_table_path = Path(master_table_path)

    # ------------------------------------------------------------------
    # Load master table — defines subject order (must be respected)
    # ------------------------------------------------------------------
    table = pd.read_csv(str(master_table_path), dtype={"subject": str})
    table = table.sort_values("volume_index").reset_index(drop=True)
    N = len(table)
    print(f"[SAM3D] {N} subjects | mode={mode} | roi={roi_dirname} | device={device}")

    is_regression = any(c.startswith("label_") for c in table.columns)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    encoder = _build_model(checkpoint_path, repo_path, device)

    # ------------------------------------------------------------------
    # Pre-allocation
    # ------------------------------------------------------------------
    D = FEATURE_DIM[mode]
    features_arr = np.empty((N, D), dtype=np.float32)

    subjects_list: List[str] = []
    folds_list: List[int] = []
    splits_list: List[str] = []
    vidx_list: List[int] = []

    if is_regression:
        label_cols = sorted([c for c in table.columns if c.startswith("label_")])
        labels_arr = np.empty((N, len(label_cols)), dtype=np.float32)
    else:
        labels_arr = np.empty(N, dtype=np.int64)

    # ------------------------------------------------------------------
    # Batch loop
    # ------------------------------------------------------------------
    offset = 0
    rows = list(table.itertuples(index=False))

    for batch_start in tqdm(
        range(0, N, batch_size),
        desc=f"[SAM3D] {mode} / {roi_dirname}",
        leave=True,
    ):
        batch_rows = rows[batch_start : batch_start + batch_size]
        B = len(batch_rows)

        # Load + preprocess on CPU
        batch_vols = np.stack(
            [
                _load_and_preprocess(crop_dir, roi_dirname, row.subject)
                for row in batch_rows
            ],
            axis=0,
        )  # [B, 128, 128, 128]

        # → tensor [B, 1, 128, 128, 128] on device
        x = torch.from_numpy(batch_vols[:, None]).to(device, non_blocking=True)

        feat = _forward(encoder, x, mode)  # [B, D]
        features_arr[offset : offset + B] = feat.cpu().numpy()

        for local_i, row in enumerate(batch_rows):
            subjects_list.append(str(row.subject))
            folds_list.append(int(row.fold))
            splits_list.append(str(row.split))
            vidx_list.append(int(row.volume_index))

            idx = offset + local_i
            if is_regression:
                labels_arr[idx] = [float(getattr(row, c)) for c in label_cols]
            else:
                labels_arr[idx] = int(row.label)

        offset += B

    assert offset == N, f"[SAM3D] Processed {offset} subjects, expected {N}"

    del encoder
    torch.cuda.empty_cache()

    return {
        "features": features_arr,
        "subjects": np.asarray(subjects_list),
        "labels": labels_arr,
        "folds": np.asarray(folds_list, dtype=np.int64),
        "splits": np.asarray(splits_list),
        "volume_indices": np.asarray(vidx_list, dtype=np.int64),
    }
