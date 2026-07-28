#!/usr/bin/env python3
"""
models/bsf/extract_features.py

Zero-shot feature extraction for BrainSegFounder (SSLHead SwinViT encoder)
on MRI crops (OFC, FIP, SC).

Architecture (confirmed from inspection + SSL_Head.py source):
  SSLHead → model.swinViT (SwinTransformer from MONAI@a23c7f54)
  in_channels=1, feature_size=48, depths=[2,2,2,2], num_heads=[3,6,12,24]
  patch_size=[2,2,2], window_size=[7,7,7], input 96³

  SwinViT forward returns list of 5 tensors:
    hs[0]: [B,  48, 48, 48, 48]
    hs[1]: [B,  96, 24, 24, 24]
    hs[2]: [B, 192, 12, 12, 12]
    hs[3]: [B, 384,  6,  6,  6]
    hs[4]: [B, 768,  3,  3,  3]  ← deepest

Preprocessing (native — from data_utils.py):
  ScaleIntensityRanged(a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True)
  → affine mapping to [0, 1], no per-subject statistics.
  Our crops are already float32 ∈ [0,1] after clip q99 → passthrough.
  No resize needed: window96 is already at 96³.

Feature dims:
  mean_pool              → [N, 768]   GAP on hs[4]
  mean_pool_multi_layers → [N, 1056]  cat(GAP(hs[1])[96], GAP(hs[2])[192], GAP(hs[4])[768])
  flatten                → [N, 20736] hs[4].flatten()  (768 × 3 × 3 × 3)

Checkpoint format (model_weights_UKB-pretrain.pt):
  state_dict with "module." prefix → strip → load into SSLHead (strict=False)
  Missing keys: 0, Unexpected keys: 0 (verified)
  Only swinViT encoder weights used — SSL heads ignored.

Import: from pretrain.models.ssl_head import SSLHead
  → requires BrainSegFounder repo on sys.path
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# =============================================================================
# Architecture constants — confirmed from inspection
# =============================================================================

FEATURE_DIM: Dict[str, int] = {
    "mean_pool": 768,  # GAP on hs[4]
    "mean_pool_multi_layers": 96 + 192 + 768,  # 1056 — hs[1,2,4] GAP concat
    "flatten": 768 * 3 * 3 * 3,  # 20736
}

SUPPORTED_MODES = list(FEATURE_DIM.keys())


# =============================================================================
# Repository path
# =============================================================================


def _add_repo_to_path(repo_path: str | Path) -> None:
    """Add BrainSegFounder repo root to sys.path for SSLHead import."""
    p = str(repo_path)
    if p not in sys.path:
        sys.path.insert(0, p)
        print(f"[BSF] Added to sys.path: {p}")


# =============================================================================
# Model builder
# =============================================================================


def _build_model(
    checkpoint_path: str | Path, repo_path: str | Path, device: str
) -> nn.Module:
    """
    Load BrainSegFounder SwinViT encoder from checkpoint.

    Instantiates SSLHead with in_channels=1 (single-channel MRI).
    Loads state dict, strips "module." prefix, strict=False.
    Extracts and returns model.swinViT only — SSL heads discarded.

    Returns: frozen SwinViT on `device`, eval mode.
    """
    _add_repo_to_path(repo_path)
    from pretrain.models.ssl_head import SSLHead

    print(f"[BSF] Loading checkpoint: {checkpoint_path}")

    args = SimpleNamespace(
        in_channels=1,
        spatial_dims=3,
        feature_size=48,
        bottleneck_depth=768,
        num_swin_blocks_per_stage=[2, 2, 2, 2],
        num_heads_per_stage=[3, 6, 12, 24],
        dropout_path_rate=0.0,
        use_checkpoint=False,
    )

    model = SSLHead(args)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[BSF] Missing keys   : {len(missing)} — {missing[:4]}")
    if unexpected:
        print(f"[BSF] Unexpected keys: {len(unexpected)} — {unexpected[:4]}")

    encoder = model.swinViT
    del model

    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[BSF] SwinViT encoder ready — {n_params:,} parameters (frozen)")
    return encoder


# =============================================================================
# Preprocessing — native passthrough
# =============================================================================


def _preprocess_volume(vol_96: np.ndarray) -> np.ndarray:
    """
    Preprocess one MRI crop for BSF input.

    Args:
        vol_96: [96, 96, 96] float32 ∈ [0,1]  (NIfTI crop, clip q99)

    Returns:
        [96, 96, 96] float32 ∈ [0,1]  — passthrough

    Native BSF preprocessing (data_utils.py):
      ScaleIntensityRanged(a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True)
      → affine normalization to [0,1], no per-subject statistics.
    Our crops are already in [0,1] after clip q99 → passthrough.
    No resize needed: window96 is already at 96³.
    """
    return vol_96.astype(np.float32)


def _load_and_preprocess(
    crop_dir: Path,
    roi_dirname: str,
    subject_id: str,
) -> np.ndarray:
    """
    Load NIfTI window96 crop and apply native BSF preprocessing.

    Returns: [96, 96, 96] float32 ∈ [0,1]
    """
    fname = f"nobias_{subject_id}_MNI09c_1mm__{roi_dirname}__crop_window96.nii.gz"
    fpath = crop_dir / roi_dirname / "window96" / fname

    if not fpath.exists():
        raise FileNotFoundError(f"NIfTI crop not found: {fpath}")

    vol_96 = nib.load(str(fpath)).get_fdata(dtype=np.float32)

    if vol_96.shape != (96, 96, 96):
        raise ValueError(f"Expected (96,96,96) for {subject_id}, got {vol_96.shape}")

    return _preprocess_volume(vol_96)


# =============================================================================
# Forward pass — feature aggregation
# =============================================================================


@torch.no_grad()
def _forward(encoder: nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Forward pass + feature aggregation.

    Args:
        x    : [B, 1, 96, 96, 96] float32 on device
        mode : "mean_pool" | "mean_pool_multi_layers" | "flatten"

    SwinViT returns list of 5 tensors — no hooks needed.

    mean_pool:
        hs[4] [B,768,3,3,3] → GAP → [B,768]

    mean_pool_multi_layers:
        GAP(hs[1])[B,96] + GAP(hs[2])[B,192] + GAP(hs[4])[B,768]
        → cat → [B,1056]

    flatten:
        hs[4] [B,768,3,3,3] → flatten(1) → [B,20736]
    """
    hs = encoder(x.contiguous())

    if mode == "mean_pool":
        return hs[4].mean(dim=[2, 3, 4])  # [B,768]

    if mode == "mean_pool_multi_layers":
        return torch.cat(
            [
                hs[1].mean(dim=[2, 3, 4]),  # [B, 96]
                hs[2].mean(dim=[2, 3, 4]),  # [B,192]
                hs[4].mean(dim=[2, 3, 4]),  # [B,768]
            ],
            dim=1,
        )  # [B,1056]

    if mode == "flatten":
        return hs[4].flatten(1)  # [B,20736]

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
    batch_size: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Extract BSF features for all subjects in master_table.

    Reads NIfTI crops from:
      crop_dir / roi_dirname / window96 / nobias_{SID}_MNI09c_1mm__{ROI}__crop_window96.nii.gz

    Preprocessing per subject (native BSF, from data_utils.py):
      ScaleIntensityRanged → [0,1]. Our crops already in [0,1] → passthrough.

    Args:
        checkpoint_path   : path to model_weights_UKB-pretrain.pt
        repo_path         : path to BrainSegFounder repo root
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
    # Load master table
    # ------------------------------------------------------------------
    table = pd.read_csv(str(master_table_path), dtype={"subject": str})
    table = table.sort_values("volume_index").reset_index(drop=True)
    N = len(table)
    print(f"[BSF] {N} subjects | mode={mode} | roi={roi_dirname} | device={device}")

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
        desc=f"[BSF] {mode} / {roi_dirname}",
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
        )  # [B, 96, 96, 96]

        # → tensor [B, 1, 96, 96, 96] on device
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

    assert offset == N, f"[BSF] Processed {offset} subjects, expected {N}"

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
