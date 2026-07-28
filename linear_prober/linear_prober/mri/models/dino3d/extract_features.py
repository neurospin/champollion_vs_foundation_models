#!/usr/bin/env python3
"""
models/dino3d/extract_features.py

Zero-shot feature extraction for 3DINO-ViT (ViT-Large, high-resolution)
on MRI crops (OFC, FIP, SC).

Architecture — hardcoded from vit3d_highres.yaml:
  img_size=112, patch_size=16, embed_dim=1024, depth=24, num_heads=16
  N_patches = (112 // 16)^3 = 7^3 = 343

Preprocessing (native — from notebooks/basic_model_use.ipynb):
  Input : window128 NIfTI crop, float32 ∈ [0,1] (clip q99 applied at masking stage)
  Step 1: Center-crop 128 → 112  (margin=8, pure numpy slicing, no interpolation)
  Step 2: Percentile normalization per subject:
            min_val = percentile(0.05%)
            max_val = percentile(99.95%)
            x = clip((x - min_val) / (max_val - min_val) * 2 - 1, -1, 1)
  Step 3: [112,112,112] → tensor [1, 1, 112, 112, 112] → GPU

Feature dims:
  mean_pool              → [N, 2048]    cat(CLS[1024], mean(patches)[1024])
  mean_pool_multi_layers → [N, 8192]    4 × cat(CLS, mean(patches)) from last 4 layers
  flatten                → [N, 352256]  cat(CLS[1024], flatten(patches)[343×1024])

Checkpoint format:
  {"teacher": {"backbone.*": weights, "dino_head.*": ..., "ibot_head.*": ...}}
  Strip "module." → strip "backbone." → load into backbone (strict=False).
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

_IMG_SIZE = 112
_PATCH_SIZE = 16
_EMBED_DIM = 1024
_N_PATCHES = (_IMG_SIZE // _PATCH_SIZE) ** 3  # 343
_N_MULTI_LAYERS = 4
_CROP_MARGIN = (128 - _IMG_SIZE) // 2  # 8

FEATURE_DIM: Dict[str, int] = {
    "mean_pool": 2 * _EMBED_DIM,  # 2048
    "mean_pool_multi_layers": _N_MULTI_LAYERS * 2 * _EMBED_DIM,  # 8192
    "flatten": (_N_PATCHES + 1) * _EMBED_DIM,  # 352256
}

SUPPORTED_MODES = list(FEATURE_DIM.keys())


# =============================================================================
# Repository path
# =============================================================================


def _add_repo_to_path(repo_path: str | Path) -> None:
    """Add 3DINO repo root to sys.path so 'dinov2.*' imports work."""
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
    Instantiate and freeze 3DINO-ViT backbone from a teacher checkpoint.

    Checkpoint format:
      {"teacher": {"backbone.*": ..., "dino_head.*": ..., "ibot_head.*": ...}}

    Returns: frozen DinoVisionTransformer3d on `device`, eval mode.
    """
    _add_repo_to_path(repo_path)
    from dinov2.models.vision_transformer import vit_large_3d

    backbone = vit_large_3d(
        img_size=_IMG_SIZE,
        patch_size=_PATCH_SIZE,
        init_values=1e-5,
        ffn_layer="mlp",
        block_chunks=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )

    print(f"[3DINO] Loading checkpoint: {checkpoint_path}")
    chkpt = torch.load(str(checkpoint_path), map_location="cpu")

    if "teacher" not in chkpt:
        raise ValueError(
            f"Checkpoint missing 'teacher' key. Found: {list(chkpt.keys())}"
        )

    state_dict = chkpt["teacher"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)

    _expected_extra = ("dino_head", "ibot_head")
    real_unexpected = [
        k for k in unexpected if not any(k.startswith(p) for p in _expected_extra)
    ]
    if real_unexpected:
        print(f"[3DINO] WARNING — unexpected keys: {real_unexpected[:5]}")
    if missing:
        print(f"[3DINO] WARNING — missing keys: {missing[:5]}")

    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"[3DINO] Backbone ready — {n_params:,} parameters (frozen)")
    return backbone


# =============================================================================
# Preprocessing — native (from notebooks/basic_model_use.ipynb)
# =============================================================================


def _preprocess_volume(vol_128: np.ndarray) -> np.ndarray:
    """
    Preprocess one MRI crop for 3DINO input.

    Args:
        vol_128: [128, 128, 128] float32 ∈ [0,1]  (NIfTI crop, clip q99)

    Returns:
        vol_112: [112, 112, 112] float32 ∈ [-1, 1]

    Steps:
      1. Center-crop 128 → 112  (margin=8, pure slicing, no interpolation)
      2. Percentile normalization per subject (native 3DINO preprocessing):
           min_val = percentile(0.05%)
           max_val = percentile(99.95%)
           x = clip((x - min_val) / (max_val - min_val) * 2 - 1, -1, 1)
    """
    # Step 1 — center-crop (no interpolation, ROI stays intact)
    m = _CROP_MARGIN  # 8
    vol_112 = vol_128[m : m + _IMG_SIZE, m : m + _IMG_SIZE, m : m + _IMG_SIZE].copy()

    # Step 2 — per-subject percentile normalization (native 3DINO)
    min_val = np.percentile(vol_112, 0.05)
    max_val = np.percentile(vol_112, 99.95)

    if max_val - min_val < 1e-6:
        # Degenerate volume (nearly constant) — return zeros
        return np.zeros_like(vol_112)

    vol_112 = (vol_112 - min_val) / (max_val - min_val)
    vol_112 = np.clip(vol_112 * 2.0 - 1.0, -1.0, 1.0)

    return vol_112


def _load_and_preprocess(
    crop_dir: Path,
    roi_dirname: str,
    subject_id: str,
) -> np.ndarray:
    """
    Load NIfTI window128 crop and apply native 3DINO preprocessing.

    Returns: [112, 112, 112] float32 ∈ [-1, 1]
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
def _forward(model: nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Forward pass + feature aggregation.

    Args:
        x    : [B, 1, 112, 112, 112] float32 ∈ [-1, 1]  on device
        mode : "mean_pool" | "mean_pool_multi_layers" | "flatten"

    mean_pool:
        forward_features → CLS[B,1024] + mean(patches[B,343,1024]) → [B, 2048]

    mean_pool_multi_layers:
        get_intermediate_layers(n=4, return_class_token=True)
        → 4 × (patches[B,N,1024], cls[B,1024])
        → per layer: cat(cls, mean(patches)) = [B, 2048]
        → concat: [B, 8192]

    flatten:
        forward_features → CLS[B,1024] + flatten(patches[B,343,1024]) → [B, 352256]
    """
    if mode in ("mean_pool", "flatten"):
        out = model.forward_features(x)
        cls = out["x_norm_clstoken"]  # [B, 1024]
        patches = out["x_norm_patchtokens"]  # [B, 343, 1024]

        if mode == "mean_pool":
            return torch.cat([cls, patches.mean(dim=1)], dim=-1)  # [B, 2048]
        else:
            return torch.cat([cls, patches.flatten(1)], dim=-1)  # [B, 352256]

    if mode == "mean_pool_multi_layers":
        intermediate = model.get_intermediate_layers(
            x, n=_N_MULTI_LAYERS, return_class_token=True
        )
        layer_feats = []
        for patches, cls in intermediate:  # (patches, cls) per layer
            layer_feats.append(
                torch.cat([cls, patches.mean(dim=1)], dim=-1)  # [B, 2048]
            )
        return torch.cat(layer_feats, dim=-1)  # [B, 8192]

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
    Extract 3DINO features for all subjects in master_table.

    Reads NIfTI crops from:
      crop_dir / roi_dirname / window128 / nobias_{SID}_MNI09c_1mm__{ROI}__crop_window128.nii.gz

    Preprocessing per subject (native 3DINO):
      1. Center-crop 128 → 112
      2. Percentile normalization → [-1, 1]

    Args:
        checkpoint_path   : path to teacher_checkpoint.pth
        repo_path         : path to 3DINO repo root (for dinov2 imports)
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
    print(f"[3DINO] {N} subjects | mode={mode} | roi={roi_dirname} | device={device}")

    # Detect task type from columns
    is_regression = any(c.startswith("label_") for c in table.columns)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = _build_model(checkpoint_path, repo_path, device)

    # ------------------------------------------------------------------
    # Feature pre-allocation
    # ------------------------------------------------------------------
    D = FEATURE_DIM[mode]
    features_arr = np.empty((N, D), dtype=np.float32)

    # Metadata accumulators
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
    # Batch loop — load NIfTI + preprocess on CPU, forward on GPU
    # ------------------------------------------------------------------
    offset = 0
    rows = list(table.itertuples(index=False))

    for batch_start in tqdm(
        range(0, N, batch_size),
        desc=f"[3DINO] {mode} / {roi_dirname}",
        leave=True,
    ):
        batch_rows = rows[batch_start : batch_start + batch_size]
        B = len(batch_rows)

        # Load + preprocess each subject in the batch (CPU, numpy)
        batch_vols = np.stack(
            [
                _load_and_preprocess(crop_dir, roi_dirname, row.subject)
                for row in batch_rows
            ],
            axis=0,
        )  # [B, 112, 112, 112]

        # → tensor [B, 1, 112, 112, 112] on device
        x = torch.from_numpy(batch_vols[:, None]).to(device, non_blocking=True)

        # Forward
        feat = _forward(model, x, mode)  # [B, D]
        features_arr[offset : offset + B] = feat.cpu().numpy()

        # Metadata
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

    assert offset == N, f"[3DINO] Processed {offset} subjects, expected {N}"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    del model
    torch.cuda.empty_cache()

    return {
        "features": features_arr,
        "subjects": np.asarray(subjects_list),
        "labels": labels_arr,
        "folds": np.asarray(folds_list, dtype=np.int64),
        "splits": np.asarray(splits_list),
        "volume_indices": np.asarray(vidx_list, dtype=np.int64),
    }
