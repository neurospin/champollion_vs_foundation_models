#!/usr/bin/env python3
"""
models/dinov3/extract_features.py

Zero-shot feature extraction for DINOv3 ViT models on MRI crops (OFC, FIP, SC).

Pipeline (one volume at a time):
  1. _preprocess_volume  — center-crop window128 → 112³ (no interpolation)
  2. get_slices          — 3D→2D slicing, tri-axial (slicers.py)
  3. _encode_axis        — normalize_imagenet + DINOv3 forward, per axis
  4. aggregate           — inter-slice aggregation (aggregator.py)

Preprocessing (native DINOv3 IRM):
  Input : window128 NIfTI crop, float32 ∈ [0,1] (clip q99 at masking stage)
  Step 1: Center-crop 128→112 (margin=8, pure numpy slicing, no interpolation)
  Step 2: normalize_imagenet on each 2D slice (mean/std ImageNet)
  No resize, no affine mapping, no standardization.

Mode convention
===============
"{extraction}__{aggregation}__{slicer}__{model_size}"

e.g. "mean_pool__mean_pool_axis__2d__vitb16"
     "flatten__concat_all__25d__vits16"

Architecture constants (112×112 input, patch_size=16)
======================================================
  N_PATCHES  = (112 // 16)² = 7² = 49
  mean_pool  → 2 × hidden_size  per slice
  flatten    → 50 × hidden_size per slice  (CLS + 49 patches)

Model configs (hidden_size, num_register_tokens):
  vits16/vits16plus : 384,  4
  vitb16            : 768,  4
  vitl16            : 1024, 4
  vith16plus        : 1280, 4
  vit7b16           : 4096, 4
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from linear_prober.mri.models.dinov3.normalizer import normalize_imagenet
from linear_prober.mri.models.dinov3.slicers import AXES, N_SLICES_PER_AXIS, get_slices

# =============================================================================
# Architecture constants
# =============================================================================

_CROP_MARGIN: int = (128 - 112) // 2  # = 8

MODEL_CONFIGS: dict[str, dict] = {
    "vits16": {"hidden_size": 384, "num_register_tokens": 4},
    "vits16plus": {"hidden_size": 384, "num_register_tokens": 4},
    "vitb16": {"hidden_size": 768, "num_register_tokens": 4},
    "vitl16": {"hidden_size": 1024, "num_register_tokens": 4},
    "vith16plus": {"hidden_size": 1280, "num_register_tokens": 4},
    "vit7b16": {"hidden_size": 4096, "num_register_tokens": 4},
}

MODEL_SIZES: set[str] = set(MODEL_CONFIGS.keys())

N_PATCHES: int = 49  # (112 // 16)² = 7²

EXTRACTION_MODES: set[str] = {"mean_pool", "flatten"}
AGGREGATION_MODES: set[str] = {"mean_pool_axis", "concat_all"}
SLICER_MODES: set[str] = {"2d", "25d"}


def get_feature_dim(model_size: str, extraction_mode: str) -> int:
    """Embedding dimension per 2D slice."""
    d = MODEL_CONFIGS[model_size]["hidden_size"]
    return 2 * d if extraction_mode == "mean_pool" else (N_PATCHES + 1) * d


def get_latent_dim(
    model_size: str, extraction_mode: str, aggregation_mode: str, slicer_mode: str
) -> int:
    """Final latent dimension after aggregation across 3 axes."""
    feat_dim = get_feature_dim(model_size, extraction_mode)
    n_per_axis = N_SLICES_PER_AXIS[slicer_mode]
    if aggregation_mode == "mean_pool_axis":
        return 3 * feat_dim
    else:
        return n_per_axis * 3 * feat_dim


# =============================================================================
# Mode parsing
# =============================================================================


def parse_mode(mode: str) -> tuple[str, str, str, str]:
    """
    Parse "{extraction}__{aggregation}__{slicer}__{model_size}".
    Returns (extraction_mode, aggregation_mode, slicer_mode, model_size).
    """
    parts = mode.split("__")
    if len(parts) != 4:
        raise ValueError(
            f"mode must have 4 '__'-separated components, got {len(parts)}: '{mode}'"
        )
    extraction_mode, aggregation_mode, slicer_mode, model_size = parts

    for name, val, valid in [
        ("extraction_mode", extraction_mode, EXTRACTION_MODES),
        ("aggregation_mode", aggregation_mode, AGGREGATION_MODES),
        ("slicer_mode", slicer_mode, SLICER_MODES),
        ("model_size", model_size, MODEL_SIZES),
    ]:
        if val not in valid:
            raise ValueError(
                f"Unknown {name} '{val}'. Expected one of: {sorted(valid)}"
            )

    return extraction_mode, aggregation_mode, slicer_mode, model_size


# =============================================================================
# Model builder
# =============================================================================


def _build_model(checkpoint_path: str | Path, device: str) -> nn.Module:
    """
    Load a frozen DINOv3 ViT model from local disk.
    checkpoint_path: directory downloaded via huggingface-cli.
    """
    from transformers import AutoModel

    print(f"[DINOv3] Loading model from: {checkpoint_path}")
    model = AutoModel.from_pretrained(str(checkpoint_path), local_files_only=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[DINOv3] Model loaded — {n_params:,} parameters (frozen)")
    return model


# =============================================================================
# Preprocessing — center-crop 128 → 112
# =============================================================================


def _preprocess_volume(vol_128: np.ndarray) -> np.ndarray:
    """
    Center-crop window128 NIfTI crop to 112³.

    Args:
        vol_128: [128, 128, 128] float32 ∈ [0,1]  (NIfTI crop, clip q99)

    Returns:
        [112, 112, 112] float32 ∈ [0,1]

    Pure numpy slicing — no interpolation, ROI stays centered and intact.
    Identical to 3DINO IRM preprocessing.
    """
    m = _CROP_MARGIN  # 8
    return vol_128[m : m + 112, m : m + 112, m : m + 112].copy()


def _load_and_preprocess(
    crop_dir: Path,
    roi_dirname: str,
    subject_id: str,
) -> np.ndarray:
    """
    Load NIfTI window128 crop and center-crop to 112³.

    Returns: [112, 112, 112] float32 ∈ [0,1]
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
# Forward pass
# =============================================================================


@torch.no_grad()
def _forward_batch(
    model: nn.Module,
    slice_batch: torch.Tensor,
    num_register_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DINOv3 forward pass on a batch of 2D slices.

    Input:  [B, 3, 112, 112] float32, ImageNet-normalized, on device
    Output: cls [B, D], patches [B, 49, D]

    last_hidden_state layout: [CLS | reg_1..reg_K | patch_1..patch_49]
    Register tokens are discarded.
    """
    outputs = model(pixel_values=slice_batch)
    last_hidden = outputs.last_hidden_state

    cls = last_hidden[:, 0, :]
    patches = last_hidden[:, 1 + num_register_tokens :, :]

    return cls, patches


@torch.no_grad()
def _embed_slices(
    cls: torch.Tensor,
    patches: torch.Tensor,
    extraction_mode: str,
) -> torch.Tensor:
    """
    mean_pool : cat(cls, mean(patches)) → [B, 2D]
    flatten   : cat(cls, flatten(patches)) → [B, 50D]
    """
    if extraction_mode == "mean_pool":
        return torch.cat([cls, patches.mean(dim=1)], dim=-1)
    else:
        return torch.cat([cls, patches.flatten(1)], dim=-1)


# =============================================================================
# Axis encoding
# =============================================================================


@torch.no_grad()
def _encode_axis(
    model: nn.Module,
    slices_axis: torch.Tensor,
    extraction_mode: str,
    num_register_tokens: int,
    device: str,
    slice_batch_size: int,
) -> torch.Tensor:
    """
    Encode all slices of one axis through DINOv3, in mini-batches.

    Input:  slices_axis [N_slices, 3, 112, 112]  float32 ∈ [0,1], on CPU
    Output: embeddings  [N_slices, feat_dim]      float32, on CPU

    Per mini-batch:
      1. normalize_imagenet → ImageNet-standardized
      2. .contiguous()      → needed after expand() in 2d mode
      3. .to(device)
      4. DINOv3 forward
      5. .cpu()
    """
    N_slices = slices_axis.shape[0]
    all_embs: list[torch.Tensor] = []

    for start in range(0, N_slices, slice_batch_size):
        batch = slices_axis[start : start + slice_batch_size]

        batch = normalize_imagenet(batch)
        batch = batch.contiguous()
        batch = batch.to(device, non_blocking=True)

        cls, patches = _forward_batch(model, batch, num_register_tokens)
        emb = _embed_slices(cls, patches, extraction_mode)
        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0)  # [N_slices, feat_dim]


# =============================================================================
# Single volume extraction
# =============================================================================


@torch.no_grad()
def _extract_volume(
    model: nn.Module,
    vol_112: np.ndarray,
    slicer_mode: str,
    extraction_mode: str,
    num_register_tokens: int,
    device: str,
    slice_batch_size: int,
) -> dict[str, torch.Tensor]:
    """
    Full single-volume extraction pipeline.

    Input:  vol_112 [112, 112, 112]  float32 ∈ [0,1]
    Output: {"D": [N_slices, feat_dim], "H": ..., "W": ...}
    """
    vol_t = torch.from_numpy(vol_112[None]).float()  # [1, 112, 112, 112]

    slices_dict = get_slices(vol_t, slicer_mode)
    # {"D": [N,3,112,112], "H": [N,3,112,112], "W": [N,3,112,112]}

    embeddings: dict[str, torch.Tensor] = {}
    for axis in AXES:
        embeddings[axis] = _encode_axis(
            model,
            slices_dict[axis],
            extraction_mode,
            num_register_tokens,
            device,
            slice_batch_size,
        )

    return embeddings


# =============================================================================
# Main extraction function
# =============================================================================


@torch.no_grad()
def extract_features(
    checkpoint_path: str | Path,
    repo_path: str | Path | None,  # unused — kept for interface consistency
    master_table_path: str | Path,
    crop_dir: str | Path,
    roi_dirname: str,
    mode: str,
    device: str = "cuda",
    batch_size: int = 8,
    slice_batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """
    Extract DINOv3 features for all subjects in master_table.

    Reads NIfTI window128 crops, center-crops to 112³, slices into 2D,
    encodes with DINOv3, aggregates per subject.

    Args:
        checkpoint_path  : directory of downloaded DINOv3 model (e.g. .../vitb16/)
        repo_path        : unused (kept for interface consistency with other models)
        master_table_path: master_table_{roi}.csv
        crop_dir         : root of crop_mni09c_1mm/
        roi_dirname      : "OFC" | "FIP" | "Central"
        mode             : "{extraction}__{aggregation}__{slicer}__{model_size}"
        device           : "cuda" | "cpu"
        batch_size       : ignored (DINOv3 processes one volume at a time)
        slice_batch_size : 2D slices per DINOv3 forward pass

    Returns dict:
        features, subjects, labels, folds, splits, volume_indices
    """
    from linear_prober.mri.models.dinov3.aggregator import aggregate

    extraction_mode, aggregation_mode, slicer_mode, model_size = parse_mode(mode)
    num_register_tokens = MODEL_CONFIGS[model_size]["num_register_tokens"]

    crop_dir = Path(crop_dir)
    master_table_path = Path(master_table_path)

    # ------------------------------------------------------------------
    # Load master table
    # ------------------------------------------------------------------
    table = pd.read_csv(str(master_table_path), dtype={"subject": str})
    table = table.sort_values("volume_index").reset_index(drop=True)
    N = len(table)

    latent_dim = get_latent_dim(
        model_size, extraction_mode, aggregation_mode, slicer_mode
    )
    print(
        f"[DINOv3] {N} subjects | mode={mode} | roi={roi_dirname}\n"
        f"         feat_per_slice={get_feature_dim(model_size, extraction_mode)}"
        f"  latent_dim={latent_dim}"
    )

    is_regression = any(c.startswith("label_") for c in table.columns)

    # ------------------------------------------------------------------
    # Build model — checkpoint_path is the per-model-size directory
    # ------------------------------------------------------------------
    model = _build_model(checkpoint_path, device)

    # ------------------------------------------------------------------
    # Pre-allocation
    # ------------------------------------------------------------------
    features_arr = np.empty((N, latent_dim), dtype=np.float32)

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
    # Subject loop — one volume at a time
    # ------------------------------------------------------------------
    rows = list(table.itertuples(index=False))

    for idx, row in enumerate(tqdm(rows, desc=f"[DINOv3] {mode[:40]} / {roi_dirname}")):

        vol_112 = _load_and_preprocess(crop_dir, roi_dirname, row.subject)

        emb_dict = _extract_volume(
            model,
            vol_112,
            slicer_mode,
            extraction_mode,
            num_register_tokens,
            device,
            slice_batch_size,
        )

        latent = aggregate(emb_dict, aggregation_mode)  # [latent_dim]
        features_arr[idx] = latent.numpy()

        subjects_list.append(str(row.subject))
        folds_list.append(int(row.fold))
        splits_list.append(str(row.split))
        vidx_list.append(int(row.volume_index))

        if is_regression:
            labels_arr[idx] = [float(getattr(row, c)) for c in label_cols]
        else:
            labels_arr[idx] = int(row.label)

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
