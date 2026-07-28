"""
models/bsf/extract_features.py

Zero-shot feature extraction for BrainSegFounder (SSLHead SwinViT encoder).

ARCHITECTURE (confirmed from inspection)
====================================================
  Backbone : SSLHead → SwinViT encoder (model.swinViT)
             in_channels=1, feature_size=48, depths=[2,2,2,2],
             num_heads=[3,6,12,24], patch_size=[2,2,2], window_size=[7,7,7]

  SwinViT forward returns list of 5 tensors (input 96³):
    hs[0]: [B,  48, 48, 48, 48]
    hs[1]: [B,  96, 24, 24, 24]
    hs[2]: [B, 192, 12, 12, 12]
    hs[3]: [B, 384,  6,  6,  6]
    hs[4]: [B, 768,  3,  3,  3]  ← deepest

  No hooks needed — stages returned directly as a list.

FEATURE MODES (confirmed from inspection)
==============================================
  mean_pool              → [B, 768]
    GAP on hs[4]: [B, 768, 3, 3, 3].mean(dim=[2,3,4])

  mean_pool_multi_layers → [B, 1056]  (96+192+768)
    GAP on hs[1], hs[2], hs[4] concatenated
    NOTE: hs[0] (48-dim) and hs[3] (384-dim) excluded —
    hs[1,2,4] give the best multi-scale representation.

  flatten                → [B, 20736]  (768 × 3 × 3 × 3)
    Flatten hs[4]

WEIGHT LOADING
==============
  checkpoint_path: local .pt file (model_weights_UKB-pretrain.pt)
  Keys: strip "module." prefix → load into SSLHead with strict=False
  Missing keys: 0, Unexpected keys: 0 (verified)

  Only swinViT encoder weights are used — SSL heads (rotation, contrastive,
  reconstruction decoder) are irrelevant and left randomly initialized.

SSL_Head.py IMPORT
==================
  SSL_Head.py must be importable. Add BrainSegFounder repo to PYTHONPATH:
    export PYTHONPATH=/path/to/BrainSegFounder:$PYTHONPATH
  Import: from pretrain.models.ssl_head import SSLHead

Config requirements:
  experiment:
    model: "bsf"
  paths:
    checkpoint_path: "/path/to/brainsegfounder_weights/model_weights_UKB-pretrain.pt"
  repositories:
    bsf: "/path/to/BrainSegFounder"
  feature_extraction:
    target_shape: [96, 96, 96]
    device:       "cuda"
    batch_size:   4
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from linear_prober.skeleton.models.bsf.normalizer import MODEL_RANGE, normalize
from linear_prober.skeleton.preprocessor import preprocess_batch

# =============================================================================
# Architecture constants — confirmed from inspection
# =============================================================================

FEATURE_DIM = {
    "mean_pool": 768,  # GAP on hs[4]
    "mean_pool_multi_layers": 96 + 192 + 768,  # 1056 — hs[1,2,4] GAP concat
    "flatten": 768 * 3 * 3 * 3,  # 20736
}


# =============================================================================
# Repo path helper
# =============================================================================


def _add_repo_to_path(repo_path: str) -> None:
    """
    Add BrainSegFounder repo root to sys.path so that
    'from pretrain.models.ssl_head import SSLHead' resolves.
    """
    if repo_path not in sys.path:
        sys.path.insert(0, str(repo_path))
        print(f"[BSF] Added to sys.path: {repo_path}")


# =============================================================================
# Model builder
# =============================================================================


def _build_model(checkpoint_path: str | Path, device: str) -> nn.Module:
    """
    Load BrainSegFounder SwinViT encoder from a local .pt checkpoint.

    Args:
        checkpoint_path : path to model_weights_UKB-pretrain.pt (local file)
        device          : "cuda" or "cpu"

    Returns:
        encoder (nn.Module) — model.swinViT, frozen, eval mode
    """
    from pretrain.models.ssl_head import SSLHead

    checkpoint_path = str(checkpoint_path)
    print(f"[BSF] Loading model from: {checkpoint_path}")

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

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    # Strip "module." prefix
    fixed = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if missing:
        print(f"[BSF] Missing keys  : {len(missing)} — {missing[:4]}")
    if unexpected:
        print(f"[BSF] Unexpected keys: {len(unexpected)} — {unexpected[:4]}")

    # Keep only the SwinViT encoder
    encoder = model.swinViT
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    encoder.to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[BSF] Encoder ready — {n_params:,} parameters (frozen)")
    return encoder


# =============================================================================
# Forward + aggregation
# =============================================================================


@torch.no_grad()
def _forward(encoder: nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Forward pass through BSF SwinViT encoder.

    Args:
        encoder : model.swinViT (frozen)
        x       : [B, 1, 96, 96, 96] float32 on device
        mode    : "mean_pool" | "mean_pool_multi_layers" | "flatten"

    Returns:
        features [B, D] float32 on CPU
    """
    # SwinViT returns list of 5 tensors — no hooks needed
    hs = encoder(x.contiguous())

    # ── mean_pool ─────────────────────────────────────────────────────────────
    if mode == "mean_pool":
        return hs[4].mean(dim=[2, 3, 4]).cpu()  # [B, 768]

    # ── mean_pool_multi_layers ────────────────────────────────────────────────
    if mode == "mean_pool_multi_layers":
        parts = [
            hs[1].mean(dim=[2, 3, 4]),  # [B,  96]
            hs[2].mean(dim=[2, 3, 4]),  # [B, 192]
            hs[4].mean(dim=[2, 3, 4]),  # [B, 768]
        ]
        return torch.cat(parts, dim=1).cpu()  # [B, 1056]

    # ── flatten ───────────────────────────────────────────────────────────────
    if mode == "flatten":
        B = x.shape[0]
        return hs[4].reshape(B, -1).cpu()  # [B, 20736]

    raise ValueError(
        f"Unknown mode '{mode}'. Supported: mean_pool, mean_pool_multi_layers, flatten"
    )


# =============================================================================
# Preprocessing helper
# =============================================================================


def _preprocess(
    volumes: torch.Tensor,
    target_shape: tuple,
    preprocessing: str,
    v0: float,
    v1: float,
    device: str,
) -> torch.Tensor:
    """
    Preprocess + normalize a batch for BSF.

    Binary-preserving (upscale_pad, nearest_neighbors):
      preprocess_batch → normalize(v0, v1)
    Continuous (trilinear):
      normalize(v0, v1) → preprocess_batch

    Returns: [B, 1, 96, 96, 96] float32 on device
    """
    is_binary_preserving = preprocessing in {"upscale_pad", "nearest_neighbors"}

    if is_binary_preserving:
        x = preprocess_batch(volumes, target_shape, preprocessing)
        x = normalize(x, v0, v1)
    else:
        x = normalize(volumes, v0, v1)
        x = preprocess_batch(x, target_shape, preprocessing)

    return x.to(device, non_blocking=True)


# =============================================================================
# Extraction loops
# =============================================================================


@torch.no_grad()
def _extract_concat(
    encoder: nn.Module,
    loader: DataLoader,
    mode: str,
    target_shape: tuple,
    preprocessing: str,
    v0: float,
    v1: float,
    device: str,
) -> Dict[str, np.ndarray]:
    """Extract features for HCP (labelled) or UKBB (unlabelled) dataloaders."""
    from datasets import UKBBSkeletonDataset

    is_ukbb = isinstance(loader.dataset, UKBBSkeletonDataset)

    all_feats: List[np.ndarray] = []
    all_labels: List = []
    all_subjects: List[str] = []
    all_folds: List[int] = []
    all_splits: List[str] = []
    all_vidx: List[int] = []

    for batch in tqdm(loader, desc=f"[BSF] {mode}/{preprocessing}", leave=True):
        x = _preprocess(batch["volume"], target_shape, preprocessing, v0, v1, device)
        feat = _forward(encoder, x, mode).numpy()
        all_feats.append(feat)
        all_subjects.extend(list(batch["subject"]))

        if not is_ukbb:
            labels = batch["label"]
            all_labels.extend(
                labels.numpy().tolist() if hasattr(labels, "numpy") else list(labels)
            )
            all_folds.extend(list(batch["fold"].numpy()))
            all_splits.extend(list(batch["split"]))
            all_vidx.extend(list(batch["volume_index"].numpy()))

    result = {
        "features": np.concatenate(all_feats, axis=0),
        "subjects": np.asarray(all_subjects),
    }
    if not is_ukbb:
        result["labels"] = np.asarray(all_labels)
        result["folds"] = np.asarray(all_folds)
        result["splits"] = np.asarray(all_splits)
        result["volume_indices"] = np.asarray(all_vidx)

    return result


@torch.no_grad()
def _extract_ukbb_prealloc(
    encoder: nn.Module,
    loader: DataLoader,
    target_shape: tuple,
    preprocessing: str,
    v0: float,
    v1: float,
    device: str,
) -> Dict[str, np.ndarray]:
    """Extract UKBB flatten features with pre-allocated array."""
    d_flat = FEATURE_DIM["flatten"]  # 20736
    n = len(loader.dataset)
    print(
        f"[BSF] Pre-allocating UKBB flatten: "
        f"{n} × {d_flat} float32 = {n * d_flat * 4 / 1e9:.2f} GB"
    )

    features_arr = np.empty((n, d_flat), dtype=np.float32)
    all_subjects: List[str] = []
    offset = 0

    for batch in tqdm(loader, desc=f"[BSF] flatten/{preprocessing} (UKBB)", leave=True):
        x = _preprocess(batch["volume"], target_shape, preprocessing, v0, v1, device)
        feat = _forward(encoder, x, "flatten").numpy()
        B = feat.shape[0]
        features_arr[offset : offset + B] = feat
        offset += B
        all_subjects.extend(list(batch["subject"]))

    assert offset == n, f"[BSF] Expected {n} subjects, got {offset}."
    return {
        "features": features_arr,
        "subjects": np.asarray(all_subjects),
    }


# =============================================================================
# Public factory
# =============================================================================


def make_extract_fn(config: dict) -> callable:
    """
    Factory: returns extract_fn(checkpoint_path, dataloader, mode, device) → dict.

    Reads from config:
      feature_extraction.target_shape   → [96, 96, 96]
      feature_extraction.preprocessing  → injected by probe scripts
      feature_extraction.v0             → injected by resolve_mapping()
      feature_extraction.v1             → injected by resolve_mapping()
    """
    target_shape = tuple(int(x) for x in config["feature_extraction"]["target_shape"])
    preprocessing = config["feature_extraction"].get("preprocessing", "upscale_pad")
    _default_v0, _default_v1 = MODEL_RANGE
    v0 = float(config["feature_extraction"].get("v0", _default_v0))
    v1 = float(config["feature_extraction"].get("v1", _default_v1))

    print(
        f"[BSF] make_extract_fn: preprocessing={preprocessing}  "
        f"target_shape={target_shape}  v0={v0}  v1={v1}"
    )

    def extract_fn(
        checkpoint_path: str | Path,
        dataloader: DataLoader,
        mode: str,
        device: str,
    ) -> Dict[str, np.ndarray]:

        from datasets import UKBBSkeletonDataset

        valid_modes = list(FEATURE_DIM.keys())
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode '{mode}'. Supported: {valid_modes}")

        encoder = _build_model(checkpoint_path, device)
        is_ukbb = isinstance(dataloader.dataset, UKBBSkeletonDataset)

        if is_ukbb and mode == "flatten":
            result = _extract_ukbb_prealloc(
                encoder, dataloader, target_shape, preprocessing, v0, v1, device
            )
        else:
            result = _extract_concat(
                encoder, dataloader, mode, target_shape, preprocessing, v0, v1, device
            )

        del encoder
        torch.cuda.empty_cache()
        return result

    return extract_fn


# =============================================================================
# Public mapping-search helper
# =============================================================================


@torch.no_grad()
def extract_mean_pool_for_mapping(
    encoder,  # SwinViT encoder from _build_model
    volumes_raw: np.ndarray,  # [N, D, H, W] uint8 binary {0,1}
    preprocessing: str,
    target_shape: tuple,
    v0: float,
    v1: float,
    device: str,
    batch_size: int,
    is_binary_preserving: bool,
) -> np.ndarray:
    """
    Extract mean_pool features [N, 768] for a specific (v0, v1) mapping.
    Called by the normaliser search.
    """
    N = len(volumes_raw)
    all_feats = []

    for start in range(0, N, batch_size):
        batch_np = volumes_raw[start : start + batch_size]
        batch_t = torch.from_numpy(
            batch_np[:, None].astype(np.float32)  # [B, 1, D, H, W]
        )

        if is_binary_preserving:
            x = preprocess_batch(batch_t, target_shape, preprocessing)
            x = normalize(x, v0, v1)
        else:
            x = normalize(batch_t, v0, v1)
            x = preprocess_batch(x, target_shape, preprocessing)

        x = x.to(device, non_blocking=True)

        feat = _forward(encoder, x, "mean_pool").numpy()  # [B, 768]
        all_feats.append(feat)

    return np.concatenate(all_feats, axis=0)  # [N, 768]
