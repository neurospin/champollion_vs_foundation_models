"""
models/dino3d/extract_features.py

Zero-shot feature extraction for 3DINO-ViT (ViT-Large, high-resolution).

Architecture — hardcoded from vit3d_highres.yaml + ssl3d_default_config.yaml:
  img_size=112, patch_size=16, embed_dim=1024, depth=24, num_heads=16
  ffn_layer="mlp", block_chunks=4, init_values=1e-5
  N_patches = (112 // 16)^3 = 7^3 = 343

Feature dims:
  mean_pool              → [B, 2048]   CLS[1024] + mean(patches)[1024]
  mean_pool_multi_layers → [B, 8192]   4 × (CLS + mean(patches)) from last 4 layers
  flatten                → [B, 352256] CLS[1024] + flatten(patches)[343 × 1024]

mean_pool_multi_layers:
  Uses model.get_intermediate_layers(x, n=4, return_class_token=True)
  Returns 4 tuples (patches [B, N, D], cls [B, D]) — one per layer.
  Each layer contributes CLS + mean(patches) = [B, 2048].
  Concatenated: [B, 4 × 2048] = [B, 8192].
  API verified in:
    vision_transformer.py: return tuple(zip(patch_outputs, class_tokens))
    linear3d.py: for _, class_token in intermediate_output  (0=patches, 1=cls)

Checkpoint format (teacher_checkpoint.pth or 3dino_vit_weights.pth):
  {"teacher": {"backbone.*": weights, "dino_head.*": ..., "ibot_head.*": ...}}
  Strip "teacher" → strip "module." → strip "backbone." → load into backbone.

Config requirements:
  repositories:
    3dino: "/path/to/3DINO/"
  feature_extraction:
    target_shape:  [112, 112, 112]
    preprocessing: "upscale_pad"   ← injected by probe scripts via CLI --preprocessing
    v0:            -1.0            ← injected by probe scripts via resolve_mapping()
    v1:            +1.0            ← injected by probe scripts via resolve_mapping()
    device:        "cuda"

Optimal mapping injection (probe scripts):
  v0, v1 = resolve_mapping(config, roi)      # from probe_utils.py
  config["feature_extraction"]["v0"] = v0
  config["feature_extraction"]["v1"] = v1
  extract_fn = make_extract_fn(config)       # captures v0, v1 at factory time

  Fallback: if v0/v1 not in config → default to MODEL_RANGE = (-1.0, +1.0)
  This preserves full backward compatibility with existing cached NPZ features.

Preprocessing dispatch (via preprocessor.py):
  "upscale_pad"          → isotropic scale + centered zero-pad
  "nearest_neighbors"    → direct F.interpolate to 112³, nearest
  "trilinear"            → direct F.interpolate to 112³, trilinear

Directory naming:
  Module lives in models/dino3d/ (not models/3dino/ — digit prefix invalid Python).
  Use model: "dino3d" in configs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from linear_prober.skeleton.models.dino3d.normalizer import MODEL_RANGE, normalize
from linear_prober.skeleton.preprocessor import preprocess_batch

# =============================================================================
# Architecture constants (hardcoded — do not change without re-validating weights)
# =============================================================================

_IMG_SIZE = 112
_PATCH_SIZE = 16
_EMBED_DIM = 1024
_N_PATCHES = (_IMG_SIZE // _PATCH_SIZE) ** 3  # 7^3 = 343
_N_MULTI_LAYERS = 4  # last N layers for multi-layer mode

FEATURE_DIM = {
    "mean_pool": 2 * _EMBED_DIM,  # 2048
    "mean_pool_multi_layers": _N_MULTI_LAYERS * 2 * _EMBED_DIM,  # 8192
    "flatten": (_N_PATCHES + 1) * _EMBED_DIM,  # 352256
}


# =============================================================================
# Repository path
# =============================================================================


def _add_repo_to_path(repo_path: str) -> None:
    """Add 3DINO repo root to sys.path so 'dinov2.*' can be imported."""
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


# =============================================================================
# Model builder
# =============================================================================


def _build_model(checkpoint_path: str | Path, device: str) -> nn.Module:
    """
    Instantiate and freeze 3DINO-ViT backbone from a teacher checkpoint.

    Architecture hardcoded (img_size=112, vit_large_3d).
    Checkpoint format: {"teacher": {"backbone.*": ..., "dino_head.*": ..., ...}}.

    Returns: frozen DinoVisionTransformer3d on `device`, in eval mode.
    """
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
# Forward pass + aggregation
# =============================================================================


@torch.no_grad()
def _forward(model: nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Forward pass + aggregation.
    x: [B, 1, 112, 112, 112] float32 in [v0, v1]

    mean_pool:
      model.forward_features(x) → CLS[B,D] + mean(patches[B,N,D])
      → [B, 2048]

    mean_pool_multi_layers:
      model.get_intermediate_layers(x, n=4, return_class_token=True)
      → tuple of 4: (patches [B,N,D], cls [B,D]) per layer
      → per layer: cat(cls, mean(patches)) = [B, 2048]
      → concat 4 layers: [B, 8192]

      API from vision_transformer.py:
        return tuple(zip(patch_outputs, class_tokens))
        → each element = (patches, cls): index 0=patches, 1=cls

    flatten:
      model.forward_features(x) → CLS[B,D] + flatten(patches[B,N,D])
      → [B, 352256]
    """
    if mode == "mean_pool":
        out = model.forward_features(x)
        cls = out["x_norm_clstoken"]  # [B, 1024]
        patches = out["x_norm_patchtokens"]  # [B, 343, 1024]
        return torch.cat([cls, patches.mean(dim=1)], dim=-1)  # [B, 2048]

    if mode == "mean_pool_multi_layers":
        intermediate = model.get_intermediate_layers(
            x, n=_N_MULTI_LAYERS, return_class_token=True
        )
        layer_feats = []
        for patches, cls in intermediate:
            layer_feats.append(
                torch.cat([cls, patches.mean(dim=1)], dim=-1)  # [B, 2048]
            )
        return torch.cat(layer_feats, dim=-1)  # [B, 8192]

    if mode == "flatten":
        out = model.forward_features(x)
        cls = out["x_norm_clstoken"]  # [B, 1024]
        patches = out["x_norm_patchtokens"]  # [B, 343, 1024]
        return torch.cat([cls, patches.flatten(1)], dim=-1)  # [B, 352256]

    raise ValueError(
        f"Unknown mode '{mode}'. "
        "Expected 'mean_pool', 'mean_pool_multi_layers', or 'flatten'."
    )


# =============================================================================
# Preprocessing helper
# =============================================================================


def _preprocess(
    batch_volume: torch.Tensor,
    target_shape: tuple,
    preprocessing: str,
    v0: float,
    v1: float,
    device: str,
) -> torch.Tensor:
    """
    Geometric preprocessing + normalization before GPU forward pass.

    Input:  [B, 1, D, H, W] float32
    Output: [B, 1, T, T, T] float32 in [v0, v1]  on device

    v0, v1 come from resolve_mapping(config, roi) injected into config by
    the probe script before make_extract_fn() is called.
    Fallback: MODEL_RANGE = (-1.0, +1.0) when not set → x*2-1 (original behaviour).

    preprocessing dispatches to preprocessor.py:
      "upscale_pad"          → isotropic scale + centered zero-pad
      "nearest_neighbors"    → direct resize, nearest
      "trilinear"            → direct resize, trilinear

    normalize(x, v0, v1) maps [0,1] → [v0, v1] for all preprocessing types.
    """
    x = preprocess_batch(batch_volume, target_shape, preprocessing)
    x = normalize(x, v0, v1)
    return x.to(device, non_blocking=True)


# =============================================================================
# Extraction — generic (list + concatenate)
# =============================================================================


@torch.no_grad()
def _extract_concat(
    model: nn.Module,
    loader: DataLoader,
    mode: str,
    target_shape: tuple,
    preprocessing: str,
    v0: float,
    v1: float,
    device: str,
) -> Dict[str, np.ndarray]:
    """
    Extract features by collecting batches and concatenating at the end.
    Used for: HCP (all modes) and UKBB mean_pool / mean_pool_multi_layers.

    Returns dict:
      Always:   "features", "subjects"
      HCP only: "labels", "folds", "splits", "volume_indices"
    """
    all_features: list = []
    tensor_lists: Dict[str, list] = {}
    string_lists: Dict[str, list] = {}

    _tensor_fields = {
        "label": "labels",
        "fold": "folds",
        "volume_index": "volume_indices",
    }
    _string_fields = {"subject": "subjects", "split": "splits"}

    for batch in tqdm(loader, desc=f"[3DINO] {mode} ({preprocessing})", leave=True):
        x = _preprocess(batch["volume"], target_shape, preprocessing, v0, v1, device)
        feat = _forward(model, x, mode)
        all_features.append(feat.cpu().numpy())

        for src, dst in _tensor_fields.items():
            if src in batch:
                tensor_lists.setdefault(dst, []).append(batch[src].cpu().numpy())

        for src, dst in _string_fields.items():
            if src in batch:
                string_lists.setdefault(dst, []).extend(list(batch[src]))

    result: Dict[str, np.ndarray] = {
        "features": np.concatenate(all_features, axis=0),
    }
    for key, chunks in tensor_lists.items():
        result[key] = np.concatenate(chunks, axis=0)
    for key, items in string_lists.items():
        result[key] = np.asarray(items)

    return result


# =============================================================================
# Extraction — UKBB flatten with pre-allocation (RAM-safe)
# =============================================================================


@torch.no_grad()
def _extract_ukbb_prealloc(
    model: nn.Module,
    loader: DataLoader,
    target_shape: tuple,
    preprocessing: str,
    v0: float,
    v1: float,
    device: str,
) -> Dict[str, np.ndarray]:
    """
    Extract UKBB flatten features with pre-allocated array.
    Peak RAM = 1× feature matrix (no concatenation spike).

    42k × 352256 × 4 bytes ≈ 59 GB — ensure sufficient RAM.
    Only used for mode="flatten" on UKBB (PCA fitting).

    np.savez (not np.savez_compressed) must be used by caller to avoid
    a 2× RAM spike (BytesIO buffer) that would OOM on Jean Zay.
    """
    n = len(loader.dataset)
    d_flat = FEATURE_DIM["flatten"]

    print(
        f"[3DINO] Pre-allocating UKBB features: "
        f"{n} × {d_flat} float32 = {n * d_flat * 4 / 1e9:.1f} GB"
    )

    features_arr = np.empty((n, d_flat), dtype=np.float32)
    all_subjects: list = []
    offset = 0

    for batch in tqdm(
        loader, desc=f"[3DINO] flatten/{preprocessing} (UKBB)", leave=True
    ):
        x = _preprocess(batch["volume"], target_shape, preprocessing, v0, v1, device)
        feat = _forward(model, x, "flatten").cpu().numpy()
        B = feat.shape[0]
        features_arr[offset : offset + B] = feat
        offset += B
        all_subjects.extend(list(batch["subject"]))

    assert offset == n, (
        f"[3DINO] Expected {n} subjects, processed {offset}. "
        "Check drop_last=False in the dataloader."
    )

    return {
        "features": features_arr,
        "subjects": np.asarray(all_subjects),
    }


# =============================================================================
# Public factory
# =============================================================================


def make_extract_fn(config: dict) -> callable:
    """
    Factory: returns a closed-over extract_fn capturing config params.

    Reads from config:
      repositories.3dino                       → added to sys.path for dinov2 imports
      feature_extraction.target_shape          → e.g. [112, 112, 112]
      feature_extraction.preprocessing         → injected by probe scripts via --preprocessing
                                                  default: "upscale_pad"
      feature_extraction.v0                    → injected by probe scripts via resolve_mapping()
      feature_extraction.v1                    → injected by probe scripts via resolve_mapping()
                                                  default: MODEL_RANGE = (-1.0, +1.0)

    Injection pattern in probe scripts (BEFORE calling make_extract_fn):
      v0, v1 = resolve_mapping(config, roi)
      config["feature_extraction"]["v0"] = v0
      config["feature_extraction"]["v1"] = v1
      extract_fn = make_extract_fn(config)

    Returns:
      extract_fn(checkpoint_path, dataloader, mode, device) -> dict
        Supported modes: "mean_pool", "mean_pool_multi_layers", "flatten"
        HCP  → {"features", "labels", "subjects", "folds", "splits", "volume_indices"}
        UKBB → {"features", "subjects"}

    Dispatch:
      UKBB + "flatten" → _extract_ukbb_prealloc  (RAM-safe, no concatenation)
      everything else  → _extract_concat
    """
    repo_path = config["repositories"]["3dino"]
    target_shape = tuple(int(x) for x in config["feature_extraction"]["target_shape"])
    preprocessing = config["feature_extraction"].get("preprocessing", "upscale_pad")

    # Optimal mapping — fallback to MODEL_RANGE if not injected
    _default_v0, _default_v1 = MODEL_RANGE
    v0 = float(config["feature_extraction"].get("v0", _default_v0))
    v1 = float(config["feature_extraction"].get("v1", _default_v1))

    _add_repo_to_path(repo_path)

    print(
        f"[3DINO] make_extract_fn: preprocessing={preprocessing}  "
        f"target_shape={target_shape}  v0={v0}  v1={v1}"
    )

    def extract_fn(
        checkpoint_path: str | Path,
        dataloader: DataLoader,
        mode: str,
        device: str,
    ) -> Dict[str, np.ndarray]:

        from datasets import UKBBSkeletonDataset

        if mode not in FEATURE_DIM:
            raise ValueError(
                f"Unknown mode '{mode}'. Supported: {list(FEATURE_DIM.keys())}"
            )

        model = _build_model(checkpoint_path, device)
        is_ukbb = isinstance(dataloader.dataset, UKBBSkeletonDataset)

        if is_ukbb and mode == "flatten":
            result = _extract_ukbb_prealloc(
                model, dataloader, target_shape, preprocessing, v0, v1, device
            )
        else:
            result = _extract_concat(
                model, dataloader, mode, target_shape, preprocessing, v0, v1, device
            )

        del model
        torch.cuda.empty_cache()
        return result

    return extract_fn


# =============================================================================
# Public mapping-search helper — used by the normaliser search
# =============================================================================


@torch.no_grad()
def extract_mean_pool_for_mapping(
    encoder: nn.Module,
    volumes_raw: np.ndarray,
    preprocessing: str,
    target_shape: tuple,
    v0: float,
    v1: float,
    device: str,
    batch_size: int,
    is_binary_preserving: bool,
) -> np.ndarray:
    """
    Extract mean_pool features [N, 2048] for a specific (v0, v1) mapping.
    Called by the normaliser search — model-specific implementation.

    3DINO mean_pool:
      forward_features(x) → CLS [B,1024] + mean(patches [B,343,1024]) → [B,2048]

    Two regimes (determined by preprocessing type):
      is_binary_preserving=True  (upscale_pad, nearest_neighbors):
        preprocess_batch → normalize(v0, v1) → encoder
        Rationale: preprocessing preserves {0,1} → mapping on exact binary values.

      is_binary_preserving=False (trilinear):
        normalize(v0, v1) → preprocess_batch → encoder
        Rationale: mapping applied on binary source BEFORE interpolation so that
        intermediate voxels are interpolated in the normalised space [v0, v1].
        For affine normalizers this is mathematically equivalent to the other order.

    normalize() for 3DINO: x = v0 + volume * (v1 - v0)
      Maps {0,1} → {v0, v1} with v0,v1 ∈ [-1, +1].

    Args:
      encoder              : frozen 3DINO backbone (from _build_model)
      volumes_raw          : [N, D, H, W] uint8, binary {0,1}
      preprocessing        : preprocessing mode name
      target_shape         : (T, T, T) — (112, 112, 112) for 3DINO
      v0, v1               : normalised values for voxels 0 and 1
      device               : "cuda" or "cpu"
      batch_size           : volumes per forward pass
      is_binary_preserving : True → preprocess then normalise
                             False → normalise then preprocess

    Returns:
      features [N, 2048] float32
    """
    N = len(volumes_raw)
    all_feats = []

    for start in range(0, N, batch_size):
        batch_np = volumes_raw[start : start + batch_size]  # [B, D, H, W]
        batch_t = torch.from_numpy(
            batch_np[:, None].astype(np.float32)  # [B, 1, D, H, W]
        )

        if is_binary_preserving:
            x = preprocess_batch(batch_t, target_shape, preprocessing)  # {0,1}
            x = normalize(x, v0, v1)  # {v0,v1}
        else:
            x = normalize(batch_t, v0, v1)  # {v0,v1}
            x = preprocess_batch(x, target_shape, preprocessing)  # resized to cube

        x = x.to(device, non_blocking=True)

        # 3DINO mean_pool: CLS + mean(patches)
        out = encoder.forward_features(x)
        cls = out["x_norm_clstoken"]  # [B, 1024]
        patches = out["x_norm_patchtokens"]  # [B, 343, 1024]
        feat = torch.cat([cls, patches.mean(dim=1)], dim=-1)  # [B, 2048]
        all_feats.append(feat.cpu().numpy())

    return np.concatenate(all_feats, axis=0)  # [N, 2048]
