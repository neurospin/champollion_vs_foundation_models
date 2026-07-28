"""
models/vista3d/extract_features.py

Zero-shot feature extraction for VISTA3D (SegResNetDS2, MONAI 1.4).

ARCHITECTURE (confirmed from inspection)
====================================================
  Backbone : SegResNetDS2 via monai.networks.nets.vista3d132
             in_channels=1, encoder_embed_dim=48
  Encoder  : model.image_encoder.encoder  (SegResEncoder)

  Encoder layer progression (input 96³):
    encoder.layers.0.downsample  →  [B,  96, 48, 48, 48]
    encoder.layers.1.downsample  →  [B, 192, 24, 24, 24]
    encoder.layers.2.downsample  →  [B, 384, 12, 12, 12]
    encoder.layers.3.downsample  →  [B, 768,  6,  6,  6]  ← deepest

  Hook placement: on each downsample output — these are the cleanest
  level boundaries in the SegResEncoder, one per resolution level.

FEATURE MODES
=============
  mean_pool              → [B, 768]
    GAP on encoder.layers.3.downsample output [B, 768, 6, 6, 6]

  mean_pool_multi_layers → [B, 1440]  (96+192+384+768)
    GAP on each of the 4 downsample outputs, concatenated

  flatten                → [B, 165888]  (768 × 6 × 6 × 6)
    Flatten encoder.layers.3.downsample output

NORMALISATION
=============
  Our normalize(x, v0, v1) = v0 + x*(v1-v0)  maps {0,1}→{v0,v1} ∈ [0,1]
  VISTA3D has NO internal normalization module — the affine mapping is all.

WEIGHT LOADING
==============
  checkpoint_path: local .safetensors file
  Strip "network." prefix from state_dict keys if present (confirmed: 0 missing keys)
  Missing keys: 0, Unexpected keys: 0 (verified by inspection)

Config requirements:
  experiment:
    model: "vista3d"
  paths:
    checkpoint_path: "/path/to/vista3d_pretrained_model/model.safetensors"
  feature_extraction:
    target_shape: [96, 96, 96]
    device:       "cuda"
    batch_size:   2
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from linear_prober.skeleton.models.vista3d.normalizer import MODEL_RANGE, normalize
from linear_prober.skeleton.preprocessor import preprocess_batch

# =============================================================================
# Architecture constants — confirmed from inspection
# =============================================================================

# Hook targets: downsample output at each encoder level
_HOOK_NAMES = [
    "encoder.layers.0.downsample",  # [B,  96, 48, 48, 48]
    "encoder.layers.1.downsample",  # [B, 192, 24, 24, 24]
    "encoder.layers.2.downsample",  # [B, 384, 12, 12, 12]
    "encoder.layers.3.downsample",  # [B, 768,  6,  6,  6]  ← deepest
]

_EMBED_DIMS = [96, 192, 384, 768]  # channels at each level
_SPATIAL_FINAL = (6, 6, 6)  # spatial size at deepest level

FEATURE_DIM = {
    "mean_pool": 768,  # GAP on deepest
    "mean_pool_multi_layers": 96 + 192 + 384 + 768,  # 1440
    "flatten": 768 * 6 * 6 * 6,  # 165888
}

_N_MULTI_LAYERS = 4  # all 4 encoder levels for mean_pool_multi_layers


# =============================================================================
# Model builder
# =============================================================================


def _add_repo_to_path(repo_path: str) -> None:
    """No-op for VISTA3D — installed via pip install monai, no sys.path needed."""
    pass


def _build_model(checkpoint_path: str | Path, device: str) -> nn.Module:
    """
    Load VISTA3D encoder from a local .safetensors checkpoint.

    Args:
        checkpoint_path : path to model.safetensors (local file)
        device          : "cuda" or "cpu"

    Returns:
        model (nn.Module) — full vista3d132 model, frozen, eval mode
        (we keep the full model to access image_encoder.encoder internals)
    """
    import monai
    from safetensors.torch import load_file as load_safetensors

    checkpoint_path = str(checkpoint_path)
    print(f"[VISTA3D] Loading model from: {checkpoint_path}")

    model = monai.networks.nets.vista3d132(
        in_channels=1,
        encoder_embed_dim=48,
    )

    state = load_safetensors(checkpoint_path)
    # Strip "network." prefix if present (MONAI bundle convention)
    state = {(k[8:] if k.startswith("network.") else k): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[VISTA3D] Missing keys  : {len(missing)} — {missing[:4]}")
    if unexpected:
        print(f"[VISTA3D] Unexpected keys: {len(unexpected)} — {unexpected[:4]}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[VISTA3D] Model ready — {n_params:,} parameters (frozen)")
    return model


# =============================================================================
# Hook-based feature extraction
# =============================================================================


class _HookExtractor:
    """
    Registers forward hooks on the 4 encoder downsample layers and
    collects their outputs during a forward pass.

    Hook targets (confirmed from inspection):
      encoder.layers.0.downsample  →  [B,  96, 48, 48, 48]
      encoder.layers.1.downsample  →  [B, 192, 24, 24, 24]
      encoder.layers.2.downsample  →  [B, 384, 12, 12, 12]
      encoder.layers.3.downsample  →  [B, 768,  6,  6,  6]
    """

    def __init__(self, image_encoder: nn.Module) -> None:
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks: List = []

        # Resolve module references from image_encoder
        encoder = image_encoder.encoder
        targets = {
            "layer0": encoder.layers[0].downsample,
            "layer1": encoder.layers[1].downsample,
            "layer2": encoder.layers[2].downsample,
            "layer3": encoder.layers[3].downsample,
        }

        for key, module in targets.items():

            def _make_hook(k):
                def hook(mod, inp, out):
                    self._features[k] = out.detach()

                return hook

            self._hooks.append(module.register_forward_hook(_make_hook(key)))

    def get(self) -> Dict[str, torch.Tensor]:
        return self._features

    def clear(self) -> None:
        self._features.clear()

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# =============================================================================
# Forward + aggregation
# =============================================================================


@torch.no_grad()
def _forward(model: nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Forward pass through VISTA3D image_encoder with hook-based feature capture.

    Args:
        model : full vista3d132 model (frozen)
        x     : [B, 1, 96, 96, 96] float32 on device
        mode  : "mean_pool" | "mean_pool_multi_layers" | "flatten"

    Returns:
        features [B, D] float32 on CPU
    """
    B = x.shape[0]
    extractor = _HookExtractor(model.image_encoder)
    extractor.clear()

    # Forward through image_encoder — hooks capture downsample outputs
    _ = model.image_encoder(x)

    feats = extractor.get()
    extractor.remove()

    # ── mean_pool ─────────────────────────────────────────────────────────────
    if mode == "mean_pool":
        f = feats["layer3"]  # [B, 768, 6, 6, 6]
        return f.mean(dim=[2, 3, 4]).cpu()  # [B, 768]

    # ── mean_pool_multi_layers ────────────────────────────────────────────────
    if mode == "mean_pool_multi_layers":
        parts = [
            feats["layer0"].mean(dim=[2, 3, 4]),  # [B,  96]
            feats["layer1"].mean(dim=[2, 3, 4]),  # [B, 192]
            feats["layer2"].mean(dim=[2, 3, 4]),  # [B, 384]
            feats["layer3"].mean(dim=[2, 3, 4]),  # [B, 768]
        ]
        return torch.cat(parts, dim=1).cpu()  # [B, 1440]

    # ── flatten ───────────────────────────────────────────────────────────────
    if mode == "flatten":
        f = feats["layer3"]  # [B, 768, 6, 6, 6]
        return f.reshape(B, -1).cpu()  # [B, 165888]

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
    Preprocess + normalize a batch for VISTA3D.

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
    Extract features for HCP (labelled) or UKBB (unlabelled) dataloaders.
    """
    from datasets import UKBBSkeletonDataset

    is_ukbb = isinstance(loader.dataset, UKBBSkeletonDataset)

    all_feats: List[np.ndarray] = []
    all_labels: List = []
    all_subjects: List[str] = []
    all_folds: List[int] = []
    all_splits: List[str] = []
    all_vidx: List[int] = []

    for batch in tqdm(loader, desc=f"[VISTA3D] {mode}/{preprocessing}", leave=True):
        x = _preprocess(batch["volume"], target_shape, preprocessing, v0, v1, device)
        feat = _forward(model, x, mode).numpy()
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
    Avoids RAM spike from concatenation.
    """
    d_flat = FEATURE_DIM["flatten"]  # 165888
    n = len(loader.dataset)
    print(
        f"[VISTA3D] Pre-allocating UKBB flatten: "
        f"{n} × {d_flat} float32 = {n * d_flat * 4 / 1e9:.2f} GB"
    )

    features_arr = np.empty((n, d_flat), dtype=np.float32)
    all_subjects: List[str] = []
    offset = 0

    for batch in tqdm(
        loader, desc=f"[VISTA3D] flatten/{preprocessing} (UKBB)", leave=True
    ):
        x = _preprocess(batch["volume"], target_shape, preprocessing, v0, v1, device)
        feat = _forward(model, x, "flatten").numpy()
        B = feat.shape[0]
        features_arr[offset : offset + B] = feat
        offset += B
        all_subjects.extend(list(batch["subject"]))

    assert offset == n, f"[VISTA3D] Expected {n} subjects, got {offset}."
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
        f"[VISTA3D] make_extract_fn: preprocessing={preprocessing}  "
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
# Public mapping-search helper
# =============================================================================


@torch.no_grad()
def extract_mean_pool_for_mapping(
    encoder,  # full vista3d132 model from _build_model
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
