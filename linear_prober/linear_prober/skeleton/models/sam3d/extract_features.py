"""
models/sam3d/extract_features.py

Zero-shot feature extraction for SAM-Med3D (vit_b_ori).

Architecture — from build_sam3D_vit_b_ori + image_encoder3D.py:
  img_size=128, patch_size=16, embed_dim=768 (ViT), out_chans=384 (neck)
  depth=12, num_heads=12
  global_attn_indexes=[2, 5, 8, 11] — others use window attention (window_size=14)
  N_patches = (128 // 16)^3 = 8^3 = 512

ImageEncoderViT3D forward (verified from image_encoder3D.py source):
  x = patch_embed(x)           [B,1,128,128,128] → [B,8,8,8,768]
  x = x + pos_embed            absolute positional embedding (learned)
  for blk in blocks: x = blk(x)  [B,8,8,8,768] → [B,8,8,8,768]
  x = neck(x.permute(0,4,1,2,3)) [B,768,8,8,8] → [B,384,8,8,8]

Feature extraction modes:

  mean_pool (384-D):
    → image_encoder(x)                  # [B,384,8,8,8]
    → mean over spatial dims (2,3,4)     # [B,384]
    Compact bottleneck embedding. Equivalent to 3DINO mean_pool
    but in the neck space optimised for segmentation.

  mean_pool_multi_layers (1152-D):
    → decomposed forward to intercept both ViT and neck spaces
    → vit_mean  = mean(vit_tokens, dim=1)        [B,768]  — ViT pure space
    → neck_mean = mean(neck_out,   dims=(2,3,4)) [B,384]  — bottleneck space
    → cat([neck_mean, vit_mean], dim=-1)          [B,1152]
    Richer representation combining both levels of abstraction.
    NOT analogous to 3DINO mean_pool_multi_layers (which uses 4 transformer
    layers) — here it combines two distinct representation spaces.

  flatten (196608-D):
    → image_encoder(x)                  # [B,384,8,8,8]
    → flatten(1)                         # [B,196608]
    Full bottleneck spatial map. Used for PCA fitting on UKBB.
    Analogous to 3DINO flatten but in neck space (384 × 512 = 196,608).

Checkpoint format (sam_med3d_turbo.pth):
  {"model_state_dict": {weights}}
  Loads full Sam3D (encoder + prompt encoder + mask decoder).
  Only image_encoder is used — others are discarded after loading.

Config requirements:
  repositories:
    sam3d: "/path/to/SAM-Med3D/"
  feature_extraction:
    target_shape: [128, 128, 128]
    preprocessing: "upscale_pad"   ← injected by probe scripts via CLI
    v0: 0.0                        ← injected by probe scripts via resolve_mapping()
    v1: 1.0                        ← injected by probe scripts via resolve_mapping()
    device: "cuda"

Decomposed forward for mean_pool_multi_layers:
  Verified safe from image_encoder3D.py source:
  - patch_embed output: [B,8,8,8,768]  (Conv3d + permute inside PatchEmbed3D)
  - pos_embed added if not None        (CRITICAL — must not be skipped)
  - each Block3D handles window partition/unpartition internally
  - neck expects [B,C,D,H,W] → permute(0,4,1,2,3) before neck call
  torch.allclose(enc(x), neck_out) == True guaranteed by this decomposition.

Directory naming:
  Module lives in models/sam3d/.
  Use model: "sam3d" in configs.
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

from linear_prober.skeleton.models.sam3d.normalizer import MODEL_RANGE, normalize
from linear_prober.skeleton.preprocessor import preprocess_batch

# =============================================================================
# Architecture constants (hardcoded — validated against build_sam3D_vit_b_ori)
# =============================================================================

_IMG_SIZE = 128
_PATCH_SIZE = 16
_EMBED_DIM = 768  # ViT embedding dim (before neck)
_NECK_DIM = 384  # neck output channels (= out_chans in ImageEncoderViT3D)
_N_PATCHES = (_IMG_SIZE // _PATCH_SIZE) ** 3  # 8^3 = 512

FEATURE_DIM = {
    "mean_pool": _NECK_DIM,  # 384
    "mean_pool_multi_layers": _NECK_DIM + _EMBED_DIM,  # 1152  (384 + 768)
    "flatten": _NECK_DIM * _N_PATCHES,  # 196608 (384 × 512)
}


# =============================================================================
# Repository path
# =============================================================================


def _add_repo_to_path(repo_path: str) -> None:
    """Add SAM-Med3D repo root to sys.path so 'segment_anything.*' can be imported."""
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


# =============================================================================
# Model builder
# =============================================================================


def _build_model(checkpoint_path: str | Path, device: str) -> nn.Module:
    """
    Instantiate and freeze SAM-Med3D image encoder from checkpoint.

    Loads the full Sam3D model (encoder + prompt encoder + mask decoder)
    from checkpoint["model_state_dict"], then extracts and returns only
    the image_encoder — prompt_encoder and mask_decoder are discarded.

    Returns: frozen ImageEncoderViT3D on `device`, in eval mode.
    """
    from segment_anything.build_sam3D import sam_model_registry3D

    print(f"[SAM3D] Loading checkpoint: {checkpoint_path}")
    chkpt = torch.load(str(checkpoint_path), map_location="cpu")

    if "model_state_dict" not in chkpt:
        raise ValueError(
            f"Checkpoint missing 'model_state_dict' key. Found: {list(chkpt.keys())}"
        )

    # Instantiate full model (needed to load state dict correctly)
    model = sam_model_registry3D["vit_b_ori"]()
    model.load_state_dict(chkpt["model_state_dict"])

    # Extract only the image encoder — discard prompt encoder and mask decoder
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
# Forward pass + aggregation
# =============================================================================


@torch.no_grad()
def _forward(
    encoder: nn.Module,
    x: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """
    Forward pass + aggregation for a preprocessed and normalised batch.

    x: [B, 1, 128, 128, 128] float32 — normalised by models/sam3d/normalizer.py

    mean_pool:
      Standard forward through the full image encoder.
      neck output [B,384,8,8,8] → spatial mean → [B,384]

    mean_pool_multi_layers:
      Decomposed forward to intercept both the ViT token space and neck space.
      Verified equivalent to full forward from image_encoder3D.py source:
        patch_embed → (+pos_embed) → blocks → permute → neck
      vit_mean  = mean(vit_tokens [B,512,768], dim=1)        → [B,768]
      neck_mean = mean(neck_out   [B,384,8,8,8], dims=(2,3,4)) → [B,384]
      output    = cat([neck_mean, vit_mean], dim=-1)          → [B,1152]

    flatten:
      Standard forward through the full image encoder.
      neck output [B,384,8,8,8] → flatten(1) → [B,196608]
    """
    if mode == "mean_pool":
        out = encoder(x)  # [B,384,8,8,8]
        return out.mean(dim=(2, 3, 4))  # [B,384]

    if mode == "flatten":
        out = encoder(x)  # [B,384,8,8,8]
        return out.flatten(1)  # [B,196608]

    if mode == "mean_pool_multi_layers":
        # ── Decomposed forward — verified against image_encoder3D.py source ──
        # Step 1: patch embedding [B,1,128,128,128] → [B,8,8,8,768]
        h = encoder.patch_embed(x)

        # Step 2: absolute positional embedding (learned, must not be skipped)
        if encoder.pos_embed is not None:
            h = h + encoder.pos_embed

        # Step 3: 12 transformer blocks (window/global attention handled internally)
        for blk in encoder.blocks:
            h = blk(h)
        # h: [B,8,8,8,768]

        # Step 4: extract ViT-space features before neck
        vit_tokens = h.flatten(1, 3)  # [B,512,768]
        vit_mean = vit_tokens.mean(dim=1)  # [B,768]

        # Step 5: neck (Conv3d 1×1 + Conv3d 3×3 with LayerNorm3d)
        # neck expects [B,C,D,H,W] — permute from [B,D,H,W,C]
        neck_out = encoder.neck(h.permute(0, 4, 1, 2, 3))  # [B,384,8,8,8]
        neck_mean = neck_out.mean(dim=(2, 3, 4))  # [B,384]

        # Step 6: concatenate neck and ViT spaces
        return torch.cat([neck_mean, vit_mean], dim=-1)  # [B,1152]

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
    Geometric preprocessing + SAM-Med3D normalisation before GPU forward pass.

    Input:  [B, 1, D, H, W] float32  — binary {0.0, 1.0}
    Output: [B, 1, T, T, T] float32  — normalised, on device

    v0, v1 come from resolve_mapping(config, roi) injected into config by
    the probe script before make_extract_fn() is called.
    Fallback: MODEL_RANGE = (0.0, 1.0) → original SAM-Med3D preprocessing.

    preprocessing dispatches to preprocessor.py:
      "upscale_pad"          → isotropic scale + centered zero-pad
      "nearest_neighbors"    → direct resize, nearest
      "trilinear"            → direct resize, trilinear

    normalize(x, v0, v1):
      step 1: x = v0 + x * (v1 - v0)   optimal mapping
      step 2: x = x * 255.0             rescale to [0,255]
      step 3: x = (x - 123.675)/58.395  fixed SAM-Med3D standardisation
    """
    x = preprocess_batch(batch_volume, target_shape, preprocessing)
    x = normalize(x, v0, v1)
    return x.to(device, non_blocking=True)


# =============================================================================
# Extraction — generic (list + concatenate)
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

    for batch in tqdm(loader, desc=f"[SAM3D] {mode} ({preprocessing})", leave=True):
        x = _preprocess(batch["volume"], target_shape, preprocessing, v0, v1, device)
        feat = _forward(encoder, x, mode)
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
    encoder: nn.Module,
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

    SAM-Med3D flatten dim = 196,608:
      42k × 196,608 × 4 bytes ≈ 33 GB — lower than 3DINO (59 GB).

    np.savez (not np.savez_compressed) must be used by caller to avoid
    a 2× RAM spike (BytesIO buffer) that would OOM on Jean Zay.
    """
    n = len(loader.dataset)
    d_flat = FEATURE_DIM["flatten"]

    print(
        f"[SAM3D] Pre-allocating UKBB features: "
        f"{n} × {d_flat} float32 = {n * d_flat * 4 / 1e9:.1f} GB"
    )

    features_arr = np.empty((n, d_flat), dtype=np.float32)
    all_subjects: list = []
    offset = 0

    for batch in tqdm(
        loader, desc=f"[SAM3D] flatten/{preprocessing} (UKBB)", leave=True
    ):
        x = _preprocess(batch["volume"], target_shape, preprocessing, v0, v1, device)
        feat = _forward(encoder, x, "flatten").cpu().numpy()
        B = feat.shape[0]
        features_arr[offset : offset + B] = feat
        offset += B
        all_subjects.extend(list(batch["subject"]))

    assert offset == n, (
        f"[SAM3D] Expected {n} subjects, processed {offset}. "
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
      repositories.sam3d                       → added to sys.path for segment_anything imports
      feature_extraction.target_shape          → [128, 128, 128]
      feature_extraction.preprocessing         → injected by probe scripts via --preprocessing
                                                  default: "upscale_pad"
      feature_extraction.v0                    → injected by probe scripts via resolve_mapping()
      feature_extraction.v1                    → injected by probe scripts via resolve_mapping()
                                                  default: MODEL_RANGE = (0.0, 1.0)

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
    repo_path = config["repositories"]["sam3d"]
    target_shape = tuple(int(x) for x in config["feature_extraction"]["target_shape"])
    preprocessing = config["feature_extraction"].get("preprocessing", "upscale_pad")

    # Optimal mapping — fallback to MODEL_RANGE if not injected by probe script
    _default_v0, _default_v1 = MODEL_RANGE
    v0 = float(config["feature_extraction"].get("v0", _default_v0))
    v1 = float(config["feature_extraction"].get("v1", _default_v1))

    _add_repo_to_path(repo_path)

    print(
        f"[SAM3D] make_extract_fn: preprocessing={preprocessing}  "
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
    Extract mean_pool features [N, 384] for a specific (v0, v1) mapping.
    Called by the normaliser search — model-specific implementation.

    SAM-Med3D mean_pool:
      image_encoder(x) → [B, 384, 8, 8, 8] → mean(dims=(2,3,4)) → [B, 384]
      (bottleneck space after the convolutional neck)

    Two regimes (determined by preprocessing type):
      is_binary_preserving=True  (upscale_pad, nearest_neighbors):
        preprocess_batch → normalize(v0, v1) → encoder
        Rationale: preprocessing preserves {0,1} → mapping on exact binary values.

      is_binary_preserving=False (trilinear):
        normalize(v0, v1) → preprocess_batch → encoder
        Rationale: mapping applied on binary source BEFORE interpolation.

    normalize() for SAM-Med3D (3 steps):
      step 1: x = v0 + volume * (v1 - v0)   optimal mapping {0,1}→{v0,v1} ∈[0,1]
      step 2: x = x * 255.0                  rescale to [0,255]
      step 3: x = (x - 123.675) / 58.395    fixed SAM-Med3D standardisation

    Critical difference vs 3DINO:
      encoder(x) instead of encoder.forward_features(x)
      mean over spatial dims (2,3,4) instead of CLS+patches aggregation

    Args:
      encoder              : frozen SAM-Med3D image_encoder (from _build_model)
      volumes_raw          : [N, D, H, W] uint8, binary {0,1}
      preprocessing        : preprocessing mode name
      target_shape         : (T, T, T) — (128, 128, 128) for SAM-Med3D
      v0, v1               : values for voxels 0 and 1 in [0, 1]
      device               : "cuda" or "cpu"
      batch_size           : volumes per forward pass
      is_binary_preserving : True → preprocess then normalise
                             False → normalise then preprocess

    Returns:
      features [N, 384] float32
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
            x = normalize(x, v0, v1)  # {v0,v1} → ×255 → standardise
        else:
            x = normalize(batch_t, v0, v1)  # standardise sur binaire
            x = preprocess_batch(x, target_shape, preprocessing)  # resized to cube

        x = x.to(device, non_blocking=True)

        # SAM-Med3D mean_pool: moyenne spatiale sur le bottleneck neck
        out = encoder(x)  # [B, 384, 8, 8, 8]
        feat = out.mean(dim=(2, 3, 4))  # [B, 384]
        all_feats.append(feat.cpu().numpy())

    return np.concatenate(all_feats, axis=0)  # [N, 384]
