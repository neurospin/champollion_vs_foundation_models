"""
models/dinov3/extract_features.py

Zero-shot feature extraction for DINOv3 ViT models on sulcal volumes.

Logical mode:
  {extraction}__{aggregation}__{slicer}__{model_size}[__dw]

Versioned cache mode:
  {logical_mode}__cache_regs4_v1

The cache suffix changes the feature .npz filename and prevents reuse of
legacy caches created before the register-token correction. The three probing
scripts must call add_feature_cache_version(mode) for the `mode` passed to
load_or_extract_hcp_features().

DINOv3 token layout:
  [CLS] [4 register tokens] [196 patch tokens]

Representations:
  mean_pool = concat(CLS, mean(patch tokens))
  flatten   = concat(CLS, flatten(patch tokens))

Register tokens are always excluded from patch representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from linear_prober.skeleton.models.dinov3.normalizer import (
    normalize_affine,
    normalize_imagenet,
)
from linear_prober.skeleton.models.dinov3.slicers import (
    AXES,
    N_SLICES_PER_AXIS,
    get_slices,
)
from linear_prober.skeleton.preprocessor import preprocess_batch

# =============================================================================
# Constants
# =============================================================================

TARGET_SHAPE: tuple[int, int, int] = (224, 224, 224)
PATCH_SIZE: int = 16
N_PATCHES: int = (224 // PATCH_SIZE) ** 2  # 196
EXPECTED_NUM_REGISTER_TOKENS: int = 4

# Increment this whenever feature semantics change.
FEATURE_CACHE_VERSION: str = "regs4_v1"
FEATURE_CACHE_TOKEN: str = f"cache_{FEATURE_CACHE_VERSION}"

MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "vits16": {"hidden_size": 384},
    "vits16plus": {"hidden_size": 384},
    "vitb16": {"hidden_size": 768},
    "vitl16": {"hidden_size": 1024},
    "vith16plus": {"hidden_size": 1280},
    "vit7b16": {"hidden_size": 4096},
}

MODEL_SIZES: set[str] = set(MODEL_CONFIGS)
EXTRACTION_MODES: set[str] = {"mean_pool", "flatten"}
AGGREGATION_MODES: set[str] = {"mean_pool_axis", "concat_all"}
SLICER_MODES: set[str] = {"2d", "25d"}

_DW_INCOMPATIBLE_PREPROCESSINGS: set[str] = {"trilinear"}


@dataclass(frozen=True)
class ModelMetadata:
    hidden_size: int
    patch_size: int
    num_register_tokens: int
    n_patches: int


# =============================================================================
# Dimensions and mode parsing
# =============================================================================


def get_feature_dim(model_size: str, extraction_mode: str) -> int:
    hidden_size = MODEL_CONFIGS[model_size]["hidden_size"]

    if extraction_mode == "mean_pool":
        return 2 * hidden_size

    if extraction_mode == "flatten":
        return (N_PATCHES + 1) * hidden_size

    raise ValueError(f"Unknown extraction_mode: {extraction_mode}")


def get_latent_dim(
    model_size: str,
    extraction_mode: str,
    aggregation_mode: str,
    slicer_mode: str,
) -> int:
    feature_dim = get_feature_dim(model_size, extraction_mode)

    if aggregation_mode == "mean_pool_axis":
        return 3 * feature_dim

    if aggregation_mode == "concat_all":
        return 3 * N_SLICES_PER_AXIS[slicer_mode] * feature_dim

    raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")


def parse_mode(mode: str) -> tuple[str, str, str, str, bool]:
    """
    Parse logical modes and cache-versioned modes.

    Supported forms:
      extraction__aggregation__slicer__model_size
      extraction__aggregation__slicer__model_size__dw
      extraction__aggregation__slicer__model_size__cache_regs4_v1
      extraction__aggregation__slicer__model_size__dw__cache_regs4_v1
    """
    parts = mode.split("__")

    if parts[-1].startswith("cache_"):
        cache_token = parts.pop()

        if cache_token != FEATURE_CACHE_TOKEN:
            raise ValueError(
                f"Unsupported cache token '{cache_token}'. "
                f"Expected '{FEATURE_CACHE_TOKEN}'."
            )

    density_weighting = False

    if parts[-1] == "dw":
        density_weighting = True
        parts.pop()

    if len(parts) != 4:
        raise ValueError(
            "Expected mode "
            "{extraction}__{aggregation}__{slicer}__{model_size}[__dw]"
            f"[__{FEATURE_CACHE_TOKEN}], got: {mode}"
        )

    extraction_mode, aggregation_mode, slicer_mode, model_size = parts

    checks = (
        ("extraction", extraction_mode, EXTRACTION_MODES),
        ("aggregation", aggregation_mode, AGGREGATION_MODES),
        ("slicer", slicer_mode, SLICER_MODES),
        ("model_size", model_size, MODEL_SIZES),
    )

    for name, value, valid_values in checks:
        if value not in valid_values:
            raise ValueError(
                f"Unknown {name} '{value}'. " f"Expected: {sorted(valid_values)}"
            )

    return (
        extraction_mode,
        aggregation_mode,
        slicer_mode,
        model_size,
        density_weighting,
    )


def add_feature_cache_version(mode: str) -> str:
    """
    Append the current cache version to a logical mode.

    Example:
      mean_pool__mean_pool_axis__2d__vit7b16
        ->
      mean_pool__mean_pool_axis__2d__vit7b16__cache_regs4_v1
    """
    if mode.endswith(f"__{FEATURE_CACHE_TOKEN}"):
        parse_mode(mode)
        return mode

    if any(part.startswith("cache_") for part in mode.split("__")):
        raise ValueError(f"Mode already contains another cache token: {mode}")

    parse_mode(mode)

    return f"{mode}__{FEATURE_CACHE_TOKEN}"


# =============================================================================
# Model loading and token-layout validation
# =============================================================================


def _normalize_patch_size(value: object) -> int:
    """
    Normalize a Hugging Face patch_size config value to a single integer.
    """
    if isinstance(value, int):
        return value

    if isinstance(value, (tuple, list)) and len(value) == 2:
        if int(value[0]) != int(value[1]):
            raise ValueError(f"Expected square patch size, got {value}")

        return int(value[0])

    raise TypeError(f"Unsupported patch_size value: {value!r}")


def _read_model_metadata(
    model: nn.Module,
    model_size: str,
) -> ModelMetadata:
    """
    Read and strictly validate the model architecture from model.config.

    Register-token handling is read from the checkpoint itself instead of
    relying on a manually maintained constant.
    """
    config = getattr(model, "config", None)

    if config is None:
        raise RuntimeError("Loaded model has no .config attribute.")

    required = (
        "hidden_size",
        "patch_size",
        "num_register_tokens",
    )

    missing = [name for name in required if not hasattr(config, name)]

    if missing:
        raise RuntimeError(
            f"model.config is missing {missing}. " "Refusing ambiguous token parsing."
        )

    hidden_size = int(config.hidden_size)
    patch_size = _normalize_patch_size(config.patch_size)
    num_register_tokens = int(config.num_register_tokens)

    expected_hidden_size = MODEL_CONFIGS[model_size]["hidden_size"]

    if hidden_size != expected_hidden_size:
        raise RuntimeError(
            f"Hidden-size mismatch for {model_size}: "
            f"got {hidden_size}, expected {expected_hidden_size}."
        )

    if patch_size != PATCH_SIZE:
        raise RuntimeError(
            f"Patch-size mismatch for {model_size}: "
            f"got {patch_size}, expected {PATCH_SIZE}."
        )

    if num_register_tokens != EXPECTED_NUM_REGISTER_TOKENS:
        raise RuntimeError(
            f"Register-token mismatch for {model_size}: "
            f"got {num_register_tokens}, "
            f"expected {EXPECTED_NUM_REGISTER_TOKENS}."
        )

    n_patches = (TARGET_SHAPE[1] // patch_size) * (TARGET_SHAPE[2] // patch_size)

    if n_patches != N_PATCHES:
        raise RuntimeError(
            f"Patch-count mismatch: " f"got {n_patches}, expected {N_PATCHES}."
        )

    return ModelMetadata(
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_register_tokens=num_register_tokens,
        n_patches=n_patches,
    )


def _build_model(
    checkpoint_path: str | Path,
    model_size: str,
    device: str,
) -> tuple[nn.Module, ModelMetadata]:
    """
    Load a local Hugging Face DINOv3 checkpoint and validate its architecture.
    """
    from transformers import AutoModel

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[DINOv3] Loading model from: {checkpoint_path}")

    model = AutoModel.from_pretrained(
        str(checkpoint_path),
        local_files_only=True,
    )

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    metadata = _read_model_metadata(
        model=model,
        model_size=model_size,
    )

    n_params = sum(parameter.numel() for parameter in model.parameters())

    print(f"[DINOv3] Model loaded — " f"{n_params:,} parameters (frozen)")

    print(
        "[DINOv3] Token layout validated — "
        f"1 CLS + {metadata.num_register_tokens} registers + "
        f"{metadata.n_patches} patches"
    )

    return model, metadata


# =============================================================================
# Forward and slice embeddings
# =============================================================================


@torch.no_grad()
def _forward_batch(
    model: nn.Module,
    slice_batch: torch.Tensor,
    metadata: ModelMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run DINOv3 and return:

      cls:
        [B, hidden_size]

      patches:
        [B, 196, hidden_size]

    Register tokens are explicitly excluded.
    """
    outputs = model(pixel_values=slice_batch)

    last_hidden = outputs.last_hidden_state

    expected_tokens = 1 + metadata.num_register_tokens + metadata.n_patches

    if last_hidden.ndim != 3:
        raise RuntimeError(
            f"Expected last_hidden_state [B,T,D], " f"got {tuple(last_hidden.shape)}"
        )

    if last_hidden.shape[1] != expected_tokens:
        raise RuntimeError(
            f"Unexpected sequence length "
            f"{last_hidden.shape[1]}; "
            f"expected {expected_tokens} = "
            f"1 CLS + "
            f"{metadata.num_register_tokens} registers + "
            f"{metadata.n_patches} patches."
        )

    if last_hidden.shape[2] != metadata.hidden_size:
        raise RuntimeError(
            f"Unexpected hidden size "
            f"{last_hidden.shape[2]}; "
            f"expected {metadata.hidden_size}."
        )

    cls = last_hidden[:, 0, :]

    patch_start = 1 + metadata.num_register_tokens
    patch_end = patch_start + metadata.n_patches

    patches = last_hidden[
        :,
        patch_start:patch_end,
        :,
    ]

    if patches.shape[1] != metadata.n_patches:
        raise RuntimeError(
            f"Extracted {patches.shape[1]} patch tokens; "
            f"expected {metadata.n_patches}."
        )

    return cls, patches


@torch.no_grad()
def _embed_slices(
    cls: torch.Tensor,
    patches: torch.Tensor,
    extraction_mode: str,
) -> torch.Tensor:
    """
    Build one representation per input slice.

    mean_pool:
      concat(CLS, mean of 196 patch tokens)

    flatten:
      concat(CLS, flattened 196 patch tokens)
    """
    if extraction_mode == "mean_pool":
        patch_mean = patches.mean(dim=1)

        return torch.cat(
            [cls, patch_mean],
            dim=-1,
        )

    if extraction_mode == "flatten":
        patch_flatten = patches.flatten(start_dim=1)

        return torch.cat(
            [cls, patch_flatten],
            dim=-1,
        )

    raise ValueError(f"Unknown extraction_mode: {extraction_mode}")


@torch.no_grad()
def _encode_axis(
    model: nn.Module,
    metadata: ModelMetadata,
    slices_axis: torch.Tensor,
    extraction_mode: str,
    device: str,
    slice_batch_size: int,
    v0: float,
    v1: float,
) -> torch.Tensor:
    """
    Encode all slices from a single anatomical axis.

    Input:
      slices_axis [N_slices, 3, 224, 224]

    Output:
      [N_slices, feature_dim] on CPU
    """
    if slice_batch_size <= 0:
        raise ValueError("slice_batch_size must be positive.")

    all_embeddings: list[torch.Tensor] = []
    n_slices = int(slices_axis.shape[0])

    for start in range(
        0,
        n_slices,
        slice_batch_size,
    ):
        batch = slices_axis[start : start + slice_batch_size]

        # Step 1:
        # affine intensity mapping, including normalizer-grid mappings.
        batch = normalize_affine(
            batch,
            v0=v0,
            v1=v1,
        )

        # Step 2:
        # ImageNet normalization expected by DINOv3.
        batch = normalize_imagenet(batch)

        batch = batch.contiguous().to(
            device,
            non_blocking=True,
        )

        cls, patches = _forward_batch(
            model=model,
            slice_batch=batch,
            metadata=metadata,
        )

        embeddings = _embed_slices(
            cls=cls,
            patches=patches,
            extraction_mode=extraction_mode,
        )

        all_embeddings.append(embeddings.float().cpu())

    encoded = torch.cat(
        all_embeddings,
        dim=0,
    )

    if extraction_mode == "mean_pool":
        expected_dim = 2 * metadata.hidden_size
    else:
        expected_dim = (metadata.n_patches + 1) * metadata.hidden_size

    expected_shape = (
        n_slices,
        expected_dim,
    )

    if encoded.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected axis feature shape "
            f"{tuple(encoded.shape)}; "
            f"expected {expected_shape}."
        )

    return encoded


# =============================================================================
# Density weighting
# =============================================================================


def _compute_density_weights(
    vol_preprocessed: np.ndarray,
    slicer_mode: str,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-slice density weights:

      w_i = d_i / d_max

    Density is calculated on the geometrically preprocessed volume before
    affine mapping and ImageNet normalization.
    """
    from linear_prober.skeleton.models.dinov3.slicers import (
        _25D_GROUP_SIZE,
        _25D_SLICE_START,
    )

    vol_binary = (vol_preprocessed > 0.0).astype(np.float32)

    result: Dict[str, torch.Tensor] = {}

    for axis in AXES:
        if axis == "D":
            planes = vol_binary

        elif axis == "H":
            planes = vol_binary.transpose(
                1,
                0,
                2,
            )

        else:
            planes = vol_binary.transpose(
                2,
                0,
                1,
            )

        if slicer_mode == "2d":
            densities = planes.mean(axis=(1, 2)).astype(np.float32)

        else:
            n_groups = N_SLICES_PER_AXIS["25d"]

            densities = np.empty(
                n_groups,
                dtype=np.float32,
            )

            for group_index in range(n_groups):
                start = _25D_SLICE_START + group_index * _25D_GROUP_SIZE

                stop = start + _25D_GROUP_SIZE

                densities[group_index] = planes[start:stop].mean()

        max_density = float(densities.max())

        if max_density > 1e-8:
            weights = densities / max_density
        else:
            weights = np.ones_like(densities)

        result[axis] = torch.from_numpy(
            weights.astype(
                np.float32,
                copy=False,
            )
        )

    return result


# =============================================================================
# Single-volume extraction
# =============================================================================


@torch.no_grad()
def _extract_volume(
    model: nn.Module,
    metadata: ModelMetadata,
    vol_preprocessed: np.ndarray,
    slicer_mode: str,
    extraction_mode: str,
    aggregation_mode: str,
    device: str,
    slice_batch_size: int,
    density_weighting: bool,
    v0: float,
    v1: float,
) -> torch.Tensor:
    """
    Extract one final DINOv3 representation from one preprocessed 3D volume.
    """
    from linear_prober.skeleton.models.dinov3.aggregator import aggregate

    if tuple(vol_preprocessed.shape) != TARGET_SHAPE:
        raise ValueError(
            f"Expected volume shape {TARGET_SHAPE}, "
            f"got {tuple(vol_preprocessed.shape)}."
        )

    volume_tensor = torch.from_numpy(
        np.ascontiguousarray(
            vol_preprocessed[None],
            dtype=np.float32,
        )
    )

    slices_by_axis = get_slices(
        volume=volume_tensor,
        slicer_mode=slicer_mode,
    )

    embeddings: Dict[str, torch.Tensor] = {}

    for axis in AXES:
        embeddings[axis] = _encode_axis(
            model=model,
            metadata=metadata,
            slices_axis=slices_by_axis[axis],
            extraction_mode=extraction_mode,
            device=device,
            slice_batch_size=slice_batch_size,
            v0=v0,
            v1=v1,
        )

    density_weights = None

    if density_weighting:
        density_weights = _compute_density_weights(
            vol_preprocessed=vol_preprocessed,
            slicer_mode=slicer_mode,
        )

    latent = aggregate(
        embeddings_dict=embeddings,
        aggregation_mode=aggregation_mode,
        density_weights=density_weights,
    )

    return latent.float().cpu()


# =============================================================================
# Public factory
# =============================================================================


def make_extract_fn(
    checkpoint_path: str | Path,
    preprocessing: str,
    v0: float,
    v1: float,
    slice_batch_size: int = 32,
    device: str = "cuda",
):
    """
    Return:

      extract_fn(
          checkpoint_path,
          hcp_dataloader,
          mode,
          device,
      )

    `mode` should be the versioned cache mode returned by:

      add_feature_cache_version(logical_mode)
    """
    if not (0.0 <= float(v0) < float(v1) <= 1.0):
        raise ValueError(f"Expected 0 <= v0 < v1 <= 1, " f"got v0={v0}, v1={v1}")

    if slice_batch_size <= 0:
        raise ValueError("slice_batch_size must be positive.")

    print(
        f"[DINOv3] make_extract_fn: "
        f"preprocessing={preprocessing}  "
        f"target_shape={TARGET_SHAPE}  "
        f"v0={v0}  "
        f"v1={v1}  "
        f"slice_batch_size={slice_batch_size}  "
        f"cache_version={FEATURE_CACHE_VERSION}"
    )

    model_cache: dict[
        tuple[str, str],
        tuple[nn.Module, ModelMetadata],
    ] = {}

    def extract_fn(
        ckpt_path,
        hcp_dataloader,
        mode,
        dev,
    ):
        (
            extraction_mode,
            aggregation_mode,
            slicer_mode,
            model_size,
            density_weighting,
        ) = parse_mode(mode)

        if density_weighting and preprocessing in _DW_INCOMPATIBLE_PREPROCESSINGS:
            raise ValueError(
                "density_weighting=True is incompatible with "
                f"preprocessing='{preprocessing}'. "
                "Use upscale_pad or nearest_neighbors."
            )

        checkpoint_dir = Path(ckpt_path) / model_size

        model_key = (
            str(checkpoint_dir.resolve()),
            str(dev),
        )

        if model_key not in model_cache:
            model_cache[model_key] = _build_model(
                checkpoint_path=checkpoint_dir,
                model_size=model_size,
                device=dev,
            )

        model, metadata = model_cache[model_key]

        if extraction_mode == "mean_pool":
            feature_dim = 2 * metadata.hidden_size
        else:
            feature_dim = (metadata.n_patches + 1) * metadata.hidden_size

        if aggregation_mode == "mean_pool_axis":
            latent_dim = 3 * feature_dim
        else:
            latent_dim = 3 * N_SLICES_PER_AXIS[slicer_mode] * feature_dim

        static_latent_dim = get_latent_dim(
            model_size=model_size,
            extraction_mode=extraction_mode,
            aggregation_mode=aggregation_mode,
            slicer_mode=slicer_mode,
        )

        if latent_dim != static_latent_dim:
            raise RuntimeError(
                "Static and runtime latent dimensions disagree: "
                f"runtime={latent_dim}, "
                f"static={static_latent_dim}."
            )

        print(
            f"[DINOv3] mode={mode}\n"
            f"         extraction={extraction_mode}  "
            f"aggregation={aggregation_mode}\n"
            f"         slicer={slicer_mode}  "
            f"model_size={model_size}\n"
            f"         density_weighting="
            f"{density_weighting}\n"
            f"         register_tokens_excluded="
            f"{metadata.num_register_tokens}\n"
            f"         patch_tokens="
            f"{metadata.n_patches}\n"
            f"         feat_dim_per_slice="
            f"{feature_dim}\n"
            f"         latent_dim="
            f"{latent_dim}\n"
            f"         feature_cache_version="
            f"{FEATURE_CACHE_VERSION}"
        )

        features_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        subjects_list: List[str] = []
        folds_list: List[int] = []
        splits_list: List[str] = []
        volume_indices_list: List[int] = []

        from tqdm import tqdm

        for batch in tqdm(
            hcp_dataloader,
            desc=f"[DINOv3] {mode}",
        ):
            volumes_raw = batch["volume"]

            volumes_preprocessed = preprocess_batch(
                volumes=volumes_raw,
                target_shape=TARGET_SHAPE,
                preprocessing=preprocessing,
            )

            labels_batch = batch["label"]
            subjects_batch = batch["subject"]
            folds_batch = batch["fold"]
            splits_batch = batch["split"]
            volume_indices_batch = batch["volume_index"]

            batch_size = int(volumes_raw.shape[0])

            for index in range(batch_size):
                volume = (
                    volumes_preprocessed[index, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                latent = _extract_volume(
                    model=model,
                    metadata=metadata,
                    vol_preprocessed=volume,
                    slicer_mode=slicer_mode,
                    extraction_mode=extraction_mode,
                    aggregation_mode=aggregation_mode,
                    device=dev,
                    slice_batch_size=slice_batch_size,
                    density_weighting=density_weighting,
                    v0=float(v0),
                    v1=float(v1),
                )

                if latent.shape != (latent_dim,):
                    raise RuntimeError(
                        f"Unexpected latent shape "
                        f"{tuple(latent.shape)}; "
                        f"expected {(latent_dim,)}."
                    )

                features_list.append(latent.numpy())

                labels_list.append(labels_batch[index].detach().cpu().numpy())

                subjects_list.append(str(subjects_batch[index]))

                folds_list.append(int(folds_batch[index]))

                splits_list.append(str(splits_batch[index]))

                volume_indices_list.append(int(volume_indices_batch[index]))

        if not features_list:
            raise RuntimeError("No DINOv3 features were extracted.")

        features = np.stack(
            features_list,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        labels = np.stack(
            labels_list,
            axis=0,
        )

        print(
            f"[DINOv3] Extracted "
            f"{len(features)} subjects — "
            f"latent_dim={features.shape[1]} — "
            f"cache_version="
            f"{FEATURE_CACHE_VERSION}"
        )

        return {
            "features": features,
            "labels": labels,
            "subjects": np.asarray(subjects_list),
            "folds": np.asarray(
                folds_list,
                dtype=np.int64,
            ),
            "splits": np.asarray(splits_list),
            "volume_indices": np.asarray(
                volume_indices_list,
                dtype=np.int64,
            ),
            # Metadata saved inside the .npz for traceability.
            "feature_cache_version": np.asarray(FEATURE_CACHE_VERSION),
            "num_register_tokens": np.asarray(
                metadata.num_register_tokens,
                dtype=np.int64,
            ),
            "n_patch_tokens": np.asarray(
                metadata.n_patches,
                dtype=np.int64,
            ),
            "hidden_size": np.asarray(
                metadata.hidden_size,
                dtype=np.int64,
            ),
        }

    return extract_fn
