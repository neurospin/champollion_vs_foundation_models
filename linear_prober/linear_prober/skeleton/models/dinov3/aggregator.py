"""
models/dinov3/aggregator.py

Inter-slice aggregation for the DINOv3 zero-shot probing pipeline.

Two aggregation modes
=====================

  "mean_pool_axis":
    Average embeddings across all slices within each axis → one vector per axis.
    Concatenate the 3 axis vectors → [3F].

    With density weighting:
      z_axis = sum(w_i * e_i) / sum(w_i)   (weighted mean, normalized)

  "concat_all":
    Concatenate all slice embeddings across all 3 axes in fixed order (D, H, W).
    Flatten → [N_slices × 3 × F].

    With density weighting:
      Each embedding is scaled: e_i* = w_i * e_i  before concatenation.

Density weights
===============
  density_weights: dict {"D": [N_slices], "H": [N_slices], "W": [N_slices]}
  w_i = d_i / d_max  where d_i = fraction of active voxels in slice i
  w_i in [0, 1], max weight = 1  → signal is never crushed to near-zero.
  If density_weights is None → standard unweighted aggregation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

AGGREGATION_MODES: set[str] = {"mean_pool_axis", "concat_all"}

_AXES: tuple[str, ...] = ("D", "H", "W")


# =============================================================================
# Internal helpers
# =============================================================================


def _mean_pool_axis(
    embeddings_dict: Dict[str, torch.Tensor],
    density_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Weighted or unweighted mean across slices per axis, then concat.

    embeddings_dict: {"D": [N, F], "H": [N, F], "W": [N, F]}
    density_weights: {"D": [N], "H": [N], "W": [N]}  or None

    Returns: [3F]
    """
    axis_vecs = []
    for ax in _AXES:
        emb = embeddings_dict[ax]  # [N, F]
        if density_weights is not None:
            w = density_weights[ax].to(emb.device)  # [N]
            w_sum = w.sum().clamp(min=1e-8)
            vec = (w.unsqueeze(1) * emb).sum(dim=0) / w_sum  # [F]
        else:
            vec = emb.mean(dim=0)  # [F]
        axis_vecs.append(vec)
    return torch.cat(axis_vecs, dim=0)  # [3F]


def _concat_all(
    embeddings_dict: Dict[str, torch.Tensor],
    density_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Scale each embedding by its density weight (if provided), then concat.

    embeddings_dict: {"D": [N, F], "H": [N, F], "W": [N, F]}
    density_weights: {"D": [N], "H": [N], "W": [N]}  or None

    Returns: [3*N*F]
    """
    all_embs = []
    for ax in _AXES:
        emb = embeddings_dict[ax]  # [N, F]
        if density_weights is not None:
            w = density_weights[ax].to(emb.device)  # [N]
            emb = w.unsqueeze(1) * emb  # [N, F]
        all_embs.append(emb)
    stacked = torch.cat(all_embs, dim=0)  # [3N, F]
    return stacked.flatten()  # [3NF]


def _validate(embeddings_dict: Dict[str, torch.Tensor]) -> None:
    missing = [ax for ax in _AXES if ax not in embeddings_dict]
    if missing:
        raise ValueError(f"embeddings_dict is missing keys: {missing}")
    shapes = {ax: tuple(embeddings_dict[ax].shape) for ax in _AXES}
    for ax, shape in shapes.items():
        if len(shape) != 2:
            raise ValueError(
                f"embeddings_dict['{ax}'] must be 2D [N_slices, F], got {shape}"
            )
    unique_shapes = set(shapes.values())
    if len(unique_shapes) > 1:
        raise ValueError(f"All axes must have the same shape, got: {shapes}")


# =============================================================================
# Public API
# =============================================================================


def aggregate(
    embeddings_dict: Dict[str, torch.Tensor],
    aggregation_mode: str,
    density_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Aggregate per-axis slice embeddings into a single latent vector.

    Arguments:
      embeddings_dict  : {"D": [N_slices, F], "H": [N_slices, F], "W": [N_slices, F]}
      aggregation_mode : "mean_pool_axis" → [3F]
                         "concat_all"     → [N_slices × 3 × F]
      density_weights  : {"D": [N_slices], "H": [N_slices], "W": [N_slices]}
                         w_i = d_i / d_max in [0, 1]. None → uniform weights.
    """
    if aggregation_mode not in AGGREGATION_MODES:
        raise ValueError(
            f"Unknown aggregation_mode '{aggregation_mode}'. "
            f"Expected one of: {sorted(AGGREGATION_MODES)}"
        )
    _validate(embeddings_dict)

    if aggregation_mode == "mean_pool_axis":
        return _mean_pool_axis(embeddings_dict, density_weights)
    else:
        return _concat_all(embeddings_dict, density_weights)


# Note: compute_density_weights is defined in extract_features.py
# (it needs access to the preprocessed 224³ volume and slicer constants)
