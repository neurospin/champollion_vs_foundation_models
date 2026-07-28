"""
models/dinov3/aggregator.py

Inter-slice aggregation for the DINOv3 zero-shot probing pipeline.
Identical to the skeleton project version — fully agnostic to slice size.

Two aggregation modes
=====================

  "mean_pool_axis":
    Average embeddings across all slices within each axis → one vector per axis.
    Concatenate the 3 axis vectors → [3F].

  "concat_all":
    Concatenate all slice embeddings across all 3 axes in fixed order (D, H, W).
    Flatten → [N_slices × 3 × F].
      2D  mode (N=112): 112 × 3 × F = 336F
      2.5D mode (N=37):  37 × 3 × F = 111F
"""

from __future__ import annotations

import torch

AGGREGATION_MODES: set[str] = {"mean_pool_axis", "concat_all"}

_AXES: tuple[str, ...] = ("D", "H", "W")


def _mean_pool_axis(embeddings_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    axis_vecs = [embeddings_dict[ax].mean(dim=0) for ax in _AXES]
    return torch.cat(axis_vecs, dim=0)  # [3F]


def _concat_all(embeddings_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    all_embs = [embeddings_dict[ax] for ax in _AXES]
    stacked = torch.cat(all_embs, dim=0)  # [3N, F]
    return stacked.flatten()  # [3NF]


def _validate(embeddings_dict: dict[str, torch.Tensor]) -> None:
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


def aggregate(
    embeddings_dict: dict[str, torch.Tensor],
    aggregation_mode: str,
) -> torch.Tensor:
    """
    Aggregate per-axis slice embeddings into a single latent vector.

    Arguments:
      embeddings_dict  : {"D": [N_slices, F], "H": [N_slices, F], "W": [N_slices, F]}
      aggregation_mode : "mean_pool_axis" → [3F]
                         "concat_all"     → [N_slices × 3 × F]
    """
    if aggregation_mode not in AGGREGATION_MODES:
        raise ValueError(
            f"Unknown aggregation_mode '{aggregation_mode}'. "
            f"Expected one of: {sorted(AGGREGATION_MODES)}"
        )
    _validate(embeddings_dict)

    if aggregation_mode == "mean_pool_axis":
        return _mean_pool_axis(embeddings_dict)
    else:
        return _concat_all(embeddings_dict)
