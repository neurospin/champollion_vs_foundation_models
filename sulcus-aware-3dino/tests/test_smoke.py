# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

"""Synthetic smoke test for the self-contained sulcal numerics.

Exercises the three model-agnostic building blocks of the sulcal data pipeline —
density-aware masking, geometric preprocessing, and the multi-crop collate — on
random tensors, with NO neuroimaging data, NO GPU and NO external 3DINO clone.

These three modules depend only on numpy/torch (the mask generator is *injected*
into the collate), so the test loads them in isolation rather than through the
package's single-import surface ``sulcus_aware_3DINO.data`` — whose ``__init__``
also pulls MONAI and the upstream ``dinov2`` package. Syntactic validity of the
whole package is covered separately by the CI byte-compile step.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_DATA = Path(__file__).resolve().parents[1] / "sulcus_aware_3DINO" / "data"


def _load(module_filename: str):
    """Load one standalone data module by file path (bypasses the package __init__)."""
    name = module_filename.replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, _DATA / module_filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


masking = _load("masking_non_empty.py")
preprocessing = _load("sulcal_preprocessing.py")
collate = _load("collate.py")


def _gen():
    # 7**3 = 343 patches for a 112**3 volume with patch_size 16.
    return masking.MaskingGenerator3d(input_size=7, patch_size=16)


# --------------------------------------------------------------------------- #
# Density-aware masking: only patches with >=1 active voxel are maskable.
# --------------------------------------------------------------------------- #
def test_masking_restricted_to_active_patches():
    vol = np.zeros((1, 112, 112, 112), dtype=np.float32)
    vol[0, 0:16, 0:16, 0:16] = 1.0   # active patch, flat index 0
    vol[0, 0:16, 16:32, 0:16] = 1.0  # active patch, flat index 7
    active = {0, 7}
    mask = _gen()(num_masking_patches=300, volume=torch.from_numpy(vol))
    masked = set(np.flatnonzero(mask.ravel()).tolist())
    assert masked, "expected at least one masked patch"
    assert masked.issubset(active), f"masked background patches: {masked - active}"


def test_masking_empty_volume_yields_empty_mask():
    empty = torch.zeros((1, 112, 112, 112))
    assert _gen()(num_masking_patches=300, volume=empty).sum() == 0


def test_masking_uniform_mode_masks_exact_count():
    assert _gen()(num_masking_patches=50, volume=None).sum() == 50


# --------------------------------------------------------------------------- #
# Geometric preprocessing: heterogeneous native shapes -> fixed binary cube.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(39, 45, 44), (30, 38, 22)])
def test_preprocessing_upscales_to_cube_and_stays_binary(shape):
    rng = np.random.default_rng(0)
    raw = (rng.random(shape) > 0.7).astype(np.float32)
    out = preprocessing.preprocess_sulcal_volume(raw, target=112, binarize_nonzero=True)
    assert tuple(out.shape) == (1, 112, 112, 112)
    assert set(torch.unique(out).tolist()).issubset({0.0, 1.0})


# --------------------------------------------------------------------------- #
# Multi-crop collate: builds the interface dict consumed by forward_backward.
# --------------------------------------------------------------------------- #
def _sample(rng):
    g = [
        torch.from_numpy((rng.random((1, 112, 112, 112)) > 0.9).astype("float32"))
        for _ in range(2)
    ]
    loc = [
        torch.from_numpy((rng.random((1, 64, 64, 64)) > 0.9).astype("float32"))
        for _ in range(8)
    ]
    return (
        {"global_crops": g, "global_crops_teacher": g, "local_crops": loc, "offsets": ()},
        None,
    )


def test_collate_produces_interface_keys_and_is_crop_major():
    rng = np.random.default_rng(1)
    samples = [_sample(rng) for _ in range(2)]  # batch of 2
    out = collate.collate_data_and_cast(
        samples,
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        dtype=torch.float32,
        n_tokens=343,
        mask_generator=_gen(),
        use_density_masking=True,
    )
    assert set(out) == {
        "collated_global_crops",
        "collated_local_crops",
        "collated_masks",
        "mask_indices_list",
        "masks_weight",
        "upperbound",
        "n_masked_patches",
    }
    # crop-major layout: 2 global crops x batch 2 = 4 rows, all crop-0 first.
    assert tuple(out["collated_global_crops"].shape) == (4, 1, 112, 112, 112)
    assert tuple(out["collated_masks"].shape) == (4, 343)
    assert torch.equal(out["collated_global_crops"][0], samples[0][0]["global_crops"][0])
    assert torch.equal(out["collated_global_crops"][1], samples[1][0]["global_crops"][0])
    assert int(out["n_masked_patches"][0]) == out["mask_indices_list"].numel()
