"""Unit tests for the skeleton modality (preprocessing + datasets).

Synthetic volumes and master tables only — no real neuroimaging data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from linear_prober.skeleton.dataset import HCPDataset, UKBBSkeletonDataset
from linear_prober.skeleton.preprocessor import ALL_PREPROCESSINGS, preprocess_batch

# =============================================================================
# preprocessor
# =============================================================================


def test_upscale_pad_reaches_cubic_target():
    vol = torch.rand(2, 1, 39, 45, 44)
    out = preprocess_batch(vol, (112, 112, 112), "upscale_pad")
    assert tuple(out.shape) == (2, 1, 112, 112, 112)


def test_nearest_neighbors_preserves_binary():
    vol = (torch.rand(1, 1, 30, 40, 50) > 0.5).float()
    out = preprocess_batch(vol, (112, 112, 112), "nearest_neighbors")
    assert tuple(out.shape) == (1, 1, 112, 112, 112)
    assert set(torch.unique(out).tolist()).issubset({0.0, 1.0})


def test_trilinear_target_shape():
    vol = torch.rand(1, 1, 30, 40, 50)
    out = preprocess_batch(vol, (112, 112, 112), "trilinear")
    assert tuple(out.shape) == (1, 1, 112, 112, 112)


def test_offline_preprocessings_removed():
    """Offline modes (skel1vox / distance_map) must no longer exist."""
    assert ALL_PREPROCESSINGS == {"upscale_pad", "nearest_neighbors", "trilinear"}
    for gone in ("upscale_pad_skel1vox", "distance_map"):
        with pytest.raises(ValueError):
            preprocess_batch(torch.rand(1, 1, 30, 40, 50), (112, 112, 112), gone)


def test_non_cubic_target_rejected():
    with pytest.raises(ValueError):
        preprocess_batch(torch.rand(1, 1, 30, 40, 50), (112, 96, 112), "upscale_pad")


def test_unknown_preprocessing_rejected():
    with pytest.raises(ValueError):
        preprocess_batch(torch.rand(1, 1, 30, 40, 50), (112, 112, 112), "bogus")


def test_neurovfm_fully_removed():
    """neurovfm_pad must not exist anywhere in the skeleton preprocessor."""
    assert not any("neurovfm" in p for p in ALL_PREPROCESSINGS)
    with pytest.raises(ValueError):
        preprocess_batch(torch.rand(1, 1, 30, 40, 50), (112, 112, 112), "neurovfm_pad")


# =============================================================================
# datasets
# =============================================================================


def _write_skeleton_data(
    tmp_path, n=10, shape=(8, 9, 10), regression=False, multilabel=True
):
    """Write a synthetic volumes .npy + master_table .csv, return their paths."""
    rng = np.random.default_rng(0)
    if multilabel:
        vals = np.array([0, 30, 60, 80], dtype=np.int16)
        vols = vals[rng.integers(0, len(vals), size=(n, *shape))].astype(np.int16)
    else:
        vols = (rng.random((n, *shape)) > 0.5).astype(np.uint8)
    vpath = tmp_path / "vols.npy"
    np.save(vpath, vols)

    table = pd.DataFrame(
        {
            "volume_index": np.arange(n),
            "subject": [f"s{i}" for i in range(n)],
            "fold": np.tile(np.arange(5), int(np.ceil(n / 5)))[:n],
            "split": ["train_val"] * (n - 2) + ["test"] * 2,
        }
    )
    if regression:
        for d in range(6):
            table[f"label_{d}"] = rng.normal(size=n)
    else:
        table["label"] = rng.integers(0, 2, size=n)
    tpath = tmp_path / "master.csv"
    table.to_csv(tpath, index=False)
    return str(vpath), str(tpath)


def test_hcp_classification_binarises(tmp_path):
    vpath, tpath = _write_skeleton_data(tmp_path, multilabel=True)
    ds = HCPDataset(vpath, tpath, task_type="classification")
    item = ds[0]
    assert item["volume"].shape[0] == 1
    assert set(torch.unique(item["volume"]).tolist()).issubset({0.0, 1.0})
    assert isinstance(item["label"], int)


def test_hcp_regression_returns_vector(tmp_path):
    vpath, tpath = _write_skeleton_data(tmp_path, regression=True)
    ds = HCPDataset(vpath, tpath, task_type="regression")
    item = ds[0]
    assert item["label"].shape == (6,)
    assert item["label"].dtype == torch.float32


def test_hcp_split_filter(tmp_path):
    vpath, tpath = _write_skeleton_data(tmp_path, n=10)
    ds_test = HCPDataset(vpath, tpath, split="test")
    assert len(ds_test) == 2
    assert all(ds_test[i]["split"] == "test" for i in range(len(ds_test)))


def test_ukbb_dataset_binarises_and_squeezes(tmp_path):
    rng = np.random.default_rng(1)
    vols = (rng.integers(0, 90, size=(6, 8, 8, 8, 1))).astype(np.int16)  # trailing dim
    vpath = tmp_path / "ukbb.npy"
    np.save(vpath, vols)
    ds = UKBBSkeletonDataset(str(vpath))
    item = ds[0]
    assert item["volume"].shape == (1, 8, 8, 8)
    assert set(torch.unique(item["volume"]).tolist()).issubset({0.0, 1.0})
