"""Integration tests for the modality runners.

Feature extraction is monkeypatched to return synthetic features, so the full
runner wiring (task resolution -> evaluate_mode -> result serialisation) is
exercised without any model checkpoint or neuroimaging data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from linear_prober.mri import runner as mri_runner
from linear_prober.skeleton import runner as skeleton_runner


def _synthetic(task_name, n_per_split=60, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per_split
    if task_name == "regression":
        W = rng.normal(size=(n_features, 6))
        X = rng.normal(size=(2 * n, n_features))
        labels = (X @ W + rng.normal(scale=0.1, size=(2 * n, 6))).astype(np.float32)
    else:
        n_classes = 4 if task_name == "multiclass" else 2
        centers = rng.normal(scale=5.0, size=(n_classes, n_features))
        labels = rng.integers(0, n_classes, size=2 * n)
        X = centers[labels] + rng.normal(scale=0.5, size=(2 * n, n_features))
    return {
        "features": X.astype(np.float32),
        "labels": labels,
        "folds": np.tile(np.arange(5), int(np.ceil(2 * n / 5)))[: 2 * n].astype(
            np.int64
        ),
        "splits": np.array(["train_val"] * n + ["test"] * n),
        "subjects": np.array([str(i) for i in range(2 * n)]),
        "volume_indices": np.arange(2 * n),
    }


def _config(tmp_path):
    return {
        "experiment": {"model": "dino3d"},
        "paths": {"output_root": str(tmp_path)},
        "feature_extraction": {},
        "probe": {
            "n_folds": 5,
            "C": [0.1, 1.0, 10.0],
            "alpha": [1.0, 10.0],
            "flatten_raw_alpha": [1.0, 10.0],
            "n_components_list": [4],
        },
        "model_normalization": {"range": [-1.0, 1.0]},
        "rois": {},
    }


# =============================================================================
# Skeleton runner
# =============================================================================


@pytest.mark.parametrize(
    "roi,task", [("fip", "binary"), ("ofc", "multiclass"), ("sc", "regression")]
)
def test_skeleton_mean_pool_writes_summary(tmp_path, monkeypatch, roi, task):
    data = _synthetic(task)
    monkeypatch.setattr(skeleton_runner, "get_hcp_features", lambda *a, **k: data)

    skeleton_runner.run(_config(tmp_path), roi, "upscale_pad", mode="mean_pool")

    summary = (
        tmp_path
        / "dino3d"
        / roi
        / "upscale_pad"
        / "results"
        / "dino3d__mean_pool_summary.csv"
    )
    assert summary.is_file()
    row = pd.read_csv(summary).iloc[0]
    assert row["preprocessing"] == "upscale_pad"
    if task == "regression":
        assert row["test_mean_r2"] > 0.8
    else:
        assert row["test_score"] > 0.9


def test_skeleton_flatten_raw(tmp_path, monkeypatch):
    data = _synthetic("binary")
    monkeypatch.setattr(skeleton_runner, "get_hcp_features", lambda *a, **k: data)

    skeleton_runner.run(_config(tmp_path), "fip", "upscale_pad", flatten_raw=True)

    summary = (
        tmp_path
        / "dino3d"
        / "fip"
        / "upscale_pad"
        / "results"
        / "dino3d__flatten_raw_summary.csv"
    )
    assert summary.is_file()
    assert pd.read_csv(summary).iloc[0]["classifier"] == "ridge_classifier"


def test_skeleton_flatten_uses_pca(tmp_path, monkeypatch):
    data = _synthetic("binary")
    monkeypatch.setattr(skeleton_runner, "get_hcp_features", lambda *a, **k: data)
    # Bypass the on-disk PCA: identity projection.
    monkeypatch.setattr(skeleton_runner, "load_pca", lambda *a, **k: "pca_stub")
    monkeypatch.setattr(skeleton_runner, "apply_pca", lambda feats, pca: feats)

    skeleton_runner.run(_config(tmp_path), "fip", "upscale_pad", mode="flatten")

    summary = (
        tmp_path
        / "dino3d"
        / "fip"
        / "upscale_pad"
        / "results"
        / "dino3d__flatten_n4_summary.csv"
    )
    assert summary.is_file()
    assert pd.read_csv(summary).iloc[0]["n_components"] == 4


# =============================================================================
# MRI runner
# =============================================================================


@pytest.mark.parametrize(
    "roi,task", [("fip", "binary"), ("ofc", "multiclass"), ("sc", "regression")]
)
def test_mri_all_modes_write_summaries(tmp_path, monkeypatch, roi, task):
    data = _synthetic(task)
    monkeypatch.setattr(mri_runner, "get_features", lambda *a, **k: data)

    mri_runner.run(_config(tmp_path), roi)

    results_dir = tmp_path / "dino3d" / roi / "results"
    for mode in ("mean_pool", "mean_pool_multi_layers", "flatten"):
        assert (results_dir / f"dino3d__{mode}_summary.csv").is_file()
    # MRI results carry no preprocessing column.
    row = pd.read_csv(results_dir / "dino3d__mean_pool_summary.csv").iloc[0]
    assert "preprocessing" not in row.index
