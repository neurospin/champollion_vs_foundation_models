"""Tests for the DINOv3 runners and probe selection.

Feature extraction is monkeypatched to synthetic features; no checkpoint or
neuroimaging data is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from linear_prober.core.dinov3 import build_mode, select_probe
from linear_prober.core.tasks import resolve_task
from linear_prober.mri import runner_dinov3 as mri_runner
from linear_prober.skeleton import runner_dinov3 as skeleton_runner


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
        "experiment": {"model": "dinov3"},
        "paths": {"output_root": str(tmp_path)},
        "feature_extraction": {
            "device": "cpu",
            "batch_size": 8,
            "slice_batch_size": 16,
            "num_workers": 0,
        },
        "probe": {
            "n_folds": 5,
            "logreg": {"C": [0.1, 1.0, 10.0]},
            "ridgeclassifier": {"alpha": [1.0, 10.0]},
            "ridge": {"alpha": [1.0, 10.0]},
        },
        "model_normalization": {"range": [0.0, 1.0]},
        "optimal_mapping": {},
        "rois": {},
    }


# =============================================================================
# select_probe / build_mode
# =============================================================================


def test_build_mode_composes_and_appends_dw():
    assert (
        build_mode("mean_pool", "mean_pool_axis", "2d", "vitb16")
        == "mean_pool__mean_pool_axis__2d__vitb16"
    )
    assert build_mode("flatten", "concat_all", "25d", "vitl16", True).endswith("__dw")


def test_select_probe_switches_on_extraction_aggregation():
    cfg = {
        "probe": {
            "logreg": {"C": [1]},
            "ridgeclassifier": {"alpha": [1]},
            "ridge": {"alpha": [1]},
        }
    }
    task = resolve_task("ofc")
    _, _, _, label = select_probe(task, "mean_pool", "mean_pool_axis", cfg)
    assert label == "logreg"
    _, _, _, label = select_probe(task, "flatten", "concat_all", cfg)
    assert label == "ridgeclassifier"
    _, _, _, label = select_probe(
        resolve_task("sc"), "mean_pool", "mean_pool_axis", cfg
    )
    assert label == "ridge"


# =============================================================================
# Skeleton DINOv3 runner
# =============================================================================


def test_skeleton_dinov3_logreg_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        skeleton_runner,
        "get_hcp_features_dinov3",
        lambda *a, **k: _synthetic("multiclass"),
    )
    skeleton_runner.run(
        _config(tmp_path),
        "ofc",
        "upscale_pad",
        model_size="vitb16",
        slicer_mode="2d",
        extraction="mean_pool",
        aggregation="mean_pool_axis",
    )
    mode = "mean_pool__mean_pool_axis__2d__vitb16"
    summary = (
        tmp_path
        / "dinov3"
        / "ofc"
        / "upscale_pad"
        / "results"
        / f"dinov3__{mode}_summary.csv"
    )
    assert summary.is_file()
    row = pd.read_csv(summary).iloc[0]
    assert row["classifier"] == "logreg" and row["model_size"] == "vitb16"
    assert row["test_score"] > 0.9


def test_skeleton_dinov3_regression(tmp_path, monkeypatch):
    monkeypatch.setattr(
        skeleton_runner,
        "get_hcp_features_dinov3",
        lambda *a, **k: _synthetic("regression"),
    )
    skeleton_runner.run(
        _config(tmp_path),
        "sc",
        "upscale_pad",
        model_size="vitl16",
        slicer_mode="25d",
        extraction="flatten",
        aggregation="concat_all",
    )
    mode = "flatten__concat_all__25d__vitl16"
    summary = (
        tmp_path
        / "dinov3"
        / "sc"
        / "upscale_pad"
        / "results"
        / f"dinov3__{mode}_summary.csv"
    )
    assert summary.is_file()
    assert pd.read_csv(summary).iloc[0]["test_mean_r2"] > 0.8


def test_skeleton_dinov3_density_weighting_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(
        skeleton_runner, "get_hcp_features_dinov3", lambda *a, **k: _synthetic("binary")
    )
    skeleton_runner.run(
        _config(tmp_path),
        "fip",
        "upscale_pad",
        model_size="vitb16",
        slicer_mode="2d",
        extraction="mean_pool",
        aggregation="mean_pool_axis",
        density_weighting=True,
    )
    mode = "mean_pool__mean_pool_axis__2d__vitb16__dw"
    assert (
        tmp_path
        / "dinov3"
        / "fip"
        / "upscale_pad"
        / "results"
        / f"dinov3__{mode}_summary.csv"
    ).is_file()


def test_skeleton_dinov3_dw_incompatible_with_trilinear(tmp_path, monkeypatch):
    monkeypatch.setattr(
        skeleton_runner, "get_hcp_features_dinov3", lambda *a, **k: _synthetic("binary")
    )
    with pytest.raises(ValueError):
        skeleton_runner.run(
            _config(tmp_path),
            "fip",
            "trilinear",
            model_size="vitb16",
            slicer_mode="2d",
            extraction="mean_pool",
            aggregation="mean_pool_axis",
            density_weighting=True,
        )


# =============================================================================
# MRI DINOv3 runner
# =============================================================================


def test_mri_dinov3_no_preprocessing_column(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mri_runner, "get_features", lambda *a, **k: _synthetic("binary")
    )
    mri_runner.run(
        _config(tmp_path),
        "fip",
        model_size="vitl16",
        slicer_mode="25d",
        extraction="mean_pool",
        aggregation="mean_pool_axis",
    )
    mode = "mean_pool__mean_pool_axis__25d__vitl16"
    summary = tmp_path / "dinov3" / "fip" / "results" / f"dinov3__{mode}_summary.csv"
    assert summary.is_file()
    row = pd.read_csv(summary).iloc[0]
    assert "preprocessing" not in row.index and row["classifier"] == "logreg"
