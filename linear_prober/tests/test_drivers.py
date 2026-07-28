"""Tests for the auxiliary skeleton drivers (PCA fit, normaliser search).

The model encoder and data loading are monkeypatched, so no checkpoint or
neuroimaging data is required.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

import linear_prober.skeleton.normalizer_search as ns


def _fake_table(n=40, regression=False):
    t = pd.DataFrame(
        {
            "volume_index": np.arange(n),
            "subject": [f"s{i}" for i in range(n)],
            "fold": np.tile(np.arange(5), n // 5),
            "split": ["train_val"] * (n - 10) + ["test"] * 10,
        }
    )
    if regression:
        for d in range(6):
            t[f"label_{d}"] = np.random.default_rng(d).normal(size=n)
    else:
        t["label"] = np.tile([0, 1], n // 2)
    return t


class _FakeDataset:
    def __init__(self, *a, task_type="classification", **k):
        self.volumes = np.zeros((40, 4, 4, 4), dtype=np.uint8)
        self.table = _fake_table(regression=(task_type == "regression"))


def _fake_module():
    m = types.SimpleNamespace()
    m._build_model = lambda ckpt, device: "encoder"
    m._add_repo_to_path = lambda p: None

    def extract_for_mapping(
        encoder,
        volumes_raw,
        preprocessing,
        target_shape,
        v0,
        v1,
        device,
        batch_size,
        is_binary_preserving,
    ):
        # Synthetic features; a slight v0-dependent shift so mappings differ.
        rng = np.random.default_rng(int(abs(v0) * 100))
        return rng.normal(size=(len(volumes_raw), 8)).astype(np.float32)

    m.extract_mean_pool_for_mapping = extract_for_mapping
    return m


def _config(tmp_path):
    return {
        "experiment": {"model": "dino3d"},
        "paths": {"output_root": str(tmp_path)},
        "feature_extraction": {
            "device": "cpu",
            "batch_size": 4,
            "target_shape": [4, 4, 4],
        },
        "probe": {"n_folds": 5, "normalizer_search_C": [0.1, 1.0]},
        "model_normalization": {"range": [-1.0, 1.0]},
        "normalizer_search": {
            "grid_step": 0.5,
            "top_k": 2,
            "checkpoints": [
                {
                    "name": "zero_shot",
                    "checkpoint_path": "/x.pth",
                    "output_model_name": "dino3d_zero_shot",
                }
            ],
        },
        "rois": {"ofc": {"hcp_volumes_native": "x.npy", "hcp_master_table": "x.csv"}},
    }


@pytest.mark.parametrize("roi", ["fip"])
def test_normalizer_search_writes_grid_and_best(tmp_path, monkeypatch, roi):
    monkeypatch.setattr(ns, "HCPDataset", _FakeDataset)
    monkeypatch.setattr(ns.importlib, "import_module", lambda name: _fake_module())

    cfg = _config(tmp_path)
    cfg["rois"] = {roi: {"hcp_volumes_native": "x.npy", "hcp_master_table": "x.csv"}}
    ns.search(cfg, roi, "upscale_pad")

    base = tmp_path / "dino3d_zero_shot" / roi / "upscale_pad" / "normalizer_search"
    grid = pd.read_csv(base / "grid_results.csv")
    best = pd.read_csv(base / "best_mapping.csv")
    # grid_step=0.5 -> 3 valid (p0<p1) couples; best keeps top_k=2.
    assert len(grid) == 3 and len(best) == 2
    # grid is sorted by descending CV score.
    assert grid["cv_mean_score"].is_monotonic_decreasing


def test_fit_pca_script_imports():
    import importlib

    mod = importlib.import_module("linear_prober.skeleton.pca")
    assert hasattr(mod, "fit_ukbb_pca") and hasattr(mod, "build_ukbb_feature_path")
