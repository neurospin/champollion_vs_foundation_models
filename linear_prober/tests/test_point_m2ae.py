"""Tests for the Point-M2AE point-cloud path.

The voxel-to-cloud conversion is exercised directly (pure numpy/scipy); the
runner is exercised on synthetic features with extraction monkeypatched, so no
checkpoint, upstream repository or GPU is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from linear_prober.skeleton import runner_point_m2ae
from linear_prober.skeleton.models.point_m2ae.extract_features import build_mode
from linear_prober.skeleton.models.point_m2ae.pointcloud import volume_to_point_cloud

# =============================================================================
# volume -> point cloud
# =============================================================================


def _volume(shape=(30, 38, 22), n_active=50, seed=0):
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, dtype=np.float32)
    flat = rng.choice(vol.size, size=n_active, replace=False)
    vol.ravel()[flat] = 1.0
    return vol


def test_conversion_yields_one_point_per_active_voxel():
    pts = volume_to_point_cloud(_volume(), upsample=1.0)
    assert pts.shape == (50, 3)
    assert pts.dtype == np.float32


def test_conversion_is_normalised_to_unit_range():
    pts = volume_to_point_cloud(_volume())
    assert np.abs(pts).max() <= 1.0


def test_conversion_squeezes_singleton_channel():
    vol = _volume()
    assert volume_to_point_cloud(vol[None]).shape == volume_to_point_cloud(vol).shape


def test_conversion_rejects_empty_volume():
    with pytest.raises(ValueError, match="active voxel"):
        volume_to_point_cloud(np.zeros((8, 8, 8), dtype=np.float32))


def test_upsampling_increases_point_count():
    vol = _volume()
    n_native = len(volume_to_point_cloud(vol, upsample=1.0))
    n_up = len(volume_to_point_cloud(vol, upsample=2.0))
    assert n_up > n_native


def test_fixed_centre_preserves_anatomical_position():
    # The same shape at two positions must yield different clouds: the centre
    # is the volume centre, not the cloud centroid.
    a = np.zeros((16, 16, 16), dtype=np.float32)
    b = np.zeros((16, 16, 16), dtype=np.float32)
    a[2:5, 2:5, 2:5] = 1.0
    b[10:13, 10:13, 10:13] = 1.0
    pa = volume_to_point_cloud(a)
    pb = volume_to_point_cloud(b)
    assert pa.shape == pb.shape
    assert not np.allclose(np.sort(pa, axis=0), np.sort(pb, axis=0))


# =============================================================================
# composite mode string
# =============================================================================


def test_build_mode_is_self_describing():
    assert build_mode("standard", "mean", 1.0) == "standard__mean__up1"
    assert build_mode("wide", "multi_level", 1.5) == "wide__multi_level__up1.5"


def test_build_mode_rejects_unknown_axes():
    with pytest.raises(ValueError, match="grouping"):
        build_mode("huge", "mean", 1.0)
    with pytest.raises(ValueError, match="aggregation"):
        build_mode("standard", "median", 1.0)
    with pytest.raises(ValueError, match="upsample"):
        build_mode("standard", "mean", 3.0)


# =============================================================================
# runner (synthetic features, extraction monkeypatched)
# =============================================================================


def _synthetic(task_name, n_per_split=60, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per_split
    if task_name == "regression":
        W = rng.normal(size=(n_features, 6))
        X = rng.normal(size=(2 * n, n_features))
        labels = (X @ W + rng.normal(scale=0.1, size=(2 * n, 6))).astype(np.float32)
    else:
        centers = rng.normal(scale=5.0, size=(2, n_features))
        labels = rng.integers(0, 2, size=2 * n)
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
        "experiment": {"model": "point_m2ae"},
        "paths": {"output_root": str(tmp_path)},
        "probe": {
            "n_folds": 5,
            "classifier": "logreg",
            "C": [0.1, 1.0],
            "alpha": [1.0, 10.0],
        },
        "rois": {},
    }


@pytest.mark.parametrize("roi,task", [("fip", "binary"), ("sc", "regression")])
def test_runner_writes_summary(tmp_path, monkeypatch, roi, task):
    data = _synthetic(task)
    monkeypatch.setattr(
        runner_point_m2ae, "get_hcp_features_point_m2ae", lambda *a, **k: data
    )

    runner_point_m2ae.run(_config(tmp_path), roi, "standard", "mean", 1.0)

    summary = (
        tmp_path
        / "point_m2ae"
        / roi
        / "results"
        / "point_m2ae__standard__mean__up1_summary.csv"
    )
    assert summary.exists()
