"""Unit tests for the modality-agnostic core engine.

These tests use small synthetic feature arrays only — no real neuroimaging data
is required. They validate the cross-validation protocol, the task registry,
path construction and the feature cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from linear_prober.core import (
    build_feature_path,
    build_results_paths,
    load_or_extract,
    resolve_task,
    run_cv_grid,
    split_tv_test,
)
from linear_prober.core.cross_validation import evaluate_test
from linear_prober.core.metrics import softmax
from linear_prober.core.tasks import ROI_TASK

# =============================================================================
# Fixtures — synthetic linearly-separable data
# =============================================================================


def _make_classification(n_per_split=60, n_features=8, n_classes=2, seed=0):
    """Linearly-separable features with pre-stratified folds and splits."""
    rng = np.random.default_rng(seed)
    n = n_per_split
    centers = rng.normal(scale=5.0, size=(n_classes, n_features))

    labels = rng.integers(0, n_classes, size=2 * n)
    features = centers[labels] + rng.normal(scale=0.5, size=(2 * n, n_features))

    splits = np.array(["train_val"] * n + ["test"] * n)
    folds = np.tile(np.arange(5), int(np.ceil(2 * n / 5)))[: 2 * n]

    return {
        "features": features.astype(np.float32),
        "labels": labels.astype(np.int64),
        "folds": folds.astype(np.int64),
        "splits": splits,
        "subjects": np.array([str(i) for i in range(2 * n)]),
        "volume_indices": np.arange(2 * n),
    }


def _make_regression(n_per_split=60, n_features=8, n_dims=6, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per_split
    W = rng.normal(size=(n_features, n_dims))
    X = rng.normal(size=(2 * n, n_features))
    Y = X @ W + rng.normal(scale=0.1, size=(2 * n, n_dims))

    splits = np.array(["train_val"] * n + ["test"] * n)
    folds = np.tile(np.arange(5), int(np.ceil(2 * n / 5)))[: 2 * n]

    return {
        "features": X.astype(np.float32),
        "labels": Y.astype(np.float32),
        "folds": folds.astype(np.int64),
        "splits": splits,
        "subjects": np.array([str(i) for i in range(2 * n)]),
        "volume_indices": np.arange(2 * n),
    }


# =============================================================================
# metrics
# =============================================================================


def test_softmax_rows_sum_to_one():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = softmax(x)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0.0)


# =============================================================================
# split_tv_test
# =============================================================================


def test_split_tv_test_partitions_correctly():
    data = _make_classification()
    f_tv, y_tv, folds_tv, f_te, y_te = split_tv_test(data)
    assert len(f_tv) == 60 and len(f_te) == 60
    assert f_tv.dtype == np.float32 and folds_tv.dtype == np.int64
    # train_val and test are disjoint and cover everything.
    assert len(f_tv) + len(f_te) == len(data["features"])


def test_split_tv_test_rejects_subject_in_both_splits():
    # A subject with volumes on both sides of the train_val/test frontier
    # would leak test information into training.
    data = _make_classification()
    subjects = data["subjects"].copy()
    subjects[-1] = subjects[0]  # row 0 is train_val, last row is test
    data["subjects"] = subjects
    with pytest.raises(ValueError, match="both train_val and test"):
        split_tv_test(data)


def test_split_tv_test_rejects_subject_spanning_folds():
    # Within train_val, a subject sitting in two folds is on both the fit and
    # validation sides of a CV iteration during hyperparameter selection.
    data = _make_classification()
    subjects = data["subjects"].copy()
    subjects[1] = subjects[0]  # rows 0 and 1 are train_val, folds 0 and 1
    data["subjects"] = subjects
    with pytest.raises(ValueError, match="more than one CV fold"):
        split_tv_test(data)


def test_split_tv_test_requires_subjects():
    data = _make_classification()
    del data["subjects"]
    with pytest.raises(KeyError, match="subjects"):
        split_tv_test(data)


# =============================================================================
# Task registry
# =============================================================================


def test_roi_task_mapping_is_complete():
    assert set(ROI_TASK) == {"fip", "lc", "ofc", "sc"}
    assert ROI_TASK["ofc"] == "multiclass"
    assert ROI_TASK["sc"] == "regression"
    assert resolve_task("fip").name == "binary"


def test_resolve_task_rejects_unknown_roi():
    with pytest.raises(KeyError):
        resolve_task("unknown")


def test_regression_task_has_six_dims():
    assert resolve_task("sc").n_dims == 6
    assert resolve_task("fip").n_dims == 1


# =============================================================================
# End-to-end probing per task family
# =============================================================================


@pytest.mark.parametrize("roi", ["fip", "ofc"])
def test_classification_probe_recovers_signal(roi):
    task = resolve_task(roi)
    n_classes = 4 if task.name == "multiclass" else 2
    data = _make_classification(n_classes=n_classes)
    f_tv, y_tv, folds_tv, f_te, y_te = split_tv_test(data)
    y_tv = y_tv.astype(task.label_cast)
    y_te = y_te.astype(task.label_cast)

    grid = [{"C": c} for c in (0.1, 1.0, 10.0)]
    grid_df, best = run_cv_grid(
        f_tv,
        y_tv,
        folds_tv,
        task.standard_model,
        task.standard_score,
        grid,
        list(range(5)),
    )
    assert "mean_score" in best
    assert len(grid_df) == 3

    test_score = evaluate_test(
        f_tv,
        y_tv,
        f_te,
        y_te,
        task.standard_model,
        best,
        task.standard_score,
    )
    # Linearly-separable data -> near-perfect ROC-AUC.
    assert test_score > 0.9


def test_regression_probe_recovers_signal():
    task = resolve_task("sc")
    data = _make_regression()
    f_tv, y_tv, folds_tv, f_te, y_te = split_tv_test(data)
    y_tv = y_tv.astype(task.label_cast)
    y_te = y_te.astype(task.label_cast)

    grid = [{"alpha": a} for a in (0.1, 1.0, 10.0)]
    # Probe each target dimension independently, as the SC runner does.
    for dim in range(task.n_dims):
        _, best = run_cv_grid(
            f_tv,
            y_tv[:, dim],
            folds_tv,
            task.standard_model,
            task.standard_score,
            grid,
            list(range(5)),
        )
        r2 = evaluate_test(
            f_tv,
            y_tv[:, dim],
            f_te,
            y_te[:, dim],
            task.standard_model,
            best,
            task.standard_score,
        )
        assert r2 > 0.9


def test_flatten_raw_classification_uses_decision_function():
    """RidgeClassifier path must score without predict_proba."""
    task = resolve_task("fip")
    data = _make_classification()
    f_tv, y_tv, folds_tv, f_te, y_te = split_tv_test(data)
    y_tv = y_tv.astype(task.label_cast)
    y_te = y_te.astype(task.label_cast)

    grid = [{"alpha": a} for a in (1.0, 10.0)]
    _, best = run_cv_grid(
        f_tv,
        y_tv,
        folds_tv,
        task.flatten_raw_model,
        task.flatten_raw_score,
        grid,
        list(range(5)),
    )
    score = evaluate_test(
        f_tv,
        y_tv,
        f_te,
        y_te,
        task.flatten_raw_model,
        best,
        task.flatten_raw_score,
    )
    assert score > 0.9


def test_cv_grid_selects_best_mean_score():
    data = _make_classification()
    f_tv, y_tv, folds_tv, _, _ = split_tv_test(data)
    task = resolve_task("fip")
    grid = [{"C": c} for c in (0.01, 1.0, 100.0)]
    grid_df, best = run_cv_grid(
        f_tv,
        y_tv.astype(np.int64),
        folds_tv,
        task.standard_model,
        task.standard_score,
        grid,
        list(range(5)),
    )
    assert best["mean_score"] == pytest.approx(grid_df["mean_score"].max())


# =============================================================================
# Paths
# =============================================================================


def test_feature_path_with_and_without_preprocessing():
    p_skel = build_feature_path(
        "/out", "dino3d", "fip", "mean_pool", preprocessing="upscale_pad"
    )
    p_mri = build_feature_path("/out", "dino3d", "fip", "mean_pool")
    assert p_skel.as_posix().endswith(
        "dino3d/fip/upscale_pad/extracted_features/dino3d__mean_pool_features.npz"
    )
    assert p_mri.as_posix().endswith(
        "dino3d/fip/extracted_features/dino3d__mean_pool_features.npz"
    )


def test_results_paths_pca_prefix(tmp_path):
    paths = build_results_paths(tmp_path, "dino3d", "sc", "flatten", n_components=256)
    assert paths["summary_csv"].name == "dino3d__flatten_n256_summary.csv"
    assert paths["results_dir"].is_dir()


# =============================================================================
# Feature cache
# =============================================================================


def test_load_or_extract_writes_then_hits_cache(tmp_path):
    calls = {"n": 0}
    payload = _make_classification(n_per_split=5)

    def thunk():
        calls["n"] += 1
        return payload

    cache = tmp_path / "feat.npz"
    first = load_or_extract(cache, thunk)
    assert calls["n"] == 1 and cache.is_file()

    second = load_or_extract(cache, thunk)
    assert calls["n"] == 1  # cache hit, thunk not called again
    assert np.array_equal(first["features"], second["features"])
