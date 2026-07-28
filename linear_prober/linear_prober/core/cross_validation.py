"""Cross-validation and held-out evaluation — modality-agnostic core.

The evaluation protocol is shared by both modalities (binary sulcal grids and
MRI crops) and follows the CHAMPOLLION V1 convention:

  - folds are *pre-stratified* in the master table; they are never re-drawn.
    :class:`~sklearn.model_selection.GridSearchCV` is deliberately not used
    because it would re-shuffle the fold assignment.
  - hyperparameters are selected by manual k-fold CV on the ``train_val`` split;
  - the ``test`` split is touched exactly once, after selection.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


def split_tv_test(
    data: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a cached feature dict into train_val / test using ``splits``.

    Returns:
        features_tv, labels_tv, folds_tv, features_test, labels_test

    Feature arrays are cast to ``float32`` and folds to ``int64``; label dtype
    is left untouched so the caller can cast per task (int for classification,
    float for regression).
    """
    splits = data["splits"].astype(str)
    tv_mask = splits == "train_val"
    test_mask = splits == "test"

    features_tv = data["features"][tv_mask].astype(np.float32)
    labels_tv = data["labels"][tv_mask]
    folds_tv = data["folds"][tv_mask].astype(np.int64)
    features_test = data["features"][test_mask].astype(np.float32)
    labels_test = data["labels"][test_mask]

    return features_tv, labels_tv, folds_tv, features_test, labels_test


def run_cv_grid(
    features_tv: np.ndarray,
    labels_tv: np.ndarray,
    folds_tv: np.ndarray,
    build_model_fn: Callable[[Dict], object],
    score_fn: Callable[[object, np.ndarray, np.ndarray], float],
    hparam_grid: List[Dict],
    expected_folds: List[int],
) -> Tuple[pd.DataFrame, Dict]:
    """Manual k-fold CV over ``hparam_grid`` using pre-stratified folds.

    For each hyperparameter setting, a model is fit on all-but-one fold and
    scored on the held-out fold; the mean over folds ranks the settings.

    Returns:
        grid_df    : one row per hyperparameter setting with per-fold and mean
                     scores.
        best_hparams: the highest-mean-score setting, plus its ``mean_score``.
    """
    grid_rows = []
    n_total = len(hparam_grid)

    for i, hparams in enumerate(hparam_grid):
        fold_scores = {}

        for k in expected_folds:
            train_mask = folds_tv != k
            val_mask = folds_tv == k

            X_train, y_train = features_tv[train_mask], labels_tv[train_mask]
            X_val, y_val = features_tv[val_mask], labels_tv[val_mask]

            model = build_model_fn(hparams)
            model.fit(X_train, y_train)
            fold_scores[f"fold_{k}"] = float(score_fn(model, X_val, y_val))

        mean_score = float(np.mean(list(fold_scores.values())))
        grid_rows.append({**hparams, "mean_score": mean_score, **fold_scores})

        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            hparam_str = "  ".join(f"{k}={v}" for k, v in hparams.items())
            print(f"  [{i + 1}/{n_total}] {hparam_str} -> mean={mean_score:.4f}")

    grid_df = pd.DataFrame(grid_rows)
    best_row = grid_df.loc[grid_df["mean_score"].idxmax()]

    best_hparams = {k: best_row[k] for k in hparam_grid[0]}
    best_hparams["mean_score"] = float(best_row["mean_score"])

    print(f"  Best: {best_hparams}")
    return grid_df, best_hparams


def evaluate_test(
    features_tv: np.ndarray,
    labels_tv: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    build_model_fn: Callable[[Dict], object],
    best_hparams: Dict,
    score_fn: Callable[[object, np.ndarray, np.ndarray], float],
) -> float:
    """Refit on the full train_val split, score once on the held-out test set."""
    hparams = {k: v for k, v in best_hparams.items() if k != "mean_score"}
    model = build_model_fn(hparams)
    model.fit(features_tv, labels_tv)
    test_score = float(score_fn(model, features_test, labels_test))
    print(f"  [Test] score = {test_score:.6f}")
    return test_score
