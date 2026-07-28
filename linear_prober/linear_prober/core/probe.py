"""Generic single-mode evaluation, shared by both modalities.

A "mode" is one feature representation (``mean_pool``, ``mean_pool_multi_layers``,
``flatten`` after PCA, or raw ``flatten``). Given the train_val/test features and
a task, this module runs the CV grid search, evaluates once on test, and returns
ready-to-serialise scalar results plus the full grid.

Classification tasks are scored once. Regression tasks probe each of the
``task.n_dims`` targets independently and report per-dimension and mean R2 — the
control-flow difference between the two families is contained here so the
modality runners stay simple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from linear_prober.core.cross_validation import evaluate_test, run_cv_grid
from linear_prober.core.tasks import TaskSpec


@dataclass
class ModeResult:
    """Scalar results of one mode, plus the CV grid for serialisation."""

    summary_core: Dict  # ready-to-merge scalar fields
    grid_df: pd.DataFrame  # full hyperparameter grid


def evaluate_mode(
    task: TaskSpec,
    features_tv: np.ndarray,
    labels_tv: np.ndarray,
    folds_tv: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    build_model_fn: Callable,
    score_fn: Callable,
    hparam_grid: List[Dict],
    expected_folds: List[int],
) -> ModeResult:
    """Run CV + held-out evaluation for one mode.

    For classification the summary holds ``cv_mean_score`` / ``test_score`` and
    the selected hyperparameters. For regression it holds ``cv_mean_r2`` /
    ``test_mean_r2`` and per-dimension scores, and the returned grid is the
    concatenation of the per-dimension grids (with a ``dim`` column).
    """
    if not task.is_regression:
        grid_df, best = run_cv_grid(
            features_tv,
            labels_tv,
            folds_tv,
            build_model_fn,
            score_fn,
            hparam_grid,
            expected_folds,
        )
        test_score = evaluate_test(
            features_tv,
            labels_tv,
            features_test,
            labels_test,
            build_model_fn,
            best,
            score_fn,
        )
        summary_core = {
            "cv_mean_score": best["mean_score"],
            "test_score": test_score,
            **{k: v for k, v in best.items() if k != "mean_score"},
        }
        return ModeResult(summary_core=summary_core, grid_df=grid_df)

    # Regression — probe each target dimension independently.
    cv_r2, test_r2, grids, best_hparams = [], [], [], []
    for dim in range(task.n_dims):
        print(f"  [regression] dim {dim}/{task.n_dims - 1}")
        grid_df, best = run_cv_grid(
            features_tv,
            labels_tv[:, dim],
            folds_tv,
            build_model_fn,
            score_fn,
            hparam_grid,
            expected_folds,
        )
        r2 = evaluate_test(
            features_tv,
            labels_tv[:, dim],
            features_test,
            labels_test[:, dim],
            build_model_fn,
            best,
            score_fn,
        )
        cv_r2.append(float(best["mean_score"]))
        test_r2.append(float(r2))
        best_hparams.append({k: v for k, v in best.items() if k != "mean_score"})
        grid_df = grid_df.copy()
        grid_df["dim"] = dim
        grids.append(grid_df)

    summary_core: Dict = {
        "cv_mean_r2": float(np.mean(cv_r2)),
        "test_mean_r2": float(np.mean(test_r2)),
    }
    for dim in range(task.n_dims):
        summary_core[f"dim_{dim}_cv_r2"] = cv_r2[dim]
        summary_core[f"dim_{dim}_test_r2"] = test_r2[dim]
        for k, v in best_hparams[dim].items():
            summary_core[f"dim_{dim}_best_{k}"] = v

    return ModeResult(
        summary_core=summary_core, grid_df=pd.concat(grids, ignore_index=True)
    )
