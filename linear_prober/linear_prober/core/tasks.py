"""Downstream-task registry — the single place that encodes what differs
between the ROIs being probed.

Every ROI maps to one of three task families. A :class:`TaskSpec` bundles
everything the runner needs to treat a task generically: how to cast labels,
which estimator to fit, which metric to score with, and which hyperparameter
grid to search. The rest of the pipeline (feature caching, cross-validation,
result saving) is entirely task-agnostic.

Two estimator/scorer pairs are stored per task:

  - ``standard``    — used for the low-dimensional regimes (``mean_pool``,
    ``mean_pool_multi_layers``) and, on the skeleton modality, ``flatten`` after
    PCA projection. Classification uses LogisticRegression + ``predict_proba``.
  - ``flatten_raw`` — used for the high-dimensional raw ``flatten`` regime
    (``D >> N``) where LogisticRegression is ill-conditioned. Classification
    uses RidgeClassifier scored from ``decision_function``.

Regression uses Ridge in both regimes; its ``n_dims`` targets are probed and
scored independently, then averaged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from linear_prober.core import hparams, metrics

# Number of independent regression targets for the SC ROI.
N_REGRESSION_DIMS = 6


# =============================================================================
# Estimator builders
# =============================================================================
# Each builder returns a fresh StandardScaler -> estimator pipeline. The scaler
# is the *supervised probe* scaler, fit inside every CV fold on train only; it
# is independent of any feature-side normalisation applied upstream.


def build_logreg(hparams_dict: Dict) -> Pipeline:
    """LogisticRegression L2 (lbfgs) — low-dimensional classification."""
    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=float(hparams_dict["C"]),
        max_iter=5000,
        random_state=42,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def build_ridge_classifier(hparams_dict: Dict) -> Pipeline:
    """RidgeClassifier — high-dimensional classification (no ``predict_proba``)."""
    clf = RidgeClassifier(alpha=float(hparams_dict["alpha"]))
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def build_ridge_regressor(hparams_dict: Dict) -> Pipeline:
    """Ridge — regression, both low- and high-dimensional regimes."""
    reg = Ridge(alpha=float(hparams_dict["alpha"]))
    return Pipeline([("scaler", StandardScaler()), ("reg", reg)])


# =============================================================================
# Task specification
# =============================================================================


@dataclass(frozen=True)
class TaskSpec:
    """Everything the runner needs to handle one task family generically."""

    name: str  # "binary" | "multiclass" | "regression"
    label_cast: type  # np.int64 (classification) | np.float32
    is_regression: bool
    n_dims: int  # regression targets; 1 for classification

    # Low-dimensional / PCA regime.
    standard_model: Callable[[Dict], Pipeline]
    standard_score: Callable[[object, np.ndarray, np.ndarray], float]
    standard_grid: Callable[[Dict], List[Dict]]

    # High-dimensional raw ``flatten`` regime.
    flatten_raw_model: Callable[[Dict], Pipeline]
    flatten_raw_score: Callable[[object, np.ndarray, np.ndarray], float]
    flatten_raw_grid: Callable[[Dict], List[Dict]]


TASKS: Dict[str, TaskSpec] = {
    "binary": TaskSpec(
        name="binary",
        label_cast=np.int64,
        is_regression=False,
        n_dims=1,
        standard_model=build_logreg,
        standard_score=metrics.binary_auc_proba,
        standard_grid=hparams.build_grid_C,
        flatten_raw_model=build_ridge_classifier,
        flatten_raw_score=metrics.binary_auc_decision,
        flatten_raw_grid=hparams.build_grid_flatten_raw,
    ),
    "multiclass": TaskSpec(
        name="multiclass",
        label_cast=np.int64,
        is_regression=False,
        n_dims=1,
        standard_model=build_logreg,
        standard_score=metrics.multiclass_auc_proba,
        standard_grid=hparams.build_grid_C,
        flatten_raw_model=build_ridge_classifier,
        flatten_raw_score=metrics.multiclass_auc_decision,
        flatten_raw_grid=hparams.build_grid_flatten_raw,
    ),
    "regression": TaskSpec(
        name="regression",
        label_cast=np.float32,
        is_regression=True,
        n_dims=N_REGRESSION_DIMS,
        standard_model=build_ridge_regressor,
        standard_score=metrics.regression_r2,
        standard_grid=hparams.build_grid_alpha,
        flatten_raw_model=build_ridge_regressor,
        flatten_raw_score=metrics.regression_r2,
        flatten_raw_grid=hparams.build_grid_flatten_raw,
    ),
}

# Which task each region-of-interest is evaluated under.
ROI_TASK: Dict[str, str] = {
    "fip": "binary",
    "lc": "binary",
    "ofc": "multiclass",
    "sc": "regression",
}


def resolve_task(roi: str) -> TaskSpec:
    """Return the :class:`TaskSpec` for a region-of-interest name."""
    if roi not in ROI_TASK:
        raise KeyError(f"Unknown roi '{roi}'. Expected one of: {sorted(ROI_TASK)}.")
    return TASKS[ROI_TASK[roi]]
