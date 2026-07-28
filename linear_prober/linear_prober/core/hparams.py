"""Hyperparameter grid builders read from the YAML ``probe`` section.

The linear probes are selected by a small manual grid search over a single
regularisation hyperparameter:

  - ``C``     for :class:`~sklearn.linear_model.LogisticRegression`
  - ``alpha`` for :class:`~sklearn.linear_model.Ridge` /
    :class:`~sklearn.linear_model.RidgeClassifier`

Each builder returns a list of one-key dicts consumed by
:func:`linear_prober.core.cross_validation.run_cv_grid`.
"""

from __future__ import annotations

from typing import Dict, List


def build_grid_C(config: Dict) -> List[Dict]:
    """C grid for LogisticRegression (``mean_pool`` / ``mean_pool_multi_layers``)."""
    return [{"C": float(c)} for c in config["probe"]["C"]]


def build_grid_alpha(config: Dict) -> List[Dict]:
    """Alpha grid for Ridge regression and RidgeClassifier."""
    return [{"alpha": float(a)} for a in config["probe"]["alpha"]]


def build_grid_flatten_raw(config: Dict) -> List[Dict]:
    """Alpha grid for the high-dimensional ``flatten`` regime (``D >> N``)."""
    return [{"alpha": float(a)} for a in config["probe"]["flatten_raw_alpha"]]


# -----------------------------------------------------------------------------
# DINOv3 uses three separate, nested grids (LogisticRegression, RidgeClassifier,
# Ridge) selected by the extraction/aggregation combination and the task family.
# -----------------------------------------------------------------------------


def build_grid_dinov3_logreg(config: Dict) -> List[Dict]:
    """C grid for the DINOv3 LogisticRegression probe."""
    return [{"C": float(c)} for c in config["probe"]["logreg"]["C"]]


def build_grid_dinov3_ridgeclf(config: Dict) -> List[Dict]:
    """Alpha grid for the DINOv3 RidgeClassifier probe."""
    return [{"alpha": float(a)} for a in config["probe"]["ridgeclassifier"]["alpha"]]


def build_grid_dinov3_ridge(config: Dict) -> List[Dict]:
    """Alpha grid for the DINOv3 Ridge regression probe."""
    return [{"alpha": float(a)} for a in config["probe"]["ridge"]["alpha"]]
