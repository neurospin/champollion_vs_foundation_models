"""Shared DINOv3 probe selection, used by both modality DINOv3 runners.

DINOv3 is the 2D-slicing path and diverges from the 3D-encoder runners: its
feature "mode" is a composite of four axes (extraction × aggregation × slicer ×
model size), and the linear probe / hyperparameter grid it uses depends on that
combination:

  - classification, ``mean_pool`` + ``mean_pool_axis`` -> LogisticRegression
    (probabilistic scoring);
  - classification, any other combination -> RidgeClassifier
    (decision-function scoring);
  - regression -> Ridge (per-target).

This module centralises the composite-mode string and that probe selection so
the skeleton and MRI DINOv3 runners stay identical where they can.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from linear_prober.core import hparams, metrics
from linear_prober.core.tasks import TaskSpec, build_ridge_regressor


def build_mode(
    extraction: str,
    aggregation: str,
    slicer_mode: str,
    model_size: str,
    density_weighting: bool = False,
) -> str:
    """Compose the DINOv3 feature-cache mode string.

    ``density_weighting`` (skeleton only) appends the ``__dw`` suffix.
    """
    mode = f"{extraction}__{aggregation}__{slicer_mode}__{model_size}"
    return mode + "__dw" if density_weighting else mode


def select_probe(
    task: TaskSpec,
    extraction: str,
    aggregation: str,
    config: Dict,
) -> Tuple[Callable, Callable, List[Dict], str]:
    """Return ``(build_model_fn, score_fn, hparam_grid, classifier_label)``.

    The label is written into the result summary (``logreg`` / ``ridgeclassifier``
    / ``ridge``).
    """
    if task.is_regression:
        return (
            build_ridge_regressor,
            metrics.regression_r2,
            hparams.build_grid_dinov3_ridge(config),
            "ridge",
        )

    is_logreg = extraction == "mean_pool" and aggregation == "mean_pool_axis"
    if is_logreg:
        return (
            task.standard_model,  # LogisticRegression
            task.standard_score,  # predict_proba scoring
            hparams.build_grid_dinov3_logreg(config),
            "logreg",
        )
    return (
        task.flatten_raw_model,  # RidgeClassifier
        task.flatten_raw_score,  # decision_function scoring
        hparams.build_grid_dinov3_ridgeclf(config),
        "ridgeclassifier",
    )
