"""Modality-agnostic probing engine: cross-validation, tasks, metrics, I/O."""

from linear_prober.core.cross_validation import (
    evaluate_test,
    run_cv_grid,
    split_tv_test,
)
from linear_prober.core.feature_cache import load_or_extract
from linear_prober.core.paths import build_feature_path, build_results_paths
from linear_prober.core.results import save_results
from linear_prober.core.tasks import ROI_TASK, TASKS, TaskSpec, resolve_task

__all__ = [
    "evaluate_test",
    "run_cv_grid",
    "split_tv_test",
    "load_or_extract",
    "build_feature_path",
    "build_results_paths",
    "save_results",
    "resolve_task",
    "TaskSpec",
    "TASKS",
    "ROI_TASK",
]
