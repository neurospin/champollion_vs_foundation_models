"""MRI probing runner.

Reproduces the per-ROI MRI pipeline (formerly three near-identical
``run_linear_probe_{fip,ofc,sc}.py`` scripts) through the shared task registry.

MRI crops use a single native preprocessing (no geometric preprocessing choice,
no intensity-mapping search), so the runner is a thin loop over feature modes:

  - ``mean_pool`` / ``mean_pool_multi_layers`` — LogisticRegression / Ridge;
  - ``flatten`` — RidgeClassifier / Ridge on raw high-dimensional features
    (equivalent to the skeleton ``flatten_raw`` regime; no PCA on MRI).
"""

from __future__ import annotations

from typing import Dict, Optional

from linear_prober.core import (
    build_results_paths,
    resolve_task,
    save_results,
    split_tv_test,
)
from linear_prober.core.probe import evaluate_mode
from linear_prober.mri.extract import get_features

STANDARD_MODES = ["mean_pool", "mean_pool_multi_layers", "flatten"]


def run(config: Dict, roi: str, mode: Optional[str] = None) -> None:
    """Run MRI probing for one ROI.

    Args:
        config: parsed YAML config.
        roi: ``fip`` | ``ofc`` | ``sc``.
        mode: single mode to run; default runs all :data:`STANDARD_MODES`.
    """
    task = resolve_task(roi)
    model_name = config["experiment"]["model"]
    output_root = config["paths"]["output_root"]
    expected_folds = list(range(int(config["probe"]["n_folds"])))
    modes = [mode] if mode else STANDARD_MODES

    print("=" * 60)
    print(f"[mri] model={model_name} roi={roi} task={task.name} modes={modes}")
    print("=" * 60)

    for m in modes:
        data = get_features(config, roi, m)
        f_tv, y_tv, folds_tv, f_te, y_te = split_tv_test(data)
        y_tv, y_te = y_tv.astype(task.label_cast), y_te.astype(task.label_cast)

        # Raw high-dimensional flatten uses the RidgeClassifier/Ridge regime.
        if m == "flatten":
            build_fn, score_fn = task.flatten_raw_model, task.flatten_raw_score
            grid = task.flatten_raw_grid(config)
        elif m in ("mean_pool", "mean_pool_multi_layers"):
            build_fn, score_fn = task.standard_model, task.standard_score
            grid = task.standard_grid(config)
        else:
            raise ValueError(f"Unknown mode '{m}'. Expected one of {STANDARD_MODES}.")

        res = evaluate_mode(
            task,
            f_tv,
            y_tv,
            folds_tv,
            f_te,
            y_te,
            build_fn,
            score_fn,
            grid,
            expected_folds,
        )
        summary = {"model": model_name, "roi": roi, "mode": m, **res.summary_core}
        save_results(
            summary, res.grid_df, build_results_paths(output_root, model_name, roi, m)
        )

    print("\nDone.")
