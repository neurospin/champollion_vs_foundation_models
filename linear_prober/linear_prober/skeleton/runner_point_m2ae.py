"""Point-M2AE probing runner — skeleton modality.

Point-M2AE is the point-cloud path: the skeleton volume is converted to a
normalised point cloud, so there is no geometric-preprocessing choice and one
run is a single composite mode (grouping × aggregation × upsampling). Feature
dimensions (384–1536) sit in the standard low-dimensional regime, so every
mode uses the standard probe family (LogisticRegression / Ridge) — the same
protocol and grids as the other encoders.
"""

from __future__ import annotations

from typing import Dict

from linear_prober.core import (
    build_results_paths,
    resolve_task,
    save_results,
    split_tv_test,
)
from linear_prober.core.probe import evaluate_mode
from linear_prober.skeleton.extract import get_hcp_features_point_m2ae
from linear_prober.skeleton.models.point_m2ae.extract_features import build_mode


def run(
    config: Dict, roi: str, grouping: str, aggregation: str, upsample: float
) -> None:
    """Run one Point-M2AE skeleton probe for one ROI and mode combination."""
    task = resolve_task(roi)
    mode = build_mode(grouping, aggregation, upsample)

    output_root = config["paths"]["output_root"]
    expected_folds = list(range(int(config["probe"]["n_folds"])))
    classifier = config["probe"].get("classifier")

    print("=" * 60)
    print(f"[skeleton/point_m2ae] roi={roi} mode={mode} task={task.name}")
    print("=" * 60)

    data = get_hcp_features_point_m2ae(
        config, roi, mode, grouping, aggregation, upsample
    )
    f_tv, y_tv, folds_tv, f_te, y_te = split_tv_test(data)
    y_tv, y_te = y_tv.astype(task.label_cast), y_te.astype(task.label_cast)

    res = evaluate_mode(
        task,
        f_tv,
        y_tv,
        folds_tv,
        f_te,
        y_te,
        task.standard_model,
        task.standard_score,
        task.standard_grid(config),
        expected_folds,
    )

    summary = {
        "model": "point_m2ae",
        "roi": roi,
        "mode": mode,
        "grouping": grouping,
        "aggregation": aggregation,
        "upsample": float(upsample),
        "n_train_val": int(len(y_tv)),
        "n_test": int(len(y_te)),
        **({} if task.is_regression else {"classifier": classifier}),
        **res.summary_core,
    }
    save_results(
        summary,
        res.grid_df,
        build_results_paths(output_root, "point_m2ae", roi, mode),
    )
    print("\nDone.")
