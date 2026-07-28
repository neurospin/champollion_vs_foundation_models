"""DINOv3 probing runner — MRI modality.

Reproduces the former ``run_linear_probe_{fip,ofc,sc}_dinov3.py`` MRI scripts.
Like the skeleton DINOv3 runner it evaluates a single composite feature mode,
but MRI crops use a single native preprocessing (no geometric preprocessing, no
intensity-mapping search, no density weighting). Feature extraction reuses the
standard MRI adapter, which already forwards the ``mode`` string and the 2D
slice batch size.
"""

from __future__ import annotations

from typing import Dict

from linear_prober.core import (
    build_results_paths,
    resolve_task,
    save_results,
    split_tv_test,
)
from linear_prober.core.dinov3 import build_mode, select_probe
from linear_prober.core.probe import evaluate_mode
from linear_prober.mri.extract import get_features


def run(
    config: Dict,
    roi: str,
    model_size: str,
    slicer_mode: str,
    extraction: str,
    aggregation: str,
) -> None:
    """Run one DINOv3 MRI probe for one ROI and mode combination."""
    task = resolve_task(roi)
    mode = build_mode(extraction, aggregation, slicer_mode, model_size)
    output_root = config["paths"]["output_root"]
    expected_folds = list(range(int(config["probe"]["n_folds"])))

    print("=" * 60)
    print(f"[mri/dinov3] roi={roi} mode={mode}")
    print("=" * 60)

    data = get_features(config, roi, mode)
    f_tv, y_tv, folds_tv, f_te, y_te = split_tv_test(data)
    y_tv, y_te = y_tv.astype(task.label_cast), y_te.astype(task.label_cast)

    build_fn, score_fn, grid, classifier = select_probe(
        task, extraction, aggregation, config
    )
    res = evaluate_mode(
        task, f_tv, y_tv, folds_tv, f_te, y_te, build_fn, score_fn, grid, expected_folds
    )

    summary = {
        "model": "dinov3",
        "roi": roi,
        "mode": mode,
        "model_size": model_size,
        "slicer_mode": slicer_mode,
        "extraction": extraction,
        "aggregation": aggregation,
        "n_train_val": int(len(y_tv)),
        "n_test": int(len(y_te)),
        **({} if task.is_regression else {"classifier": classifier}),
        **res.summary_core,
    }
    save_results(
        summary, res.grid_df, build_results_paths(output_root, "dinov3", roi, mode)
    )
    print("\nDone.")
