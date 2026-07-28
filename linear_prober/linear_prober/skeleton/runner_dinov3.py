"""DINOv3 probing runner — skeleton modality.

Reproduces the former ``run_linear_probe_{fip_lc,ofc,sc}_dinov3.py`` scripts.
DINOv3 is the 2D-slicing path: one run is a single composite feature mode
(extraction × aggregation × slicer × model size, optionally density-weighted),
not the three standard 3D modes. The probe and grid are selected by
:func:`linear_prober.core.dinov3.select_probe`.
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
from linear_prober.skeleton.extract import get_hcp_features_dinov3
from linear_prober.skeleton.models.dinov3.extract_features import (
    _DW_INCOMPATIBLE_PREPROCESSINGS,
)
from linear_prober.skeleton.pca import resolve_mapping


def run(
    config: Dict,
    roi: str,
    preprocessing: str,
    model_size: str,
    slicer_mode: str,
    extraction: str,
    aggregation: str,
    density_weighting: bool = False,
) -> None:
    """Run one DINOv3 skeleton probe for one ROI and mode combination."""
    task = resolve_task(roi)
    mode = build_mode(
        extraction, aggregation, slicer_mode, model_size, density_weighting
    )

    if density_weighting and preprocessing in _DW_INCOMPATIBLE_PREPROCESSINGS:
        raise ValueError(
            f"--density_weighting is incompatible with preprocessing={preprocessing}. "
            "Use upscale_pad or nearest_neighbors."
        )

    v0, v1 = resolve_mapping(config, roi)
    output_root = config["paths"]["output_root"]
    expected_folds = list(range(int(config["probe"]["n_folds"])))

    print("=" * 60)
    print(f"[skeleton/dinov3] roi={roi} mode={mode} preprocessing={preprocessing}")
    print("=" * 60)

    data = get_hcp_features_dinov3(config, roi, preprocessing, mode)
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
        "preprocessing": preprocessing,
        "mode": mode,
        "model_size": model_size,
        "slicer_mode": slicer_mode,
        "extraction": extraction,
        "aggregation": aggregation,
        "density_weighting": density_weighting,
        "v0": v0,
        "v1": v1,
        "n_train_val": int(len(y_tv)),
        "n_test": int(len(y_te)),
        **({} if task.is_regression else {"classifier": classifier}),
        **res.summary_core,
    }
    save_results(
        summary,
        res.grid_df,
        build_results_paths(
            output_root, "dinov3", roi, mode, preprocessing=preprocessing
        ),
    )
    print("\nDone.")
