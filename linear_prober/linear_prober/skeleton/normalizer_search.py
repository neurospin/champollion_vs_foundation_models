"""Per-ROI intensity-mapping search (skeleton modality).

Binary sulcal grids ``V ∈ {0,1}`` are mapped to a frozen encoder's input range
before extraction::

    phi(x) = v0 + x * (v1 - v0),   v0 = alpha + p0 (beta - alpha),
                                   v1 = alpha + p1 (beta - alpha),   0 <= p0 < p1 <= 1.

This module sweeps all valid ``(p0, p1)`` couples at a given ``grid_step``, and
for each one extracts mean-pool features and runs the standard 5-fold CV probe.
The full grid and the top-k mappings (ranked by CV score only — the held-out
test score is reported but never used for selection) are written to disk. The
selected mapping is then copied into ``config.optimal_mapping`` for the main
probing runs.

The mapping is applied relative to preprocessing depending on whether the
preprocessing preserves binary values (mapping applied after) or produces
continuous ones (mapping applied before). Extraction itself is delegated to each
model's ``extract_mean_pool_for_mapping``.
"""

from __future__ import annotations

import importlib
from typing import Dict

import numpy as np
import pandas as pd

from linear_prober.core.probe import evaluate_mode
from linear_prober.core.tasks import resolve_task
from linear_prober.skeleton.dataset import HCPDataset
from linear_prober.skeleton.pca import (
    build_normalizer_grid,
    build_normalizer_search_hparam_grid,
    build_normalizer_search_paths,
)

# upscale_pad / nearest_neighbors keep {0,1}; trilinear produces continuous values.
BINARY_PRESERVING = {"upscale_pad", "nearest_neighbors"}


def _load_volumes_and_metadata(roi_cfg: Dict, task_type: str):
    """Return raw volumes plus split/fold/label arrays aligned by row order."""
    dataset = HCPDataset(
        volumes_path=roi_cfg["hcp_volumes_native"],
        master_table_path=roi_cfg["hcp_master_table"],
        task_type=task_type,
    )
    table = dataset.table
    splits = table["split"].values
    folds = table["fold"].values
    if task_type == "regression":
        label_cols = sorted(c for c in table.columns if c.startswith("label_"))
        labels = table[label_cols].values.astype(np.float32)
    else:
        labels = table["label"].values.astype(np.int64)
    return dataset.volumes, splits, folds, labels


def _search_checkpoint(config, roi, preprocessing, module, checkpoint_cfg) -> None:
    task = resolve_task(roi)
    task_type = "regression" if task.is_regression else "classification"

    output_root = config["paths"]["output_root"]
    device = config["feature_extraction"]["device"]
    batch_size = int(config["feature_extraction"]["batch_size"])
    target_shape = tuple(int(v) for v in config["feature_extraction"]["target_shape"])
    expected_folds = list(range(int(config["probe"]["n_folds"])))
    top_k = int(config["normalizer_search"].get("top_k", 5))
    model_range = tuple(float(x) for x in config["model_normalization"]["range"])
    grid = build_normalizer_grid(
        float(config["normalizer_search"]["grid_step"]), model_range
    )
    hparam_grid = build_normalizer_search_hparam_grid(config)
    is_binary_preserving = preprocessing in BINARY_PRESERVING

    output_dir = str(checkpoint_cfg["output_model_name"])
    checkpoint_path = str(checkpoint_cfg["checkpoint_path"])

    if config["experiment"]["model"] == "bsf":
        module._add_repo_to_path(config["repositories"]["bsf"])
    encoder = module._build_model(checkpoint_path, device)
    extract_for_mapping = module.extract_mean_pool_for_mapping

    volumes_raw, splits, folds, labels = _load_volumes_and_metadata(
        config["rois"][roi], task_type
    )
    tv_mask, test_mask = splits == "train_val", splits == "test"

    print(
        f"[NormSearch] {output_dir} roi={roi} preprocessing={preprocessing} "
        f"couples={len(grid)} regime={'binary' if is_binary_preserving else 'continuous'}"
    )

    rows = []
    for i, couple in enumerate(grid, start=1):
        v0, v1 = couple["v0"], couple["v1"]
        print(f"[NormSearch] mapping {i}/{len(grid)}  v0={v0:.4f} v1={v1:.4f}")

        features = extract_for_mapping(
            encoder=encoder,
            volumes_raw=volumes_raw,
            preprocessing=preprocessing,
            target_shape=target_shape,
            v0=v0,
            v1=v1,
            device=device,
            batch_size=batch_size,
            is_binary_preserving=is_binary_preserving,
        )
        f_tv, f_te = features[tv_mask].astype(np.float32), features[test_mask].astype(
            np.float32
        )
        res = evaluate_mode(
            task,
            f_tv,
            labels[tv_mask],
            folds[tv_mask],
            f_te,
            labels[test_mask],
            task.standard_model,
            task.standard_score,
            hparam_grid,
            expected_folds,
        )
        cv = res.summary_core.get("cv_mean_score", res.summary_core.get("cv_mean_r2"))
        test = res.summary_core.get("test_score", res.summary_core.get("test_mean_r2"))
        rows.append(
            {
                "roi": roi,
                **{k: couple[k] for k in ("p0", "p1", "v0", "v1")},
                "cv_mean_score": float(cv),
                "test_score": float(test),
            }
        )

    grid_df = pd.DataFrame(rows).sort_values("cv_mean_score", ascending=False)
    paths = build_normalizer_search_paths(output_root, output_dir, roi, preprocessing)
    grid_df.to_csv(paths["grid_csv"], index=False)
    grid_df.head(top_k).to_csv(paths["best_csv"], index=False)
    best = grid_df.iloc[0]
    print(
        f"[NormSearch] best: p0={best['p0']} p1={best['p1']} "
        f"cv={best['cv_mean_score']:.4f} -> {paths['best_csv']}"
    )


def search(config: Dict, roi: str, preprocessing: str) -> None:
    """Run the mapping search for every checkpoint in ``normalizer_search.checkpoints``."""
    module = importlib.import_module(
        f"linear_prober.skeleton.models.{config['experiment']['model']}.extract_features"
    )
    for checkpoint_cfg in config["normalizer_search"]["checkpoints"]:
        _search_checkpoint(config, roi, preprocessing, module, checkpoint_cfg)
