"""Skeleton probing runner.

Reproduces the per-ROI skeleton pipeline (formerly three near-identical
``run_linear_probe_{fip_lc,ofc,sc}.py`` scripts) through the shared task
registry. One entry point handles all ROIs; the task family (binary /
multiclass / regression) is resolved from the ROI.

Feature modes:
  - ``mean_pool`` / ``mean_pool_multi_layers`` — low-dimensional, standard probe;
  - ``flatten``    — high-dimensional, projected through per-``n_components`` PCA;
  - ``flatten_raw`` (``--flatten-raw``) — high-dimensional, RidgeClassifier / Ridge
    with no PCA.
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
from linear_prober.skeleton.extract import get_hcp_features
from linear_prober.skeleton.pca import apply_pca, load_pca, resolve_mapping

STANDARD_MODES = ["mean_pool", "mean_pool_multi_layers", "flatten"]


def run(
    config: Dict,
    roi: str,
    preprocessing: str,
    mode: Optional[str] = None,
    flatten_raw: bool = False,
) -> None:
    """Run skeleton probing for one ROI under one preprocessing.

    Args:
        config: parsed YAML config.
        roi: ``fip`` | ``lc`` | ``ofc`` | ``sc``.
        preprocessing: one geometric preprocessing name.
        mode: single mode to run; default runs all :data:`STANDARD_MODES`.
        flatten_raw: run the no-PCA high-dimensional probe instead.
    """
    task = resolve_task(roi)
    task_type = "regression" if task.is_regression else "classification"

    # Per-ROI intensity mapping is injected into the config so extract_features
    # can pick it up when building the model's normaliser.
    v0, v1 = resolve_mapping(config, roi)
    config["feature_extraction"]["preprocessing"] = preprocessing
    config["feature_extraction"]["v0"] = v0
    config["feature_extraction"]["v1"] = v1

    model_name = config["experiment"]["model"]
    output_dir = config["experiment"].get("output_model_name", model_name)
    output_root = config["paths"]["output_root"]
    expected_folds = list(range(int(config["probe"]["n_folds"])))
    classifier = config["probe"].get("classifier")

    def base_summary(mode_name: str, n_components) -> Dict:
        return {
            "model": output_dir,
            "roi": roi,
            "preprocessing": preprocessing,
            "mode": mode_name,
            "n_components": n_components,
            "v0": v0,
            "v1": v1,
        }

    print("=" * 60)
    print(
        f"[skeleton] model={model_name} roi={roi} preprocessing={preprocessing} "
        f"task={task.name} mapping=({v0}, {v1})"
    )
    print("=" * 60)

    def _split(data):
        f_tv, y_tv, folds_tv, f_te, y_te = split_tv_test(data)
        return (
            f_tv,
            y_tv.astype(task.label_cast),
            folds_tv,
            f_te,
            y_te.astype(task.label_cast),
        )

    # ── flatten_raw (no PCA) ────────────────────────────────────────────────
    if flatten_raw:
        data = get_hcp_features(config, roi, preprocessing, "flatten", task_type)
        f_tv, y_tv, folds_tv, f_te, y_te = _split(data)
        res = evaluate_mode(
            task,
            f_tv,
            y_tv,
            folds_tv,
            f_te,
            y_te,
            task.flatten_raw_model,
            task.flatten_raw_score,
            task.flatten_raw_grid(config),
            expected_folds,
        )
        summary = {
            **base_summary("flatten_raw", None),
            "classifier": "ridge_classifier",
            **res.summary_core,
        }
        save_results(
            summary,
            res.grid_df,
            build_results_paths(
                output_root, output_dir, roi, "flatten_raw", preprocessing=preprocessing
            ),
        )
        return

    modes = [mode] if mode else STANDARD_MODES

    for m in modes:
        if m in ("mean_pool", "mean_pool_multi_layers"):
            data = get_hcp_features(config, roi, preprocessing, m, task_type)
            f_tv, y_tv, folds_tv, f_te, y_te = _split(data)
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
                **base_summary(m, None),
                "classifier": classifier,
                **res.summary_core,
            }
            save_results(
                summary,
                res.grid_df,
                build_results_paths(
                    output_root, output_dir, roi, m, preprocessing=preprocessing
                ),
            )

        elif m == "flatten":
            data = get_hcp_features(config, roi, preprocessing, "flatten", task_type)
            f_tv, y_tv, folds_tv, f_te, y_te = _split(data)
            for n_comp in sorted(int(n) for n in config["probe"]["n_components_list"]):
                pca = load_pca(output_root, output_dir, roi, preprocessing, n_comp)
                f_tv_p, f_te_p = apply_pca(f_tv, pca), apply_pca(f_te, pca)
                res = evaluate_mode(
                    task,
                    f_tv_p,
                    y_tv,
                    folds_tv,
                    f_te_p,
                    y_te,
                    task.standard_model,
                    task.standard_score,
                    task.standard_grid(config),
                    expected_folds,
                )
                summary = {
                    **base_summary("flatten", n_comp),
                    "classifier": classifier,
                    **res.summary_core,
                }
                save_results(
                    summary,
                    res.grid_df,
                    build_results_paths(
                        output_root,
                        output_dir,
                        roi,
                        "flatten",
                        preprocessing=preprocessing,
                        n_components=n_comp,
                    ),
                )
        else:
            raise ValueError(f"Unknown mode '{m}'. Expected one of {STANDARD_MODES}.")

    print("\nDone.")
