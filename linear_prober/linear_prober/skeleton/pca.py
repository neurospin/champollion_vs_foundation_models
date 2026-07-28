"""Skeleton-only extras: UKBB PCA, intensity mapping and normaliser search.

These steps exist only for the skeleton modality:

  - PCA on UKBB ``flatten`` features (fit once, reused by the ``flatten`` probe
    regime). The saved object is a ``StandardScaler -> PCA`` pipeline so the
    PCA axes are computed on scaled features;
  - :func:`resolve_mapping` — the per-ROI intensity mapping ``[0,1] -> [v0,v1]``
    selected by the normaliser search;
  - :func:`build_normalizer_grid` and helpers — the ``(p0, p1)`` grid explored
    by that search.

MRI crops need none of this (native percentile normalisation, no UKBB set).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# PCA paths
# =============================================================================


def build_pca_path(
    output_root: str | Path,
    model_name: str,
    roi: str,
    preprocessing: str,
    n_components: int,
) -> Path:
    """Path of the saved ``StandardScaler -> PCA`` pipeline for ``n_components``."""
    return (
        Path(output_root)
        / model_name
        / roi
        / preprocessing
        / "pca_ukbb"
        / f"{model_name}__pca_n{n_components}.pkl"
    )


def build_explained_variance_path(
    output_root: str | Path,
    model_name: str,
    roi: str,
    preprocessing: str,
) -> Path:
    """Path of the explained-variance summary CSV."""
    return (
        Path(output_root)
        / model_name
        / roi
        / preprocessing
        / "pca_ukbb"
        / f"{model_name}__explained_variance.csv"
    )


def build_ukbb_feature_path(
    output_root: str | Path,
    model_name: str,
    roi: str,
    preprocessing: str,
    mode: str = "flatten",
) -> Path:
    """Path of the cached UKBB feature archive (kept separate from HCP features)."""
    return (
        Path(output_root)
        / model_name
        / roi
        / preprocessing
        / "extracted_features_ukbb"
        / f"{model_name}__{mode}_features.npz"
    )


# =============================================================================
# PCA fit / load / apply
# =============================================================================


def fit_ukbb_pca(
    features: np.ndarray,
    n_components_list: List[int],
    output_root: str | Path,
    model_name: str,
    roi: str,
    preprocessing: str,
) -> pd.DataFrame:
    """Fit and save one ``StandardScaler -> PCA`` pipeline per ``n_components``.

    Args:
        features: UKBB ``flatten`` features ``[N, D]``.
        n_components_list: PCA dimensionalities to fit.

    Returns:
        Explained-variance-ratio summary (also written to disk).
    """
    n_subjects, d_flat = features.shape
    print(f"[PCA] Fitting on {n_subjects} subjects x {d_flat} dims")

    for n in n_components_list:
        if n > d_flat:
            raise ValueError(f"n_components={n} > feature_dim={d_flat}.")

    ev_rows = []
    for n_components in sorted(n_components_list):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=n_components,
                        svd_solver="randomized",
                        random_state=42,
                    ),
                ),
            ]
        )
        pipeline.fit(features)

        explained_total = float(
            pipeline.named_steps["pca"].explained_variance_ratio_.sum()
        )
        print(
            f"[PCA] n_components={n_components}: explained variance {explained_total * 100:.2f}%"
        )

        pkl_path = build_pca_path(
            output_root, model_name, roi, preprocessing, n_components
        )
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, str(pkl_path))

        ev_rows.append(
            {
                "n_components": n_components,
                "explained_variance_total": explained_total,
                "pre_pca_scaling": "StandardScaler_fit_on_UKBB",
            }
        )

    ev_df = pd.DataFrame(ev_rows)
    ev_path = build_explained_variance_path(output_root, model_name, roi, preprocessing)
    ev_path.parent.mkdir(parents=True, exist_ok=True)
    ev_df.to_csv(str(ev_path), index=False)
    print(f"[PCA] Explained variance -> {ev_path}")
    return ev_df


def load_pca(
    output_root: str | Path,
    model_name: str,
    roi: str,
    preprocessing: str,
    n_components: int,
):
    """Load a saved ``StandardScaler -> PCA`` pipeline; error if missing."""
    path = build_pca_path(output_root, model_name, roi, preprocessing, n_components)
    if not path.is_file():
        raise FileNotFoundError(f"PCA not found: {path}. Fit it first (fit_ukbb_pca).")

    transformer = joblib.load(str(path))
    if not hasattr(transformer, "transform"):
        raise TypeError(
            f"Loaded PCA object from {path} has no .transform(); got {type(transformer)}."
        )
    print(f"[PCA] Loaded n_components={n_components} from {path}")
    return transformer


def apply_pca(features: np.ndarray, pca_model) -> np.ndarray:
    """Project ``[N, D]`` features through a loaded PCA pipeline."""
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features [N, D], got shape {features.shape}.")
    if not hasattr(pca_model, "transform"):
        raise TypeError(f"pca_model must expose .transform(), got {type(pca_model)}.")
    return np.asarray(pca_model.transform(features), dtype=np.float32)


# =============================================================================
# Intensity mapping and normaliser search
# =============================================================================


def resolve_mapping(config: Dict, roi: str) -> Tuple[float, float]:
    """Resolve the per-ROI intensity mapping ``(v0, v1)`` from config.

    Reads ``optimal_mapping[roi].{p0,p1}`` as fractions of the model range
    ``model_normalization.range = [alpha, beta]``. Falls back to the full range
    when the mapping is absent or null.
    """
    alpha, beta = [float(x) for x in config["model_normalization"]["range"]]

    mapping_cfg = config.get("optimal_mapping", {}).get(roi, {})
    p0 = mapping_cfg.get("p0") if mapping_cfg else None
    p1 = mapping_cfg.get("p1") if mapping_cfg else None

    if p0 is None or p1 is None:
        print(
            f"[resolve_mapping] roi={roi}: no mapping -> full range ({alpha}, {beta})"
        )
        return alpha, beta

    p0, p1 = float(p0), float(p1)
    if not (0.0 <= p0 < p1 <= 1.0):
        raise ValueError(
            f"[resolve_mapping] roi={roi}: invalid p0={p0}, p1={p1}. "
            "Must satisfy 0 <= p0 < p1 <= 1."
        )

    v0 = alpha + p0 * (beta - alpha)
    v1 = alpha + p1 * (beta - alpha)
    return round(v0, 6), round(v1, 6)


def build_normalizer_grid(
    grid_step: float, model_range: Tuple[float, float]
) -> List[Dict]:
    """Enumerate valid ``(p0, p1)`` couples (``p0 < p1``) over [0, 1] at ``grid_step``.

    Each couple is mapped to absolute values ``(v0, v1)`` in ``model_range``.
    """
    alpha, beta = model_range
    n_steps = round(1.0 / grid_step)
    percentages = [round(i / n_steps, 10) for i in range(n_steps + 1)]

    couples = []
    for p0, p1 in itertools.combinations(percentages, 2):
        v0 = alpha + p0 * (beta - alpha)
        v1 = alpha + p1 * (beta - alpha)
        couples.append(
            {
                "p0": round(p0, 6),
                "p1": round(p1, 6),
                "v0": round(v0, 6),
                "v1": round(v1, 6),
            }
        )
    return couples


def build_normalizer_search_hparam_grid(config: Dict) -> List[Dict]:
    """C grid for the LogisticRegression probe used inside the normaliser search."""
    return [{"C": float(c)} for c in config["probe"]["normalizer_search_C"]]


def build_normalizer_search_paths(
    output_root: str | Path,
    model_name: str,
    roi: str,
    preprocessing: str,
) -> Dict[str, Path]:
    """Grid/best CSV paths for the normaliser search of one (model, roi, preproc)."""
    d = Path(output_root) / model_name / roi / preprocessing / "normalizer_search"
    d.mkdir(parents=True, exist_ok=True)
    return {
        "results_dir": d,
        "grid_csv": d / "grid_results.csv",
        "best_csv": d / "best_mapping.csv",
    }
