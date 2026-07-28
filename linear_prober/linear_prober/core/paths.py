"""Output path construction, shared by both modalities.

The on-disk layout is::

    {output_root}/{model}/{roi}[/{preprocessing}]/extracted_features/...
    {output_root}/{model}/{roi}[/{preprocessing}]/results/...

The ``preprocessing`` level is present only for the skeleton modality, where the
same ROI is probed under several geometric preprocessings. MRI crops have a
single native preprocessing, so that level is omitted (``preprocessing=None``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def _roi_dir(
    output_root: str | Path,
    model_name: str,
    roi: str,
    preprocessing: Optional[str],
) -> Path:
    """Base directory for one (model, roi[, preprocessing]) combination."""
    base = Path(output_root) / model_name / roi
    return base / preprocessing if preprocessing is not None else base


def build_feature_path(
    output_root: str | Path,
    model_name: str,
    roi: str,
    mode: str,
    preprocessing: Optional[str] = None,
) -> Path:
    """Path of the cached feature archive ``{model}__{mode}_features.npz``."""
    return (
        _roi_dir(output_root, model_name, roi, preprocessing)
        / "extracted_features"
        / f"{model_name}__{mode}_features.npz"
    )


def build_results_paths(
    output_root: str | Path,
    model_name: str,
    roi: str,
    mode: str,
    preprocessing: Optional[str] = None,
    n_components: Optional[int] = None,
) -> Dict[str, Path]:
    """Summary/grid CSV paths for one probing run.

    ``n_components`` selects the ``flatten_n{N}`` prefix used by the skeleton
    PCA regime; otherwise the prefix is ``{model}__{mode}``.
    """
    results_dir = _roi_dir(output_root, model_name, roi, preprocessing) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if n_components is None:
        prefix = f"{model_name}__{mode}"
    else:
        prefix = f"{model_name}__flatten_n{n_components}"

    return {
        "results_dir": results_dir,
        "summary_csv": results_dir / f"{prefix}_summary.csv",
        "grid_csv": results_dir / f"{prefix}_grid.csv",
    }
