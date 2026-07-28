"""MRI input adapter: build features from intensity crops.

Each MRI model reads its own NIfTI crops and applies its own preprocessing
(window size and intensity normalisation differ per model), so extraction is
delegated to the per-model ``extract_features`` entry point. This wrapper only
resolves config paths and plugs the result into the shared feature cache.
"""

from __future__ import annotations

import importlib
from typing import Dict

from linear_prober.core.feature_cache import load_or_extract
from linear_prober.core.paths import build_feature_path


def get_features(config: Dict, roi: str, mode: str) -> Dict:
    """Return cached (or freshly extracted) MRI features for one (roi, mode).

    Features are cached under ``{output_root}/{model}/{roi}/extracted_features/``
    (no preprocessing level — MRI crops have a single native preprocessing).
    """
    model_name = config["experiment"]["model"]
    output_root = config["paths"]["output_root"]
    checkpoint_path = config["paths"]["checkpoint_path"]
    device = config["feature_extraction"]["device"]
    batch_size = int(config["feature_extraction"]["batch_size"])
    roi_cfg = config["rois"][roi]

    module = importlib.import_module(
        f"linear_prober.mri.models.{model_name}.extract_features"
    )

    # Repository key name varies per model (3dino, sam3d, ...); take the first.
    repos = config.get("repositories", {})
    repo_path = next(iter(repos.values())) if repos else None

    def thunk() -> Dict:
        kwargs = dict(
            checkpoint_path=checkpoint_path,
            repo_path=repo_path,
            master_table_path=roi_cfg["master_table"],
            crop_dir=roi_cfg["crop_dir"],
            roi_dirname=roi_cfg["roi_dirname"],
            mode=mode,
            device=device,
            batch_size=batch_size,
        )
        # Only the 2D DINOv3 path slices volumes and needs a slice batch size.
        if model_name == "dinov3":
            kwargs["slice_batch_size"] = int(
                config["feature_extraction"].get("slice_batch_size", 32)
            )
        return module.extract_features(**kwargs)

    cache_path = build_feature_path(output_root, model_name, roi, mode)
    return load_or_extract(cache_path, thunk)
