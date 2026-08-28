"""Skeleton input adapter: build features from binary sulcal-skeleton grids.

Wires the modality-specific pieces (dataloader over binary volumes, per-model
``make_extract_fn`` factory) into the shared feature cache. The frozen encoder
and its intensity normaliser live in ``skeleton/models/<model>/``.
"""

from __future__ import annotations

import importlib
from typing import Dict

from linear_prober.core.feature_cache import load_or_extract
from linear_prober.core.paths import build_feature_path
from linear_prober.skeleton.dataset import build_hcp_dataloader, build_ukbb_dataloader
from linear_prober.skeleton.pca import build_ukbb_feature_path, resolve_mapping


def get_hcp_features(
    config: Dict,
    roi: str,
    preprocessing: str,
    mode: str,
    task_type: str,
) -> Dict:
    """Return cached (or freshly extracted) HCP features for one (roi, mode).

    ``task_type`` is ``"classification"`` or ``"regression"`` and controls how
    the dataloader reads labels. Features are cached under
    ``{output_root}/{output_dir}/{roi}/{preprocessing}/extracted_features/``.
    """
    model_name = config["experiment"]["model"]
    output_dir = config["experiment"].get("output_model_name", model_name)
    checkpoint_path = config["paths"]["checkpoint_path"]
    output_root = config["paths"]["output_root"]
    device = config["feature_extraction"]["device"]
    batch_size = int(config["feature_extraction"]["batch_size"])
    num_workers = int(config["feature_extraction"]["num_workers"])

    roi_cfg = config["rois"][roi]
    volumes_path = roi_cfg["hcp_volumes_native"]
    master_table = roi_cfg["hcp_master_table"]

    module = importlib.import_module(
        f"linear_prober.skeleton.models.{model_name}.extract_features"
    )
    if model_name == "bsf":
        module._add_repo_to_path(config["repositories"]["bsf"])
    extract_fn = module.make_extract_fn(config)

    dataloader = build_hcp_dataloader(
        volumes_path=volumes_path,
        master_table_path=master_table,
        task_type=task_type,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    cache_path = build_feature_path(
        output_root, output_dir, roi, mode, preprocessing=preprocessing
    )
    return load_or_extract(
        cache_path, lambda: extract_fn(checkpoint_path, dataloader, mode, device)
    )


def get_ukbb_features(config: Dict, roi: str, preprocessing: str) -> Dict:
    """Return cached (or extracted) UKBB ``flatten`` features for PCA fitting.

    UKBB volumes are unlabelled and used only to fit the PCA basis for the
    skeleton ``flatten`` regime. Features are cached separately from HCP under
    ``extracted_features_ukbb/``.
    """
    model_name = config["experiment"]["model"]
    output_dir = config["experiment"].get("output_model_name", model_name)
    checkpoint_path = config["paths"]["checkpoint_path"]
    output_root = config["paths"]["output_root"]
    device = config["feature_extraction"]["device"]
    batch_size = int(config["feature_extraction"]["batch_size"])
    num_workers = int(config["feature_extraction"]["num_workers"])

    # Inject preprocessing and intensity mapping so make_extract_fn captures them.
    v0, v1 = resolve_mapping(config, roi)
    config["feature_extraction"]["preprocessing"] = preprocessing
    config["feature_extraction"]["v0"] = v0
    config["feature_extraction"]["v1"] = v1

    ukbb_volumes = config["rois"][roi]["ukbb_volumes_native"]

    module = importlib.import_module(
        f"linear_prober.skeleton.models.{model_name}.extract_features"
    )
    if model_name == "bsf":
        module._add_repo_to_path(config["repositories"]["bsf"])
    extract_fn = module.make_extract_fn(config)

    dataloader = build_ukbb_dataloader(
        ukbb_volumes_path=ukbb_volumes,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    cache_path = build_ukbb_feature_path(output_root, output_dir, roi, preprocessing)
    return load_or_extract(
        cache_path, lambda: extract_fn(checkpoint_path, dataloader, "flatten", device)
    )


def get_hcp_features_point_m2ae(
    config: Dict, roi: str, mode: str, grouping: str, aggregation: str, upsample: float
) -> Dict:
    """Return cached (or extracted) Point-M2AE features for one composite mode.

    Point-M2AE consumes point clouds built from the native volumes, so there is
    no geometric-preprocessing level: features are cached under
    ``{output_root}/point_m2ae/{roi}/extracted_features/`` directly (the MRI
    layout). The upstream repository is put on ``sys.path`` from the config
    ``repositories.point_m2ae`` key.
    """
    checkpoint_path = config["paths"]["checkpoint_path"]
    output_root = config["paths"]["output_root"]
    device = config["feature_extraction"]["device"]
    batch_size = int(config["feature_extraction"]["batch_size"])
    num_workers = int(config["feature_extraction"].get("num_workers", 4))

    roi_cfg = config["rois"][roi]

    module = importlib.import_module(
        "linear_prober.skeleton.models.point_m2ae.extract_features"
    )
    module._add_repo_to_path(config["repositories"]["point_m2ae"])
    extract_fn = module.make_extract_fn(
        grouping=grouping, aggregation=aggregation, upsample=upsample, device=device
    )

    dataloader = build_hcp_dataloader(
        volumes_path=roi_cfg["hcp_volumes_native"],
        master_table_path=roi_cfg["hcp_master_table"],
        task_type=roi_cfg["task_type"],
        batch_size=batch_size,
        num_workers=num_workers,
    )

    cache_path = build_feature_path(output_root, "point_m2ae", roi, mode)
    return load_or_extract(
        cache_path, lambda: extract_fn(checkpoint_path, dataloader, mode, device)
    )


def get_hcp_features_dinov3(
    config: Dict, roi: str, preprocessing: str, mode: str
) -> Dict:
    """Return cached (or extracted) DINOv3 HCP features for one composite mode.

    DINOv3's ``make_extract_fn`` has a bespoke signature (it captures the
    intensity mapping and 2D slice batch size) and its ``extract_fn`` parses the
    composite ``mode`` string to decide extraction / aggregation / slicer /
    model size. Features are cached under the fixed model name ``dinov3``.
    """
    checkpoint_path = config["paths"]["checkpoint_path"]
    output_root = config["paths"]["output_root"]
    device = config["feature_extraction"]["device"]
    batch_size = int(config["feature_extraction"]["batch_size"])
    slice_batch_size = int(config["feature_extraction"].get("slice_batch_size", 32))
    num_workers = int(config["feature_extraction"].get("num_workers", 4))

    v0, v1 = resolve_mapping(config, roi)
    roi_cfg = config["rois"][roi]

    module = importlib.import_module(
        "linear_prober.skeleton.models.dinov3.extract_features"
    )
    extract_fn = module.make_extract_fn(
        checkpoint_path=checkpoint_path,
        preprocessing=preprocessing,
        v0=v0,
        v1=v1,
        slice_batch_size=slice_batch_size,
        device=device,
    )

    dataloader = build_hcp_dataloader(
        volumes_path=roi_cfg["hcp_volumes_native"],
        master_table_path=roi_cfg["hcp_master_table"],
        task_type=roi_cfg["task_type"],
        batch_size=batch_size,
        num_workers=num_workers,
    )

    cache_path = build_feature_path(
        output_root, "dinov3", roi, mode, preprocessing=preprocessing
    )
    return load_or_extract(
        cache_path, lambda: extract_fn(checkpoint_path, dataloader, mode, device)
    )
