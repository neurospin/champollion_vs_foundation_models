"""Structural validation of the shipped configs against what the code reads.

These tests load every real config in ``configs/`` and assert that the keys the
runners / extractors / probe selection actually access are present. They guard
against config/code drift (e.g. a runner reading a nested hyperparameter grid
that a config only provides in flat form).

Values are not checked (paths are ``/path/to`` placeholders); only the presence
and nesting of required keys.
"""

from __future__ import annotations

import glob
import os

import pytest
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
CONFIGS = sorted(glob.glob(os.path.join(CONFIG_DIR, "**", "*.yaml"), recursive=True))


def _has(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return False
        d = d[k]
    return True


def _label(path):
    return os.path.relpath(path, CONFIG_DIR)


@pytest.mark.parametrize("path", CONFIGS, ids=_label)
def test_config_has_required_keys(path):
    cfg = yaml.safe_load(open(path))
    modality = "mri" if os.sep + "mri" + os.sep in path else "skeleton"
    model = cfg.get("experiment", {}).get("model")

    # Common keys read by every runner / extractor.
    for keys in [
        ("experiment", "model"),
        ("paths", "checkpoint_path"),
        ("paths", "output_root"),
        ("feature_extraction", "device"),
        ("feature_extraction", "batch_size"),
        ("probe", "n_folds"),
    ]:
        assert _has(cfg, *keys), f"{_label(path)}: missing {'.'.join(keys)}"

    # ROI-level keys differ by modality.
    roi_keys = (
        ["hcp_volumes_native", "hcp_master_table", "task_type"]
        if modality == "skeleton"
        else ["master_table", "crop_dir", "roi_dirname"]
    )
    for roi, roi_cfg in (cfg.get("rois") or {}).items():
        for k in roi_keys:
            assert k in roi_cfg, f"{_label(path)}: rois.{roi} missing {k}"

    if modality == "skeleton":
        assert _has(cfg, "model_normalization", "range"), _label(path)

    # Hyperparameter grids: DINOv3 uses three nested grids; the 3D models use
    # the flat C / alpha / flatten_raw_alpha grids.
    if model == "dinov3":
        for keys in [
            ("probe", "logreg", "C"),
            ("probe", "ridgeclassifier", "alpha"),
            ("probe", "ridge", "alpha"),
        ]:
            assert _has(cfg, *keys), f"{_label(path)}: missing {'.'.join(keys)}"
    else:
        for keys in [("probe", "C"), ("probe", "alpha"), ("probe", "flatten_raw_alpha")]:
            assert _has(cfg, *keys), f"{_label(path)}: missing {'.'.join(keys)}"
        if modality == "skeleton":
            assert _has(cfg, "probe", "n_components_list"), _label(path)


def test_all_configs_discovered():
    # Sanity: the glob actually found the shipped configs (5 skeleton + 4 mri).
    assert len(CONFIGS) == 9
