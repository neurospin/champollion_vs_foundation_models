# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)
#
# Portions of this file are derived from DINOv2 (Meta AI Research),
# licensed under the Apache License 2.0.

"""Configuration utilities for the anisotropic sulcal SSL pipeline.

Geometry validation/resolution is original; the generic setup trio
(``apply_scaling_rules_to_cfg``, ``default_setup``, ``write_config``) is reused
from :mod:`sulcus_aware_3DINO.training.setup`.
"""

from __future__ import annotations

import logging
import math
import os
from numbers import Integral
from pathlib import Path
from typing import Any, Tuple

from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from sulcus_aware_3DINO.training.setup import (
    apply_scaling_rules_to_cfg,
    default_setup,
    write_config,
)

logger = logging.getLogger("dinov2")


# Absolute path resolved from this file:
#
#   sulcus-aware-3dino/sulcus_aware_3DINO/training/setup_anisotropic.py
#              ↓ parents[2]
#   sulcus-aware-3dino/                       (repo root, next to train.py)
#              ↓
#   sulcus-aware-3dino/configs/ssl3d_default_config_anisotropic.yaml
DEFAULT_ANISOTROPIC_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "ssl3d_default_config_anisotropic.yaml"
)


def _require_positive_int(value: Any, name: str) -> int:
    """
    Validate and normalize one strictly positive integer.

    Booleans are explicitly rejected even though bool is a subclass of int
    in Python.

    Args:
        value:
            Value to validate.

        name:
            Human-readable configuration field name used in error messages.

    Returns:
        The validated value as a Python int.

    Raises:
        TypeError:
            If value is not an integer.

        ValueError:
            If value is zero or negative.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"{name} must be an integer, got " f"{value!r} ({type(value).__name__})."
        )

    value = int(value)

    if value <= 0:
        raise ValueError(f"{name} must be strictly positive, got {value}.")

    return value


def normalize_shape_3d(
    value: Any,
    name: str = "shape",
) -> Tuple[int, int, int]:
    """
    Validate and normalize an explicit three-dimensional shape.

    Accepted inputs:
        - list[int] of length 3
        - tuple[int, int, int]
        - OmegaConf ListConfig of length 3

    A scalar is deliberately not expanded to a cubic shape. The dedicated
    anisotropic pipeline requires the three spatial dimensions to be written
    explicitly.

    Args:
        value:
            Shape-like object to normalize.

        name:
            Configuration field name used in error messages.

    Returns:
        A Python tuple:
            (dim_0, dim_1, dim_2)

    Raises:
        TypeError:
            If value is not a supported sequence or contains non-integers.

        ValueError:
            If the sequence does not contain exactly three strictly positive
            dimensions.
    """
    if not isinstance(value, (list, tuple, ListConfig)):
        raise TypeError(
            f"{name} must be an explicit sequence of three integers, "
            f"got {value!r} ({type(value).__name__})."
        )

    if len(value) != 3:
        raise ValueError(
            f"{name} must contain exactly three dimensions, "
            f"got {len(value)} values: {list(value)!r}."
        )

    dimensions = tuple(
        _require_positive_int(dimension, f"{name}[{axis}]")
        for axis, dimension in enumerate(value)
    )

    return dimensions


def _select_required(cfg: DictConfig, path: str) -> Any:
    """
    Read one required OmegaConf field and raise a clear error if it is absent.
    """
    value = OmegaConf.select(cfg, path, default=None)

    if value is None:
        raise KeyError(f"Missing required anisotropic configuration field: {path}")

    return value


def _reject_legacy_geometry_keys(cfg: DictConfig) -> None:
    """
    Prevent legacy cubic geometry fields from becoming concurrent sources of
    truth in the dedicated anisotropic pipeline.

    The anisotropic pipeline uses only:
        - crops.backbone_reference_size
        - crops.global_crops_shape
        - crops.local_crops_size
    """
    if "crops" in cfg and "global_crops_size" in cfg.crops:
        raise ValueError(
            "Legacy field crops.global_crops_size is not supported by the "
            "dedicated anisotropic pipeline. "
            "Use crops.global_crops_shape for the real input geometry and "
            "crops.backbone_reference_size for backbone construction."
        )

    if "train" in cfg and "preprocess_target_size" in cfg.train:
        raise ValueError(
            "Legacy field train.preprocess_target_size is not supported by "
            "the dedicated anisotropic pipeline. "
            "The preprocessing target is defined exclusively by "
            "crops.global_crops_shape."
        )


def resolve_anisotropic_geometry(cfg: DictConfig) -> DictConfig:
    """
    Validate the anisotropic geometry and add all derived runtime fields.

    Required source fields:
        crops.backbone_reference_size
        crops.global_crops_shape
        crops.local_crops_size
        student.patch_size

    Example:
        backbone_reference_size = 112
        global_crops_shape       = (32, 112, 96)
        local_crops_size         = 32
        patch_size               = 16

    Derived values:
        global_patch_grid        = (2, 7, 6)
        n_global_patch_tokens    = 84

        local_crop_shape         = (32, 32, 32)
        local_patch_grid         = (2, 2, 2)
        n_local_patch_tokens     = 8

    The resulting values are stored under:

        cfg.runtime_geometry

    Lists are used inside OmegaConf so the resolved configuration can be
    serialized cleanly to YAML.

    Args:
        cfg:
            Merged anisotropic OmegaConf configuration.

    Returns:
        The same configuration object with cfg.runtime_geometry added.

    Raises:
        TypeError, ValueError, KeyError:
            On invalid or missing geometry fields.
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError(
            "resolve_anisotropic_geometry expects an OmegaConf DictConfig, "
            f"got {type(cfg).__name__}."
        )

    _reject_legacy_geometry_keys(cfg)

    backbone_reference_size = _require_positive_int(
        _select_required(cfg, "crops.backbone_reference_size"),
        "crops.backbone_reference_size",
    )

    patch_size = _require_positive_int(
        _select_required(cfg, "student.patch_size"),
        "student.patch_size",
    )

    global_crop_shape = normalize_shape_3d(
        _select_required(cfg, "crops.global_crops_shape"),
        name="global_crops_shape",
    )

    local_crops_size = _require_positive_int(
        _select_required(cfg, "crops.local_crops_size"),
        "crops.local_crops_size",
    )

    # ---------------------------------------------------------------------
    # Backbone reference geometry
    # ---------------------------------------------------------------------
    if backbone_reference_size % patch_size != 0:
        raise ValueError(
            "crops.backbone_reference_size must be divisible by "
            "student.patch_size. "
            f"Got backbone_reference_size={backbone_reference_size}, "
            f"patch_size={patch_size}."
        )

    # ---------------------------------------------------------------------
    # Global crop geometry
    # ---------------------------------------------------------------------
    for axis, dimension in enumerate(global_crop_shape):
        if dimension % patch_size != 0:
            raise ValueError(
                f"global_crops_shape[{axis}]={dimension} is not divisible "
                f"by patch_size={patch_size}"
            )

    global_patch_grid = tuple(
        dimension // patch_size for dimension in global_crop_shape
    )
    n_global_patch_tokens = math.prod(global_patch_grid)

    # ---------------------------------------------------------------------
    # Local crop geometry
    # ---------------------------------------------------------------------
    if local_crops_size % patch_size != 0:
        raise ValueError(
            "crops.local_crops_size must be divisible by "
            "student.patch_size. "
            f"Got local_crops_size={local_crops_size}, "
            f"patch_size={patch_size}."
        )

    for axis, global_dimension in enumerate(global_crop_shape):
        if local_crops_size > global_dimension:
            raise ValueError(
                f"local_crops_size={local_crops_size} exceeds "
                f"global_crops_shape[{axis}]={global_dimension}. "
                "A local crop must fit inside every global spatial axis when "
                "allow_smaller=False."
            )

    local_crop_shape = (
        local_crops_size,
        local_crops_size,
        local_crops_size,
    )

    local_patch_grid = tuple(dimension // patch_size for dimension in local_crop_shape)
    n_local_patch_tokens = math.prod(local_patch_grid)

    # ---------------------------------------------------------------------
    # Derived runtime geometry
    # ---------------------------------------------------------------------
    runtime_geometry = OmegaConf.create(
        {
            "backbone_reference_size": backbone_reference_size,
            "global_crop_shape": list(global_crop_shape),
            "global_patch_grid": list(global_patch_grid),
            "n_global_patch_tokens": int(n_global_patch_tokens),
            "local_crop_shape": list(local_crop_shape),
            "local_patch_grid": list(local_patch_grid),
            "n_local_patch_tokens": int(n_local_patch_tokens),
        }
    )

    # open_dict keeps this robust even if struct mode is enabled later.
    # Any manually supplied runtime_geometry block is intentionally
    # overwritten because these values must always be derived.
    with open_dict(cfg):
        cfg.runtime_geometry = runtime_geometry

    return cfg


def _log_resolved_geometry(cfg: DictConfig) -> None:
    """
    Log the resolved geometry after logging has been initialized.
    """
    geometry = cfg.runtime_geometry

    logger.info("############################################")
    logger.info("Resolved anisotropic geometry")
    logger.info("Backbone reference size : " f"{int(geometry.backbone_reference_size)}")
    logger.info("Global crop shape       : " f"{tuple(geometry.global_crop_shape)}")
    logger.info("Global patch grid       : " f"{tuple(geometry.global_patch_grid)}")
    logger.info("Global patch tokens     : " f"{int(geometry.n_global_patch_tokens)}")
    logger.info("Local crop shape        : " f"{tuple(geometry.local_crop_shape)}")
    logger.info("Local patch grid        : " f"{tuple(geometry.local_patch_grid)}")
    logger.info("Local patch tokens      : " f"{int(geometry.n_local_patch_tokens)}")
    logger.info("############################################")


def get_cfg_from_args_3d_anisotropic(args) -> DictConfig:
    """
    Build the anisotropic configuration using this merge order:

        dedicated anisotropic default
            <
        experiment YAML
            <
        command-line overrides
            <
        explicit --output-dir

    The dedicated default is loaded directly from:

        dinov2/configs/ssl3d_default_config_anisotropic.yaml

    This avoids modifying dinov2/configs/__init__.py or the historical cubic
    configuration machinery.

    Args:
        args:
            Parsed command-line arguments. Expected fields:
                - config_file
                - output_dir
                - opts

    Returns:
        Merged but not yet geometry-resolved DictConfig.
    """
    if not DEFAULT_ANISOTROPIC_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            "Dedicated anisotropic default configuration not found: "
            f"{DEFAULT_ANISOTROPIC_CONFIG_PATH}"
        )

    args.output_dir = os.path.abspath(args.output_dir)

    default_cfg = OmegaConf.load(DEFAULT_ANISOTROPIC_CONFIG_PATH)

    config_file = str(getattr(args, "config_file", "") or "").strip()

    if config_file:
        experiment_path = Path(config_file).expanduser()

        if not experiment_path.is_file():
            raise FileNotFoundError(
                f"Anisotropic experiment configuration not found: " f"{experiment_path}"
            )

        experiment_cfg = OmegaConf.load(experiment_path)
    else:
        experiment_cfg = OmegaConf.create({})

    cli_options = list(getattr(args, "opts", None) or [])
    cli_cfg = OmegaConf.from_cli(cli_options) if cli_options else OmegaConf.create({})

    # --output-dir must remain authoritative even if the YAML or CLI opts
    # contain train.output_dir.
    runtime_cfg = OmegaConf.create(
        {
            "train": {
                "output_dir": args.output_dir,
            }
        }
    )

    cfg = OmegaConf.merge(
        default_cfg,
        experiment_cfg,
        cli_cfg,
        runtime_cfg,
    )

    return cfg


def setup_3d_anisotropic(args) -> DictConfig:
    """
    Create, validate, initialize and save the dedicated anisotropic config.

    Execution order:
        1. load and merge configuration sources
        2. validate and resolve all geometry
        3. create output directory
        4. initialize distributed/logging/random seeds
        5. apply learning-rate scaling
        6. log the resolved geometry
        7. save the complete resolved configuration

    Geometry validation is intentionally performed before distributed setup so
    malformed configurations fail immediately.

    Args:
        args:
            Parsed training command-line arguments.

    Returns:
        Fully resolved anisotropic configuration.
    """
    cfg = get_cfg_from_args_3d_anisotropic(args)
    cfg = resolve_anisotropic_geometry(cfg)

    os.makedirs(args.output_dir, exist_ok=True)

    # Reused from sulcus_aware_3DINO.training.setup (wraps upstream 3DINO).
    # This initializes distributed execution, logging and random seeds.
    default_setup(args, cfg)

    # Must run after distributed initialization because the scaling rule uses
    # distributed.get_global_size().
    apply_scaling_rules_to_cfg(cfg)

    _log_resolved_geometry(cfg)

    write_config(
        cfg,
        args.output_dir,
        name="config.yaml",
    )

    return cfg
