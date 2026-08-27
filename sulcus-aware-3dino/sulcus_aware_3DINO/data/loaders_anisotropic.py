# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from omegaconf import DictConfig, OmegaConf

from dinov2.data import (
    SamplerType,
    make_data_loader,
)
from .sulcal_npy_dataset_anisotropic import (
    SulcalNpyArrayDatasetAnisotropic,
)

logger = logging.getLogger("dinov2")


__all__ = [
    "SamplerType",
    "make_data_loader",
    "make_sulcal_npy_dataset_anisotropic_3d",
]


def _select_required(cfg: DictConfig, path: str):
    """
    Read one mandatory configuration field.

    Args:
        cfg:
            OmegaConf configuration.

        path:
            Dot-separated field path.

    Returns:
        The selected configuration value.

    Raises:
        KeyError:
            If the field is absent or null.
    """
    value = OmegaConf.select(
        cfg,
        path,
        default=None,
    )

    if value is None:
        raise KeyError("Missing required anisotropic configuration field: " f"{path}")

    return value


def make_sulcal_npy_dataset_anisotropic_3d(
    *,
    cfg: DictConfig,
    transform: Optional[Callable] = None,
) -> SulcalNpyArrayDatasetAnisotropic:
    """
    Build the dedicated anisotropic sulcal SSL dataset.

    This first implementation intentionally supports only:

        train.dataset_format = "npy_array"

    Expected raw NPY layouts:
        - [N,D,H,W]
        - [N,D,H,W,1]

    The final preprocessing target is read exclusively from:

        cfg.runtime_geometry.global_crop_shape

    For the A. Cingulate configuration:

        raw sample:
            [18,73,57] or [18,73,57,1]

        deterministic preprocessed sample:
            [1,32,112,96]

        transformed sample:
            whatever is returned by the supplied SSL augmentation transform.

    Args:
        cfg:
            Fully resolved anisotropic configuration.

            It must already have been passed through:

                resolve_anisotropic_geometry(cfg)

        transform:
            Optional transform applied after deterministic anisotropic
            preprocessing.

            During SSL training this will be the dedicated anisotropic
            multi-crop augmentation.

    Returns:
        A SulcalNpyArrayDatasetAnisotropic instance.

    Raises:
        TypeError:
            If cfg is not an OmegaConf DictConfig.

        ValueError:
            If dataset_format is not "npy_array" or dataset_path is empty.

        KeyError:
            If a required configuration field is absent.
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError(
            "make_sulcal_npy_dataset_anisotropic_3d expects an "
            f"OmegaConf DictConfig, got {type(cfg).__name__}."
        )

    dataset_format = str(_select_required(cfg, "train.dataset_format")).strip()

    if dataset_format != "npy_array":
        raise ValueError(
            "The dedicated anisotropic SSL pipeline supports only "
            "train.dataset_format='npy_array'. "
            f"Got train.dataset_format={dataset_format!r}. "
            "The historical JSON datalist pathway is intentionally disabled."
        )

    dataset_path_raw = str(_select_required(cfg, "train.dataset_path")).strip()

    if not dataset_path_raw:
        raise ValueError(
            "train.dataset_path must point to the large raw Rskeleton.npy "
            "array when train.dataset_format='npy_array'."
        )

    dataset_path = Path(dataset_path_raw).expanduser()

    target_shape = tuple(
        int(dimension)
        for dimension in _select_required(
            cfg,
            "runtime_geometry.global_crop_shape",
        )
    )

    mmap_mode = OmegaConf.select(
        cfg,
        "train.npy_mmap_mode",
        default="r",
    )

    input_layout = str(
        OmegaConf.select(
            cfg,
            "train.npy_input_layout",
            default="auto",
        )
    )

    binarize_nonzero = OmegaConf.select(
        cfg,
        "train.binarize_nonzero",
        default=True,
    )

    if not isinstance(binarize_nonzero, bool):
        raise TypeError(
            "train.binarize_nonzero must be a boolean, "
            f"got {binarize_nonzero!r} "
            f"({type(binarize_nonzero).__name__})."
        )

    logger.info("##################################################")
    logger.info("Creating anisotropic sulcal SSL NPY dataset")
    logger.info("Dataset format        : %s", dataset_format)
    logger.info("Dataset path          : %s", dataset_path)
    logger.info("Target spatial shape  : %s", target_shape)
    logger.info("NPY mmap mode         : %s", mmap_mode)
    logger.info("NPY input layout      : %s", input_layout)
    logger.info("Binarize non-zero     : %s", binarize_nonzero)
    logger.info("Transform             : %s", type(transform).__name__)
    logger.info("JSON datalist         : disabled")
    logger.info("MONAI cache           : disabled")
    logger.info("Individual NPY files  : disabled")
    logger.info("Axis permutation      : none")
    logger.info("##################################################")

    dataset = SulcalNpyArrayDatasetAnisotropic(
        npy_path=dataset_path,
        transform=transform,
        target_shape=target_shape,
        mmap_mode=mmap_mode,
        input_layout=input_layout,
        binarize_nonzero=binarize_nonzero,
    )

    if dataset.target_shape != target_shape:
        raise RuntimeError(
            "The constructed dataset target shape does not match the "
            "resolved runtime geometry: "
            f"expected {target_shape}, got {dataset.target_shape}."
        )

    if dataset.transform is not transform:
        raise RuntimeError(
            "The supplied transform was not retained by the anisotropic " "dataset."
        )

    logger.info(
        "Anisotropic dataset ready | samples: %s",
        f"{len(dataset):,d}",
    )

    return dataset
