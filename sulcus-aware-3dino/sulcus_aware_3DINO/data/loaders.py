# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

"""Sulcal SSL dataset factory.

Only the sulcal dataset factory lives here. The sampler/loader entry points
(``SamplerType``, ``make_data_loader``) are the unmodified upstream 3DINO ones
and are imported directly from ``dinov2.data`` by callers that need them.
"""

import logging
from typing import Callable, Optional

from .sulcal_npy_dataset import SulcalNpyArrayDataset

logger = logging.getLogger("dinov2")


def make_sulcal_npy_dataset_3d(
    *,
    dataset_path: str,
    target_size: int = 112,
    transform: Optional[Callable] = None,
    mmap_mode: str = "r",
    input_layout: str = "auto",
    binarize_nonzero: bool = True,
):
    """
    Create the SSL pretraining dataset from one large raw .npy array.

    Expected .npy format:
        - [N, D, H, W]
        - [N, D, H, W, 1]

    Per-sample preprocessing:
        1. index internally along the first dimension
        2. remove trailing parasite singleton channel if present
        3. binarize with x != 0
        4. isotropic upscale + centered zero-padding to target_size^3
        5. return torch.Tensor [1, target_size, target_size, target_size]
        6. apply the optional transform / DINO augmentation

    Notes:
        - No JSON datalist is required.
        - No individual .npy files are required.
        - No MONAI cache is used.
        - The large .npy is opened with numpy memmap by default.
        - No axis permutation is applied.

    Args:
        dataset_path:
            Path to the large raw .npy file.
        target_size:
            Final cubic spatial size, typically 112 for 3DINO-ViT.
        transform:
            Transform applied after deterministic preprocessing.
            For SSL, this should be DataAugmentationDINO3d_sulcal.
        mmap_mode:
            Numpy mmap mode. Use "r" to avoid loading the full array.
        input_layout:
            "auto", "NDHW", or "NDHW1".
        binarize_nonzero:
            If True, raw values are binarized with x != 0.

    Returns:
        A SulcalNpyArrayDataset.
    """
    logger.info("Creating SSL 3D sulcal dataset from single NPY array")
    logger.info(f"Dataset path      : {dataset_path}")
    logger.info(f"Target size       : {target_size}")
    logger.info(f"mmap_mode         : {mmap_mode}")
    logger.info(f"input_layout      : {input_layout}")
    logger.info(f"binarize_nonzero  : {binarize_nonzero}")
    logger.info("MONAI cache       : disabled")
    logger.info("JSON datalist     : disabled")
    logger.info("Axis permutation  : none")

    dataset = SulcalNpyArrayDataset(
        npy_path=dataset_path,
        transform=transform,
        target_size=target_size,
        mmap_mode=mmap_mode,
        input_layout=input_layout,
        binarize_nonzero=binarize_nonzero,
    )

    logger.info(f"Dataset samples   : {len(dataset):,d}")

    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)

    return dataset
