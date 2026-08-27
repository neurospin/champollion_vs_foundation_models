# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

"""Sulcal data pipeline: datasets, preprocessing, augmentations, masking, collate.

The sulcal factories live here; ``SamplerType`` and ``make_data_loader`` are the
unmodified upstream 3DINO entry points, re-exported for a single import surface.
"""

from dinov2.data import SamplerType, make_data_loader

from .augmentations import DataAugmentationDINO3d_sulcal
from .collate import collate_data_and_cast
from .loaders import make_sulcal_npy_dataset_3d
from .masking_non_empty import MaskingGenerator3d

__all__ = [
    "SamplerType",
    "make_data_loader",
    "make_sulcal_npy_dataset_3d",
    "collate_data_and_cast",
    "MaskingGenerator3d",
    "DataAugmentationDINO3d_sulcal",
]
