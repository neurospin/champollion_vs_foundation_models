# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .sulcal_preprocessing import preprocess_sulcal_volume

logger = logging.getLogger("dinov2")


class SulcalNpyArrayDataset(Dataset):
    """
    Dataset for sulcal SSL training from one large .npy array.

    Expected file format:
      - [N, D, H, W]
      - [N, D, H, W, 1]

    Per-sample output after preprocessing:
      - torch.Tensor [1, target, target, target]
      - float32
      - binary {0,1}

    Important:
      - The final parasite singleton channel is removed if present.
      - No axis permutation is applied.
      - The .npy is opened with numpy memmap by default.
      - Each dataloader worker lazily opens its own memmap handle.
    """

    def __init__(
        self,
        npy_path: str | Path,
        transform: Optional[Callable] = None,
        target_size: int = 112,
        mmap_mode: str = "r",
        input_layout: str = "auto",
        binarize_nonzero: bool = True,
    ):
        self.npy_path = Path(npy_path)
        self.transform = transform
        self.target_size = int(target_size)
        self.mmap_mode = mmap_mode
        self.input_layout = input_layout
        self.binarize_nonzero = bool(binarize_nonzero)

        self._array = None

        self._validate_path()
        self._read_and_validate_metadata()

        logger.info("###################################")
        logger.info("Using SulcalNpyArrayDataset")
        logger.info(f"NPY path              : {self.npy_path}")
        logger.info(f"Raw array shape       : {self.shape}")
        logger.info(f"Raw array dtype       : {self.dtype}")
        logger.info(f"Number of subjects    : {self.num_samples:,d}")
        logger.info(f"Target size           : {self.target_size}")
        logger.info(f"mmap_mode             : {self.mmap_mode}")
        logger.info(f"input_layout          : {self.input_layout}")
        logger.info(f"binarize_nonzero      : {self.binarize_nonzero}")
        logger.info("Convention            : [N,D,H,W] or [N,D,H,W,1]")
        logger.info("Axis permutation      : none")
        logger.info("Trailing channel      : squeezed if present")
        logger.info("###################################")

    def _validate_path(self) -> None:
        if not self.npy_path.exists():
            raise FileNotFoundError(f"NPY file not found: {self.npy_path}")

        if self.npy_path.suffix != ".npy":
            raise ValueError(
                f"SulcalNpyArrayDataset expects a .npy file, got: {self.npy_path}"
            )

    def _read_and_validate_metadata(self) -> None:
        arr = np.load(
            self.npy_path,
            mmap_mode=self.mmap_mode,
            allow_pickle=False,
        )

        self.shape = tuple(arr.shape)
        self.dtype = arr.dtype
        self.num_samples = int(arr.shape[0])

        # Do not keep this initial handle. Each worker will reopen lazily.
        del arr

        if self.dtype == np.dtype("O"):
            raise ValueError(
                "Object dtype arrays are not supported. "
                "Expected a dense fixed-shape array [N,D,H,W] or [N,D,H,W,1]."
            )

        if len(self.shape) not in (4, 5):
            raise ValueError(
                "Expected raw array shape [N,D,H,W] or [N,D,H,W,1], "
                f"got {self.shape}"
            )

        if len(self.shape) == 5 and self.shape[-1] != 1:
            raise ValueError(
                "Expected final singleton channel for 5D array [N,D,H,W,1], "
                f"got shape {self.shape}"
            )

        if self.num_samples <= 0:
            raise ValueError(f"Dataset is empty: shape={self.shape}")

        if self.input_layout not in ("auto", "NDHW", "NDHW1"):
            raise ValueError(
                "input_layout must be one of: 'auto', 'NDHW', 'NDHW1', "
                f"got {self.input_layout}"
            )

        if self.input_layout == "NDHW" and len(self.shape) != 4:
            raise ValueError(
                f"input_layout='NDHW' requires shape [N,D,H,W], got {self.shape}"
            )

        if self.input_layout == "NDHW1" and len(self.shape) != 5:
            raise ValueError(
                f"input_layout='NDHW1' requires shape [N,D,H,W,1], got {self.shape}"
            )

    def _lazy_init_array(self) -> None:
        """
        Open the memmap lazily.

        This is important with torch DataLoader workers:
        each worker should own its own numpy memmap handle.
        """
        if self._array is None:
            self._array = np.load(
                self.npy_path,
                mmap_mode=self.mmap_mode,
                allow_pickle=False,
            )

    def __getstate__(self):
        """
        Make the dataset safely picklable for multiprocessing workers.

        The memmap handle is intentionally dropped before pickling.
        Each worker reopens it lazily in _lazy_init_array().
        """
        state = self.__dict__.copy()
        state["_array"] = None
        return state

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        self._lazy_init_array()

        if index < 0:
            index = self.num_samples + index

        if index < 0 or index >= self.num_samples:
            raise IndexError(
                f"Index {index} out of range for dataset of size {self.num_samples}"
            )

        volume_np = self._array[index]

        # Ensure a normal ndarray view/array before torch conversion.
        # This keeps the preprocessing function independent from memmap specifics.
        volume_np = np.asarray(volume_np)

        x = preprocess_sulcal_volume(
            volume_np,
            target=self.target_size,
            binarize_nonzero=self.binarize_nonzero,
        )

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"preprocess_sulcal_volume should return torch.Tensor, got {type(x)}"
            )

        if tuple(x.shape) != (
            1,
            self.target_size,
            self.target_size,
            self.target_size,
        ):
            raise RuntimeError(
                "Unexpected preprocessed tensor shape: "
                f"expected {(1, self.target_size, self.target_size, self.target_size)}, "
                f"got {tuple(x.shape)}"
            )

        if self.transform is not None:
            return self.transform(x)

        return x
