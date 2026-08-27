# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .sulcal_preprocessing_anisotropic import (
    normalize_target_shape,
    preprocess_sulcal_volume_anisotropic,
)

logger = logging.getLogger("dinov2")


class SulcalNpyArrayDatasetAnisotropic(Dataset):
    """
    Dataset for anisotropic sulcal SSL training from one large NPY array.

    Supported raw array layouts:
        - [N, D, H, W]
        - [N, D, H, W, 1]

    Per-sample deterministic preprocessing output:
        - torch.Tensor [1, target_d, target_h, target_w]
        - float32
        - binary {0,1} when binarize_nonzero=True

    Default target:
        [1, 32, 112, 96]

    Important guarantees:
        - no axis permutation is applied;
        - an optional trailing singleton channel is removed;
        - the NPY file is memory-mapped by default;
        - the initial metadata handle is not retained;
        - each DataLoader worker lazily opens its own array handle;
        - the memmap handle is excluded when the dataset is pickled.
    """

    SUPPORTED_INPUT_LAYOUTS = (
        "auto",
        "NDHW",
        "NDHW1",
    )

    SUPPORTED_MMAP_MODES = (
        "r",
        "r+",
        "w+",
        "c",
        None,
    )

    def __init__(
        self,
        npy_path: str | Path,
        transform: Optional[Callable] = None,
        target_shape=(32, 112, 96),
        mmap_mode: Optional[str] = "r",
        input_layout: str = "auto",
        binarize_nonzero: bool = True,
    ) -> None:
        """
        Args:
            npy_path:
                Path to one dense fixed-shape NPY array.

            transform:
                Optional transform applied after deterministic anisotropic
                preprocessing.

                For SSL, this will later be the dedicated anisotropic DINO
                augmentation.

            target_shape:
                Final spatial shape in [D,H,W] order.

                Default:
                    (32, 112, 96)

            mmap_mode:
                NumPy memory-map mode. Use "r" for read-only training access.

            input_layout:
                - "auto": accept NDHW or NDHW1;
                - "NDHW": require [N,D,H,W];
                - "NDHW1": require [N,D,H,W,1].

            binarize_nonzero:
                If True, every raw non-zero voxel becomes 1 before resizing.
        """
        self.npy_path = Path(npy_path).expanduser()
        self.transform = transform

        self.target_shape = normalize_target_shape(
            target_shape,
            name="target_shape",
        )

        self.mmap_mode = mmap_mode
        self.input_layout = str(input_layout)
        self.binarize_nonzero = bool(binarize_nonzero)

        # The array is never opened permanently in the parent process during
        # construction. Every worker initializes this field lazily.
        self._array = None

        self._validate_constructor_arguments()
        self._validate_path()
        self._read_and_validate_metadata()

        logger.info("##################################################")
        logger.info("Using SulcalNpyArrayDatasetAnisotropic")
        logger.info("NPY path              : %s", self.npy_path)
        logger.info("Raw array shape       : %s", self.shape)
        logger.info("Raw array dtype       : %s", self.dtype)
        logger.info("Number of subjects    : %s", f"{self.num_samples:,d}")
        logger.info("Raw spatial shape     : %s", self.raw_spatial_shape)
        logger.info("Target spatial shape  : %s", self.target_shape)
        logger.info("mmap_mode             : %s", self.mmap_mode)
        logger.info("input_layout          : %s", self.input_layout)
        logger.info("binarize_nonzero      : %s", self.binarize_nonzero)
        logger.info("Accepted convention   : NDHW or NDHW1")
        logger.info("Axis permutation      : none")
        logger.info("Trailing channel      : squeezed if present")
        logger.info("Worker memmap opening : lazy and worker-local")
        logger.info("##################################################")

    def _validate_constructor_arguments(self) -> None:
        """
        Validate non-file constructor arguments.
        """
        if self.input_layout not in self.SUPPORTED_INPUT_LAYOUTS:
            raise ValueError(
                "input_layout must be one of "
                f"{self.SUPPORTED_INPUT_LAYOUTS}, "
                f"got {self.input_layout!r}."
            )

        if self.mmap_mode not in self.SUPPORTED_MMAP_MODES:
            raise ValueError(
                "mmap_mode must be one of "
                f"{self.SUPPORTED_MMAP_MODES}, "
                f"got {self.mmap_mode!r}."
            )

    def _validate_path(self) -> None:
        """
        Validate the NPY path before opening it.
        """
        if not self.npy_path.exists():
            raise FileNotFoundError(f"NPY file not found: {self.npy_path}")

        if not self.npy_path.is_file():
            raise ValueError(f"Expected a file, got: {self.npy_path}")

        if self.npy_path.suffix.lower() != ".npy":
            raise ValueError(
                "SulcalNpyArrayDatasetAnisotropic expects a .npy file, "
                f"got: {self.npy_path}"
            )

    def _read_and_validate_metadata(self) -> None:
        """
        Read and validate only the NPY metadata.

        This handle is deliberately discarded. The actual data handle is
        reopened lazily by each process that calls __getitem__().
        """
        array = np.load(
            self.npy_path,
            mmap_mode=self.mmap_mode,
            allow_pickle=False,
        )

        try:
            shape = tuple(int(value) for value in array.shape)
            dtype = np.dtype(array.dtype)
        finally:
            # Do not retain the constructor-time array or memmap handle.
            del array

        if len(shape) not in (4, 5):
            raise ValueError(
                "Expected raw array shape [N,D,H,W] or [N,D,H,W,1], " f"got {shape}."
            )

        num_samples = int(shape[0])

        if num_samples <= 0:
            raise ValueError(f"Dataset is empty: shape={shape}.")

        if len(shape) == 4:
            raw_spatial_shape = shape[1:4]
            inferred_layout = "NDHW"
        else:
            if shape[-1] != 1:
                raise ValueError(
                    "Expected a final singleton channel for a five-dimensional "
                    "array [N,D,H,W,1], "
                    f"got shape {shape}."
                )

            raw_spatial_shape = shape[1:4]
            inferred_layout = "NDHW1"

        if any(dimension <= 0 for dimension in raw_spatial_shape):
            raise ValueError(
                "All raw spatial dimensions must be strictly positive, "
                f"got {raw_spatial_shape}."
            )

        if self.input_layout == "NDHW" and inferred_layout != "NDHW":
            raise ValueError(
                "input_layout='NDHW' requires shape [N,D,H,W], " f"got {shape}."
            )

        if self.input_layout == "NDHW1" and inferred_layout != "NDHW1":
            raise ValueError(
                "input_layout='NDHW1' requires shape [N,D,H,W,1], " f"got {shape}."
            )

        # Supported sulcal volumes are bool, integer or real floating-point
        # arrays. Object, string, structured and complex arrays are rejected.
        dtype_is_supported = (
            np.issubdtype(dtype, np.bool_)
            or np.issubdtype(dtype, np.integer)
            or np.issubdtype(dtype, np.floating)
        )

        if not dtype_is_supported:
            raise ValueError(
                "Unsupported NPY dtype. Expected a boolean, integer or "
                f"floating-point dense array, got dtype={dtype}."
            )

        self.shape = shape
        self.dtype = dtype
        self.num_samples = num_samples
        self.raw_spatial_shape = tuple(raw_spatial_shape)
        self.inferred_layout = inferred_layout

    def _lazy_init_array(self) -> None:
        """
        Open the NPY array lazily inside the process using the dataset.

        With a multi-worker DataLoader, each worker starts with _array=None and
        therefore opens its own NumPy array or memmap handle.
        """
        if self._array is not None:
            return

        array = np.load(
            self.npy_path,
            mmap_mode=self.mmap_mode,
            allow_pickle=False,
        )

        # Protect against the file being replaced or modified after dataset
        # construction.
        if tuple(array.shape) != self.shape:
            actual_shape = tuple(array.shape)
            del array

            raise RuntimeError(
                "The NPY array shape changed after dataset initialization: "
                f"expected {self.shape}, got {actual_shape}."
            )

        if np.dtype(array.dtype) != self.dtype:
            actual_dtype = np.dtype(array.dtype)
            del array

            raise RuntimeError(
                "The NPY array dtype changed after dataset initialization: "
                f"expected {self.dtype}, got {actual_dtype}."
            )

        self._array = array

    def __getstate__(self):
        """
        Return a picklable dataset state without an open array handle.

        This method is especially important when DataLoader workers use a
        multiprocessing start method that pickles the dataset.
        """
        state = self.__dict__.copy()
        state["_array"] = None
        return state

    def __len__(self) -> int:
        """
        Return the number of subjects.
        """
        return self.num_samples

    def _normalize_index(self, index: int) -> int:
        """
        Normalize and validate one dataset index.
        """
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError(
                "Dataset index must be an integer, "
                f"got {index!r} ({type(index).__name__})."
            )

        index = int(index)

        if index < 0:
            index += self.num_samples

        if index < 0 or index >= self.num_samples:
            raise IndexError(
                f"Index {index} out of range for dataset of "
                f"size {self.num_samples}."
            )

        return index

    def __getitem__(self, index: int):
        """
        Load, preprocess and optionally augment one subject.

        Deterministic output before transform:
            [1,target_d,target_h,target_w]
        """
        index = self._normalize_index(index)

        self._lazy_init_array()

        volume_np = self._array[index]

        # Keep the original axis order. np.asarray does not transpose or
        # permute the sample.
        volume_np = np.asarray(volume_np)

        expected_raw_sample_shape = (
            self.raw_spatial_shape
            if self.inferred_layout == "NDHW"
            else (*self.raw_spatial_shape, 1)
        )

        if tuple(volume_np.shape) != tuple(expected_raw_sample_shape):
            raise RuntimeError(
                f"Unexpected raw sample shape at index {index}: "
                f"expected {expected_raw_sample_shape}, "
                f"got {tuple(volume_np.shape)}."
            )

        x = preprocess_sulcal_volume_anisotropic(
            volume_np=volume_np,
            target_shape=self.target_shape,
            binarize_nonzero=self.binarize_nonzero,
        )

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "preprocess_sulcal_volume_anisotropic must return a "
                f"torch.Tensor, got {type(x).__name__}."
            )

        expected_output_shape = (
            1,
            *self.target_shape,
        )

        if tuple(x.shape) != expected_output_shape:
            raise RuntimeError(
                "Unexpected preprocessed tensor shape: "
                f"expected {expected_output_shape}, "
                f"got {tuple(x.shape)}."
            )

        if x.dtype != torch.float32:
            raise RuntimeError(
                "The anisotropic dataset expects float32 preprocessing output, "
                f"got {x.dtype}."
            )

        if self.binarize_nonzero:
            is_binary = torch.logical_or(
                x == 0,
                x == 1,
            ).all()

            if not bool(is_binary.item()):
                unique_values = torch.unique(x.detach().cpu())

                raise RuntimeError(
                    "The preprocessed sulcal volume must be binary {0,1}, "
                    f"got values such as {unique_values[:10].tolist()}."
                )

        if self.transform is not None:
            return self.transform(x)

        return x
