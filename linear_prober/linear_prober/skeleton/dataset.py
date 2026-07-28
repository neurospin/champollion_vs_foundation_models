"""Datasets for the binary sulcal-skeleton modality.

Two datasets, both model-agnostic (no model-specific normalisation — that lives
in each model's ``normalizer``; geometric transforms live in
:mod:`linear_prober.skeleton.preprocessor`):

  :class:`HCPDataset`          — labelled HCP/ACC volumes for linear probing;
  :class:`UKBBSkeletonDataset` — unlabelled UKBB volumes for PCA fitting.

Binarisation policy
-------------------
Both datasets binarise in ``__getitem__``::

    x = (x != 0).float()   ->   {0.0, 1.0}

BrainVISA skeletons are multi-label ({0, 30, 35, ...}); binarising yields the
{0,1} skeleton the frozen encoders expect. It is a no-op for already-binary
volumes. The binarisation operates on an in-memory copy; the ``.npy`` on disk is
never modified (UKBB uses ``mmap_mode='r'``, and per-sample copies are detached).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# HCP dataset (classification + regression)
# =============================================================================


class HCPDataset(Dataset):
    """HCP/ACC sulcal-skeleton dataset for linear probing.

    Returns per sample a ``[1, D, H, W]`` float32 binarised volume plus its
    label, fold, split and indices. Classification labels are int scalars;
    regression labels are float32 ``[n_targets]`` tensors read from ``label_*``
    columns.
    """

    def __init__(
        self,
        volumes_path: str,
        master_table_path: str,
        task_type: str = "classification",  # "classification" | "regression"
        split: Optional[str] = None,  # None | "train_val" | "test"
        expected_shape: Optional[tuple] = None,
        strict_binary_check: bool = False,
    ) -> None:
        super().__init__()

        assert task_type in (
            "classification",
            "regression",
        ), f"task_type must be 'classification' or 'regression', got '{task_type}'"

        self.task_type = task_type
        self.expected_shape = expected_shape
        self.strict_binary_check = strict_binary_check

        # Volumes fit in RAM for HCP (~400-1100 subjects).
        self.volumes = np.load(str(volumes_path))  # [N, D, H, W]
        if self.volumes.ndim != 4:
            raise ValueError(f"Expected [N, D, H, W], got {self.volumes.shape}")

        if expected_shape is not None:
            if tuple(self.volumes.shape[1:]) != tuple(expected_shape):
                raise ValueError(
                    f"Expected per-volume shape {expected_shape}, "
                    f"got {tuple(self.volumes.shape[1:])}"
                )

        dtype_map = {"volume_index": int, "subject": str, "fold": int, "split": str}
        if task_type == "classification":
            dtype_map["label"] = int
        self.table = pd.read_csv(str(master_table_path), dtype=dtype_map)

        if split is not None:
            assert split in (
                "train_val",
                "test",
            ), f"split must be 'train_val' or 'test', got '{split}'"
            self.table = self.table[self.table["split"] == split].copy()

        self.table = self.table.sort_values("volume_index").reset_index(drop=True)
        if len(self.table) == 0:
            raise ValueError("No samples after split filter.")

        if task_type == "regression":
            self._label_cols = sorted(
                [c for c in self.table.columns if c.startswith("label_")]
            )
            if not self._label_cols:
                raise ValueError(
                    "No label_* columns found in master_table for regression task."
                )

        print(
            f"[HCPDataset] {len(self.table)} subjects | "
            f"volume shape: {self.volumes.shape[1:]} | task: {task_type}"
        )

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> dict:
        row = self.table.iloc[idx]
        vol = self.volumes[int(row["volume_index"])]  # [D, H, W]
        x = torch.from_numpy(vol[None].astype(np.float32))  # [1, D, H, W]
        x = (x != 0).float()

        if self.strict_binary_check:
            unique_vals = x.unique()
            if not set(unique_vals.tolist()).issubset({0.0, 1.0}):
                raise ValueError(
                    f"Non-binary volume at index {idx} after binarisation: "
                    f"{unique_vals[:5].tolist()}"
                )

        if self.task_type == "classification":
            label = int(row["label"])
        else:
            label = torch.tensor(
                [float(row[c]) for c in self._label_cols], dtype=torch.float32
            )

        return {
            "volume": x,
            "label": label,
            "subject": str(row["subject"]),
            "fold": int(row["fold"]),
            "split": str(row["split"]),
            "volume_index": int(row["volume_index"]),
        }


def build_hcp_dataloader(
    volumes_path: str,
    master_table_path: str,
    task_type: str = "classification",
    split: Optional[str] = None,
    expected_shape: Optional[tuple] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    strict_binary_check: bool = False,
) -> DataLoader:
    """Deterministic (never shuffled) dataloader over :class:`HCPDataset`.

    Shuffling is disabled because folds are pre-stratified and extraction order
    must be reproducible.
    """
    dataset = HCPDataset(
        volumes_path=volumes_path,
        master_table_path=master_table_path,
        task_type=task_type,
        split=split,
        expected_shape=expected_shape,
        strict_binary_check=strict_binary_check,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


# =============================================================================
# UKBB dataset (unsupervised, for PCA fitting)
# =============================================================================


class UKBBSkeletonDataset(Dataset):
    """UKBB sulcal-skeleton volumes for PCA fitting (no labels).

    The ``.npy`` is memory-mapped read-only (``mmap_mode='r'``); only the slice
    for each requested index is copied into RAM. A trailing singleton channel
    dimension ``[N, D, H, W, 1]`` is squeezed automatically. Volumes are
    binarised in ``__getitem__``.
    """

    def __init__(
        self,
        ukbb_volumes_path: str,
        expected_shape: Optional[tuple] = None,
        strict_binary_check: bool = False,
    ) -> None:
        super().__init__()

        self.strict_binary_check = strict_binary_check

        path = Path(ukbb_volumes_path)
        if not path.is_file():
            raise FileNotFoundError(f"UKBB volumes not found: {path}")

        raw = np.load(str(path), mmap_mode="r")

        if raw.ndim == 5 and raw.shape[-1] == 1:
            self._squeeze_last = True
            self.volumes = raw
            self._vol_shape = raw.shape[1:4]
        elif raw.ndim == 4:
            self._squeeze_last = False
            self.volumes = raw
            self._vol_shape = raw.shape[1:]
        else:
            raise ValueError(
                f"Expected [N, D, H, W] or [N, D, H, W, 1], got {raw.shape}"
            )

        if expected_shape is not None:
            if tuple(self._vol_shape) != tuple(expected_shape):
                raise ValueError(
                    f"Expected per-volume shape {expected_shape}, "
                    f"got {tuple(self._vol_shape)}"
                )

        print(
            f"[UKBB] {len(self.volumes)} subjects | volume shape: {self._vol_shape} | "
            f"mmap_mode='r'"
        )

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, idx: int) -> dict:
        vol = np.array(self.volumes[idx])  # copies the mmap slice into RAM
        if self._squeeze_last:
            vol = vol[..., 0]

        x = torch.from_numpy(vol[None].astype(np.float32))  # [1, D, H, W]
        x = (x != 0).float()

        if self.strict_binary_check:
            unique_vals = x.unique()
            if not set(unique_vals.tolist()).issubset({0.0, 1.0}):
                raise ValueError(
                    f"Non-binary volume at index {idx} after binarisation: "
                    f"{unique_vals[:5].tolist()}"
                )

        return {"volume": x, "subject": str(idx)}


def build_ukbb_dataloader(
    ukbb_volumes_path: str,
    expected_shape: Optional[tuple] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    strict_binary_check: bool = False,
) -> DataLoader:
    """Deterministic (never shuffled) dataloader over :class:`UKBBSkeletonDataset`."""
    dataset = UKBBSkeletonDataset(
        ukbb_volumes_path=ukbb_volumes_path,
        expected_shape=expected_shape,
        strict_binary_check=strict_binary_check,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
