"""Feature cache — extract once, reuse across probing runs.

Frozen-encoder features are expensive to compute and are reused across every
mode, task and hyperparameter sweep, so they are cached to ``.npz`` on first
extraction. The cache is keyed by the file path built in
:mod:`linear_prober.core.paths`; the actual extraction is supplied by the
caller as a zero-argument thunk, which keeps this module agnostic to how each
modality reads its data.

``np.savez`` (not ``np.savez_compressed``) is used on purpose: compression
triples peak RAM on large ``flatten`` archives for a marginal disk saving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import numpy as np


def load_or_extract(
    cache_path: str | Path,
    extract_thunk: Callable[[], Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Return cached features if present, otherwise extract, cache and return.

    Args:
        cache_path: destination ``.npz`` path (see
            :func:`linear_prober.core.paths.build_feature_path`).
        extract_thunk: zero-argument callable returning the feature dict
            (keys: ``features``, ``labels``, ``folds``, ``splits``,
            ``subjects``, ``volume_indices``).
    """
    cache_path = Path(cache_path)

    if cache_path.is_file():
        print(f"[Features] Cache hit: {cache_path}")
        data = np.load(str(cache_path), allow_pickle=True)
        result = {k: data[k] for k in data.files}
        print(f"[Features] Shape: {result['features'].shape}")
        return result

    print(f"[Features] Extracting -> {cache_path}")
    result = extract_thunk()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cache_path), **result)
    print(f"[Features] Saved: {cache_path}")
    return result
