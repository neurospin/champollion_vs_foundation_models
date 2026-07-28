#!/usr/bin/env python3
"""Search the per-ROI voxel intensity mapping for the skeleton modality.

Sweeps all valid ``(p0, p1)`` mappings, extracts mean-pool features under each,
runs the 5-fold CV probe, and writes the full grid plus the top-k mappings
(ranked by CV score only). Copy the selected mapping into the config's
``optimal_mapping`` section for the main probing runs.

Skeleton modality only. Reads ``config.normalizer_search`` (grid_step, top_k,
checkpoints) and ``config.probe.normalizer_search_C``.

Example::

    python scripts/normalizer_search.py \
        --config configs/skeleton/config_probe_dino3d.yaml \
        --roi ofc --preprocessing upscale_pad
"""

from __future__ import annotations

import argparse

import yaml

from linear_prober.skeleton.normalizer_search import search
from linear_prober.skeleton.preprocessor import ALL_PREPROCESSINGS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Skeleton intensity-mapping grid search.")
    p.add_argument("--config", required=True)
    p.add_argument("--roi", required=True, choices=["fip", "lc", "ofc", "sc"])
    p.add_argument("--preprocessing", required=True, choices=sorted(ALL_PREPROCESSINGS))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    search(config, args.roi, args.preprocessing)


if __name__ == "__main__":
    main()
