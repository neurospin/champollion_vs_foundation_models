#!/usr/bin/env python3
"""Unified linear-probing entry point for both modalities.

Examples
--------
Skeleton (binary sulcal grids) — a geometric preprocessing is required::

    python scripts/run_probe.py --modality skeleton \
        --config configs/skeleton/config_probe_dino3d.yaml \
        --roi ofc --preprocessing upscale_pad

MRI (intensity crops) — single native preprocessing, no ``--preprocessing``::

    python scripts/run_probe.py --modality mri \
        --config configs/mri/config_probe_dino3d.yaml --roi ofc

``--roi`` valid values: skeleton = {fip, lc, ofc, sc}; MRI = {fip, ofc, sc}.
Omit ``--mode`` to run all feature modes.
"""

from __future__ import annotations

import argparse

import yaml

from linear_prober.core.tasks import ROI_TASK
from linear_prober.mri import runner as mri_runner
from linear_prober.skeleton import runner as skeleton_runner
from linear_prober.skeleton.preprocessor import ALL_PREPROCESSINGS

STANDARD_MODES = ["mean_pool", "mean_pool_multi_layers", "flatten"]
MRI_ROIS = {"fip", "ofc", "sc"}  # MRI has no LC ROI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot linear probing (skeleton | mri)."
    )
    p.add_argument("--modality", required=True, choices=["skeleton", "mri"])
    p.add_argument("--config", required=True)
    p.add_argument("--roi", required=True, choices=sorted(ROI_TASK))
    p.add_argument(
        "--preprocessing",
        default=None,
        choices=sorted(ALL_PREPROCESSINGS),
        help="Skeleton only: geometric preprocessing (required for skeleton).",
    )
    p.add_argument(
        "--mode",
        default=None,
        choices=STANDARD_MODES,
        help="Single feature mode. Default: all modes.",
    )
    p.add_argument(
        "--flatten-raw",
        action="store_true",
        help="Skeleton only: high-dimensional flatten probe without PCA.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Validate the modality-specific argument combination before touching disk.
    if args.modality == "skeleton":
        if args.preprocessing is None:
            raise SystemExit("--preprocessing is required for --modality skeleton")
    else:
        if args.roi not in MRI_ROIS:
            raise SystemExit(
                f"--roi {args.roi} is not available for MRI (use {sorted(MRI_ROIS)})"
            )
        if args.preprocessing is not None:
            raise SystemExit("--preprocessing is not applicable to --modality mri")
        if args.flatten_raw:
            raise SystemExit("--flatten-raw is not applicable to --modality mri")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.modality == "skeleton":
        skeleton_runner.run(
            config,
            args.roi,
            args.preprocessing,
            mode=args.mode,
            flatten_raw=args.flatten_raw,
        )
    else:
        mri_runner.run(config, args.roi, mode=args.mode)


if __name__ == "__main__":
    main()
