#!/usr/bin/env python3
"""DINOv3 linear-probing entry point (the 2D-slicing path).

DINOv3 is kept on a dedicated entry point because its run is parameterised by a
composite feature mode rather than the standard 3D modes. The 3D encoders
(dino3d, sam3d, vista3d, bsf) use ``scripts/run_probe.py`` instead.

Examples
--------
Skeleton — a geometric preprocessing is required::

    python scripts/run_probe_dinov3.py --modality skeleton \
        --config configs/skeleton/config_probe_dinov3.yaml \
        --roi ofc --preprocessing upscale_pad \
        --model_size vitb16 --slicer_mode 2d \
        --extraction mean_pool --aggregation mean_pool_axis

MRI — no preprocessing, no density weighting::

    python scripts/run_probe_dinov3.py --modality mri \
        --config configs/mri/config_probe_dinov3.yaml --roi ofc \
        --model_size vitl16 --slicer_mode 25d \
        --extraction mean_pool --aggregation mean_pool_axis
"""

from __future__ import annotations

import argparse

import yaml

from linear_prober.core.tasks import ROI_TASK
from linear_prober.mri import runner_dinov3 as mri_runner
from linear_prober.skeleton import runner_dinov3 as skeleton_runner
from linear_prober.skeleton.models.dinov3.extract_features import MODEL_SIZES
from linear_prober.skeleton.preprocessor import ALL_PREPROCESSINGS

MRI_ROIS = {"fip", "ofc", "sc"}  # MRI has no LC ROI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DINOv3 zero-shot linear probing (skeleton | mri)."
    )
    p.add_argument("--modality", required=True, choices=["skeleton", "mri"])
    p.add_argument("--config", required=True)
    p.add_argument("--roi", required=True, choices=sorted(ROI_TASK))
    p.add_argument(
        "--preprocessing",
        default=None,
        choices=sorted(ALL_PREPROCESSINGS),
        help="Skeleton only (required for skeleton).",
    )
    p.add_argument("--model_size", required=True, choices=sorted(MODEL_SIZES))
    p.add_argument("--slicer_mode", required=True, choices=["2d", "25d"])
    p.add_argument("--extraction", required=True, choices=["mean_pool", "flatten"])
    p.add_argument(
        "--aggregation", required=True, choices=["mean_pool_axis", "concat_all"]
    )
    p.add_argument(
        "--density_weighting",
        action="store_true",
        help="Skeleton only: append the __dw suffix.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

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
        if args.density_weighting:
            raise SystemExit("--density_weighting is not applicable to --modality mri")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.modality == "skeleton":
        skeleton_runner.run(
            config,
            args.roi,
            args.preprocessing,
            model_size=args.model_size,
            slicer_mode=args.slicer_mode,
            extraction=args.extraction,
            aggregation=args.aggregation,
            density_weighting=args.density_weighting,
        )
    else:
        mri_runner.run(
            config,
            args.roi,
            model_size=args.model_size,
            slicer_mode=args.slicer_mode,
            extraction=args.extraction,
            aggregation=args.aggregation,
        )


if __name__ == "__main__":
    main()
