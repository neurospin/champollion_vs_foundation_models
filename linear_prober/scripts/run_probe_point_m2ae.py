#!/usr/bin/env python3
"""Point-M2AE linear-probing entry point (the point-cloud path, skeleton only).

Point-M2AE consumes point clouds built from the binary skeleton volumes, so
there is no ``--preprocessing`` choice and no MRI variant. One run is a single
composite mode; the defaults select the configuration used for the reported
runs (official pretraining geometry, mean aggregation, native resolution).

Examples
--------
Reported configuration::

    python scripts/run_probe_point_m2ae.py \
        --config configs/skeleton/config_probe_point_m2ae.yaml --roi ofc

Exploration axes (studied on the OFC ROI)::

    python scripts/run_probe_point_m2ae.py \
        --config configs/skeleton/config_probe_point_m2ae.yaml --roi ofc \
        --grouping wide --aggregation multi_level --upsample 1.5
"""

from __future__ import annotations

import argparse

import yaml

from linear_prober.core.tasks import ROI_TASK
from linear_prober.skeleton import runner_point_m2ae as runner
from linear_prober.skeleton.models.point_m2ae.extract_features import (
    AGGREGATIONS,
    GROUPINGS,
)
from linear_prober.skeleton.models.point_m2ae.pointcloud import UPSAMPLE_FACTORS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Point-M2AE zero-shot linear probing (skeleton)."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--roi", required=True, choices=sorted(ROI_TASK))
    p.add_argument(
        "--grouping",
        default="standard",
        choices=sorted(GROUPINGS),
        help="FPS/KNN tokeniser geometry (default: the pretraining one).",
    )
    p.add_argument(
        "--aggregation",
        default="mean",
        choices=sorted(AGGREGATIONS),
        help="Frozen-token pooling (default: mean, 384D).",
    )
    p.add_argument(
        "--upsample",
        type=float,
        default=1.0,
        choices=list(UPSAMPLE_FACTORS),
        help="Isotropic nearest-neighbour voxel upsampling before conversion.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    runner.run(config, args.roi, args.grouping, args.aggregation, args.upsample)


if __name__ == "__main__":
    main()
