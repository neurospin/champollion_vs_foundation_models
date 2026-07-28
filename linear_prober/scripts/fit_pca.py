#!/usr/bin/env python3
"""Fit the UKBB PCA basis used by the skeleton ``flatten`` probe.

For a given (model, ROI, preprocessing), this extracts UKBB ``flatten`` features
once and fits one ``StandardScaler -> PCA`` pipeline per ``n_components`` in
``config.probe.n_components_list``. The pipelines are cached to disk and reused
by ``scripts/run_probe.py --mode flatten``.

Skeleton modality only — MRI has no UKBB set and probes raw ``flatten`` features.

Example::

    python scripts/fit_pca.py \
        --config configs/skeleton/config_probe_dino3d.yaml \
        --roi ofc --preprocessing upscale_pad
"""

from __future__ import annotations

import argparse

import yaml

from linear_prober.skeleton.extract import get_ukbb_features
from linear_prober.skeleton.pca import fit_ukbb_pca
from linear_prober.skeleton.preprocessor import ALL_PREPROCESSINGS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit UKBB PCA for the skeleton flatten probe."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--roi", required=True, choices=["fip", "lc", "ofc", "sc"])
    p.add_argument("--preprocessing", required=True, choices=sorted(ALL_PREPROCESSINGS))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["experiment"]["model"]
    output_dir = config["experiment"].get("output_model_name", model_name)
    output_root = config["paths"]["output_root"]
    n_components_list = [int(n) for n in config["probe"]["n_components_list"]]

    features = get_ukbb_features(config, args.roi, args.preprocessing)["features"]
    fit_ukbb_pca(
        features=features,
        n_components_list=n_components_list,
        output_root=output_root,
        model_name=output_dir,
        roi=args.roi,
        preprocessing=args.preprocessing,
    )


if __name__ == "__main__":
    main()
