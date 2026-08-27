# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)
#
# Portions of this file are derived from DINOv2 (Meta AI Research),
# licensed under the Apache License 2.0.

"""Training setup helpers.

These wrap 3DINO's config/setup utilities. The only behavioural change over
upstream is that the random seed is read from ``cfg.train.seed`` (rather than
``args.seed``), so a run is reproducible from its config alone. Everything else
is delegated to the frozen upstream implementation.
"""

import os

from dinov2.utils.config import apply_scaling_rules_to_cfg
from dinov2.utils.config import default_setup as _upstream_default_setup
from dinov2.utils.config import get_cfg_from_args_3d, write_config

__all__ = [
    "setup_3d",
    "default_setup",
    "apply_scaling_rules_to_cfg",
    "write_config",
    "get_cfg_from_args_3d",
]


def default_setup(args, cfg):
    """3DINO's ``default_setup``, but seed the run from ``cfg.train.seed``.

    Injects the config seed into ``args.seed`` and delegates the actual setup
    (distributed init, logging, ``fix_random_seeds``) to the upstream function.
    """
    args.seed = cfg.train.seed
    return _upstream_default_setup(args)


def setup_3d(args):
    """Build the config and run basic 3D training/evaluation setup.

    Mirrors upstream ``setup_3d`` step for step, but calls the seed-from-config
    ``default_setup`` above. Every other step is the unmodified upstream helper.
    """
    cfg = get_cfg_from_args_3d(args)
    os.makedirs(args.output_dir, exist_ok=True)
    default_setup(args, cfg)
    apply_scaling_rules_to_cfg(cfg)
    write_config(cfg, args.output_dir)
    return cfg
