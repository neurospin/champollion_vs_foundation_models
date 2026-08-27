# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

"""sulcus_aware_3DINO — a sulcus-aware self-supervised extension layer over 3DINO.

This package contains *only* original code: sulcal data pipelines, PEFT adapters,
and thin subclasses/wrappers that plug into a frozen, unmodified upstream 3DINO
checkout (imported as ``dinov2.*``). No 3DINO source is vendored here; clone 3DINO
separately and put it on the ``PYTHONPATH`` (see the README).
"""
