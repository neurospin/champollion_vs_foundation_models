# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

"""
Dedicated model builder for the anisotropic 3DINO SSL pipeline.

Important geometry distinction:

- cfg.runtime_geometry.backbone_reference_size
    Public cubic size used to construct the pretrained backbone and its learned
    positional embedding.

- cfg.runtime_geometry.global_crop_shape
    Real anisotropic input shape used later during preprocessing and forward
    passes.

For the A. Cingulate anisotropic pipeline:

    backbone_reference_size = 112
    global_crop_shape        = (32, 112, 96)

The model must be constructed with img_size=112 so that its learned positional
embedding retains the public 7 x 7 x 7 patch grid:

    1 CLS token + 343 patch tokens = 344 positional tokens.
"""

from dinov2.models import build_model


def build_model_from_anisotropic_cfg(cfg, only_teacher: bool = False):
    """
    Build the student and teacher backbones in the public cubic geometry.

    The real anisotropic global crop shape must not be passed as img_size here.
    It will be handled dynamically during the forward pass by the existing 3D
    positional-encoding interpolation.

    Args:
        cfg:
            Fully resolved anisotropic OmegaConf configuration. It must contain:

                cfg.runtime_geometry.backbone_reference_size

        only_teacher:
            If True, build only the teacher model. The return format remains
            identical to dinov2.models.build_model().

    Returns:
        If only_teacher is False:
            student, teacher, embed_dim

        If only_teacher is True:
            teacher, embed_dim
    """
    backbone_reference_size = int(cfg.runtime_geometry.backbone_reference_size)

    return build_model(
        cfg.student,
        only_teacher=only_teacher,
        img_size=backbone_reference_size,
    )
