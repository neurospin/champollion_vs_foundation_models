# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

"""Sulcus-aware 3D data augmentation for binary sulcal skeleton volumes."""

import logging

from monai.transforms import (
    Compose,
    RandAffine,
    RandCropByLabelClasses,
)

logger = logging.getLogger("dinov2")


class DataAugmentationDINO3d_sulcal(object):
    """
    Data augmentation for preprocessed binary {0,1} sulcal skeleton volumes.

    Assumptions:
        - Volumes are already preprocessed offline.
        - Final input shape is (1, 112, 112, 112).
        - Values are binary {0,1}.
        - No resampling, padding, intensity normalization, flip, or 90-degree
          rotation is performed here.

    Augmentation strategy:
        - Global crops:
            Two independent affine-augmented views of the full 112^3 volume.
        - Local crops:
            Foreground-centered 64^3 crops, followed by small affine
            perturbations.

    Rationale:
        Sulcal skeletons are anatomically oriented. Therefore, flips and 90-degree
        rotations would inject unrealistic invariances. We only use small rigid
        perturbations: small rotations and small translations.

    Important:
        RandAffine uses nearest-neighbor interpolation to preserve binary values.
    """

    def __init__(
        self,
        local_crops_number,
        global_crops_size=112,
        local_crops_size=64,
        affine_prob_global=1.0,
        translate_range_global=(1, 1, 1),
        rotate_range_global=(0.314159, 0.314159, 0.314159),  # ±18°
        affine_prob_local=1.0,
        translate_range_local=(1, 1, 1),
        rotate_range_local=(0.314159, 0.314159, 0.314159),  # ±18°
        max_local_crop_retries=3,
    ):
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.max_local_crop_retries = max_local_crop_retries

        logger.info("###################################")
        logger.info("Using sulcal 3D data augmentation:")
        logger.info("No flips. No 90-degree rotations.")
        logger.info("Only small rigid affine perturbations.")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"affine_prob_global: {affine_prob_global}")
        logger.info(f"translate_range_global: {translate_range_global}")
        logger.info(f"rotate_range_global: {rotate_range_global}")
        logger.info(f"affine_prob_local: {affine_prob_local}")
        logger.info(f"translate_range_local: {translate_range_local}")
        logger.info(f"rotate_range_local: {rotate_range_local}")
        logger.info(f"max_local_crop_retries: {max_local_crop_retries}")
        logger.info("###################################")

        self.geometric_augmentation_global = Compose(
            [
                RandAffine(
                    prob=affine_prob_global,
                    translate_range=translate_range_global,
                    rotate_range=rotate_range_global,
                    mode="nearest",
                    padding_mode="zeros",
                ),
            ]
        )

        self.foreground_crop = RandCropByLabelClasses(
            spatial_size=(local_crops_size, local_crops_size, local_crops_size),
            ratios=[0, 1],
            num_classes=2,
            num_samples=1,
            allow_smaller=False,
        )

        self.geometric_augmentation_local = Compose(
            [
                RandAffine(
                    prob=affine_prob_local,
                    translate_range=translate_range_local,
                    rotate_range=rotate_range_local,
                    mode="nearest",
                    padding_mode="zeros",
                ),
            ]
        )

    def _augment_global(self, image):
        """
        Apply a small rigid affine perturbation to the full 112^3 volume.
        """
        return self.geometric_augmentation_global(image)

    def _augment_local(self, image):
        """
        Extract one foreground-centered local crop, then apply a small rigid
        affine perturbation.

        A retry mechanism avoids empty crops after the affine transformation.
        """
        for _ in range(self.max_local_crop_retries):
            crops = self.foreground_crop(img=image, label=image)
            crop = crops[0]

            crop = self.geometric_augmentation_local(crop)

            # Guard against empty crops.
            if crop.sum() > 0:
                return crop

        # Fallback (very rare): return the last crop even if empty.
        return crop

    def __call__(self, image):
        """
        Return DINO/iBOT-style multi-crop views.

        Input may be either:
            - a MONAI dict: {"image": tensor}
            - a tensor directly

        Returns:
            (output_dict, None)
        """
        if isinstance(image, dict):
            image = image["image"]

        global_crop_1 = self._augment_global(image)
        global_crop_2 = self._augment_global(image)

        local_crops = [
            self._augment_local(image) for _ in range(self.local_crops_number)
        ]

        output = {
            "global_crops": [global_crop_1, global_crop_2],
            "global_crops_teacher": [global_crop_1, global_crop_2],
            "local_crops": local_crops,
            "offsets": (),
        }

        return output, None
