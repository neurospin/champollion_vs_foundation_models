# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

import random

import numpy as np


class MaskingGenerator3d:
    def __init__(
        self,
        input_size,
        patch_size=16,
    ):
        """
        Create a masking generator for 3D data.

        In standard mode (volume=None), masking is sampled uniformly over all
        patches.

        In sulcal mode (volume provided), masking is restricted to patches
        containing at least one active voxel. The mask ratio from
        mask_ratio_min_max is applied to the number of active patches, so it
        represents the fraction of the visible sulcus that is masked.

        If the supplied volume contains no active patch, an entirely empty mask
        is returned. iBOT must not generate reconstruction targets from pure
        background patches.

        Args:
            input_size:
                Size of the patch grid as an int or a 3-tuple.
                Example: 7 for a 112^3 volume with patch_size=16,
                corresponding to 7^3 = 343 patches.

            patch_size:
                Spatial size of each patch in voxels. Default: 16.
        """
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3

        self.height, self.width, self.depth = input_size
        self.num_patches = self.height * self.width * self.depth
        self.patch_size = patch_size

    def __repr__(self):
        return "Generator(%d, %d, %d)" % (
            self.height,
            self.width,
            self.depth,
        )

    def get_shape(self):
        return self.height, self.width, self.depth

    def _mask(self, mask, n_masked):
        """
        Apply uniform random masking over all patch positions.

        This method is used only in standard masking mode, when no volume is
        supplied to the generator.
        """
        mask_inds = random.sample(
            range(self.num_patches),
            k=n_masked,
        )
        mask.ravel()[mask_inds] = True

    def _mask_active_only(self, mask, n_masked, volume):
        """
        Apply random masking only to active sulcal patches.

        Only patches containing at least one non-zero voxel are eligible for
        masking. The ratio recovered from n_masked / num_patches is applied to
        the number of active patches.

        If the volume contains no active patch, the mask remains entirely empty.
        No uniform-background fallback is performed.

        Args:
            mask:
                Boolean array of shape (H, W, D), modified in place.

            n_masked:
                Number of patches initially requested by the collate function,
                based on the total number of patches.

            volume:
                Tensor of shape (1, D, H, W), typically
                (1, 112, 112, 112) for global crops.
        """
        ps = self.patch_size
        h, w, d = self.height, self.width, self.depth

        # Vectorized patch-activity computation.
        #
        # Example for a 112^3 volume and patch_size=16:
        #   volume[0]                      : (112, 112, 112)
        #   reshaped                      : (7, 16, 7, 16, 7, 16)
        #   summed over patch dimensions  : (7, 7, 7)
        #   flattened                     : (343,)
        v = volume[0]

        if hasattr(v, "numpy"):
            v = v.float().numpy()
        else:
            v = np.asarray(v, dtype=np.float32)

        patch_density = v.reshape(h, ps, w, ps, d, ps).sum(axis=(1, 3, 5)).ravel()

        # A patch is active as soon as it contains at least one non-zero voxel.
        active_patch_inds = np.where(patch_density > 0)[0]
        n_active = len(active_patch_inds)

        if n_active == 0:
            # No active sulcal patch exists in this view.
            # Keep the mask entirely empty: iBOT must not learn from
            # pure-background patches.
            return

        # Recover the original masking ratio and apply it to active patches.
        #
        # Example:
        #   n_masked = 154 over 343 total patches -> ratio ~= 0.45
        #   n_active = 30                        -> mask int(30 * 0.45) = 13
        ratio = n_masked / self.num_patches
        n_to_mask = int(n_active * ratio)

        if n_to_mask == 0:
            return

        mask_inds = np.random.choice(
            active_patch_inds,
            size=n_to_mask,
            replace=False,
        )
        mask.ravel()[mask_inds] = True

    def __call__(self, num_masking_patches=0, volume=None):
        """
        Generate a boolean patch mask.

        Args:
            num_masking_patches:
                Number of patches requested by the collate function, generally
                computed as int(num_patches * ratio).

                In sulcal mode, this value is converted back into a ratio and
                applied to the number of active patches.

            volume:
                Optional tensor of shape (1, D, H, W).

                If provided, active-patch-only masking is used.
                If omitted, uniform masking over all patches is used.

        Returns:
            Boolean NumPy array of shape (H, W, D).
        """
        mask = np.zeros(
            shape=self.get_shape(),
            dtype=bool,
        )

        if num_masking_patches == 0:
            return mask

        if volume is not None:
            self._mask_active_only(
                mask,
                num_masking_patches,
                volume,
            )
        else:
            self._mask(
                mask,
                num_masking_patches,
            )

        return mask
