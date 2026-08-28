"""Binary skeleton volume -> Point-M2AE input point cloud.

Modality adapter for the point-cloud path: the frozen encoder consumes (N, 3)
coordinates, not voxel grids, so the geometric voxel preprocessings do not
apply. Conversion:

  1. optional isotropic nearest-neighbour upsampling of the voxel grid —
     increases point density without creating new structures (nearest keeps
     the volume strictly binary);
  2. active voxels -> (N, 3) float coordinates;
  3. fixed-centre normalisation: coordinates are centred on the *volume*
     centre (not the cloud centroid) and scaled by the max absolute
     coordinate into [-1, 1].

Fixing the centre preserves the anatomical position of the skeleton inside
its crop: two identical shapes at different positions yield different clouds,
mirroring the centred-padding convention of the voxel pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Upsampling factors explored on the OFC ROI (1.0 = native resolution).
UPSAMPLE_FACTORS = (1.0, 1.25, 1.5, 1.75, 2.0)


def volume_to_point_cloud(volume: np.ndarray, upsample: float = 1.0) -> np.ndarray:
    """Convert one binary skeleton volume to a normalised (N, 3) point cloud.

    Args:
        volume: binary volume of shape ``[D, H, W]``; a leading or trailing
            singleton channel is squeezed.
        upsample: isotropic nearest-neighbour upsampling factor applied to the
            voxel grid before conversion.

    Returns:
        ``float32`` array of shape ``(N, 3)`` with coordinates in ``[-1, 1]``.

    Raises:
        ValueError: on a non-3D volume or a volume with no active voxel.
    """
    vol = np.asarray(volume)
    if vol.ndim == 4 and vol.shape[0] == 1:
        vol = vol[0]
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {tuple(vol.shape)}")

    if float(upsample) != 1.0:
        vol = ndimage.zoom(vol, float(upsample), order=0)  # nearest keeps binary

    points = np.argwhere(vol > 0).astype(np.float32)
    if points.size == 0:
        raise ValueError("Empty skeleton: no active voxel to convert to points.")

    # Fixed volume-centre normalisation (not the cloud centroid).
    centre = np.array(vol.shape, dtype=np.float32)[None, :] / 2.0
    points -= centre
    points /= float(np.abs(points).max()) + 1e-6
    return points
