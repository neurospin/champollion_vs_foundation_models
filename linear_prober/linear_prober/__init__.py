"""linear_prober — zero-shot linear probing of frozen 3D encoders.

Two modalities share one evaluation engine:

  - ``skeleton`` : binary 3D grids of cortical sulcal skeletons (4 ROIs);
  - ``mri``      : MRI intensity crops centred on ROIs (3 ROIs).

The shared engine lives in :mod:`linear_prober.core`; each modality supplies
only its input adapter (data loading + preprocessing) in
:mod:`linear_prober.skeleton` and :mod:`linear_prober.mri`.
"""

__version__ = "1.0.0"
