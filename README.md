# Champollion vs foundation models

Code for a study comparing sulcus-specific self-supervised models with
general-purpose 3D foundation models on cortical sulcal morphology.

## What's in this repository

| Component | Role |
|---|---|
| [`sulcus-aware-3dino/`](sulcus-aware-3dino/) | **Pretraining** — self-supervised (DINO/iBOT) continual pretraining of the public 3DINO-ViT backbone on binary 3D sulcal skeletons: isotropic and anisotropic crop geometries, density-aware masking, PEFT (LoRA, LoRA + unfrozen last block, full fine-tuning). Built as an *extension layer* over an unmodified upstream [3DINO](https://github.com/AICONSlab/3DINO) clone. |
| [`linear_prober/`](linear_prober/) | **Evaluation** — zero-shot linear probing of frozen 3D encoders (3DINO-ViT, SAM-Med3D, VISTA3D, BrainSegFounder, DINOv3, Point-M2AE) on two modalities (binary sulcal skeletons, MRI crops), under one shared, leakage-guarded protocol. |

Upstream model repositories and checkpoints are **not** redistributed here; each
component's README explains what to clone or download and how to point the
configs at it.

## For reviewers — what can be verified without the data

The neuroimaging data is sensitive medical data and is not distributed, so
training and probing runs cannot be reproduced from this repository alone.
Everything else can be checked directly:

```bash
# Evaluation engine — full test suite on synthetic features
# (no data, no checkpoints; validates the CV protocol, task registry,
#  metrics, leakage guards and both modality runners)
cd linear_prober
pip install -e ".[test]"
pytest

# Pretraining pipeline — synthetic smoke test (numpy + CPU torch only)
cd ../sulcus-aware-3dino
pip install numpy pytest torch
pytest tests/

# Extension-layer contract — every dinov2.* import resolves in the pinned
# upstream 3DINO clone (pure AST, executes nothing; needs the clone next to
# this repository, see sulcus-aware-3dino/README.md)
python tests/check_import_wiring.py
```

The same checks run in CI on every push (Actions tab): byte-compilation of every
source file, both test suites, and the import-wiring check against a fresh clone
of 3DINO at the pinned commit.

## Method claims → where in the code

| Claim | Code |
|---|---|
| iBOT masking restricted to non-empty (sulcus-containing) patches | [`sulcus-aware-3dino/.../data/masking_non_empty.py`](sulcus-aware-3dino/sulcus_aware_3DINO/data/masking_non_empty.py) |
| Anatomy-preserving augmentations: small rigid affines only, no flips / 90° rotations | [`sulcus-aware-3dino/.../data/augmentations.py`](sulcus-aware-3dino/sulcus_aware_3DINO/data/augmentations.py) |
| Anisotropic (non-cubic) crop geometry for elongated sulci | [`.../data/*_anisotropic.py`](sulcus-aware-3dino/sulcus_aware_3DINO/data/), [`.../training/meta_arch_anisotropic.py`](sulcus-aware-3dino/sulcus_aware_3DINO/training/meta_arch_anisotropic.py) |
| PEFT: LoRA on the fused qkv (optionally with the last transformer block unfrozen), full fine-tuning | [`sulcus-aware-3dino/.../models/peft/`](sulcus-aware-3dino/sulcus_aware_3DINO/models/peft/) |
| Isotropic upscale + centered padding, identical at training and probing time | [`.../data/sulcal_preprocessing.py`](sulcus-aware-3dino/sulcus_aware_3DINO/data/sulcal_preprocessing.py), [`linear_prober/.../skeleton/preprocessor.py`](linear_prober/linear_prober/skeleton/preprocessor.py) |
| Pre-stratified folds, test split touched once, subject-level leakage guard | [`linear_prober/.../core/cross_validation.py`](linear_prober/linear_prober/core/cross_validation.py) |
| One shared evaluation engine for every encoder (fair comparison) | [`linear_prober/.../core/`](linear_prober/linear_prober/core/) |
| Per-ROI tasks and metrics (binary / multiclass / 6-target regression) | [`linear_prober/.../core/tasks.py`](linear_prober/linear_prober/core/tasks.py), [`.../core/metrics.py`](linear_prober/linear_prober/core/metrics.py) |
| Hyperparameters of every experiment | [`sulcus-aware-3dino/configs/`](sulcus-aware-3dino/configs/), [`linear_prober/configs/`](linear_prober/configs/) |

## Data availability

The neuroimaging data used in the study is sensitive medical data and is not
distributed with this repository. The expected input formats (single `.npy`
volume arrays, per-ROI master tables) are documented in each component's README.

## Licenses

Original code in both components is released under the MIT license (see each
component's `LICENSE`). Upstream model code is not redistributed: 3DINO
(CC BY-NC-ND 4.0) and the other foundation-model repositories remain external
dependencies under their own licenses — see the per-component READMEs.
