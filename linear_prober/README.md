# linear_prober

Zero-shot **linear probing** of frozen 3D encoders on two neuroimaging
modalities:

- **skeleton** — binary 3D grids of cortical sulcal skeletons (4 ROIs: `fip`, `lc`, `ofc`, `sc`);
- **mri** — MRI intensity crops centred on ROIs (3 ROIs: `fip`, `ofc`, `sc`).

Regions of interest:

| ROI | Region |
|-----|--------|
| `fip` | Intraparietal |
| `lc`  | Anterior Cingulate |
| `ofc` | Orbitofrontal |
| `sc`  | Central |

Both modalities share one evaluation engine. For each frozen encoder, features
are extracted once, cached, and probed with simple linear models under a common
5-fold cross-validation protocol.

## Downstream tasks

| ROI | Region | Task | Metric |
|-----|--------|------|--------|
| `fip` | Intraparietal | binary classification | ROC-AUC |
| `lc` | Anterior Cingulate | binary classification | ROC-AUC |
| `ofc` | Orbitofrontal | 4-class classification | ROC-AUC, one-vs-rest, weighted |
| `sc` | Central | 6-target regression | R² (per target, averaged) |

Hyperparameters (`C` for LogisticRegression, `alpha` for Ridge / RidgeClassifier)
are selected by manual k-fold CV on the `train_val` split using **pre-stratified
folds** from the master table; the `test` split is evaluated exactly once.

## Feature modes

- `mean_pool` — CLS ⊕ mean(patch tokens);
- `mean_pool_multi_layers` — the same, concatenated over the last layers;
- `flatten` — CLS ⊕ all patch tokens (high-dimensional). On the skeleton
  modality this is projected through a UKBB-fit PCA; on MRI it is probed raw
  with a RidgeClassifier / Ridge.

## Architecture

```
linear_prober/
├── core/        # modality-agnostic engine: cross-validation, task registry,
│                #   metrics, hyperparameter grids, feature cache, result I/O
├── skeleton/    # binary-grid modality: dataset, geometric preprocessing,
│                #   UKBB PCA, extraction adapter, runner, models/
└── mri/         # MRI-crop modality: extraction adapter, runner, models/
```

The boundary between the two modalities is the **input adapter** (data loading +
preprocessing). The probing engine in `core/` is fully shared; each modality
keeps its own `models/<model>/` because the frozen encoders' preprocessing and
forward passes genuinely differ between binary grids and MRI crops.

## Foundation models

| Model | `model` key | Upstream code required |
|-------|-------------|------------------------|
| 3DINO-ViT | `dino3d` | [3DINO](https://github.com/AICONSlab/3DINO) |
| SAM-Med3D | `sam3d` | [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) |
| VISTA3D | `vista3d` | [MONAI VISTA3D](https://github.com/Project-MONAI/VISTA) *(skeleton only)* |
| BrainSegFounder | `bsf` | [BrainSegFounder](https://github.com/lab-smile/BrainSegFounder) (bundled `SSL_Head.py`) |
| DINOv3 | `dinov3` | [DINOv3](https://github.com/facebookresearch/dinov3) weights (2D slicing path) |

Each model loads a frozen checkpoint and, for models that import an external
repository, needs that repository on the Python path — set via the config
`repositories:` and `paths.checkpoint_path` keys. These external repositories and
checkpoints are **not** bundled here; install them separately and point the
config at them.

## Installation

```bash
pip install -e .          # or: pip install -r requirements.txt
```

Python ≥ 3.9. Running the encoders additionally requires the per-model upstream
repositories above and a CUDA-capable GPU.

## Usage

The 3D encoders (`dino3d`, `sam3d`, `vista3d`, `bsf`) use `run_probe.py`; DINOv3
(the 2D-slicing path) has a dedicated entry point `run_probe_dinov3.py` because
its run is parameterised by a composite feature mode.

```bash
# 3D encoders — skeleton (a geometric preprocessing is required)
python scripts/run_probe.py --modality skeleton \
    --config configs/skeleton/config_probe_dino3d.yaml \
    --roi ofc --preprocessing upscale_pad

# 3D encoders — MRI (single native preprocessing, no --preprocessing)
python scripts/run_probe.py --modality mri \
    --config configs/mri/config_probe_dino3d.yaml --roi ofc

# DINOv3 — composite mode via the dedicated entry point
python scripts/run_probe_dinov3.py --modality skeleton \
    --config configs/skeleton/config_probe_dinov3.yaml \
    --roi ofc --preprocessing upscale_pad \
    --model_size vitb16 --slicer_mode 2d \
    --extraction mean_pool --aggregation mean_pool_axis
```

Omit `--mode` to run all feature modes. `--flatten-raw` (skeleton, 3D encoders)
probes the raw high-dimensional features without PCA.

Two optional skeleton pre-steps (3D encoders):

```bash
# Fit the UKBB PCA basis required by the `flatten` mode
python scripts/fit_pca.py --config configs/skeleton/config_probe_dino3d.yaml \
    --roi ofc --preprocessing upscale_pad

# Search the per-ROI intensity mapping (fills config.optimal_mapping)
python scripts/normalizer_search.py --config configs/skeleton/config_probe_dino3d.yaml \
    --roi ofc --preprocessing upscale_pad
```

Results are written as CSV under
`{output_root}/{model}/{roi}[/{preprocessing}]/results/`.

## Data format

The runners consume a per-ROI **master table** (CSV) with columns:
`volume_index`, `subject`, `fold`, `split` (`train_val` / `test`), and either
`label` (classification) or `label_0…label_5` (regression). The skeleton
modality additionally reads the binary volumes referenced in the config; the MRI
modality reads NIfTI crops from `crop_dir`.

> The neuroimaging data used in the paper is sensitive medical data and is not
> distributed with this repository.

## Tests

```bash
pip install pytest
PYTHONPATH="$PWD" python -m pytest -q
```

The test suite validates the engine and both runners on synthetic features — no
model checkpoint or neuroimaging data is required.

## License

The 3DINO-derived encoder code is subject to the upstream 3DINO / DINOv2 license
(CC BY-NC-ND 4.0); each other model wrapper is subject to its upstream
repository's license. See `LICENSE`.
