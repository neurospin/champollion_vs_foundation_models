# sulcus_aware_3DINO — sulcal self-supervised pretraining as a 3DINO extension layer

This repository provides self-supervised (DINO/iBOT) continual-pretraining of the
public **3DINO-ViT** backbone on **binary 3D cortical sulcal-skeleton volumes**.

It is the **pretraining / fine-tuning component** of a study comparing a
sulcus-specific model to general-purpose 3D foundation models. Downstream linear
probing lives in a separate package.

## What this is (and is not)

This is an **extension layer**, not a fork. It contains **only original code**
and does **not** redistribute 3DINO. At runtime it plugs into an *unmodified*
upstream 3DINO installation through `dinov2.*` imports, which are resolved from a
separate clone that you obtain yourself (see below). Concretely, this package
adds, on top of frozen 3DINO:

- a sulcal data pipeline (single-`.npy` datasets, MONAI-based 3D augmentations,
  density-aware "non-empty patch" masking) — isotropic and anisotropic variants;
- a parameter-efficient fine-tuning layer (LoRA / LoRA+last-block / additional
  blocks / full fine-tune) injected in-place into the frozen backbone;
- a thin `SSLMetaArch` subclass and two training entry points that wire the above
  to 3DINO's SSL objective.

This design is deliberate: 3DINO is released under **CC BY-NC-ND 4.0**
(NoDerivatives), so a modified redistribution is not permitted. Keeping 3DINO
external and shipping only original code respects that license (see
[Licensing](#licensing)).

## Installation

Python 3.9 and a CUDA-capable GPU are required (the reference runs used a single
H100).

**1. Clone the upstream 3DINO at the pinned commit and install its requirements.**

```shell
git clone https://github.com/AICONSlab/3DINO.git
git -C 3DINO checkout 85bd4435c1b2ada41cd34cd15cad17c4d3c88d89
pip install -r 3DINO/requirements.txt
```

This provides the full runtime stack (`torch`, `xformers`, `torchvision`,
`iopath`, `monai`, …) and the `dinov2` package this extension layer imports.

**2. Install this package's direct requirements** (a subset of the above, listed
for reproducibility):

```shell
pip install -r requirements.txt
```

**3. Make both packages importable.** Point `PYTHONPATH` at the 3DINO clone (so
`import dinov2` resolves to it) and at this repository:

```shell
export PYTHONPATH="/path/to/3DINO:/path/to/sulcus-aware-3dino:$PYTHONPATH"
```

The public 3DINO-ViT checkpoint used as the starting point
(`student.pretrained_weights`) is available on
[HuggingFace](https://huggingface.co/AICONSlab/3DINO-ViT).

## Data

Training reads a single `.npy` array of shape `[N, D, H, W]` (one binary skeleton
per sample), pointed to by `train.dataset_path`.

> The neuroimaging data used in the study is sensitive medical data and is **not**
> distributed with this repository.

## Training

Two entry points share the DINO/iBOT SSL objective and differ only in the crop
geometry. Both are launched with `torchrun` from the repository root, with
`PYTHONPATH` set as above.

**Isotropic crops** — cubic global crops (e.g. 112³):

```shell
torchrun --nproc_per_node=1 train.py \
    --config-file configs/train/full_finetuning.yaml \
    --output-dir /path/to/output \
    train.dataset_format=npy_array \
    train.dataset_path=/path/to/dataset.npy \
    student.pretrained_weights=/path/to/3dino_vit_weights.pth \
    peft.enable=true peft.method=full_finetune
```

**Anisotropic crops** — non-cubic global crops (e.g. 32×112×96), for elongated
structures such as the cingulate sulcus (single process):

```shell
torchrun --nproc_per_node=1 train_anisotropic.py \
    --config-file configs/train/full_finetuning_anisotropic_cingulate.yaml \
    --output-dir /path/to/output \
    train.dataset_format=npy_array \
    train.dataset_path=/path/to/dataset.npy \
    student.pretrained_weights=/path/to/3dino_vit_weights.pth \
    'crops.global_crops_shape=[32,112,96]' \
    peft.enable=true peft.method=full_finetune
```

All hyperparameters are read from the config file and can be overridden on the
command line (OmegaConf dotlist syntax). The base defaults live in
[`configs/ssl3d_default_config.yaml`](configs/ssl3d_default_config.yaml)
(isotropic) and
[`configs/ssl3d_default_config_anisotropic.yaml`](configs/ssl3d_default_config_anisotropic.yaml)
(anisotropic).

The teacher checkpoint is saved to
`<output-dir>/eval/<iteration>/teacher_checkpoint.pth` during training.

### PEFT

Full fine-tuning (`peft.method=full_finetune`) is the default. LoRA and adapter
variants are available through the configs in
[`configs/train/`](configs/train/) (`peft_lora_*`, `peft_additional_blocks_ofc`).

## Licensing

The original code in this repository is released under the **MIT** license — see
[`LICENSE`](LICENSE).

This package is an extension layer and does **not** include 3DINO. At runtime it
depends on the upstream 3DINO package, which you install separately and which is
licensed under **CC BY-NC-ND 4.0** (NonCommercial, NoDerivatives). When you use
this package together with 3DINO, your use of the combined work is subject to
3DINO's **NonCommercial** terms.

3DINO itself derives substantial portions from
[DINOv2](https://github.com/facebookresearch/dinov2) (Meta AI Research,
Apache-2.0). Files in this package that contain or directly extend DINOv2-derived
expression carry an Apache-2.0 attribution note in their header.

## Acknowledgements

This work builds on [3DINO](https://github.com/AICONSlab/3DINO)
([Xu et al., *npj Digital Medicine* 2025](https://doi.org/10.1038/s41746-025-02035-w))
and the original [DINOv2](https://github.com/facebookresearch/dinov2).

## Citing 3DINO

If you use this code, please cite the original 3DINO paper:

```
@article{xu3dino2025,
  title={A generalizable 3D framework and model for self-supervised learning in medical imaging},
  author={Xu, Tony and Hosseini, Sepehr and Anderson, Chris and Rinaldi, Anthony and Krishnan, Rahul G. and Martel, Anne L. and Goubran, Maged},
  journal={npj Digital Medicine},
  year={2025},
  doi={10.1038/s41746-025-02035-w},
}
```
