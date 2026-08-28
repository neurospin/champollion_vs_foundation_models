"""Point-M2AE frozen-encoder feature extraction — skeleton modality only.

Point-M2AE (Zhang et al., 2022; MIT license) is a hierarchical point-cloud
MAE: its encoder consumes (N, 3) coordinates, so skeleton volumes are
converted to fixed-centre normalised point clouds (see ``pointcloud``) instead
of going through the voxel geometric preprocessings.

One run is a composite mode:

  grouping     — FPS/KNN tokeniser geometry: ``standard`` (the official
                 pretraining configuration, i.e. the one the released
                 checkpoint was trained with — used for the reported runs) or
                 ``wide`` (larger group sizes, explored on the OFC ROI);
  aggregation  — pooling of the frozen tokens: ``mean`` (384D),
                 ``mean_std_min_max`` (1536D) or ``multi_level`` (stage-2 mean
                 concatenated with stage-3 mean, 576D);
  upsample     — isotropic nearest-neighbour upsampling of the voxel grid
                 before conversion (1.0 … 2.0), controlling point density.

The upstream repository is added to ``sys.path`` via the config
``repositories: point_m2ae:`` key; extraction additionally needs its runtime
dependencies (timm, knn_cuda, CUDA point ops) and a GPU. Probing cached
features does not.
"""

from __future__ import annotations

import sys
from typing import Callable, Dict

import numpy as np
import torch
from tqdm import tqdm

from linear_prober.skeleton.models.point_m2ae.pointcloud import (
    UPSAMPLE_FACTORS,
    volume_to_point_cloud,
)

# Architecture constants tied to the public ``pre-train.pth`` checkpoint.
_FIXED_ARCH = {
    "encoder_depths": [5, 5, 5],
    "encoder_dims": [96, 192, 384],
    "drop_path_rate": 0.1,
    "num_heads": 6,
}

# Tokeniser geometries. "standard" is the official pretraining configuration.
GROUPINGS: Dict[str, Dict] = {
    "standard": {
        "num_groups": [512, 256, 64],
        "group_sizes": [16, 8, 8],
        "local_radius": [0.32, 0.64, 1.28],
    },
    "wide": {
        "num_groups": [512, 256, 64],
        "group_sizes": [24, 16, 16],
        "local_radius": [0.32, 0.64, 1.28],
    },
}

# Aggregation name -> output feature dimension.
AGGREGATIONS: Dict[str, int] = {
    "mean": 384,
    "mean_std_min_max": 1536,
    "multi_level": 576,
}


def build_mode(grouping: str, aggregation: str, upsample: float) -> str:
    """Compose the self-describing feature-cache mode string."""
    if grouping not in GROUPINGS:
        raise ValueError(f"Unknown grouping '{grouping}'. Expected {sorted(GROUPINGS)}.")
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. Expected {sorted(AGGREGATIONS)}."
        )
    if float(upsample) not in UPSAMPLE_FACTORS:
        raise ValueError(
            f"Unknown upsample factor {upsample}. Expected one of {UPSAMPLE_FACTORS}."
        )
    return f"{grouping}__{aggregation}__up{float(upsample):g}"


def _add_repo_to_path(repo_path: str) -> None:
    """Put the upstream Point-M2AE clone first on sys.path (absolute imports)."""
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def _apply_knn_compat_shim() -> None:
    """Adapt older 3-argument ``knn_cuda.knn`` signatures.

    Must run before the upstream ``Group`` module is used; harmless when the
    installed knn_cuda already accepts ``transpose_mode``.
    """
    import importlib
    import inspect

    modules_mod = importlib.import_module("models.modules")
    try:
        from knn_cuda import knn as knn_native

        if len(inspect.signature(knn_native).parameters) == 3:

            def knn_compat(x, y, k, transpose_mode=True):
                return knn_native(x, y, k)

            modules_mod.knn = knn_compat
    except Exception:
        pass


def _build_model(checkpoint_path: str, grouping: str, device: str):
    """Build the frozen hierarchical encoder + groupers and load its weights.

    Only the ``h_encoder.*`` weights of the pretraining checkpoint are loaded
    (the decoder is not needed for feature extraction).
    """
    _apply_knn_compat_shim()
    from models.modules import Group
    from models.Point_M2AE_Finetune import H_Encoder

    class _Cfg:
        pass

    cfg = _Cfg()
    for key, value in {**_FIXED_ARCH, **GROUPINGS[grouping]}.items():
        setattr(cfg, key, value)

    encoder = H_Encoder(cfg).to(device).eval()
    groupers = (
        torch.nn.ModuleList(
            [
                Group(num_group=ng, group_size=gs)
                for ng, gs in zip(cfg.num_groups, cfg.group_sizes)
            ]
        )
        .to(device)
        .eval()
    )

    obj = torch.load(str(checkpoint_path), map_location="cpu")
    sd = None
    for key in ("state_dict", "base_model", "model", "module"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            sd = obj[key]
            break
    if sd is None and isinstance(obj, dict):
        sd = obj

    encoder_sd = {
        k.replace("h_encoder.", ""): v
        for k, v in sd.items()
        if k.startswith("h_encoder.")
    }
    if not encoder_sd:
        raise ValueError(
            f"No 'h_encoder.*' weights found in {checkpoint_path}; "
            "expected the official Point-M2AE pretraining checkpoint."
        )
    encoder.load_state_dict(encoder_sd, strict=False)
    return encoder, groupers


@torch.no_grad()
def _forward_one(encoder, groupers, points: torch.Tensor, aggregation: str) -> torch.Tensor:
    """One cloud ``[1, N, 3]`` -> one feature row ``[1, D]``."""
    neighborhoods, centers, idxs = [], [], []
    cur = points
    for grouper in groupers:
        nei, ctr, idx = grouper(cur.contiguous())
        neighborhoods.append(nei)
        centers.append(ctr)
        idxs.append(idx)
        cur = ctr

    if aggregation == "multi_level":
        # Capture the stage-2 tokens with a forward hook — single forward pass.
        captured: Dict[str, torch.Tensor] = {}
        handle = encoder.encoder_blocks[1].register_forward_hook(
            lambda module, inputs, output: captured.__setitem__(
                "stage2", output.detach()
            )
        )
        try:
            x_vis = encoder(neighborhoods, centers, idxs, eval=True)
        finally:
            handle.remove()
        return torch.cat([captured["stage2"].mean(1), x_vis.mean(1)], dim=1)

    x_vis = encoder(neighborhoods, centers, idxs, eval=True)
    if aggregation == "mean":
        return x_vis.mean(1)
    # mean_std_min_max
    return torch.cat(
        [x_vis.mean(1), x_vis.std(1), x_vis.min(1).values, x_vis.max(1).values],
        dim=1,
    )


def make_extract_fn(
    grouping: str, aggregation: str, upsample: float, device: str
) -> Callable:
    """Return the extraction function consumed by the feature cache.

    Samples are processed one cloud at a time (FPS/KNN grouping is
    per-sample: clouds have variable point counts).
    """

    def extract_fn(checkpoint_path, dataloader, mode, device=device):
        encoder, groupers = _build_model(checkpoint_path, grouping, device)

        all_features: list = []
        tensor_lists: Dict[str, list] = {}
        string_lists: Dict[str, list] = {}
        tensor_fields = {
            "label": "labels",
            "fold": "folds",
            "volume_index": "volume_indices",
        }
        string_fields = {"subject": "subjects", "split": "splits"}

        for batch in tqdm(dataloader, desc=f"[Point-M2AE] {mode}", leave=True):
            volumes = batch["volume"]  # [B, 1, D, H, W]
            for i in range(volumes.shape[0]):
                cloud = volume_to_point_cloud(
                    volumes[i, 0].numpy(), upsample=upsample
                )
                pts = torch.from_numpy(cloud).unsqueeze(0).to(device).contiguous()
                feature = _forward_one(encoder, groupers, pts, aggregation)
                all_features.append(feature.cpu().numpy())

            for src, dst in tensor_fields.items():
                if src in batch:
                    tensor_lists.setdefault(dst, []).append(np.asarray(batch[src]))
            for src, dst in string_fields.items():
                if src in batch:
                    string_lists.setdefault(dst, []).extend(list(batch[src]))

        result: Dict[str, np.ndarray] = {
            "features": np.concatenate(all_features, axis=0)
        }
        for dst, chunks in tensor_lists.items():
            result[dst] = np.concatenate(chunks, axis=0)
        for dst, values in string_lists.items():
            result[dst] = np.array(values)
        return result

    return extract_fn
