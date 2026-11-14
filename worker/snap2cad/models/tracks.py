"""Suivi de points de VGGT (tracker VGGSfM, DINOv2, ALIKED, SuperPoint) hors ligne (plan §6.2).

`vggt.dependency.track_predict.predict_tracks` télécharge ses poids par `torch.hub` (et le code de DINOv2
depuis GitHub). Dans `offline()`, ces adresses sont servies depuis le magasin de modèles d'après le
manifeste ; toute autre adresse est refusée.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import torch
from vggt.layers.vision_transformer import vit_base

from snap2cad.models import manifest, models_dir

MODELS = ("vggsfm-v2-tracker", "dinov2-vitb14-reg", "aliked-n16", "superpoint-v1")


def local_path(url: str) -> Path:
    for name in MODELS:
        for f in manifest()[name]["files"]:
            if f["url"] == url:
                return models_dir() / f["path"]
    raise RuntimeError(f"Téléchargement refusé (fonctionnement hors ligne) : {url}")


def _load_state_dict(url: str, model_dir: Any = None, map_location: Any = None, **_: Any) -> Any:
    return torch.load(local_path(url), map_location=map_location, weights_only=True)


def _hub_load(repo: str, model: str, *_: Any, **__: Any) -> torch.nn.Module:
    if (repo, model) != ("facebookresearch/dinov2", "dinov2_vitb14_reg"):
        raise RuntimeError(f"Modèle torch.hub non embarqué : {repo} {model}")
    # Réglages de `dinov2_vitb14_reg` (dinov2/hub/backbones.py) ; le ViT de VGGT est une copie de celui de DINOv2.
    m = vit_base(
        patch_size=14,
        img_size=518,
        init_values=1.0,
        block_chunks=0,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )
    url = manifest()["dinov2-vitb14-reg"]["files"][0]["url"]
    m.load_state_dict(_load_state_dict(url, map_location="cpu"))
    return m


@contextmanager
def offline() -> Iterator[None]:
    with (
        mock.patch.object(torch.hub, "load_state_dict_from_url", _load_state_dict),
        mock.patch.object(torch.hub, "load", _hub_load),
    ):
        yield
