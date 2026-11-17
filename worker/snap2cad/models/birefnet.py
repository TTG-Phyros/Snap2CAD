"""Adaptateur BiRefNet (détourage de l'objet saillant), repli automatique de `segment` (plan §4.2, étape 3).

Le code du modèle est celui publié avec les poids (Hugging Face, révision figée, SHA-256 dans le manifeste) et
vit dans le magasin de modèles. Il n'utilise `transformers` que pour deux classes de base : une cale les
remplace, sans ajouter `transformers` aux dépendances.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from snap2cad import device
from snap2cad.models import models_dir

NAME = "birefnet"  # complet sur 6 et 8 Go : lite moins bon (13 vues perdues sur 579 contre 6), 3,5 Go au pic
SIZE = 1024  # entrée du modèle (carré)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _shim_transformers() -> None:
    if "transformers" in sys.modules:
        return
    shim = types.ModuleType("transformers")

    class PretrainedConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class PreTrainedModel(torch.nn.Module):
        def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__()

        def post_init(self) -> None:  # initialisation des poids : inutile, ils sont chargés ensuite
            pass

    shim.PretrainedConfig, shim.PreTrainedModel = PretrainedConfig, PreTrainedModel  # type: ignore[attr-defined]
    sys.modules["transformers"] = shim


def load(name: str, info: device.DeviceInfo) -> torch.nn.Module:
    """Modèle `name` du magasin (`birefnet` ou `birefnet-lite`), en évaluation sur `info.device`."""
    _shim_transformers()
    package = f"_snap2cad_{name.replace('-', '_')}"  # le code importe `.BiRefNet_config` : paquet requis
    if package not in sys.modules:
        pkg = types.ModuleType(package)
        pkg.__path__ = [str(models_dir() / name)]
        sys.modules[package] = pkg
    module = importlib.import_module(f"{package}.birefnet")
    model = module.BiRefNet(config=module.BiRefNetConfig(bb_pretrained=False))
    model.load_state_dict(load_file(models_dir() / name / "model.safetensors"))
    return model.eval().to(info.device)


@torch.inference_mode()
def predict(model: torch.nn.Module, rgb: np.ndarray, info: device.DeviceInfo) -> np.ndarray:
    """Probabilité (H, W) d'appartenir à l'objet saillant, à la taille de la photo `rgb` (H, W, 3, uint8)."""
    x = torch.from_numpy(np.array(rgb)).to(info.device).permute(2, 0, 1)[None].float() / 255
    x = F.interpolate(x, (SIZE, SIZE), mode="bilinear", align_corners=False)
    mean, std = (torch.tensor(v, device=info.device)[:, None, None] for v in (MEAN, STD))
    x = (x - mean) / std
    with device.autocast(info):
        logits = model(x)[-1]
    p = F.interpolate(logits.float().sigmoid(), rgb.shape[:2], mode="bilinear", align_corners=False)
    return p[0, 0].cpu().numpy()


def largest_component(mask: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
    """Plus grande composante connexe, hors celles couvertes à plus de moitié par `exclude` : BiRefNet prend
    souvent aussi la pièce ou la carte posée à côté, parfois plus grande que l'objet à l'image."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA].astype(float)
    area[0] = 0  # fond
    if exclude is not None and exclude.any():
        inside = np.bincount(labels[exclude], minlength=n)
        area[inside > 0.5 * area] = 0
    return labels == int(np.argmax(area)) if area.max() > 0 else np.zeros_like(mask, bool)
