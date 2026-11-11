"""Adaptateur VGGT-1B : conversion unique en bf16 et chargement hors ligne (plan §6.2)."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from vggt.models.vggt import VGGT

from snap2cad.device import DeviceInfo
from snap2cad.models import models_dir


def bf16_path() -> Path:
    return models_dir() / "vggt-1b-bf16" / "model.safetensors"


def convert_bf16() -> Path:
    """fp32 (5,0 Go) → bf16 (2,5 Go), une seule fois, dans le magasin de modèles."""
    dst = bf16_path()
    state = load_file(models_dir() / "vggt-1b" / "model.safetensors")
    state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state.items()}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    save_file(state, tmp)
    os.replace(tmp, dst)  # pas de fichier à moitié écrit si la conversion est interrompue
    return dst


def load(info: DeviceInfo) -> VGGT:
    path = bf16_path()
    if not path.is_file():
        convert_bf16()
    model = VGGT()
    # assign=True : les paramètres gardent le bf16 du fichier, les buffers (moyenne/écart-type) restent en fp32.
    model.load_state_dict(load_file(path), assign=True)
    # Les têtes tournent hors autocast (vggt.py) : on les garde en fp32 comme en amont.
    for head in (model.camera_head, model.depth_head, model.point_head, model.track_head):
        if head is not None:
            head.float()
    return model.eval().to(info.device)
