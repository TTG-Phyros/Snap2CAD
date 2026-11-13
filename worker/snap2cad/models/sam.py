"""Adaptateur SAM 2.1 (mode vidéo) : modèle selon le profil mémoire, hors ligne (plan §6.3)."""

from __future__ import annotations

from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from snap2cad.device import DeviceInfo
from snap2cad.models import models_dir

# Plan §6.3 : `large` sur 8 Go, `base_plus` sur 6 Go.
MODELS = {
    "full": ("sam2.1-hiera-large", "configs/sam2.1/sam2.1_hiera_l.yaml"),
    "degraded": ("sam2.1-hiera-base-plus", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
}


def load(name: str, info: DeviceInfo) -> SAM2VideoPredictor:
    config = dict(MODELS.values())[name]
    return build_sam2_video_predictor(config, str(models_dir() / name / "model.pt"), device=info.device)
