"""Choix du périphérique, de la précision et du profil mémoire (plan §3.3).

Les étapes passent par ce module : aucun `torch.cuda` en dur ailleurs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Une carte « 8 Go » expose ~7,99 Gio, une « 6 Go » ~6,0 Gio.
FULL_MIN_GIB = 7.5
DEGRADED_MIN_GIB = 5.5


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    dtype: torch.dtype
    name: str
    memory_gib: float
    profile: str  # "full" | "degraded" | "unsupported"


def profile_for(memory_gib: float, kind: str) -> str:
    if kind == "cpu":
        return "unsupported"  # pas de mode tout-processeur pour l'IA (plan §2)
    if memory_gib >= FULL_MIN_GIB:
        return "full"
    if memory_gib >= DEGRADED_MIN_GIB:
        return "degraded"
    return "unsupported"


def detect() -> DeviceInfo:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        mem = props.total_memory / 2**30
        dtype = torch.bfloat16 if props.major >= 8 else torch.float16
        return DeviceInfo(torch.device("cuda:0"), dtype, props.name, mem, profile_for(mem, "cuda"))
    if torch.backends.mps.is_available():
        mem = torch.mps.recommended_max_memory() / 2**30
        return DeviceInfo(torch.device("mps"), torch.float16, "Apple MPS", mem, profile_for(mem, "mps"))
    return DeviceInfo(torch.device("cpu"), torch.float32, "CPU", 0.0, "unsupported")


def reset_peak_memory(info: DeviceInfo) -> None:
    if info.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(info.device)


def peak_memory_mb(info: DeviceInfo) -> float | None:
    """Pic de mémoire allouée depuis le dernier reset (enregistré dans project.json)."""
    if info.device.type == "cuda":
        return torch.cuda.max_memory_allocated(info.device) / 2**20
    return None  # shortcut: pas de mesure de pic sur MPS/CPU, à ajouter avec le portage Mac


def autocast(info: DeviceInfo) -> torch.autocast:
    return torch.autocast(device_type=info.device.type, dtype=info.dtype, enabled=info.device.type != "cpu")
