"""Vérifie l'environnement de Snap2CAD (jalon M0).

Usage : uv run --project worker python tools/check_env.py
Code de sortie 1 si un contrôle obligatoire échoue.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import torch

from snap2cad import device, models

ROOT = Path(__file__).resolve().parents[1]
MIN_DRIVER = 570  # torch cu128
failures = 0
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # console Windows en cp1252


def check(label: str, ok: bool, detail: str, required: bool = True) -> None:
    global failures
    failures += required and not ok
    mark = "OK  " if ok else ("ERR " if required else "WARN")
    print(f"[{mark}] {label:<22} {detail}")


def run(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.rstrip()
    except (OSError, subprocess.CalledProcessError):
        return ""


check("OS", True, f"{platform.system()} {platform.release()} {platform.machine()}")
check("Python", sys.version_info[:2] == (3, 11), sys.version.split()[0])
check("torch", torch.__version__.startswith("2.7"), f"{torch.__version__} (CUDA {torch.version.cuda})")
for pkg in ("torchvision", "numpy", "safetensors", "vggt"):
    check(pkg, True, version(pkg))
check("numpy < 2", version("numpy").startswith("1."), version("numpy"))

info = device.detect()
check("Périphérique", info.device.type != "cpu", f"{info.device} · {info.name} · {info.dtype}")
check("Mémoire / profil", info.profile != "unsupported", f"{info.memory_gib:.2f} Gio → {info.profile}")

if info.device.type == "cuda":
    drv = run("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").splitlines()
    major = int(drv[0].split(".")[0]) if drv else 0
    check("Pilote NVIDIA", major >= MIN_DRIVER, f"{drv[0] if drv else '?'} (≥ R{MIN_DRIVER} requis)")
    x = torch.randn(256, 256, device=info.device, dtype=info.dtype)
    check("Calcul GPU", bool(torch.isfinite(x @ x).all()), "matmul OK")

status = run("git", "-C", str(ROOT), "submodule", "status")
for line in status.splitlines():
    # ' ' = commit figé, '+' = autre commit, '-' = non initialisé
    check(f"third_party {line.split()[1].split('/')[-1]}", line[0] == " ", line[1:9])
if not status:
    check("third_party", False, "git submodule status vide")

check("Magasin de modèles", models.models_dir().is_dir(), str(models.models_dir()))
for name, m in models.manifest().items():
    problems = models.verify(name)
    check(
        f"Poids {name}", not problems, "; ".join(problems) or "présents", required=m["group"] == "reference"
    )

print("\nEnvironnement OK" if not failures else f"\n{failures} contrôle(s) en échec")
sys.exit(1 if failures else 0)
