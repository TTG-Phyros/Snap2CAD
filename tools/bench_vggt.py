"""Mesure VRAM et temps de VGGT selon le nombre d'images (jalon M0, plan §8).

Usage : uv run --project worker python tools/bench_vggt.py <nom> "<glob des images>" [N ...]
Exemple : ... tools/bench_vggt.py kitchen "third_party/vggt/examples/kitchen/images/*.png" 4 8 16 25
Ajoute les résultats dans docs/benchmarks/vggt.json (une entrée par GPU et jeu d'images).

Même modèle que l'étape `poses_init` (sans tête de points ni de suivi) et même plafond VRAM
(`device.cap_cuda_memory`) : un N « tient » s'il passe sans OutOfMemoryError sous ce plafond.
"""

from __future__ import annotations

import gc
import glob
import json
import sys
import time
from pathlib import Path

import torch
from vggt.utils.load_fn import load_and_preprocess_images

from snap2cad import device
from snap2cad.models import vggt as vggt_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmarks" / "vggt.json"


def run(model: torch.nn.Module, info: device.DeviceInfo, paths: list[str]) -> tuple[float, float, float]:
    images = load_and_preprocess_images(paths).to(info.device)
    gc.collect()
    torch.cuda.empty_cache()
    device.reset_peak_memory(info)
    t = time.perf_counter()
    with torch.no_grad(), device.autocast(info):
        model(images)
    torch.cuda.synchronize()
    return time.perf_counter() - t, device.peak_memory_mb(info) or 0.0, device.peak_reserved_mb(info) or 0.0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows en cp1252 quand la sortie est redirigée
    name, pattern, *ns = sys.argv[1:]
    paths = sorted(glob.glob(pattern))
    assert paths, f"aucune image pour {pattern}"
    info = device.detect()
    assert info.device.type == "cuda" and info.budget_mb, "banc prévu pour NVIDIA"
    model = vggt_model.load(info)
    model.point_head = None  # comme poses_init
    model.track_head = None
    run(model, info, paths[:2])  # préchauffage (init cuDNN), non enregistré

    rows = []
    for n in [int(n) for n in ns] or [4, 8, 16, 24, 32]:
        if n > len(paths):
            break
        try:
            secs, peak, reserved = run(model, info, paths[:n])
        except torch.OutOfMemoryError:
            rows.append({"n_images": n, "fits": False})
            print(f"N={n:3d}  OOM sous le plafond de {info.budget_mb:.0f} Mo")
            break
        rows.append(
            {
                "n_images": n,
                "time_s": round(secs, 2),
                "peak_mb": round(peak),
                "reserved_mb": round(reserved),
                "fits": True,
            }
        )
        print(f"N={n:3d}  {secs:6.1f} s  alloué {peak:6.0f} Mo  réservé {reserved:6.0f} Mo")

    ok = [r["n_images"] for r in rows if r["fits"]]
    results = json.loads(OUT.read_text("utf-8")) if OUT.exists() else []
    results = [r for r in results if not (r["gpu"] == info.name and r["dataset"] == name)]
    results.append(
        {
            "gpu": info.name,
            "memory_gib": round(info.memory_gib, 2),
            "profile": info.profile,
            "dataset": name,
            "image_shape": list(load_and_preprocess_images(paths[:1]).shape[-2:]),
            "torch": torch.__version__,
            "heads": "camera+depth",
            "budget_mb": round(info.budget_mb),
            "budget_rule": f"VRAM libre au lancement - {device.VRAM_MARGIN_MB} Mo, plafond par processus",
            "max_n_fitting": max(ok, default=0),
            "rows": rows,
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"N max sous le plafond de {info.budget_mb:.0f} Mo : {max(ok, default=0)} → {OUT}")


if __name__ == "__main__":
    main()
