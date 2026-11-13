"""Critère M1 : erreur de rotation médiane de VGGT (poses_init) sur DTU et MobileBrick.

Usage : uv run --project worker python -m tools.eval.poses [scène ...] [--n 20] [--work DOSSIER]
Scènes : dtu:1, dtu:6, mobilebrick:aston, synth:<série>, ... ; par défaut DTU 1 et 6 + les 18 séquences MobileBrick.
Résultats ajoutés dans docs/benchmarks/poses.json (référence pour l'ajustement de faisceaux, M3).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from snap2cad import device
from snap2cad.project import Project
from snap2cad.steps import import_, poses_init
from tools.eval import datasets, metrics

ROOT = Path(__file__).resolve().parents[2]
DTU = ROOT / "datasets/dtu/SampleSet"
MOBILEBRICK = ROOT / "datasets/mobilebrick/test"
SYNTH = ROOT / "datasets/synth"
OUT = ROOT / "docs/benchmarks/poses.json"


def load(spec: str) -> datasets.Scene:
    kind, arg = spec.split(":")
    if kind == "synth":
        return datasets.synth(SYNTH, arg)
    return datasets.dtu(DTU, int(arg)) if kind == "dtu" else datasets.mobilebrick(MOBILEBRICK, arg)


def evaluate(scene: datasets.Scene, work: Path, n: int, info: device.DeviceInfo) -> dict[str, Any]:
    root = work / f"{scene.name}.s2c"
    shutil.rmtree(root, ignore_errors=True)
    datasets.write_project(scene, root, n)
    project = Project(root)
    import_.run(project)
    rec = poses_init.run(project, info)
    gt = datasets.read_gt(root)
    with np.load(root / "vggt/predictions.npz") as pred:
        assert list(pred["names"]) == list(gt["names"]), "ordre des vues différent"
        err = metrics.pose_errors(pred["extrinsic"].astype(np.float64), gt["extrinsic"])
        f_ratio = np.median(pred["K_full"][:, 0, 0] / gt["K"][:, 0, 0])
    rot = err["rotation_deg"]
    return {
        "scene": scene.name,
        "n_images": len(rot),
        "rotation_deg": {
            "median": round(float(np.median(rot)), 3),
            "mean": round(float(rot.mean()), 3),
            "max": round(float(rot.max()), 3),
        },
        "center_mm_median": round(float(np.median(err["center"])), 2),
        "focal_ratio_median": round(float(f_ratio), 4),  # fx VGGT / fx vérité terrain
        "time_s": rec["duration_s"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--n", type=int, default=20, help="vues réparties sur la série (VGGT : ≤ N₆)")
    ap.add_argument("--work", type=Path, default=ROOT / "datasets/_eval")
    a = ap.parse_args()
    scenes = a.scenes or ["dtu:1", "dtu:6", *(f"mobilebrick:{p.name}" for p in sorted(MOBILEBRICK.iterdir()))]
    info = device.detect()
    rows = []
    for spec in scenes:
        r = evaluate(load(spec), a.work, a.n, info)
        rows.append(r)
        print(
            f"{r['scene']:28s} rot méd {r['rotation_deg']['median']:6.2f}°  "
            f"max {r['rotation_deg']['max']:6.2f}°  centre {r['center_mm_median']:7.2f} mm  "
            f"f {r['focal_ratio_median']:.3f}"
        )
    med = {
        kind: round(
            float(np.median([r["rotation_deg"]["median"] for r in rows if r["scene"].startswith(kind)])), 3
        )
        for kind in ("dtu", "mobilebrick", "synth")
        if any(r["scene"].startswith(kind) for r in rows)
    }
    print("médiane par dataset :", med)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True, cwd=ROOT
    )
    entry = {
        "date": time.strftime("%Y-%m-%d"),
        "commit": commit.stdout.strip(),
        "gpu": info.name,
        "method": "vggt-1b bf16 (poses_init), alignement Umeyama sur les centres",
        "n_views": a.n,
        "rotation_median_deg": med,
        "scenes": rows,
    }
    data = json.loads(OUT.read_text("utf-8")) if OUT.exists() else []
    OUT.write_text(json.dumps([*data, entry], indent=2, ensure_ascii=False) + "\n", "utf-8")


if __name__ == "__main__":
    main()
