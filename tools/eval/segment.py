"""Critère M2 : IoU des masques SAM 2.1 sur le jeu synthétique (≥ 0,9) et aucune perte de l'objet.

Usage : uv run --project worker python -m tools.eval.segment [série ...] [--n 20] [--work DOSSIER]
Clic simulé : un point au cœur de la cible (érosions du masque de vérité terrain) sur la première photo,
dans l'ordre EXIF, où elle est bien visible. Par défaut : toutes les séries de datasets/synth.
Résultats ajoutés dans docs/benchmarks/segment.json.
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
from PIL import Image

from snap2cad import device
from snap2cad.project import Project
from snap2cad.steps import import_, segment
from tools.eval import datasets

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / "datasets/synth"
OUT = ROOT / "docs/benchmarks/segment.json"
LABELS = {"object": 1, "reference": 2}  # valeurs de `labels/*.png` (tools/synth)
MIN_CLICK_PX = 2000  # cible assez visible pour y cliquer
LOST_IOU = 0.5


def inner_point(mask: np.ndarray, step: int = 8) -> tuple[float, float]:
    """Point (x, y) loin du bord de `mask` : dernier pixel survivant aux érosions 3×3 (image réduite)."""
    m = np.pad(mask[::step, ::step], 1)
    if not m.any():
        raise ValueError("masque vide")
    while True:
        e = m.copy()
        e[1:-1, 1:-1] &= m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
        if not e.any():
            break
        m = e
    i, j = np.argwhere(m)[len(np.argwhere(m)) // 2] - 1
    return step * (j + 0.5), step * (i + 0.5)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def evaluate(series: str, work: Path, n: int, info: device.DeviceInfo) -> dict[str, Any]:
    scene = datasets.synth(SYNTH, series)
    root = work / f"{scene.name}.s2c"
    shutil.rmtree(root, ignore_errors=True)
    datasets.write_project(scene, root, n)
    project = Project(root)
    import_.run(project)
    frames = segment.frame_order(json.loads((root / "exif.json").read_text("utf-8")))
    stems = [Path(e["file"]).stem for e in frames]
    gt = {s: np.asarray(Image.open(SYNTH / series / "labels" / f"{s}.png")) for s in stems}
    clicks = {}
    for target, label in LABELS.items():
        s = next((s for s in stems if (gt[s] == label).sum() >= MIN_CLICK_PX), None)
        if s is not None:
            x, y = inner_point(gt[s] == label)
            clicks[target] = [{"image": f"{s}.jpg", "x": x, "y": y, "positive": True}]
    rec = segment.run(project, info, {"clicks": clicks})

    out: dict[str, Any] = {"series": series, "n_images": len(stems), "time_s": rec["duration_s"]}
    for target in clicks:
        scores = {}
        for s in stems:
            g = gt[s] == LABELS[target]
            if g.mean() >= segment.PARAMS["min_area_ratio"]:  # IoU indéfinie si la cible est absente
                scores[s] = iou(np.asarray(Image.open(root / f"masks/{target}/{s}.png")) > 0, g)
        v = np.array(list(scores.values()))
        out[target] = {
            "views": len(v),
            "iou_mean": round(float(v.mean()), 4),
            "iou_min": round(float(v.min()), 4),
            "lost": sorted(s for s, x in scores.items() if x < LOST_IOU),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("series", nargs="*")
    ap.add_argument("--n", type=int, default=20, help="vues réparties sur la série")
    ap.add_argument("--work", type=Path, default=ROOT / "datasets/_eval")
    a = ap.parse_args()
    series = a.series or sorted(p.parent.name for p in SYNTH.glob("*/scene.json"))
    info = device.detect()
    rows = []
    for s in series:
        r = evaluate(s, a.work, a.n, info)
        rows.append(r)
        print(
            f"{s:40s}"
            + "".join(
                f"  {t} IoU moy {r[t]['iou_mean']:.3f} min {r[t]['iou_min']:.3f} perdues {len(r[t]['lost'])}"
                for t in LABELS
                if t in r
            )
        )
    summary = {
        t: {
            "iou_mean": round(float(np.mean([r[t]["iou_mean"] for r in rows if t in r])), 4),
            "series_lost": sum(bool(r[t]["lost"]) for r in rows if t in r),
        }
        for t in LABELS
        if any(t in r for r in rows)
    }
    print("résumé :", summary)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True, cwd=ROOT
    )
    entry = {
        "date": time.strftime("%Y-%m-%d"),
        "commit": commit.stdout.strip(),
        "gpu": info.name,
        "method": "SAM 2.1 vidéo (segment), 1 clic positif simulé par cible",
        "n_views": a.n,
        "summary": summary,
        "series": rows,
    }
    data = json.loads(OUT.read_text("utf-8")) if OUT.exists() else []
    OUT.write_text(json.dumps([*data, entry], indent=2, ensure_ascii=False) + "\n", "utf-8")


if __name__ == "__main__":
    main()
