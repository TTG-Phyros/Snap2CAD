"""Critère M2 : IoU des masques SAM 2.1 sur le jeu synthétique (≥ 0,9) et aucune perte de l'objet.

Usage : uv run --project worker python -m tools.eval.segment [série ...] [--n 20] [--work DOSSIER]
Clics simulés comme un utilisateur, sur la première photo (ordre EXIF) où la cible est bien visible : un
point au cœur de la cible, puis des clics de correction au cœur de la plus grande erreur (SAM en mode
image) jusqu'à IoU ≥ 0,95, au plus 5 clics. Puis correction manuelle simulée : tant qu'une vue reste sous
IoU 0,8, l'utilisateur clique de même sur la pire (au plus 3 vues par cible) et relance l'étape.
Par défaut : toutes les séries de datasets/synth.
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
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from snap2cad import device
from snap2cad.models import models_dir
from snap2cad.models import sam as sam_model
from snap2cad.project import Project
from snap2cad.steps import import_, segment
from tools.eval import datasets

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / "datasets/synth"
OUT = ROOT / "docs/benchmarks/segment.json"
LABELS = {"object": 1, "reference": 2}  # valeurs de `labels/*.png` (tools/synth)
MIN_CLICK_PX = 2000  # cible assez visible pour y cliquer
LOST_IOU = 0.5
MAX_CLICKS, CLICK_IOU = 5, 0.95
MAX_CORRECTIONS, CORRECT_IOU = 3, 0.8  # vues corrigées par cible ; une vue sous ce seuil se voit à l'écran


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


def user_clicks(predictor: SAM2ImagePredictor, gt: np.ndarray) -> list[tuple[float, float, bool]]:
    """Clic au cœur de `gt`, puis corrections au cœur de l'erreur dominante (manqué -> +, en trop -> −)."""
    clicks = [(*inner_point(gt), True)]
    while len(clicks) < MAX_CLICKS:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([c[:2] for c in clicks]),
            point_labels=np.array([int(c[2]) for c in clicks]),
            multimask_output=len(clicks) == 1,  # un seul clic est ambigu : SAM propose 3 masques
        )
        m = masks[scores.argmax()] > 0
        missed, extra = gt & ~m, m & ~gt
        err, positive = (missed, True) if missed.sum() >= extra.sum() else (extra, False)
        if iou(m, gt) >= CLICK_IOU or not err[::8, ::8].any():
            break
        clicks.append((*inner_point(err), positive))
    return clicks


def to_correct(scores: dict[str, float], clickable: set[str], clicked: set[str]) -> str | None:
    """Vue que l'utilisateur corrige : la pire sous CORRECT_IOU, où la cible se voit, pas encore cliquée."""
    bad = [s for s, x in scores.items() if x < CORRECT_IOU and s in clickable and s not in clicked]
    return min(bad, key=scores.__getitem__, default=None)


def evaluate(
    series: str, work: Path, n: int, info: device.DeviceInfo, predictor: SAM2ImagePredictor
) -> dict[str, Any]:
    scene = datasets.synth(SYNTH, series)
    root = work / f"{scene.name}.s2c"
    shutil.rmtree(root, ignore_errors=True)
    datasets.write_project(scene, root, n)
    project = Project(root)
    import_.run(project)
    frames = segment.frame_order(json.loads((root / "exif.json").read_text("utf-8")))
    stems = [Path(e["file"]).stem for e in frames]
    gt = {s: np.asarray(Image.open(SYNTH / series / "labels" / f"{s}.png")) for s in stems}

    def click(target: str, s: str) -> list[dict[str, Any]]:
        with torch.inference_mode(), device.autocast(info):
            predictor.set_image(np.asarray(Image.open(root / "images" / f"{s}.jpg").convert("RGB")))
            cs = user_clicks(predictor, gt[s] == LABELS[target])
        return [{"image": f"{s}.jpg", "x": x, "y": y, "positive": p} for x, y, p in cs]

    clickable = {
        t: {s for s in stems if (gt[s] == label).sum() >= MIN_CLICK_PX} for t, label in LABELS.items()
    }
    clicks = {t: click(t, s) for t in LABELS if (s := next((s for s in stems if s in clickable[t]), None))}
    corrected: dict[str, list[str]] = {t: [] for t in clicks}
    runs = []
    while True:
        rec = segment.run(project, info, {"clicks": clicks})
        scores = {}
        for target in clicks:
            scores[target] = {
                s: iou(np.asarray(Image.open(root / f"masks/{target}/{s}.png")) > 0, gt[s] == LABELS[target])
                for s in stems
                if (gt[s] == LABELS[target]).mean() >= segment.PARAMS["min_area_ratio"]  # IoU indéfinie sinon
            }
        runs.append((rec["duration_s"], scores))
        todo = {
            t: v
            for t in clicks
            if len(corrected[t]) < MAX_CORRECTIONS
            and (v := to_correct(scores[t], clickable[t], {c["image"][:-4] for c in clicks[t]}))
        }
        if not todo:
            break
        for t, v in todo.items():
            corrected[t].append(v)
            clicks[t] += click(t, v)

    def stats(sc: dict[str, float]) -> dict[str, Any]:
        v = np.array(list(sc.values()))
        lost = sorted(s for s, x in sc.items() if x < LOST_IOU)
        return {"iou_mean": round(float(v.mean()), 4), "iou_min": round(float(v.min()), 4), "lost": lost}

    out: dict[str, Any] = {"series": series, "n_images": len(stems), "time_s": runs[0][0], "runs": len(runs)}
    for target, tc in clicks.items():
        first, final = runs[0][1][target], runs[-1][1][target]
        out[target] = {
            "views": len(final),
            "clicks": len(tc),
            "clicked_iou": round(final[tc[0]["image"][:-4]], 4),
            "corrected_views": corrected[target],
            **stats(final),
            "before_correction": stats(first),
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
    name, config = sam_model.MODELS[info.profile]
    predictor = SAM2ImagePredictor(build_sam2(config, str(models_dir() / name / "model.pt"), info.device))
    rows = []
    for s in series:
        r = evaluate(s, a.work, a.n, info, predictor)
        rows.append(r)
        print(
            f"{s:40s}"
            + "".join(
                f"  {t} {r[t]['clicks']} clic(s) {len(r[t]['corrected_views'])} vue(s) corrigée(s) "
                f"moy {r[t]['before_correction']['iou_mean']:.3f} -> {r[t]['iou_mean']:.3f} "
                f"min {r[t]['iou_min']:.3f} perdues {len(r[t]['before_correction']['lost'])} -> {len(r[t]['lost'])}"
                for t in LABELS
                if t in r
            )
        )

    def agg(rs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "iou_mean": round(float(np.mean([r["iou_mean"] for r in rs])), 4),
            "series_iou_090": sum(r["iou_mean"] >= 0.9 for r in rs),
            "series_lost": sum(bool(r["lost"]) for r in rs),
        }

    summary = {}
    for t in LABELS:
        if rs := [r[t] for r in rows if t in r]:
            summary[t] = {
                **agg(rs),
                "corrected_views_mean": round(float(np.mean([len(r["corrected_views"]) for r in rs])), 2),
                "before_correction": agg([r["before_correction"] for r in rs]),
            }
    print("résumé :", summary)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True, cwd=ROOT
    )
    entry = {
        "date": time.strftime("%Y-%m-%d"),
        "commit": commit.stdout.strip(),
        "gpu": info.name,
        "method": f"SAM 2.1 vidéo (segment, {name}), clics simulés sur la 1re vue (≤ {MAX_CLICKS}, IoU ≥ {CLICK_IOU}), "
        f"puis correction des vues sous IoU {CORRECT_IOU} (≤ {MAX_CORRECTIONS} par cible)",
        "n_views": a.n,
        "summary": summary,
        "series": rows,
    }
    data = json.loads(OUT.read_text("utf-8")) if OUT.exists() else []
    OUT.write_text(json.dumps([*data, entry], indent=2, ensure_ascii=False) + "\n", "utf-8")


if __name__ == "__main__":
    main()
