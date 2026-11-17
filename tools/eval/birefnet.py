"""BiRefNet seul (objet saillant) sur le jeu synthétique : IoU de l'objet par vue, part de la référence prise
pour l'objet, temps et VRAM. Sert à choisir la variante et à savoir quand s'y fier en repli de `segment`.

Usage : uv run --project worker python -m tools.eval.birefnet [--models birefnet birefnet-lite]
Vues : les 20 de chaque projet `datasets/_eval/synth_*.s2c` (comme tools.eval.segment).
Résultats ajoutés dans docs/benchmarks/birefnet.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from snap2cad import device
from snap2cad.models import birefnet

from tools.eval.segment import LABELS, LOST_IOU, SYNTH, iou

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "datasets/_eval"
OUT = ROOT / "docs/benchmarks/birefnet.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[birefnet.NAME])  # birefnet-lite : hors manifeste
    a = ap.parse_args()
    info = device.detect()
    projects = sorted(EVAL.glob("synth_*.s2c"))
    entry: dict = {
        "date": time.strftime("%Y-%m-%d"),
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout.strip(),
        "gpu": info.name,
        "method": f"BiRefNet seul, {birefnet.SIZE} px, seuil 0,5 ; IoU objet ; vue perdue : IoU < {LOST_IOU}",
        "models": {},
    }
    for name in a.models:
        device.reset_peak_memory(info)
        model = birefnet.load(name, info)
        rows, times = [], []
        for proj in projects:
            series = proj.name[len("synth_") : -len(".s2c")]
            for img in sorted((proj / "images").glob("*.jpg")):
                lab = np.asarray(
                    Image.open(SYNTH / series / "labels" / f"{img.stem}.png")
                )
                gt, ref = lab == LABELS["object"], lab == LABELS["reference"]
                if gt.mean() < 1e-4:
                    continue
                t0 = time.perf_counter()
                m = (
                    birefnet.predict(
                        model, np.asarray(Image.open(img).convert("RGB")), info
                    )
                    > 0.5
                )
                times.append(time.perf_counter() - t0)
                rows.append({
                    "series": series,
                    "view": img.stem,
                    "iou": round(iou(m, gt), 4),
                    "iou_largest": round(iou(birefnet.largest_component(m), gt), 4),
                    "iou_largest_no_ref": round(iou(birefnet.largest_component(m, ref), gt), 4),  # référence connue
                    "reference_taken": round(float(m[ref].mean()), 3) if ref.any() else None,
                })  # fmt: skip
        del model
        torch.cuda.empty_cache()
        v = np.array([r["iou"] for r in rows])
        by_series = {}
        for r in rows:
            by_series.setdefault(r["series"], []).append(r["iou"])
        refs = [r["reference_taken"] for r in rows if r["reference_taken"] is not None]
        entry["models"][name] = {
            "views": len(rows),
            "iou_mean": round(float(v.mean()), 4),
            "iou_median": round(float(np.median(v)), 4),
            "views_lost": int((v < LOST_IOU).sum()),
            "iou_mean_largest": round(
                float(np.mean([r["iou_largest"] for r in rows])), 4
            ),
            "views_lost_largest": int(sum(r["iou_largest"] < LOST_IOU for r in rows)),
            "iou_mean_largest_no_ref": round(
                float(np.mean([r["iou_largest_no_ref"] for r in rows])), 4
            ),
            "views_lost_largest_no_ref": int(
                sum(r["iou_largest_no_ref"] < LOST_IOU for r in rows)
            ),
            "series_iou_090": int(sum(np.mean(s) >= 0.9 for s in by_series.values())),
            "reference_taken_over_half": int(sum(x > 0.5 for x in refs)),
            "time_s_median": round(float(np.median(times[1:])), 3),
            "peak_vram_mb": round(p) if (p := device.peak_memory_mb(info)) else None,
            "series": {s: round(float(np.mean(x)), 4) for s, x in by_series.items()},
            "rows": rows,
        }
        print(
            name,
            {
                k: x
                for k, x in entry["models"][name].items()
                if k not in ("series", "rows")
            },
        )
    data = json.loads(OUT.read_text("utf-8")) if OUT.exists() else []
    OUT.write_text(
        json.dumps([*data, entry], indent=2, ensure_ascii=False) + "\n", "utf-8"
    )


if __name__ == "__main__":
    main()
