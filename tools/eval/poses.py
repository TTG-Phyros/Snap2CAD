"""Critère M1 : erreur de rotation médiane de VGGT (poses_init) sur DTU et MobileBrick.

Usage : uv run --project worker python -m tools.eval.poses [scène ...] [--n 20] [--work DOSSIER] [--ba] [--resume]
Scènes : dtu:1, dtu:6, mobilebrick:aston, synth:<série>, ... ; par défaut DTU 1 et 6 + les 18 séquences MobileBrick.
Avec --ba : aussi l'étape `ba` (M3), comparée à VGGT seul. Résultats ajoutés dans docs/benchmarks/poses.json.
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
import pycolmap

from snap2cad import device
from snap2cad.project import Project
from snap2cad.steps import ba, import_, poses_init
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


def rotation_stats(rot: np.ndarray) -> dict[str, float]:
    return {
        k: round(float(f(rot)), 3) for k, f in (("median", np.median), ("mean", np.mean), ("max", np.max))
    }


def evaluate(
    scene: datasets.Scene, work: Path, n: int, info: device.DeviceInfo, run_ba: bool, resume: bool = False
) -> dict[str, Any]:
    root = work / f"{scene.name}.s2c"
    if not (resume and (root / "exif.json").exists()):  # exif.json : import fait, donc entrées complètes
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
    out = {
        "scene": scene.name,
        "n_images": len(rot),
        "rotation_deg": rotation_stats(rot),
        "center_mm_median": round(float(np.median(err["center"])), 2),
        "focal_ratio_median": round(float(f_ratio), 4),  # fx VGGT / fx vérité terrain
        "time_s": rec["duration_s"],
    }
    if run_ba:
        # DTU : images rectifiées, point principal loin du centre (≈ 23 px) ; MobileBrick comme un smartphone.
        rec = ba.run(project, info, {"refine_principal_point": scene.name.startswith("dtu")})
        sparse = pycolmap.Reconstruction(root / "sparse")
        by_name = {im.name: im for im in sparse.images.values()}
        extr = np.stack([by_name[n].cam_from_world.matrix() for n in gt["names"]])
        f = np.array([sparse.cameras[by_name[n].camera_id].params[0] for n in gt["names"]])
        err = metrics.pose_errors(extr, gt["extrinsic"])
        out["ba"] = {
            "rotation_deg": rotation_stats(err["rotation_deg"]),
            "center_mm_median": round(float(np.median(err["center"])), 2),
            "focal_ratio_median": round(float(np.median(f / gt["K"][:, 0, 0])), 4),
            "reproj_px": rec["result"]["reproj_px"],
            "n_points": rec["result"]["n_points"],
            "time_s": rec["duration_s"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--n", type=int, default=20, help="vues réparties sur la série (VGGT : ≤ N₆)")
    ap.add_argument("--work", type=Path, default=ROOT / "datasets/_eval")
    ap.add_argument("--ba", action="store_true", help="ajustement de faisceaux (étape ba, M3)")
    ap.add_argument("--resume", action="store_true", help="garder les projets existants (étapes en cache)")
    a = ap.parse_args()
    scenes = a.scenes or ["dtu:1", "dtu:6", *(f"mobilebrick:{p.name}" for p in sorted(MOBILEBRICK.iterdir()))]
    info = device.detect()
    rows = []
    for spec in scenes:
        r = evaluate(load(spec), a.work, a.n, info, a.ba, a.resume)
        rows.append(r)
        for label, x in (("vggt", r), ("ba", r.get("ba"))):
            if x:
                print(
                    f"{r['scene']:28s} {label:4s} rot méd {x['rotation_deg']['median']:6.2f} deg  "
                    f"max {x['rotation_deg']['max']:6.2f} deg  centre {x['center_mm_median']:7.2f} mm  "
                    f"f {x['focal_ratio_median']:.3f}"
                    + (f"  reproj méd {x['reproj_px']['median']:.3f} px" if label == "ba" else "")
                )

    def per_dataset(key: str | None) -> dict[str, float]:
        return {
            kind: round(float(np.median([(r[key] if key else r)["rotation_deg"]["median"] for r in rs])), 3)
            for kind in ("dtu", "mobilebrick", "synth")
            if (rs := [r for r in rows if r["scene"].startswith(kind)])
        }

    med = per_dataset(None)
    print("médiane par dataset :", med)
    if a.ba:
        med_ba = per_dataset("ba")
        reproj = round(float(np.median([r["ba"]["reproj_px"]["median"] for r in rows])), 3)
        print("médiane par dataset après ba :", med_ba, " reprojection médiane :", reproj)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True, cwd=ROOT
    )
    entry = {
        "date": time.strftime("%Y-%m-%d"),
        "commit": commit.stdout.strip(),
        "gpu": info.name,
        "method": "vggt-1b bf16 (poses_init)"
        + (" puis ba (suivis VGGSfM ALIKED+SP, pycolmap)" if a.ba else "")
        + ", alignement Umeyama sur les centres",
        "n_views": a.n,
        "rotation_median_deg": med,
        **({"ba_rotation_median_deg": med_ba, "ba_reproj_median_px": reproj} if a.ba else {}),
        "scenes": rows,
    }
    data = json.loads(OUT.read_text("utf-8")) if OUT.exists() else []
    OUT.write_text(json.dumps([*data, entry], indent=2, ensure_ascii=False) + "\n", "utf-8")


if __name__ == "__main__":
    main()
