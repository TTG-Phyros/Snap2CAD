"""Jalon M4, contour par vue : arête haute de la référence comparée à la vérité terrain du jeu synthétique.

Usage : uv run --project worker python -m tools.eval.scale [série ...] [--no-save]
Masques de référence de l'étape `segment` (projets `datasets/_eval/synth_*.s2c`), caméras exactes de
`datasets/synth/<série>/cameras.npz` (k1 compris) et normale de la table projetée : on isole l'erreur du contour
de celle des poses et du plan.
- Pièce : demi-axes de l'ellipse ajustée vs ceux de la face haute projetée (cercle à z = épaisseur).
- Carte : longueur des côtés ajustés vs ceux de la face haute (coins vifs, 85,60 × 53,98 mm).
L'erreur relative de taille d'une vue donne l'ordre de grandeur de sa contribution à l'erreur d'échelle.
Puis cercle ou rectangle 3D : caméras exactes, et étape `scale` complète (caméras `ba`, plan de table, repère
monde) alignée sur la vérité par similitude des centres des caméras.
Aussi : distance des points retenus à l'arête haute et à la silhouette exactes (enveloppe des faces haute et
basse). Par défaut : toutes les séries ; résultats ajoutés dans docs/benchmarks/scale.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap
from PIL import Image

from snap2cad.project import Project
from snap2cad.steps import scale
from tools.eval import metrics

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / "datasets/synth"
EVAL = ROOT / "datasets/_eval"
OUT = ROOT / "docs/benchmarks/scale.json"
MIN_MASK_PX = 500
FAIL_PCT = 2.0  # vue ratée : erreur de taille > 2 % (ou ajustement impossible)


def reference_outline(ref: dict[str, Any], z: float, n: int = 720) -> np.ndarray:
    """Contour de la face de la référence à la hauteur z (mm, monde) : cercle, ou 4 coins vifs de la carte."""
    d, (cx, cy, _) = ref["dims_mm"], ref["center_mm"]
    if ref["type"] == "card":
        w, h = d["width"] / 2, d["height"] / 2
        local = np.array([[w, h], [-w, h], [-w, -h], [w, -h]])
    else:
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        local = d["diameter"] / 2 * np.stack([np.cos(a), np.sin(a)], 1)
    t = np.radians(ref["yaw_deg"])
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return np.c_[local @ R.T + [cx, cy], np.full(len(local), z)]


def pinhole(X: np.ndarray, w2c: np.ndarray, K: np.ndarray) -> np.ndarray:
    Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
    return Xc[:, :2] / Xc[:, 2:] @ K[:2, :2].T + K[:2, 2]


def match_corners(fit: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Coins de `gt` réordonnés pour correspondre à `fit` (sens et départ du contour)."""
    cands = [np.roll(g, k, 0) for g in (gt, gt[::-1]) for k in range(4)]
    return min(cands, key=lambda g: float(np.linalg.norm(fit - g, axis=1).mean()))


def sides(c: np.ndarray) -> np.ndarray:
    return np.linalg.norm(c - np.roll(c, -1, 0), axis=1)


def evaluate(series: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    proj = EVAL / f"synth_{series}.s2c"
    scene = json.loads((SYNTH / series / "scene.json").read_text("utf-8"))
    ref = scene["reference"]
    with np.load(SYNTH / series / "cameras.npz") as f:
        cams = {
            Path(str(n)).stem: (e, K, float(k1))
            for n, e, K, k1 in zip(f["names"], f["extrinsic"], f["K"], f["k1"], strict=True)
        }
    W, H = scene["camera"]["width"], scene["camera"]["height"]
    c0 = np.array(ref["center_mm"])
    views, fits = [], []
    for mpath in sorted((proj / "masks/reference").glob("*.png")):
        mask = np.asarray(Image.open(mpath)) > 0
        if mask.sum() < MIN_MASK_PX:  # référence absente ou perdue par `segment`
            continue
        w2c, K, k1 = cams[mpath.stem]
        cam = pycolmap.Camera(
            model="SIMPLE_RADIAL",
            width=W,
            height=H,
            params=[K[0, 0], K[0, 2], K[1, 2], k1],
        )
        rgb = np.asarray(Image.open(proj / "images" / f"{mpath.stem}.jpg").convert("RGB"))
        top = pinhole(reference_outline(ref, ref["dims_mm"]["thickness"]), w2c, K)
        hull = cv2.convexHull(np.r_[top, pinhole(reference_outline(ref, 0.0), w2c, K)].astype(np.float32))
        # « Haut » : normale de la table projetée au centre de la référence (en production : plan et poses).
        a, b = pinhole(np.stack([c0, c0 + [0, 0, 10.0]]), w2c, K)
        v: dict[str, Any] = {
            "view": mpath.stem,
            "size_px": round(float(np.ptp(hull[:, 0], 0).max()), 1),
        }
        try:
            fit = scale.fit_view(rgb, mask, cam, ref["type"], up=(b - a), params=params)
        except ValueError as e:
            views.append({**v, "error": str(e)})
            continue
        for key, poly in (("top_px", top.astype(np.float32)), ("silhouette_px", hull)):
            d = np.array([-cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in fit["points"]])
            v[key] = {
                "median": round(float(np.median(d)), 3),
                "p90_abs": round(float(np.percentile(np.abs(d), 90)), 3),
            }
        if ref["type"] == "card":
            c = np.array(fit["corners"])
            gt = sides(match_corners(c, top))
            v["size_err_pct"] = np.round((sides(c) - gt) / gt * 100, 3).tolist()
        else:
            _, (gw, gh), _ = cv2.fitEllipseDirect((top - top.mean(0)).astype(np.float32))
            fa, ga = np.sort(fit["ellipse"][2:4]), np.sort([gw / 2, gh / 2])
            v["axes_err_px"] = np.round(fa - ga, 3).tolist()  # [petit, grand] demi-axe
            v["size_err_pct"] = np.round((fa - ga) / ga * 100, 3).tolist()
        v["failed"] = bool(np.abs(v["size_err_pct"]).max() > FAIL_PCT)
        views.append(v)
        fits.append({"name": mpath.stem, "K": K, "cam_from_world": w2c, **fit})
    exact = fit3d(fits, ref, 1.0)
    if "rejected" in exact:  # tri par la cohérence entre vues vs vues ratées (erreur de taille > 2 %)
        failed = {v["view"] for v in views if v.get("failed")}
        exact["kept_failed"] = sorted(failed - set(exact["rejected"]))
        exact["rejected_ok"] = sorted(set(exact["rejected"]) - failed)
    out = {
        "series": series,
        "reference": ref["type"],
        "views": views,
        "fit3d": {"exact": exact},
    }
    if (proj / "sparse").is_dir():
        out["fit3d"]["step"] = step(proj, scene, cams)
    return out


def fit3d(fits: list[dict[str, Any]], ref: dict[str, Any], mm_per_unit: float) -> dict[str, Any]:
    """Cercle ou rectangle 3D sur les vues ajustées ; erreur d'échelle (%) vs `mm_per_unit` (vérité)."""
    try:
        f = scale.fit_rectangle(fits) if ref["type"] == "card" else scale.fit_circle(fits, ref["type"])
    except ValueError as e:
        return {"error": str(e)}
    err = (np.atleast_1d(f["mm_per_unit"]) / mm_per_unit - 1) * 100
    s = float(np.atleast_1d(f["mm_per_unit"])[0])
    return {
        "scale_err_pct": np.round(err, 4).tolist(),  # pièce : diamètre ; carte : grand puis petit côté
        "sigma_rel_pct": [round(scale.sigma_rel(ref["type"], f) * 100, 4)],  # jackknife ⊕ plancher
        "sigma_views_pct": np.round(np.atleast_1d(f["sigma_rel"]) * 100, 4).tolist(),  # jackknife seul
        # contrôles avec la table exacte (z = 0, monde en mm)
        "checks": {k: c["value"] for k, c in scale.checks(ref["type"], f, np.array([0.0, 0, 1]), 0.0, s).items()},
        "n_views": len(f["views"]),
        "rejected": f["rejected"],
        "rms_px_median": round(float(np.median([f["rms_px"][n] for n in f["views"]])), 4),
        "rms_rel_pct": round(f["rms_rel"] * 100, 3),
    }


def step(proj: Path, scene: dict[str, Any], cams: dict[str, Any]) -> dict[str, Any]:
    """Étape `scale` complète (caméras `ba`, plan de table, « haut » par vue) ; vérité : similitude des centres
    des caméras `ba` → vérité terrain (table : z = 0). Plan de table mesuré seul (même si l'étape échoue), puis
    erreurs d'échelle, d'origine (centre de la référence sur la table) et d'axe X (grand axe de l'objet)."""
    ref = scene["reference"]
    rec = pycolmap.Reconstruction(proj / "sparse")
    names = [Path(im.name).stem for im in rec.images.values()]
    src = np.array([im.projection_center() for im in rec.images.values()])
    s, R, t = metrics.umeyama(src, metrics.camera_centers(np.array([cams[n][0] for n in names])))
    xyz, obj = scale.sparse_points(rec, proj / "masks/object")
    ba = json.loads((proj / "ba_report.json").read_text("utf-8")) if (proj / "ba_report.json").is_file() else {}
    out: dict[str, Any] = {
        "n_points": len(xyz),
        "n_object": int(obj.sum()),
        "pose_center_mm_median": round(float(np.median(np.linalg.norm(
            s * src @ R.T + t - metrics.camera_centers(np.array([cams[n][0] for n in names])), axis=1))), 3),
        "ba_checks": [c["code"] for c in ba.get("checks", [])],  # avertissements de `ba` (pose_drift…)
    }  # fmt: skip
    try:
        n, _, inl = scale.table_plane(xyz[~obj], src)
    except ValueError as e:
        return {**out, "error": str(e)}
    z = (s * xyz[~obj][inl] @ R.T + t)[:, 2]
    out["table"] = {
        "normal_err_deg": round(float(np.degrees(np.arccos(min((R @ n)[2], 1.0)))), 3),
        "inliers": int(inl.sum()),
        "z_median_mm": round(float(np.median(z)), 3),
    }
    try:
        scale.run(Project(proj), {"reference_type": ref["type"]})
    except ValueError as e:
        return {**out, "error": str(e)}
    res = json.loads((proj / "scale.json").read_text("utf-8"))
    M = np.array(res["world_from_sparse"])
    origin = s * R @ np.linalg.solve(M[:3, :3], -M[:3, 3]) + t
    o2w = np.array(scene["object"]["object_to_world_mm"])[:3, :3]
    long_axis = o2w[:, int(np.argmax(scene["object"]["overall_mm"]))]
    x, lx = (R @ M[0, :3])[:2], long_axis[:2]  # angle dans le plan de la table, au signe près
    fit = res["reference_fit"]
    return {
        **out,
        "scale_err_pct": np.round((np.atleast_1d(fit["mm_per_unit"]) / s - 1) * 100, 4).tolist(),
        "sigma_rel_pct": [round(res["sigma_rel"] * 100, 4)],
        "sigma_views_pct": np.round(np.atleast_1d(fit["sigma_rel"]) * 100, 4).tolist(),
        "checks": {k: c["value"] for k, c in res["checks"].items()},
        "checks_ok": res["checks_ok"],
        "n_views": len(fit["views"]),
        "rejected": fit["rejected"],
        "rms_px_median": round(float(np.median([fit["rms_px"][n] for n in fit["views"]])), 4),
        "rms_rel_pct": round(fit["rms_rel"] * 100, 3),
        "origin_err_mm": np.round(origin - [*ref["center_mm"][:2], 0.0], 3).tolist(),
        "reference_height_mm": res["reference_height_mm"],
        # grand axe presque vertical (objet debout) : axe X non défini par la vérité
        "x_axis_err_deg": round(
            float(np.degrees(np.arccos(min(abs(x @ lx) / np.hypot(*x) / np.hypot(*lx), 1.0)))), 2
        )
        if np.hypot(*lx) > 0.5
        else None,
    }


def summarize_3d(rows: list[dict[str, Any]], kind: str, cams: str) -> dict[str, Any]:
    fs = [
        r["fit3d"][cams]
        for r in rows
        if (r["reference"] == "card") == (kind == "card") and cams in r["fit3d"]
    ]
    ok = [f for f in fs if "error" not in f]
    plane = {}
    if tables := [f["table"] for f in fs if "table" in f]:
        ang = np.array([t["normal_err_deg"] for t in tables])
        plane = {"table": {"series": len(tables), "normal_err_deg_median": round(float(np.median(ang)), 3),
                           "below_1deg": int((ang < 1).sum())}}  # fmt: skip
    if not ok:
        return {"series": len(fs), "ok": 0, **plane}
    err = np.array([f["scale_err_pct"] for f in ok])  # (séries, 1 ou 2)
    a = np.abs(err)
    z = a[:, 0] / np.maximum([f["sigma_rel_pct"][0] for f in ok], 1e-9)  # échelle retenue : |erreur| / σ
    lim = scale.CHECKS
    passed = [all(abs(v) <= lim[k] for k, v in f["checks"].items()) for f in ok]
    return {
        "series": len(fs),
        "ok": len(ok),
        "scale_err_pct": {
            "median": np.round(np.median(a, 0), 4).tolist(),
            "max": np.round(a.max(0), 4).tolist(),
            "mean_signed": np.round(err.mean(0), 4).tolist(),
        },
        "sigma_rel_pct_median": round(float(np.median([f["sigma_rel_pct"][0] for f in ok])), 4),
        "err_over_sigma": {
            "median": round(float(np.median(z)), 2),
            "within_1": int((z <= 1).sum()),
            "within_2": int((z <= 2).sum()),
        },
        "checks": {
            k: round(float(np.max(np.abs([f["checks"][k] for f in ok]))), 4) for k in ok[0]["checks"]
        },  # écart maximal par contrôle
        "checks_ok": int(sum(passed)),
        "kept_failed": sum(len(f.get("kept_failed", [])) for f in ok),
        **plane,
        "rejected_ok": sum(len(f.get("rejected_ok", [])) for f in ok),
    }


def summarize(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    vs = [v for r in rows if (r["reference"] == "card") == (kind == "card") for v in r["views"]]
    ok = [v for v in vs if "error" not in v and not v["failed"]]
    if not ok:
        return {"views": len(vs), "ok": 0}
    # Pièce : grand demi-axe (non raccourci, il porte l'échelle) ; carte : les 4 côtés.
    err = np.abs([v["size_err_pct"][1] if kind == "coin" else np.mean(v["size_err_pct"]) for v in ok])
    return {
        "views": len(vs),
        "ok": len(ok),
        "failed": sum(v.get("failed", True) for v in vs),
        "size_err_pct": {
            "median": round(float(np.median(err)), 3),
            "p90": round(float(np.percentile(err, 90)), 3),
        },
        "signed_err_pct_mean": round(
            float(
                np.mean([v["size_err_pct"][1] if kind == "coin" else np.mean(v["size_err_pct"]) for v in ok])
            ),
            3,
        ),
        "top_px_p90_median": round(float(np.median([v["top_px"]["p90_abs"] for v in ok])), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("series", nargs="*")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    series = a.series or sorted(p.name[len("synth_") : -len(".s2c")] for p in EVAL.glob("synth_*.s2c"))
    rows = []
    for s in series:
        t0 = time.perf_counter()
        r = evaluate(s)
        rows.append(r)
        ok = [v for v in r["views"] if "error" not in v and not v["failed"]]
        f3 = {c: f.get("scale_err_pct", f.get("error")) for c, f in r["fit3d"].items()}
        print(
            f"{s} ({r['reference']}) : {len(ok)}/{len(r['views'])} vues, 3D {f3}, {time.perf_counter() - t0:.0f} s"
        )
    summary = {k: summarize(rows, k) for k in ("coin", "card")}
    for k in ("coin", "card"):
        summary[k]["fit3d"] = {c: summarize_3d(rows, k, c) for c in ("exact", "step")}
    print("résumé :", json.dumps(summary, ensure_ascii=False))
    if a.no_save:
        return
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    entry = {
        "date": time.strftime("%Y-%m-%d"),
        "commit": commit.stdout.strip(),
        "method": "scale.fit_view : arête haute par rayons (silhouette côté éloigné, modèle de tranche côté caméra), "
        "masques SAM de `segment`, normale de table exacte ; puis scale.fit_circle / fit_rectangle (caméras exactes, "
        "et étape `scale` complète : caméras `ba`, plan de table RANSAC, repère monde ; vérité par similitude des centres) ; "
        "σ d'échelle par jackknife sur les vues et contrôles de cohérence (hauteur, inclinaison, côtés de la carte)",
        "params": scale.PARAMS,
        "summary": summary,
        "series": rows,
    }
    data = json.loads(OUT.read_text("utf-8")) if OUT.exists() else []
    OUT.write_text(json.dumps([*data, entry], indent=2, ensure_ascii=False) + "\n", "utf-8")


if __name__ == "__main__":
    main()
