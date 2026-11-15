"""Jalon M4, contour par vue : arête haute de la référence comparée à la vérité terrain du jeu synthétique.

Usage : uv run --project worker python -m tools.eval.scale [série ...] [--no-save]
Masques de référence de l'étape `segment` (projets `datasets/_eval/synth_*.s2c`), caméras exactes de
`datasets/synth/<série>/cameras.npz` (k1 compris) et normale de la table projetée : on isole l'erreur du contour
de celle des poses et du plan.
- Pièce : demi-axes de l'ellipse ajustée vs ceux de la face haute projetée (cercle à z = épaisseur).
- Carte : longueur des côtés ajustés vs ceux de la face haute (coins vifs, 85,60 × 53,98 mm).
L'erreur relative de taille d'une vue donne l'ordre de grandeur de sa contribution à l'erreur d'échelle.
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
    views, fits, ups = [], [], {}
    for mpath in sorted((proj / "masks/reference").glob("*.png")):
        mask = np.asarray(Image.open(mpath)) > 0
        if mask.sum() < MIN_MASK_PX:  # référence absente ou perdue par `segment`
            continue
        w2c, K, k1 = cams[mpath.stem]
        cam = pycolmap.Camera(
            model="SIMPLE_RADIAL", width=W, height=H, params=[K[0, 0], K[0, 2], K[1, 2], k1]
        )
        rgb = np.asarray(Image.open(proj / "images" / f"{mpath.stem}.jpg").convert("RGB"))
        top = pinhole(reference_outline(ref, ref["dims_mm"]["thickness"]), w2c, K)
        hull = cv2.convexHull(np.r_[top, pinhole(reference_outline(ref, 0.0), w2c, K)].astype(np.float32))
        # « Haut » : normale de la table projetée au centre de la référence (en production : plan et poses).
        a, b = pinhole(np.stack([c0, c0 + [0, 0, 10.0]]), w2c, K)
        ups[mpath.stem] = b - a
        v: dict[str, Any] = {"view": mpath.stem, "size_px": round(float(np.ptp(hull[:, 0], 0).max()), 1)}
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
    out = {"series": series, "reference": ref["type"], "views": views, "fit3d": {"exact": exact}}
    if (proj / "sparse").is_dir():
        out["fit3d"]["ba"] = fit3d_ba(proj, ref, cams, ups, params)
    return out


def fit3d(fits: list[dict[str, Any]], ref: dict[str, Any], mm_per_unit: float) -> dict[str, Any]:
    """Cercle ou rectangle 3D sur les vues ajustées ; erreur d'échelle (%) vs `mm_per_unit` (vérité)."""
    try:
        f = scale.fit_rectangle(fits) if ref["type"] == "card" else scale.fit_circle(fits, ref["type"])
    except ValueError as e:
        return {"error": str(e)}
    err = (np.atleast_1d(f["mm_per_unit"]) / mm_per_unit - 1) * 100
    return {
        "scale_err_pct": np.round(err, 4).tolist(),  # pièce : diamètre ; carte : grand puis petit côté
        "n_views": len(f["views"]),
        "rejected": f["rejected"],
        "rms_px_median": round(float(np.median([f["rms_px"][n] for n in f["views"]])), 4),
    }


def fit3d_ba(
    proj: Path,
    ref: dict[str, Any],
    cams: dict[str, Any],
    ups: dict[str, np.ndarray],
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Même chose avec les caméras de `ba` (distorsion comprise) ; vérité : similitude des centres des caméras
    `ba` → vérité terrain (mm par unité `ba`)."""
    rec = pycolmap.Reconstruction(proj / "sparse")
    fits, src, dst = [], [], []
    for im in rec.images.values():
        stem = Path(im.name).stem
        w2c = im.cam_from_world.matrix()
        src.append(-w2c[:, :3].T @ w2c[:, 3])
        dst.append(metrics.camera_centers(cams[stem][0][None])[0])
        if stem not in ups:
            continue
        cam = rec.cameras[im.camera_id]
        mask = np.asarray(Image.open(proj / "masks/reference" / f"{stem}.png")) > 0
        rgb = np.asarray(Image.open(proj / "images" / im.name).convert("RGB"))
        try:
            fit = scale.fit_view(rgb, mask, cam, ref["type"], up=ups[stem], params=params)
        except ValueError:
            continue
        fits.append({"name": stem, "K": cam.calibration_matrix(), "cam_from_world": w2c, **fit})
    s, _, _ = metrics.umeyama(np.array(src), np.array(dst))
    return fit3d(fits, ref, s)


def summarize_3d(rows: list[dict[str, Any]], kind: str, cams: str) -> dict[str, Any]:
    fs = [
        r["fit3d"][cams]
        for r in rows
        if (r["reference"] == "card") == (kind == "card") and cams in r["fit3d"]
    ]
    ok = [f for f in fs if "error" not in f]
    if not ok:
        return {"series": len(fs), "ok": 0}
    err = np.array([f["scale_err_pct"] for f in ok])  # (séries, 1 ou 2)
    a = np.abs(err)
    return {
        "series": len(fs),
        "ok": len(ok),
        "scale_err_pct": {
            "median": np.round(np.median(a, 0), 4).tolist(),
            "max": np.round(a.max(0), 4).tolist(),
            "mean_signed": np.round(err.mean(0), 4).tolist(),
        },
        "kept_failed": sum(len(f.get("kept_failed", [])) for f in ok),
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
        summary[k]["fit3d"] = {c: summarize_3d(rows, k, c) for c in ("exact", "ba")}
    print("résumé :", json.dumps(summary, ensure_ascii=False))
    if a.no_save:
        return
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True, cwd=ROOT
    )
    entry = {
        "date": time.strftime("%Y-%m-%d"),
        "commit": commit.stdout.strip(),
        "method": "scale.fit_view : arête haute par rayons (silhouette côté éloigné, modèle de tranche côté caméra), "
        "masques SAM de `segment`, normale de table exacte ; puis scale.fit_circle / fit_rectangle (caméras exactes, "
        "et caméras `ba` alignées par similitude des centres)",
        "params": scale.PARAMS,
        "summary": summary,
        "series": rows,
    }
    data = json.loads(OUT.read_text("utf-8")) if OUT.exists() else []
    OUT.write_text(json.dumps([*data, entry], indent=2, ensure_ascii=False) + "\n", "utf-8")


if __name__ == "__main__":
    main()
