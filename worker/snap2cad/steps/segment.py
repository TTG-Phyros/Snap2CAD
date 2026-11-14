"""Étape `segment` (plan §4.2 n° 3) : SAM 2.1 en mode vidéo → `masks/object/*.png`, `masks/reference/*.png`.

Les photos sont ordonnées par heure de prise de vue (EXIF) : la propagation suit ainsi le tour de l'objet.
Les clics (pixels de `images/`, origine au coin haut gauche) sont normalement sur la 1ʳᵉ photo, mais
peuvent porter sur n'importe laquelle : la propagation part de la première photo cliquée vers la fin,
puis revient vers le début.

Repli guidé en 3D (risque R3, vues très espacées) : le masque des photos cliquées est remonté en 3D avec la
profondeur VGGT (`poses_init`), puis reprojeté dans chaque photo. Si la boîte reprojetée et celle du masque
propagé ne se recouvrent pas assez, SAM refait la photo en mode image avec la boîte reprojetée.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

from snap2cad import device
from snap2cad.models import manifest
from snap2cad.models import sam as sam_model
from snap2cad.project import Project

STEP = "segment"
TARGETS = {"object": 1, "reference": 2}  # identifiant d'objet SAM par cible
REFERENCES = ("coin2", "coin1", "card")  # pièce de 2 €, de 1 €, carte ID-1
PARAMS: dict[str, Any] = {
    "model": None,  # None : selon le profil (large sur 8 Go, base_plus sur 6 Go)
    "reference_type": "coin2",
    # {"object": [{"image": "IMG_0001.jpg", "x": 1520.0, "y": 980.0, "positive": true}, ...], "reference": [...]}
    "clicks": {},
    "min_area_ratio": 1e-4,  # masque plus petit que cette part de l'image : cible absente de la photo
    "min_reference_views": 5,  # plan §1 : référence visible sur au moins 5 photos
    "fallback_box_iou": 0.5,  # en dessous : masque propagé jugé perdu, repli sur la boîte reprojetée
    "box_margin": 0.1,  # marge ajoutée à la boîte reprojetée (part de sa taille)
    "depth_tolerance": 0.03,  # point caché s'il est plus loin que la profondeur VGGT de plus de 3 %
}
Box = tuple[float, float, float, float]  # x0, y0, x1, y1 en pixels de la photo


def frame_order(exif: dict[str, Any]) -> list[dict[str, Any]]:
    """Photos retenues par l'import, triées par heure EXIF ; celles sans heure à la fin, par nom."""
    kept = [e for e in exif["images"] if e["status"] != "rejected"]
    # "AAAA:MM:JJ HH:MM:SS.fraction" : l'ordre lexicographique est l'ordre chronologique.
    return sorted(kept, key=lambda e: (e["datetime"] is None, e["datetime"] or "", e["file"]))


def lift(mask: np.ndarray, pred: dict[str, np.ndarray], k: int) -> np.ndarray:
    """Points 3D (repère VGGT) des pixels de `mask` (photo pleine résolution) dans la vue VGGT `k`.

    On garde la moitié la plus confiante de la profondeur sous le masque.
    """
    depth, conf, K = pred["depth"][k], pred["depth_conf"][k], pred["intrinsic"][k]
    sx, sy, tx, ty = pred["affine"][k]
    v, u = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]] + 0.5  # centres des pixels VGGT
    U, V = np.floor((u - tx) / sx).astype(int), np.floor((v - ty) / sy).astype(int)
    sel = (U >= 0) & (U < mask.shape[1]) & (V >= 0) & (V < mask.shape[0])  # hors complément blanc
    sel[sel] = mask[V[sel], U[sel]]
    if not sel.any():
        return np.zeros((0, 3))
    sel &= conf >= np.median(conf[sel])
    xc = np.linalg.solve(K, np.stack([u[sel], v[sel], np.ones(sel.sum())])) * depth[sel]
    R, t = pred["extrinsic"][k][:, :3], pred["extrinsic"][k][:, 3]
    return (xc - t[:, None]).T @ R  # Rᵀ·(x_cam − t)


def guide_box(
    X: np.ndarray,
    pred: dict[str, np.ndarray],
    k: int,
    hw: tuple[int, int],
    p: dict[str, Any],
    occlusion: bool,
) -> Box | None:
    """Boîte de la projection de `X` dans la photo `k` ; None si la cible n'y est pas (ou cachée)."""
    R, t, K = pred["extrinsic"][k][:, :3], pred["extrinsic"][k][:, 3], pred["intrinsic"][k]
    xc = X @ R.T + t
    z = xc[:, 2]
    uv = (xc @ K.T)[:, :2] / np.where(z > 0, z, np.inf)[:, None]
    sx, sy, tx, ty = pred["affine"][k]
    full = (uv - [tx, ty]) / [sx, sy]
    keep = (z > 0) & (full[:, 0] >= 0) & (full[:, 0] < hw[1]) & (full[:, 1] >= 0) & (full[:, 1] < hw[0])
    if occlusion:
        # La référence, à plat sur la table, peut être cachée par l'objet. Pas l'objet : ses points vus
        # depuis la photo cliquée sont souvent derrière sa propre face avant, sans qu'il soit caché.
        d = pred["depth"][k]
        i = np.clip(uv[:, 1].astype(int), 0, d.shape[0] - 1)
        j = np.clip(uv[:, 0].astype(int), 0, d.shape[1] - 1)
        keep &= z <= d[i, j] * (1 + p["depth_tolerance"])
    if keep.sum() < max(10, 0.2 * len(X)):
        return None
    lo, hi = np.percentile(full[keep], 2, axis=0), np.percentile(full[keep], 98, axis=0)
    pad = p["box_margin"] * (hi - lo)
    lo, hi = np.maximum(lo - pad, 0), np.minimum(hi + pad, [hw[1], hw[0]])
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def mask_box(m: np.ndarray) -> Box | None:
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    if not len(rows):
        return None
    return float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)


def box_iou(a: Box | None, b: Box) -> float:
    if a is None:
        return 0.0
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union


def run(
    project: Project, info: device.DeviceInfo | None = None, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    p = {**PARAMS, **(params or {})}
    root = project.root
    if p["reference_type"] not in REFERENCES:
        raise ValueError(f"Référence inconnue : {p['reference_type']} (choix : {', '.join(REFERENCES)})")
    if unknown := set(p["clicks"]) - TARGETS.keys():
        raise ValueError(f"Cible inconnue : {', '.join(sorted(unknown))}")
    if not p["clicks"].get("object"):
        raise ValueError("Cliquer au moins une fois sur l'objet.")
    frames = frame_order(json.loads((root / "exif.json").read_text("utf-8")))
    index = {Path(e["file"]).name: i for i, e in enumerate(frames)}
    if unknown := {c["image"] for cs in p["clicks"].values() for c in cs} - index.keys():
        raise ValueError(f"Clic sur une photo absente de la série : {', '.join(sorted(unknown))}")
    if not (root / "vggt/predictions.npz").is_file():
        raise RuntimeError("Lancer l'étape poses_init avant segment (repli guidé en 3D).")

    info = info or device.detect()
    if info.profile == "unsupported":
        raise RuntimeError(f"GPU non pris en charge pour SAM 2.1 : {info.name} ({info.memory_gib:.1f} Gio)")
    p["model"] = p["model"] or sam_model.MODELS[info.profile][0]
    model_sha = manifest()[p["model"]]["files"][0]["sha256"]
    versions = {"snap2cad": version("snap2cad"), "sam2": version("sam-2"), "model": model_sha}
    key = project.inputs_hash(["images", "exif.json", "vggt/predictions.npz"], p, versions)
    if not project.needs_run(STEP, key):
        return project.data["steps"][STEP]

    t0 = time.perf_counter()
    targets = {t: TARGETS[t] for t in TARGETS if p["clicks"].get(t)}
    masks: dict[str, dict[int, np.ndarray]] = {t: {} for t in targets}  # n° de photo → masque
    predictor = sam_model.load(p["model"], info)
    device.reset_peak_memory(info)
    with tempfile.TemporaryDirectory() as tmp, torch.inference_mode(), device.autocast(info):
        for i, e in enumerate(frames):  # SAM lit un dossier « <n° de trame>.jpg »
            shutil.copyfile(root / e["file"], Path(tmp) / f"{i:05d}.jpg")
        # Trames et état sur le processeur : la VRAM ne dépend pas du nombre de photos.
        state = predictor.init_state(tmp, offload_video_to_cpu=True, offload_state_to_cpu=True)
        clicked: dict[str, list[int]] = {}
        for target, obj_id in targets.items():
            by_frame = defaultdict(list)
            for c in p["clicks"][target]:
                by_frame[index[c["image"]]].append(c)
            for i, cs in by_frame.items():
                size = np.array([frames[i]["width"], frames[i]["height"]], np.float32)
                predictor.add_new_points_or_box(
                    state,
                    i,
                    obj_id,
                    points=np.array([[c["x"], c["y"]] for c in cs], np.float32) / size,
                    labels=np.array([1 if c["positive"] else 0 for c in cs], np.int32),
                    # Photos de tailles différentes : SAM normaliserait par une seule, on le fait par photo.
                    normalize_coords=False,
                )
            clicked[target] = list(by_frame)
        names = {v: k for k, v in targets.items()}
        for reverse in (False, True):
            for i, obj_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=min(i for cs in clicked.values() for i in cs), reverse=reverse
            ):
                hw = (frames[i]["height"], frames[i]["width"])
                if tuple(logits.shape[-2:]) != hw:  # SAM ramène tout à la taille d'une seule photo
                    logits = F.interpolate(logits.float(), hw, mode="bilinear", align_corners=False)
                for obj_id, m in zip(obj_ids, (logits[:, 0] > 0).cpu().numpy(), strict=True):
                    masks[names[obj_id]][i] = m

        with np.load(root / "vggt/predictions.npz") as f:
            pred = {k: f[k] for k in ("names", "extrinsic", "intrinsic", "depth", "depth_conf", "affine")}
        vi = {str(n): k for k, n in enumerate(pred["names"])}
        image_predictor = SAM2ImagePredictor(predictor)  # même modèle, en mode image
        fallback: dict[str, list[str]] = {}
        for target in targets:
            ms = masks[target]
            X = np.concatenate([lift(ms[i], pred, vi[Path(frames[i]["file"]).name]) for i in clicked[target]])
            fallback[target] = []
            for i, e in enumerate(frames):
                hw = (e["height"], e["width"])
                box = guide_box(X, pred, vi[Path(e["file"]).name], hw, p, occlusion=target == "reference")
                if box is None or box_iou(mask_box(ms[i]), box) >= p["fallback_box_iou"]:
                    continue
                image_predictor.set_image(np.asarray(Image.open(root / e["file"]).convert("RGB")))
                out, _, _ = image_predictor.predict(box=np.array(box), multimask_output=False)
                ms[i] = out[0] > 0
                fallback[target].append(Path(e["file"]).name)
    peak = device.peak_memory_mb(info)
    del predictor, state, image_predictor

    shutil.rmtree(root / "masks", ignore_errors=True)
    outputs: list[str] = []
    checks: list[dict[str, Any]] = []
    area: dict[str, list[float]] = {}
    for target, ms in masks.items():
        (root / "masks" / target).mkdir(parents=True)
        area[target] = []
        assert len(ms) == len(frames)  # propagation avant + arrière : toutes les photos
        for i, e in enumerate(frames):
            m = ms[i]
            rel = f"masks/{target}/{Path(e['file']).stem}.png"
            Image.fromarray(m.astype(np.uint8) * 255).save(root / rel)
            outputs.append(rel)
            area[target].append(round(float(m.mean()), 6))
    absent = {
        t: [Path(e["file"]).name for e, a in zip(frames, area[t]) if a < p["min_area_ratio"]] for t in area
    }
    if absent["object"]:
        checks.append(
            {
                "level": "warning",
                "code": "object_lost",
                "message": "Objet non retrouvé sur certaines photos : ajouter un clic sur l'une d'elles.",
                "images": absent["object"],
            }
        )
    n_ref = len(frames) - len(absent.get("reference", frames))
    if n_ref < p["min_reference_views"]:
        checks.append(
            {
                "level": "warning",
                "code": "reference_few_views",
                "message": f"Référence visible sur {n_ref} photos, il en faut au moins "
                f"{p['min_reference_views']} pour l'échelle.",
                "images": absent.get("reference", []),
            }
        )
    project.record(
        STEP,
        key,
        p,
        outputs,
        model_sha256=model_sha,
        code_version=f"snap2cad {versions['snap2cad']} + sam2 {versions['sam2']}",
        device=info.name,
        duration_s=round(time.perf_counter() - t0, 2),
        peak_vram_mb=round(peak) if peak else None,
        result={
            "order": [Path(e["file"]).name for e in frames],
            "area_ratio": area,
            "fallback_3d": fallback,
            "checks": checks,
        },
    )
    return project.data["steps"][STEP]
