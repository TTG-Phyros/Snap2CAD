"""Étape `segment` (plan §4.2 n° 3) : SAM 2.1 en mode vidéo → `masks/object/*.png`, `masks/reference/*.png`.

Les photos sont ordonnées par heure de prise de vue (EXIF) : la propagation suit ainsi le tour de l'objet.
Les clics (pixels de `images/`, origine au coin haut gauche) sont normalement sur la 1ʳᵉ photo, mais
peuvent porter sur n'importe laquelle : la propagation part de la première photo cliquée vers la fin,
puis revient vers le début.

Repli guidé en 3D (risque R3, vues très espacées), si `sparse/` existe (étape `ba`, qui n'utilise pas les
masques) : les points 3D de la cible sont ceux dont les observations tombent dans les masques des photos
fiables (cliquées, puis celles où la propagation s'accorde avec ces points). Une photo où la plupart des
observations de ces points tombent hors du masque est perdue : la boîte de leur projection y devient une
invite, comme un clic de correction, et la propagation est relancée.
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
import pycolmap
import torch
import torch.nn.functional as F
from PIL import Image

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
    "guide_min_points": 15,  # points de la cible projetés dans la photo, en dessous : photo non jugée
    "guide_trust": 0.8,  # part des projections dans le masque : photo fiable (sert à remonter la cible)
    "guide_lost": 0.5,  # en dessous : photo perdue, relancée avec la boîte reprojetée
    "guide_rounds": 3,  # relances au plus par cible (comme la correction manuelle)
}
Box = tuple[float, float, float, float]  # x0, y0, x1, y1 en pixels de la photo


def frame_order(exif: dict[str, Any]) -> list[dict[str, Any]]:
    """Photos retenues par l'import, triées par heure EXIF ; celles sans heure à la fin, par nom."""
    kept = [e for e in exif["images"] if e["status"] != "rejected"]
    # "AAAA:MM:JJ HH:MM:SS.fraction" : l'ordre lexicographique est l'ordre chronologique.
    return sorted(kept, key=lambda e: (e["datetime"] is None, e["datetime"] or "", e["file"]))


class Tracks:
    """Points de `sparse/` et leurs observations, par n° de photo (ordre de `frame_order`)."""

    def __init__(self, rec: pycolmap.Reconstruction, frames: list[dict[str, Any]]):
        index = {Path(e["file"]).name: i for i, e in enumerate(frames)}
        self.xyz = {pid: p.xyz for pid, p in rec.points3D.items()}
        self.obs: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # photo → (ids des points, pixels (N, 2))
        self.images: dict[int, Any] = {}
        self.cameras: dict[int, Any] = {}
        for im in rec.images.values():
            if (i := index.get(Path(im.name).name)) is None:
                continue
            o = [(p.point3D_id, p.xy) for p in im.points2D if p.has_point3D()]
            self.obs[i] = (np.array([a for a, _ in o], np.int64), np.array([b for _, b in o]).reshape(-1, 2))
            self.images[i], self.cameras[i] = im, rec.cameras[im.camera_id]

    def lift(self, masks: dict[int, np.ndarray], trusted: set[int]) -> np.ndarray:
        """Points de la cible : observés dans au moins une photo fiable, dans le masque pour plus de la moitié."""
        votes: dict[int, list[bool]] = defaultdict(list)
        for i in trusted & self.obs.keys():
            for pid, ok in zip(self.obs[i][0], inside(masks[i], self.obs[i][1]), strict=True):
                votes[int(pid)].append(bool(ok))
        return np.array(sorted(pid for pid, v in votes.items() if np.mean(v) > 0.5), np.int64)

    def project(self, i: int, ids: np.ndarray, hw: tuple[int, int] | None = None) -> np.ndarray:
        """Projection (distorsion comprise) des points `ids` devant la photo i ; avec `hw`, ceux dans l'image."""
        im, cam = self.images[i], self.cameras[i]
        X = np.array([self.xyz[int(p)] for p in ids]).reshape(-1, 3)
        Xc = X @ im.cam_from_world.rotation.matrix().T + im.cam_from_world.translation
        Xc = Xc[Xc[:, 2] > 0]
        if not len(Xc):
            return np.zeros((0, 2))
        uv = np.asarray(cam.img_from_cam(Xc[:, :2] / Xc[:, 2:]), np.float64).reshape(-1, 2)
        return uv[(uv >= 0).all(1) & (uv[:, 0] < hw[1]) & (uv[:, 1] < hw[0])] if hw else uv

    def box(self, i: int, ids: np.ndarray, hw: tuple[int, int]) -> Box | None:
        """Boîte (1ᵉʳ – 99ᵉ centiles) de la projection des points `ids` dans la photo i."""
        uv = self.project(i, ids, hw)
        if len(uv) < 3:
            return None
        lo, hi = np.percentile(uv, 1, axis=0), np.percentile(uv, 99, axis=0)
        return None if (hi <= lo).any() else (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def inside(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Pixels `xy` (N, 2), convention COLMAP, dans `mask` ?"""
    x = np.clip(xy[:, 0].astype(int), 0, mask.shape[1] - 1)
    y = np.clip(xy[:, 1].astype(int), 0, mask.shape[0] - 1)
    return mask[y, x]


def lost_views(
    tracks: Tracks,
    masks: dict[int, np.ndarray],
    clicked: set[int],
    p: dict[str, Any],
    hidden: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[int, float]]:
    """Points de la cible et photos perdues (n° → part des projections dans le masque). Les points cachés
    se projettent aussi dans la silhouette : toutes les projections comptent, pas seulement les observations.
    Photo jugée si la cible y est entière (≥ 90 % des projections dans l'image) et, avec `hidden` (masques de
    l'objet, pour la référence posée sur la table), pas cachée à plus de moitié. Les photos fiables s'ajoutent
    aux photos cliquées tant que l'ensemble change."""
    trusted = set(clicked)
    for _ in range(10):
        ids = tracks.lift(masks, trusted)
        frac = {}
        for i in tracks.obs:
            front = tracks.project(i, ids)
            uv = tracks.project(i, ids, masks[i].shape)
            if len(uv) < max(p["guide_min_points"], 0.9 * len(front)):
                continue
            if hidden is not None:
                uv = uv[~inside(hidden[i], uv)]
                if len(uv) < max(p["guide_min_points"], 0.5 * len(front)):
                    continue
            frac[i] = float(inside(masks[i], uv).mean())
        new = clicked | {i for i, f in frac.items() if f >= p["guide_trust"]}
        if new == trusted:
            break
        trusted = new
    return ids, {i: f for i, f in frac.items() if f < p["guide_lost"] and i not in clicked}


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

    info = info or device.detect()
    if info.profile == "unsupported":
        raise RuntimeError(f"GPU non pris en charge pour SAM 2.1 : {info.name} ({info.memory_gib:.1f} Gio)")
    p["model"] = p["model"] or sam_model.MODELS[info.profile][0]
    model_sha = manifest()[p["model"]]["files"][0]["sha256"]
    versions = {"snap2cad": version("snap2cad"), "sam2": version("sam-2"), "model": model_sha}
    guided = (root / "sparse").is_dir()
    key = project.inputs_hash(["images", "exif.json", *(["sparse"] if guided else [])], p, versions)
    if not project.needs_run(STEP, key):
        return project.data["steps"][STEP]

    t0 = time.perf_counter()
    targets = {t: TARGETS[t] for t in TARGETS if p["clicks"].get(t)}
    masks: dict[str, dict[int, np.ndarray]] = {t: {} for t in targets}  # n° de photo → masque
    tracks = Tracks(pycolmap.Reconstruction(root / "sparse"), frames) if guided else None
    prompts: dict[str, dict[str, list[float]]] = {t: {} for t in targets}  # photo → boîte reprojetée
    predictor = sam_model.load(p["model"], info)
    device.reset_peak_memory(info)
    with tempfile.TemporaryDirectory() as tmp, torch.inference_mode(), device.autocast(info):
        for i, e in enumerate(frames):  # SAM lit un dossier « <n° de trame>.jpg »
            shutil.copyfile(root / e["file"], Path(tmp) / f"{i:05d}.jpg")
        # Trames et état sur le processeur : la VRAM ne dépend pas du nombre de photos.
        state = predictor.init_state(tmp, offload_video_to_cpu=True, offload_state_to_cpu=True)
        by_frame = {t: defaultdict(list) for t in targets}
        for t in targets:
            for c in p["clicks"][t]:
                by_frame[t][index[c["image"]]].append(c)
        clicked = {t: set(by_frame[t]) for t in targets}
        boxes: dict[str, dict[int, Box]] = {t: {} for t in targets}  # n° de photo → boîte reprojetée
        names = {v: k for k, v in targets.items()}

        def prompt_and_propagate() -> None:
            # Toutes les invites sur un état vierge : une invite posée sur une photo déjà suivie ne serait
            # qu'une correction, oubliée à la propagation suivante.
            predictor.reset_state(state)
            for target, obj_id in targets.items():
                for i, cs in by_frame[target].items():
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
                # Boîte seule : des points positifs en plus aident l'enceinte noire mais perdent la pièce du mug,
                # des points seuls dégradent (8 séries, M2-segment.md).
                for i, b in boxes[target].items():
                    w, h = frames[i]["width"], frames[i]["height"]
                    predictor.add_new_points_or_box(
                        state, i, obj_id, box=np.array(b, np.float32) / [w, h, w, h], normalize_coords=False
                    )
            start = min(i for cs in clicked.values() for i in cs)
            for reverse in (False, True):
                for i, obj_ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=start, reverse=reverse
                ):
                    hw = (frames[i]["height"], frames[i]["width"])
                    if tuple(logits.shape[-2:]) != hw:  # SAM ramène tout à la taille d'une seule photo
                        logits = F.interpolate(logits.float(), hw, mode="bilinear", align_corners=False)
                    for obj_id, m in zip(obj_ids, (logits[:, 0] > 0).cpu().numpy(), strict=True):
                        masks[names[obj_id]][i] = m

        prompt_and_propagate()
        for _ in range(p["guide_rounds"] if tracks else 0):
            added = False
            for target in targets:
                # Graines : les photos cliquées. Une photo relancée ne compte que si elle s'accorde ensuite.
                hidden = masks["object"] if target == "reference" and "object" in masks else None
                ids, lost = lost_views(tracks, masks[target], clicked[target], p, hidden)
                for i in sorted(lost, key=lost.__getitem__):  # la plus perdue d'abord, une par tour
                    hw = (frames[i]["height"], frames[i]["width"])
                    if i in boxes[target] or (box := tracks.box(i, ids, hw)) is None:
                        continue
                    boxes[target][i] = box
                    prompts[target][Path(frames[i]["file"]).name] = [round(v, 1) for v in box]
                    added = True
                    break
            if not added:
                break
            prompt_and_propagate()
    peak = device.peak_memory_mb(info)
    predictor = state = None  # libère la VRAM ; pas de del : la fonction imbriquée les référence

    shutil.rmtree(root / "masks", ignore_errors=True)
    outputs: list[str] = []
    checks: list[dict[str, Any]] = []
    area: dict[str, list[float]] = {}
    for target, ms in masks.items():
        (root / "masks" / target).mkdir(parents=True)
        area[target] = []
        assert len(ms) == len(frames)  # propagation avant puis arrière : toutes les photos
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
            "guided_3d": prompts if guided else None,  # photos relancées avec une boîte reprojetée
            "checks": checks,
        },
    )
    return project.data["steps"][STEP]
