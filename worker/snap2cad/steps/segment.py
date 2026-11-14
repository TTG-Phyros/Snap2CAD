"""Étape `segment` (plan §4.2 n° 3) : SAM 2.1 en mode vidéo → `masks/object/*.png`, `masks/reference/*.png`.

Les photos sont ordonnées par heure de prise de vue (EXIF) : la propagation suit ainsi le tour de l'objet.
Les clics (pixels de `images/`, origine au coin haut gauche) sont normalement sur la 1ʳᵉ photo, mais
peuvent porter sur n'importe laquelle : la propagation part de la première photo cliquée vers la fin,
puis revient vers le début.
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
}


def frame_order(exif: dict[str, Any]) -> list[dict[str, Any]]:
    """Photos retenues par l'import, triées par heure EXIF ; celles sans heure à la fin, par nom."""
    kept = [e for e in exif["images"] if e["status"] != "rejected"]
    # "AAAA:MM:JJ HH:MM:SS.fraction" : l'ordre lexicographique est l'ordre chronologique.
    return sorted(kept, key=lambda e: (e["datetime"] is None, e["datetime"] or "", e["file"]))


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
    key = project.inputs_hash(["images", "exif.json"], p, versions)
    if not project.needs_run(STEP, key):
        return project.data["steps"][STEP]

    t0 = time.perf_counter()
    targets = {t: TARGETS[t] for t in TARGETS if p["clicks"].get(t)}
    masks: dict[str, list[np.ndarray | None]] = {t: [None] * len(frames) for t in targets}
    predictor = sam_model.load(p["model"], info)
    device.reset_peak_memory(info)
    with tempfile.TemporaryDirectory() as tmp, torch.inference_mode(), device.autocast(info):
        for i, e in enumerate(frames):  # SAM lit un dossier « <n° de trame>.jpg »
            shutil.copyfile(root / e["file"], Path(tmp) / f"{i:05d}.jpg")
        # Trames et état sur le processeur : la VRAM ne dépend pas du nombre de photos.
        state = predictor.init_state(tmp, offload_video_to_cpu=True, offload_state_to_cpu=True)
        clicked = []
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
                clicked.append(i)
        names = {v: k for k, v in targets.items()}
        for reverse in (False, True):
            for i, obj_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=min(clicked), reverse=reverse
            ):
                hw = (frames[i]["height"], frames[i]["width"])
                if tuple(logits.shape[-2:]) != hw:  # SAM ramène tout à la taille d'une seule photo
                    logits = F.interpolate(logits.float(), hw, mode="bilinear", align_corners=False)
                for obj_id, m in zip(obj_ids, (logits[:, 0] > 0).cpu().numpy(), strict=True):
                    masks[names[obj_id]][i] = m
    peak = device.peak_memory_mb(info)
    del predictor, state

    shutil.rmtree(root / "masks", ignore_errors=True)
    outputs: list[str] = []
    checks: list[dict[str, Any]] = []
    area: dict[str, list[float]] = {}
    for target, ms in masks.items():
        (root / "masks" / target).mkdir(parents=True)
        area[target] = []
        for e, m in zip(frames, ms, strict=True):
            assert m is not None
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
        result={"order": [Path(e["file"]).name for e in frames], "area_ratio": area, "checks": checks},
    )
    return project.data["steps"][STEP]
