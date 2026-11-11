"""Étape `import` (plan §4.2 n° 1) : `input/` → `images/` normalisées + `exif.json` et contrôles."""

from __future__ import annotations

import json
import math
import shutil
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pillow_heif
from PIL import Image, ImageOps
from PIL.ExifTags import IFD, Base

from snap2cad.project import Project

pillow_heif.register_heif_opener()

STEP = "import"
EXTENSIONS = {".jpg", ".jpeg", ".heic", ".heif", ".png"}  # PNG : datasets publics (DTU)
DIAG_35MM = math.hypot(36.0, 24.0)  # la focale équivalente 35 mm se rapporte à la diagonale
PARAMS: dict[str, Any] = {
    "min_photos": 8,  # en dessous : la série est refusée
    "recommended_photos": 15,  # plan §9.3 : 15 à 40 vues
    "min_focal_35mm": 20.0,  # en dessous : ultra grand-angle, photo écartée
    # shortcut: focale physique < 3 mm = module ultra grand-angle de smartphone (heuristique, pas de table
    # par modèle) ; à remplacer par une table Make/Model/LensModel si des faux positifs apparaissent.
    "max_uw_focal_mm": 3.0,
    "blur_ratio": 0.3,  # flou probable si variance du laplacien < ratio × médiane de la série
    "blur_long_side": 1024,
    "jpeg_quality": 95,
}


def _text(v: Any) -> str | None:
    return (str(v).strip("\x00 ") or None) if v is not None else None


def read_exif(img: Image.Image) -> dict[str, Any]:
    ifd0 = img.getexif()
    ex = ifd0.get_ifd(IFD.Exif)
    f, f35 = ex.get(Base.FocalLength), ex.get(Base.FocalLengthIn35mmFilm)
    dt, sub = _text(ex.get(Base.DateTimeOriginal)), _text(ex.get(Base.SubsecTimeOriginal))
    return {
        "make": _text(ifd0.get(Base.Make)),
        "model": _text(ifd0.get(Base.Model)),
        "lens_model": _text(ex.get(Base.LensModel)),
        "focal_mm": float(f) if f else None,
        "focal_35mm": float(f35) if f35 else None,
        "datetime": f"{dt}.{sub}" if dt and sub else dt,  # tri des photos au M2
        "orientation": int(ifd0.get(Base.Orientation, 1)),
    }


def intrinsics(w: int, h: int, focal_35mm: float | None) -> dict[str, Any]:
    if focal_35mm:
        f, source = focal_35mm * math.hypot(w, h) / DIAG_35MM, "exif_35mm"
    else:
        f, source = 1.2 * max(w, h), "default"  # valeur par défaut de COLMAP, affinée par l'ajustement
    # Point principal au centre, convention COLMAP (centre du pixel à 0,5).
    return {"fx": f, "fy": f, "cx": w / 2, "cy": h / 2, "source": source}


def blur_score(img: Image.Image, long_side: int) -> float:
    g = img.convert("L")
    g.thumbnail((long_side, long_side))  # score indépendant de la résolution d'origine
    a = np.asarray(g, np.float32)
    lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
    return float(lap.var())


def run(project: Project, params: dict[str, Any] | None = None) -> dict[str, Any]:
    p = {**PARAMS, **(params or {})}
    root = project.root
    key = project.inputs_hash(
        ["input"], p, {"snap2cad": version("snap2cad"), "pillow_heif": pillow_heif.__version__}
    )
    if not project.needs_run(STEP, key):
        return json.loads((root / "exif.json").read_text("utf-8"))

    t0 = time.perf_counter()
    out = root / "images"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    checks: list[dict[str, Any]] = []

    def check(level: str, code: str, message: str, images: list[str]) -> None:
        checks.append({"level": level, "code": code, "message": message, "images": images})

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for src in sorted(
        (q for q in (root / "input").iterdir() if q.suffix.lower() in EXTENSIONS), key=lambda q: q.name
    ):
        e: dict[str, Any] = {"source": f"input/{src.name}", "status": "ok"}
        entries.append(e)
        name = src.stem + ".jpg"
        if name.lower() in seen:
            e["status"] = "rejected"
            check("error", "duplicate_name", f"Nom en double, photo écartée : {src.name}", [src.name])
            continue
        seen.add(name.lower())
        with Image.open(src) as img:
            e.update(read_exif(img))
            if e["focal_35mm"] and e["focal_35mm"] < p["min_focal_35mm"]:
                e["status"] = "rejected"
                check(
                    "error",
                    "ultra_wide",
                    f"Ultra grand-angle ({e['focal_35mm']:.0f} mm éq.), photo écartée : "
                    "utiliser l'objectif principal (1×).",
                    [src.name],
                )
                continue
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            original_ok = (
                src.suffix.lower() in (".jpg", ".jpeg") and e["orientation"] == 1 and img.mode == "RGB"
            )
        if original_ok:
            shutil.copyfile(src, out / name)  # pas de réencodage : aucune perte
        else:
            rgb.save(out / name, quality=p["jpeg_quality"], subsampling=0)
        e["file"] = f"images/{name}"
        e["width"], e["height"] = rgb.size
        e["K"] = intrinsics(*rgb.size, e["focal_35mm"])
        e["blur_var"] = blur_score(rgb, p["blur_long_side"])

    kept = [e for e in entries if e["status"] != "rejected"]
    names = lambda es: [Path(e["source"]).name for e in es]

    if no_focal := [e for e in kept if e["K"]["source"] == "default"]:
        check("warning", "no_focal", "Focale absente de l'EXIF : valeur par défaut.", names(no_focal))
    # Bascule macro de l'iPhone : le module ultra grand-angle est recadré pour imiter le 1×.
    if macro := [e for e in kept if e["focal_mm"] and e["focal_mm"] < p["max_uw_focal_mm"]]:
        check(
            "warning",
            "cropped_ultra_wide",
            "Photos prises avec le module ultra grand-angle recadré (bascule macro ?) : "
            "désactiver le mode macro automatique ou s'éloigner de l'objet.",
            names(macro),
        )
    lenses = Counter((e["lens_model"], e["focal_35mm"]) for e in kept)
    if len(lenses) > 1:
        main = lenses.most_common(1)[0][0]
        others = [e for e in kept if (e["lens_model"], e["focal_35mm"]) != main]
        check("warning", "mixed_lenses", "Plusieurs objectifs ou zooms dans la série.", names(others))
    if len(kept) >= 3:
        med = float(np.median([e["blur_var"] for e in kept]))
        if blurry := [e for e in kept if e["blur_var"] < p["blur_ratio"] * med]:
            for e in blurry:
                e["status"] = "warning"
            check("warning", "blurry", "Photos probablement floues : les reprendre.", names(blurry))
    if len(kept) < p["min_photos"]:
        check(
            "error",
            "too_few_photos",
            f"{len(kept)} photos exploitables, il en faut au moins {p['min_photos']}.",
            [],
        )
    elif len(kept) < p["recommended_photos"]:
        check(
            "warning",
            "few_photos",
            f"{len(kept)} photos : {p['recommended_photos']} ou plus recommandées.",
            [],
        )

    result = {
        "ok": len(kept) >= p["min_photos"],
        "n_images": len(kept),
        "images": entries,
        "checks": checks,
    }
    (root / "exif.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
    project.record(
        STEP,
        key,
        p,
        ["exif.json", *(e["file"] for e in kept)],
        duration_s=round(time.perf_counter() - t0, 3),
        result={"ok": result["ok"], "n_images": len(kept)},
    )
    return result
