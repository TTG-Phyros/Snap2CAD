"""Étape `poses_init` (plan §4.2 n° 2) : VGGT sur les images entières → poses, intrinsèques, profondeur.

Sorties : `vggt/predictions.npz`, `vggt/points.ply` (coloré par confiance), `vggt/preview.glb`.
Unités VGGT, sans échelle réelle ; repère = 1re caméra, convention OpenCV.
"""

from __future__ import annotations

import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from snap2cad import device
from snap2cad.models import manifest
from snap2cad.models import vggt as vggt_model
from snap2cad.project import Project
from snap2cad.project.ply import write_ply

STEP = "poses_init"
SIZE = 518  # largeur d'entrée de VGGT
PARAMS: dict[str, Any] = {
    "model": "vggt-1b",
    "dtype": "bf16",
    "resolution": SIZE,
    "conf_percentile": 50.0,  # points gardés pour le PLY/GLB : confiance au-dessus de ce centile
    "max_points": 1_000_000,
}
# Mesuré sous le plafond VRAM de device.cap_cuda_memory (docs/benchmarks/M0-vggt.md) : N₈ = 28 sur la 4060.
# shortcut: N₆ = 20 date d'avant le plafond, à remesurer sur la 3060. Au-delà : lots + ajustement global (plan R5).
MAX_IMAGES = {"full": 28, "degraded": 20}


def preprocess(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Même règle que `vggt.utils.load_fn` (mode crop), mais on garde la transformation.

    Retourne les images (S, H, W, 3) uint8 et `affine` (S, 4) = (sx, sy, tx, ty) telles que
    u_vggt = sx·u + tx et v_vggt = sy·v + ty (coordonnées continues, origine au coin de l'image).
    """
    imgs, affine = [], []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
            nh = round(h * SIZE / w / 14) * 14  # multiple de 14 (taille des patchs)
            a = np.asarray(im.convert("RGB").resize((SIZE, nh), Image.Resampling.BICUBIC))
        top = max(0, (nh - SIZE) // 2)  # portrait : recadrage central à 518
        imgs.append(a[top : top + SIZE])
        affine.append([SIZE / w, nh / h, 0.0, -top])
    hmax = max(a.shape[0] for a in imgs)
    for i, a in enumerate(imgs):  # formats mélangés : complément blanc centré, comme VGGT
        pad = (hmax - a.shape[0]) // 2
        imgs[i] = np.pad(a, ((pad, hmax - a.shape[0] - pad), (0, 0), (0, 0)), constant_values=255)
        affine[i][3] += pad
    return np.stack(imgs), np.asarray(affine, np.float64)


def full_res_intrinsics(K: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Intrinsèques VGGT (S, 3, 3) → pixels de l'image pleine résolution."""
    sx, sy, tx, ty = affine.T
    out = K.astype(np.float64).copy()
    out[:, 0, 0], out[:, 0, 2] = K[:, 0, 0] / sx, (K[:, 0, 2] - tx) / sx
    out[:, 1, 1], out[:, 1, 2] = K[:, 1, 1] / sy, (K[:, 1, 2] - ty) / sy
    return out


def confidence_colors(conf: np.ndarray) -> np.ndarray:
    """Rang de confiance → rouge (faible) à vert (forte)."""
    c = np.argsort(np.argsort(conf)) / max(len(conf) - 1, 1)
    return np.stack([255 * (1 - c), 255 * c, np.full_like(c, 40)], 1).astype(np.uint8)


def select_points(
    points: np.ndarray, conf: np.ndarray, rgb: np.ndarray, p: dict[str, Any]
) -> tuple[np.ndarray, ...]:
    pts, c, col = points.reshape(-1, 3), conf.reshape(-1), rgb.reshape(-1, 3)
    keep = np.flatnonzero((c >= np.percentile(c, p["conf_percentile"])) & np.isfinite(pts).all(1))
    if len(keep) > p["max_points"]:
        keep = np.sort(np.random.default_rng(0).choice(keep, p["max_points"], replace=False))
    return pts[keep].astype(np.float32), c[keep].astype(np.float32), col[keep]


def write_outputs(root: Path, names: list[str], preds: dict[str, np.ndarray], p: dict[str, Any]) -> None:
    out = root / "vggt"
    out.mkdir(exist_ok=True)
    np.savez(out / "predictions.npz", names=np.asarray(names), **preds)
    points = unproject_depth_map_to_point_map(
        preds["depth"][..., None], preds["extrinsic"], preds["intrinsic"]
    )
    pts, conf, rgb = select_points(points, preds["depth_conf"], preds["images"], p)
    cc = confidence_colors(conf)
    write_ply(out / "points.ply", pts, red=cc[:, 0], green=cc[:, 1], blue=cc[:, 2], confidence=conf)

    # glTF : y vers le haut, z vers l'arrière → on retourne y et z du repère OpenCV.
    cloud = trimesh.PointCloud(pts * np.array([1, -1, -1], np.float32), colors=rgb)
    trimesh.Scene(cloud).export(out / "preview.glb")


def run(
    project: Project, info: device.DeviceInfo | None = None, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    p = {**PARAMS, **(params or {})}
    root = project.root
    model_sha = manifest()[p["model"]]["files"][0]["sha256"]
    versions = {"snap2cad": version("snap2cad"), "vggt": version("vggt"), "model": model_sha}
    key = project.inputs_hash(["images"], p, versions)
    if not project.needs_run(STEP, key):
        return project.data["steps"][STEP]

    paths = sorted((root / "images").glob("*.jpg"), key=lambda q: q.name)
    info = info or device.detect()
    if info.profile == "unsupported":
        raise RuntimeError(f"GPU non pris en charge pour VGGT : {info.name} ({info.memory_gib:.1f} Gio)")
    if len(paths) > MAX_IMAGES[info.profile]:
        raise RuntimeError(
            f"{len(paths)} images : VGGT en traite au plus {MAX_IMAGES[info.profile]} sur ce GPU "
            "(traitement par lots pas encore disponible)."
        )

    t0 = time.perf_counter()
    images, affine = preprocess(paths)
    model = vggt_model.load(info)
    model.point_head = None  # on reprojette la profondeur (plus précis selon VGGT) : économise VRAM et temps
    model.track_head = None
    x = torch.from_numpy(images).to(info.device).permute(0, 3, 1, 2).float().div(255)
    device.reset_peak_memory(info)
    with torch.no_grad(), device.autocast(info):
        out = model(x)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(out["pose_enc"], x.shape[-2:])
    preds = {
        "extrinsic": extrinsic[0].float().cpu().numpy(),  # (S, 3, 4) monde → caméra
        "intrinsic": intrinsic[0].float().cpu().numpy(),  # (S, 3, 3) en pixels VGGT
        "K_full": full_res_intrinsics(intrinsic[0].float().cpu().numpy(), affine),
        "affine": affine,
        "depth": out["depth"][0, ..., 0].float().cpu().numpy(),  # (S, H, W)
        "depth_conf": out["depth_conf"][0].float().cpu().numpy(),
        "images": images,
    }
    peak = device.peak_memory_mb(info)
    del model, out, x
    write_outputs(root, [q.name for q in paths], preds, p)
    outputs = ["vggt/predictions.npz", "vggt/points.ply", "vggt/preview.glb"]
    project.record(
        STEP,
        key,
        p,
        outputs,
        model_sha256=model_sha,
        code_version=f"snap2cad {versions['snap2cad']} + vggt {versions['vggt']}",
        device=info.name,
        duration_s=round(time.perf_counter() - t0, 2),
        peak_vram_mb=round(peak) if peak else None,
        result={"n_images": len(paths), "image_hw": list(images.shape[1:3])},
    )
    return project.data["steps"][STEP]
