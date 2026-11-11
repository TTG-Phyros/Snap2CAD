"""Adaptateurs DTU et MobileBrick → format de vérité terrain Snap2CAD (plan §9.2, jalon M1).

Format (dossier projet) :
- `input/` : images de la série (lues par l'étape `import`) ;
- `gt/cameras.npz` : `names` (noms dans `images/` après import), `extrinsic` (N, 3, 4) monde → caméra
  en convention OpenCV et en mm, `K` (N, 3, 3) en pixels pleine résolution, `geometry` (chemin du nuage
  ou mesh de référence), `geometry_to_mm` (facteur d'unité de ce fichier).
Même convention que `vggt/predictions.npz` (`extrinsic`, `K_full`) pour comparer directement.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Scene:
    name: str
    images: list[Path]
    K: np.ndarray  # (N, 3, 3) pixels
    w2c: np.ndarray  # (N, 4, 4) monde → caméra, OpenCV, translation en mm
    geometry: Path  # nuage / mesh de référence
    geometry_to_mm: float


def _homog(rt: np.ndarray) -> np.ndarray:
    out = np.tile(np.eye(4), (len(rt), 1, 1))
    out[:, :3] = rt[:, :3]
    return out


def decompose_projection(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P (3, 4) = λ·K[R|t] → K (fx > 0, fy > 0, K[2,2] = 1) et [R|t] (3, 4) avec det R = +1."""
    P = P / np.linalg.norm(P[2, :3]) * np.sign(np.linalg.det(P[:, :3]))  # λ = 1 et det R > 0
    q, r = np.linalg.qr(np.flipud(P[:, :3]).T)  # RQ par QR de la matrice retournée
    K, R = np.flipud(np.fliplr(r.T)), np.flipud(q.T)
    s = np.diag(np.sign(np.diag(K)))
    K, R = K @ s, s @ R
    return K / K[2, 2], np.c_[R, np.linalg.solve(K, P[:, 3])]


def dtu(root: Path, scan: int, light: int = 3) -> Scene:
    """DTU MVS 2014 : images rectifiées (sans distorsion), éclairage `light` (3 = le plus diffus), mm."""
    d = root / "MVS Data"
    images = sorted((d / f"Rectified/scan{scan}").glob(f"rect_*_{light}_r5000.png"))
    if not images:
        raise FileNotFoundError(d / f"Rectified/scan{scan}")
    Ks, rts = zip(
        *(
            decompose_projection(np.loadtxt(d / f"Calibration/cal18/pos_{p.name.split('_')[1]}.txt"))
            for p in images
        ),
        strict=True,
    )
    return Scene(
        f"dtu_scan{scan}",
        images,
        np.stack(Ks),
        _homog(np.stack(rts)),
        d / f"Points/stl/stl{scan:03d}_total.ply",
        1.0,
    )


def mobilebrick(root: Path, seq: str) -> Scene:
    """MobileBrick : `pose/` = caméra → monde en mètres, `intrinsic/` en pixels (1920 × 1440)."""
    d = root / seq
    images = sorted((d / "image").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(d / "image")
    c2w = np.stack([np.loadtxt(d / "pose" / f"{p.stem}.txt") for p in images])
    c2w[:, :3, 3] *= 1000.0
    K = np.stack([np.loadtxt(d / "intrinsic" / f"{p.stem}.txt") for p in images])
    return Scene(f"mobilebrick_{seq}", images, K, np.linalg.inv(c2w), d / "mesh/gt_mesh.ply", 1000.0)


def write_project(scene: Scene, dest: Path, n: int | None = None) -> Path:
    """Copie `n` vues réparties sur la série dans `dest/input/` et écrit `dest/gt/cameras.npz`."""
    idx = np.arange(len(scene.images))
    if n and n < len(idx):
        idx = np.round(np.linspace(0, len(idx) - 1, n)).astype(int)
    (dest / "input").mkdir(parents=True, exist_ok=True)
    (dest / "gt").mkdir(exist_ok=True)
    for i in idx:
        shutil.copyfile(scene.images[i], dest / "input" / scene.images[i].name)
    np.savez(
        dest / "gt/cameras.npz",
        names=np.asarray([scene.images[i].stem + ".jpg" for i in idx]),
        extrinsic=scene.w2c[idx, :3],
        K=scene.K[idx],
        geometry=str(scene.geometry),
        geometry_to_mm=scene.geometry_to_mm,
    )
    return dest


def read_gt(project_root: Path) -> dict[str, np.ndarray]:
    with np.load(project_root / "gt/cameras.npz") as f:
        return dict(f)
