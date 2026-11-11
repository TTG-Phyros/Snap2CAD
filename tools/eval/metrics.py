"""Métriques de poses (plan §9.4) : erreur de rotation après alignement avec échelle."""

from __future__ import annotations

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similitude (s, R, t) minimisant Σ‖dst − (s·R·src + t)‖² (Umeyama 1991)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    e = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    R = u @ e @ vt
    s = float(np.trace(np.diag(d) @ e) / (a**2).sum(1).mean())
    return s, R, md - s * R @ ms


def camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    R, t = extrinsic[:, :3, :3], extrinsic[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t)


def pose_errors(est: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    """`est`, `gt` : (N, 3, 4) monde → caméra (OpenCV). Alignement par les centres des caméras.

    Retourne l'erreur de rotation (degrés) et de position (unités de `gt`) par vue.
    """
    s, Ra, ta = umeyama(camera_centers(est), camera_centers(gt))
    R_al = est[:, :3, :3] @ Ra.T  # rotation monde gt → caméra estimée
    cos = (np.einsum("nij,nij->n", R_al, gt[:, :3, :3]) - 1) / 2  # trace(R_al·R_gtᵀ)
    rot = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    centers = s * camera_centers(est) @ Ra.T + ta
    return {"rotation_deg": rot, "center": np.linalg.norm(centers - camera_centers(gt), axis=1)}
