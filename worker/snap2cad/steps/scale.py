"""Étape `scale` (plan §4.2 n° 5, §4.3) : échelle à partir de l'objet de référence.

Par vue (jalon M4, première tâche) : arête de la face haute de la référence au sous-pixel, distorsion retirée,
puis ellipse (pièce) ou quadrilatère (carte : 4 droites, coins arrondis écartés). La face haute se projette
exactement en ellipse (ou en quadrilatère), contrairement à la silhouette.
Bords cherchés le long de rayons normaux à la forme initiale (ajustée au masque SAM) :
- côté éloigné de la caméra, la face haute borde la silhouette : bord le plus extérieur (la frappe de la pièce
  et le motif de la carte donnent des bords intérieurs plus forts) ;
- côté caméra, la tranche est visible et la silhouette passe par l'arête basse, décalée vers l'extérieur par
  la tranche sombre et l'ombre de contact (≈ 1 px sur le synthétique) : on prend le bord précédent, dans une
  fenêtre bornée par l'épaisseur projetée.
Puis (`run`) : plan de table par RANSAC (le « haut » de chaque vue), cercle ou rectangle 3D et repère monde.
Pixels en convention COLMAP (centre du pixel 0 en 0,5), comme `sparse/`.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap
from PIL import Image

from snap2cad.project import Project

PARAMS: dict[str, Any] = {
    "sigma_px": 1.0,  # lissage gaussien avant le gradient
    "band": 0.06,  # demi-longueur des rayons au premier passage, × taille de la référence dans l'image
    "band_px": [6.0, 30.0],  # bornes de cette demi-longueur
    "refine_band_px": [3.0, 1.5],  # passages suivants, autour de la forme ajustée
    "edge_rel": 0.15,  # bord retenu : dérivée ≥ edge_rel × 99e centile des dérivées le long des rayons
    "ray_rel": 0.3,  # et ≥ ray_rel × maximum du rayon (écarte les pics du bruit sur la rampe du bord)
    "corner_margin": 0.12,  # part de chaque côté de la carte écartée près des coins (rayon 3,18 mm)
}
# Dimensions (mm) : pièces de 2 € et 1 €, carte ID-1 (ISO/IEC 7810).
REFERENCES: dict[str, dict[str, float]] = {
    "coin2": {"diameter": 25.75, "thickness": 2.20},
    "coin1": {"diameter": 23.25, "thickness": 2.33},
    "card": {"width": 85.60, "height": 53.98, "thickness": 0.76, "corner_radius": 3.18},
}
STEP_PX = 0.25  # pas d'échantillonnage le long des rayons


def ray_edges(
    rgb: np.ndarray,
    q: np.ndarray,
    n: np.ndarray,
    band: float,
    wall: np.ndarray | None = None,
    sigma: float = 1.0,
    rel: float = 0.15,
    ray_rel: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Arête de la face haute le long de chaque rayon q + s·n, s ∈ [−band, band] (q en pixels COLMAP, n unitaire
    vers l'extérieur). Bords : maxima de la dérivée couleur le long du rayon, √Σ_c (n·∇I_c)², affinés par une
    parabole. Avec `wall` (px, par rayon) : le plus extérieur, ou, si un autre bord le précède à moins de
    `wall` px (tranche visible côté caméra), ce dernier. Sans `wall` : le plus proche de q (forme déjà
    ajustée). Retourne les points (M, 2) et les indices des rayons qui en ont un."""
    s = np.arange(-band, band + STEP_PX / 2, STEP_PX)
    x = q[:, 0, None] + s * n[:, 0, None] - 0.5  # coordonnées tableau (centre du pixel 0 en 0)
    y = q[:, 1, None] + s * n[:, 1, None] - 0.5
    m = int(np.ceil(3 * sigma)) + 3
    x0, y0 = max(int(np.floor(x.min())) - m, 0), max(int(np.floor(y.min())) - m, 0)
    x1 = min(int(np.ceil(x.max())) + m + 1, rgb.shape[1])
    y1 = min(int(np.ceil(y.max())) + m + 1, rgb.shape[0])
    img = cv2.GaussianBlur(rgb[y0:y1, x0:x1].astype(np.float32), (0, 0), sigma)
    mx, my = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    sample = [
        cv2.remap(cv2.Sobel(img, cv2.CV_32F, dx, 1 - dx, ksize=3) / 8, mx, my, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REPLICATE).reshape(*mx.shape, -1)
        for dx in (1, 0)
    ]  # fmt: skip
    d = np.sqrt(((sample[0] * n[:, None, :1] + sample[1] * n[:, None, 1:]) ** 2).sum(-1))
    peak = np.zeros_like(d, bool)
    thr = np.maximum(rel * np.percentile(d, 99), ray_rel * d.max(1, keepdims=True))
    peak[:, 1:-1] = (d[:, 1:-1] >= d[:, :-2]) & (d[:, 1:-1] > d[:, 2:]) & (d[:, 1:-1] > thr)
    i, k = np.nonzero(peak)
    a, b, c = d[i, k - 1], d[i, k], d[i, k + 1]
    den = a - 2 * b + c
    sub = np.full(d.shape, -np.inf)  # position sous-pixel de chaque bord le long du rayon
    sub[i, k] = s[k] + np.clip(0.5 * (a - c) / np.where(den < 0, den, -1.0), -0.5, 0.5) * STEP_PX
    if wall is None:
        pick = np.take_along_axis(sub, np.abs(sub).argmin(1)[:, None], 1)[:, 0]
    else:
        out = sub.max(1, keepdims=True)
        inner = np.where((sub < out - 0.5) & (sub >= out - wall[:, None]), sub, -np.inf).max(1)
        pick = np.where(np.isfinite(inner), inner, out[:, 0])
    rays = np.flatnonzero(np.isfinite(pick))
    return q[rays] + pick[rays, None] * n[rays], rays


def undistort(camera: pycolmap.Camera, pts: np.ndarray) -> np.ndarray:
    """Pixels de l'image → pixels d'une caméra sténopé de même matrice K (distorsion retirée)."""
    if not len(pts):
        return pts
    xy = np.asarray(camera.cam_from_img(np.asarray(pts, np.float64)), np.float64).reshape(-1, 2)
    K = camera.calibration_matrix()
    return xy @ K[:2, :2].T + K[:2, 2]


def distort(camera: pycolmap.Camera, pts: np.ndarray) -> np.ndarray:
    """Inverse de `undistort`."""
    K = camera.calibration_matrix()
    return np.asarray(camera.img_from_cam((np.asarray(pts, np.float64) - K[:2, 2]) / np.diag(K)[:2]))


def mask_contour(mask: np.ndarray) -> np.ndarray:
    """Contour extérieur de la plus grande composante du masque, en pixels (convention COLMAP)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        raise ValueError("masque vide")
    return max(cs, key=cv2.contourArea)[:, 0].astype(np.float64) + 0.5


def ellipse_distance(e: tuple[float, ...] | list[float], pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance signée (approximation de Sampson, > 0 dehors) des points à l'ellipse (cx, cy, a, b, θ en
    degrés ; demi-axes), et normale extérieure unitaire de l'ellipse au point."""
    cx, cy, a, b, t = e
    c, s = np.cos(np.radians(t)), np.sin(np.radians(t))
    d = pts - [cx, cy]
    u, v = d @ [c, s], d @ [-s, c]
    gu, gv = 2 * u / a**2, 2 * v / b**2
    g = np.hypot(gu, gv) + 1e-12
    n = np.stack([gu * c - gv * s, gu * s + gv * c], 1) / g[:, None]
    return (u**2 / a**2 + v**2 / b**2 - 1) / g, n


def ellipse_points(e: tuple[float, ...] | list[float], step_px: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Points de l'ellipse espacés d'environ `step_px`, et normales extérieures."""
    cx, cy, a, b, t = e
    phi = np.linspace(0, 2 * np.pi, max(int(np.pi * (a + b) / step_px), 64), endpoint=False)
    c, s = np.cos(np.radians(t)), np.sin(np.radians(t))
    u, v = a * np.cos(phi), b * np.sin(phi)
    p = np.stack([cx + u * c - v * s, cy + u * s + v * c], 1)
    return p, ellipse_distance(e, p)[1]


def _fit_ellipse(pts: np.ndarray) -> tuple[float, ...]:
    o = pts.mean(0)  # coordonnées centrées : OpenCV travaille en float32
    (cx, cy), (w, h), t = cv2.fitEllipseDirect((pts - o).astype(np.float32))
    return (cx + o[0], cy + o[1], w / 2, h / 2, t)


def robust_ellipse(pts: np.ndarray, iters: int = 5) -> tuple[tuple[float, ...], np.ndarray]:
    """Ellipse aux moindres carrés, points à plus de 3 MAD (au moins 0,5 px) écartés à chaque tour."""
    inl = np.ones(len(pts), bool)
    for _ in range(iters):
        if inl.sum() < 20:
            raise ValueError(f"trop peu de points de contour ({int(inl.sum())})")
        e = _fit_ellipse(pts[inl])
        r = ellipse_distance(e, pts)[0]
        tol = max(3 * 1.4826 * float(np.median(np.abs(r[inl]))), 0.5)
        inl = np.abs(r) < tol
    return e, inl


def silhouette_ellipse(
    pts: np.ndarray, n: np.ndarray, up: np.ndarray, thick_ratio: float, iters: int = 6
) -> tuple[tuple[float, ...], float]:
    """Arête haute E d'une pièce à partir du bord le plus extérieur (normales `n` de la forme initiale). Côté
    caméra (n·up < 0), ce bord est l'arête basse, approchée par E translatée de t le long de −up. La silhouette
    seule ne sépare pas t de la taille de E ; t vient de l'épaisseur connue :
    t = (épaisseur / rayon) · a · √(1 − (b/a)²) (demi-axes a ≥ b : a donne les px/mm, b/a l'inclinaison).
    Points à plus de 3 MAD écartés. Échoue si la tranche se confond avec la table (arête basse invisible)."""
    shift = (n @ up < 0)[:, None] * up
    t = 0.0
    keep = np.ones(len(pts), bool)
    for _ in range(iters):
        e, inl = robust_ellipse(pts[keep] + t * shift[keep])  # arête basse ramenée sur l'arête haute
        keep[np.flatnonzero(keep)[~inl]] = False
        a, b = max(e[2], e[3]), min(e[2], e[3])
        t = thick_ratio * a * np.sqrt(1 - (b / a) ** 2)
    return e, float(t)


def _line(p: np.ndarray) -> np.ndarray:
    """Droite (a, b, c) normalisée (a² + b² = 1), moindres carrés totaux."""
    m = p.mean(0)
    n = np.linalg.svd(p - m)[2][1]
    return np.r_[n, -n @ m]


def robust_line(pts: np.ndarray, iters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    inl = np.ones(len(pts), bool)
    for _ in range(iters):
        if inl.sum() < 10:
            raise ValueError(f"trop peu de points sur un côté ({int(inl.sum())})")
        line = _line(pts[inl])
        r = pts @ line[:2] + line[2]
        inl = np.abs(r) < max(3 * 1.4826 * float(np.median(np.abs(r[inl]))), 0.5)
    return line, inl


def quad_corners(contour: np.ndarray) -> np.ndarray:
    """4 coins approchés du contour du masque, dans le sens trigonométrique de l'image."""
    hull = cv2.convexHull(contour.astype(np.float32))
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 19):
        q = cv2.approxPolyDP(hull, eps * peri, True)
        if len(q) == 4:
            c = q[:, 0].astype(np.float64)
            a = c - c.mean(0)
            return c[np.argsort(np.arctan2(a[:, 1], a[:, 0]))]
    raise ValueError("contour de la carte non quadrilatère")


def side_rays(
    c: np.ndarray, margin: float, step_px: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points espacés de `step_px` sur les 4 côtés (hors coins), normales extérieures, numéro du côté."""
    q, n, k = [], [], []
    center = c.mean(0)
    for i in range(4):
        p0, p1 = c[i], c[(i + 1) % 4]
        L = np.linalg.norm(p1 - p0)
        t = np.linspace(margin, 1 - margin, max(int(L * (1 - 2 * margin) / step_px), 10))
        nn = np.array([p1[1] - p0[1], p0[0] - p1[0]]) / L
        nn = nn if nn @ (p0 - center) > 0 else -nn
        q.append(p0 + t[:, None] * (p1 - p0))
        n.append(np.tile(nn, (len(t), 1)))
        k.append(np.full(len(t), i))
    return np.concatenate(q), np.concatenate(n), np.concatenate(k)


def _intersect(lines: list[np.ndarray]) -> np.ndarray:
    h = [np.cross(lines[i - 1], lines[i]) for i in range(4)]
    return np.array([v[:2] / v[2] for v in h])


def fit_view(
    rgb: np.ndarray,
    mask: np.ndarray,
    camera: pycolmap.Camera,
    kind: str,
    up: tuple[float, float] = (0.0, -1.0),
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Arête de la face haute de la référence dans une vue, distorsion retirée : ellipse (`coin1`, `coin2`) ou
    quadrilatère (`card`). `up` : direction de l'image qui s'éloigne de la caméra sur la table (normale de la
    table projetée ; par défaut le haut de l'image, téléphone tenu droit). Rayons autour de la forme du masque,
    puis autour de la forme ajustée (`refine_band_px`). Retourne aussi les points retenus (`points`, pixels
    sans distorsion)."""
    p = {**PARAMS, **(params or {})}
    dims = REFERENCES[kind]
    contour = undistort(camera, mask_contour(mask))
    size = float(np.ptp(contour, 0).max())
    thick_px = dims["thickness"] * size / max(dims.get("diameter", 0), dims.get("width", 0))
    band = float(np.clip(p["band"] * size, *p["band_px"]))
    u = np.asarray(up, np.float64) / np.linalg.norm(up)

    def edges(
        q: np.ndarray, n: np.ndarray, band: float, wall: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        # Rayons tracés dans l'image sans distorsion, échantillonnés dans l'image d'origine.
        qd = distort(camera, q)
        nd = distort(camera, q + n) - qd
        nd /= np.linalg.norm(nd, axis=1, keepdims=True)
        pts, rays = ray_edges(rgb, qd, nd, band, wall, p["sigma_px"], p["edge_rel"], p["ray_rel"])
        return undistort(camera, pts), rays

    # Puis bord le plus proche de la forme ajustée, sur des rayons de plus en plus courts.
    bands = [band, *p["refine_band_px"]]
    if kind == "card":
        c = quad_corners(contour)
        for k, b in enumerate(bands):
            q, n, side = side_rays(c, p["corner_margin"])
            # Tranche fine et sombre : au premier passage, bord précédant la silhouette côté caméra.
            wall = 1.3 * thick_px * np.maximum(-(n @ u), 0) + 1 if k == 0 else None
            pts, rays = edges(q, n, b, wall)
            fits = [robust_line(pts[side[rays] == i]) for i in range(4)]
            c = _intersect([f[0] for f in fits])
        res = [pts[side[rays] == i][f[1]] @ f[0][:2] + f[0][2] for i, f in enumerate(fits)]
        return {
            "corners": np.round(c, 4).tolist(),
            "n_points": [int(f[1].sum()) for f in fits],
            "rms_px": [round(float(np.sqrt((r**2).mean())), 4) for r in res],
            "points": np.concatenate([pts[side[rays] == i][f[1]] for i, f in enumerate(fits)]),
        }
    # Pièce : tranche épaisse et réfléchissante (bords dans la tranche) ; premier passage sur la silhouette.
    q, n = ellipse_points(_fit_ellipse(contour))
    pts, rays = edges(q, n, band, np.zeros(len(q)))
    e, _ = silhouette_ellipse(pts, n[rays], u, dims["thickness"] / (dims["diameter"] / 2))
    for b in bands[1:]:
        pts, _ = edges(*ellipse_points(e), b)
        e, inl = robust_ellipse(pts)
    r = ellipse_distance(e, pts[inl])[0]
    return {
        "ellipse": [round(float(v), 4) for v in e],
        "n_points": int(inl.sum()),
        "rms_px": round(float(np.sqrt((r**2).mean())), 4),
        "points": pts[inl],
    }


# --- Ajustement 3D sur plusieurs vues ---------------------------------------------------------------------------
# Vues : dicts {"name", "K" (3, 3), "cam_from_world" (3, 4), et "ellipse" (pièce) ou "corners" (carte)}, pixels
# sans distorsion (sortie de `fit_view`). Unités du monde : celles des poses (`sparse/`, unités VGGT).
FIT3D: dict[str, Any] = {
    "huber_px": 1.0,  # perte de Huber sur les résidus
    "view_rms_px": 0.5,  # vue écartée si son rms dépasse max(view_rms_px, view_rms_rel × médiane des vues gardées)
    "view_rms_rel": 3.0,
    "max_rms_px": 1.5,  # échec si le rms médian des vues gardées le dépasse (caméras exactes : 0,16 à 1,1 px)
    "min_views": {"coin": 5, "card": 3},
}


def _lm(f: Any, x: np.ndarray, iters: int = 100) -> np.ndarray:
    """Levenberg-Marquardt, jacobien par différences finies (x normalisé : composantes d'ordre 1)."""
    r = f(x)
    cost, lam = r @ r, 1e-3
    for _ in range(iters):
        J = np.stack([(f(x + h) - r) / 1e-7 for h in np.eye(len(x)) * 1e-7], 1)
        A, g = J.T @ J, J.T @ r
        while True:
            dx = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r2 = f(x + dx)
            if (c2 := r2 @ r2) < cost:
                x, r, cost, lam = x + dx, r2, c2, lam / 3
                break
            lam *= 4
            if lam > 1e10:
                return x
        if np.abs(dx).max() < 1e-10:
            break
    return x


def _huber(r: np.ndarray, d: float) -> np.ndarray:
    """Résidus transformés dont la somme des carrés est la perte de Huber (seuil d)."""
    a = np.abs(r)
    return np.where(a <= d, r, np.sign(r) * np.sqrt(np.maximum(2 * d * a - d * d, 0)))


def _center(view: dict[str, Any]) -> np.ndarray:
    P = np.asarray(view["cam_from_world"], np.float64)
    return -P[:, :3].T @ P[:, 3]


def _homography(view: dict[str, Any], C: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Plan (C ; u, v) → pixels sans distorsion : (x, y, 1) ↦ K·(R·(C + x·u + y·v) + t)."""
    P = np.asarray(view["cam_from_world"], np.float64)
    return np.asarray(view["K"]) @ np.c_[P[:, :3] @ u, P[:, :3] @ v, P[:, :3] @ C + P[:, 3]]


def _rays(view: dict[str, Any], px: np.ndarray) -> np.ndarray:
    """Directions monde (non normées) des rayons passant par les pixels `px` (N, 2)."""
    P = np.asarray(view["cam_from_world"], np.float64)
    return (P[:, :3].T @ np.linalg.solve(view["K"], np.c_[px, np.ones(len(px))].T)).T


def _ray_point(views: list[dict[str, Any]], px: list[np.ndarray]) -> np.ndarray:
    """Point 3D le plus proche (moindres carrés) des rayons passant par `px` dans chaque vue."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for v, p in zip(views, px, strict=True):
        d = _rays(v, np.asarray(p, np.float64)[None])[0]
        M = np.eye(3) - np.outer(d, d) / (d @ d)
        A, b = A + M, b + M @ _center(v)
    return np.linalg.solve(A, b)


def _up(views: list[dict[str, Any]], C: np.ndarray) -> np.ndarray:
    """Normale initiale : direction moyenne vers les caméras (toutes du même côté de la table)."""
    m = np.mean([(o - C) / np.linalg.norm(o - C) for o in map(_center, views)], 0)
    return m / np.linalg.norm(m)


def _basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _robust_views(
    views: list[dict[str, Any]], kind: str, residuals: Any, x: np.ndarray, params: dict[str, Any] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ajustement sur les vues gardées, puis rejet des vues incohérentes (rms trop grand) jusqu'à stabilité.
    `residuals(x, i)` : résidus (px) de la vue i. Retourne x, le rms par vue et le masque des vues gardées."""
    p = {**FIT3D, **(params or {})}
    keep = np.ones(len(views), bool)
    for _ in range(10):
        if keep.sum() < p["min_views"][kind]:
            raise ValueError(f"trop peu de vues cohérentes ({int(keep.sum())} < {p['min_views'][kind]})")
        idx = np.flatnonzero(keep)
        x = _lm(lambda y, idx=idx: _huber(np.concatenate([residuals(y, i) for i in idx]), p["huber_px"]), x)
        rms = np.array([np.sqrt(np.mean(residuals(x, i) ** 2)) for i in range(len(views))])
        new = rms < max(p["view_rms_px"], p["view_rms_rel"] * float(np.median(rms[keep])))
        if (new == keep).all():
            break
        keep = new
    # Le rejet relatif ne voit rien si toutes les vues sont fausses (poses `ba` incohérentes) : garde-fou absolu.
    if (med := float(np.median(rms[keep]))) > p["max_rms_px"]:
        raise ValueError(f"vues incohérentes entre elles (rms médian {med:.2f} px > {p['max_rms_px']} px)")
    return x, rms, keep


def _views_out(views: list[dict[str, Any]], rms: np.ndarray, keep: np.ndarray) -> dict[str, Any]:
    return {
        "views": [v["name"] for v, k in zip(views, keep, strict=True) if k],
        "rejected": [v["name"] for v, k in zip(views, keep, strict=True) if not k],
        "rms_px": {v["name"]: round(float(e), 4) for v, e in zip(views, rms, strict=True)},
    }


def fit_circle(
    views: list[dict[str, Any]], kind: str = "coin2", params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Cercle 3D (centre, normale, rayon) dont les projections reproduisent au mieux les ellipses par vue
    (arête de la face haute). Résidus : distance de Sampson de 64 points de chaque ellipse observée à la conique
    projetée (poids égal par vue). `mm_per_unit` : diamètre connu / diamètre ajusté."""
    obs = [np.c_[ellipse_points(v["ellipse"], 1e9)[0], np.ones(64)] for v in views]  # 64 points par ellipse
    C0 = _ray_point(views, [v["ellipse"][:2] for v in views])
    n0 = _up(views, C0)
    t1, t2 = _basis(n0)
    depth = [(np.asarray(v["cam_from_world"])[2] @ [*C0, 1.0]) for v in views]
    r0 = float(
        np.median([max(v["ellipse"][2:4]) * z / v["K"][0][0] for v, z in zip(views, depth, strict=True)])
    )

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        n = n0 + x[3] * t1 + x[4] * t2
        return C0 + r0 * x[:3], n / np.linalg.norm(n), r0 * float(np.exp(x[5]))

    def residuals(x: np.ndarray, i: int) -> np.ndarray:
        C, n, r = unpack(x)
        u, w = _basis(n)
        Hi = np.linalg.inv(_homography(views[i], C, r * u, r * w))  # plan en rayons : cercle unité
        g = obs[i] @ (Hi.T @ np.diag([1.0, 1.0, -1.0]) @ Hi)
        return (g * obs[i]).sum(1) / (2 * np.linalg.norm(g[:, :2], axis=1))

    x, rms, keep = _robust_views(views, "coin", residuals, np.zeros(6), params)
    C, n, r = unpack(x)
    return {
        "center": C.tolist(),
        "normal": n.tolist(),
        "radius": r,
        "mm_per_unit": REFERENCES[kind]["diameter"] / (2 * r),
        **_views_out(views, rms, keep),
    }


def fit_rectangle(views: list[dict[str, Any]], params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rectangle 3D (centre, axes, côtés libres) dont les coins projetés reproduisent les coins observés
    (intersections des 4 droites par vue). `mm_per_unit` : deux estimations, grand côté et petit côté."""
    obs = [np.asarray(v["corners"], np.float64) for v in views]
    # Intersection des diagonales : projection exacte du centre du rectangle.
    diag = [np.cross(np.cross([*c[0], 1], [*c[2], 1]), np.cross([*c[1], 1], [*c[3], 1])) for c in obs]
    C0 = _ray_point(views, [d[:2] / d[2] for d in diag])
    n0 = _up(views, C0)
    # Coins de la vue où la carte est la plus grande, remontés sur le plan (C0, n0) : axes et côtés initiaux.
    ref = int(np.argmax([np.ptp(c, 0).max() for c in obs]))
    o, d = _center(views[ref]), _rays(views[ref], obs[ref])
    X = o + d * (((C0 - o) @ n0) / (d @ n0))[:, None]
    a1 = X[1] - X[0] + X[2] - X[3]
    a1 -= (a1 @ n0) * n0
    a1 /= np.linalg.norm(a1)
    R0 = np.c_[a1, np.cross(n0, a1), n0]
    w0 = float(np.linalg.norm(X[1] - X[0]) + np.linalg.norm(X[2] - X[3])) / 2
    h0 = float(np.linalg.norm(X[2] - X[1]) + np.linalg.norm(X[3] - X[0])) / 2
    local = np.array([[-1.0, -1], [1, -1], [1, 1], [-1, 1]]) / 2

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        return (
            C0 + w0 * x[:3],
            R0 @ cv2.Rodrigues(x[3:6])[0],
            w0 * float(np.exp(x[6])),
            h0 * float(np.exp(x[7])),
        )

    def project(x: np.ndarray, i: int) -> np.ndarray:
        C, R, w, h = unpack(x)
        q = np.c_[local * [w, h], np.ones(4)] @ _homography(views[i], C, R[:, 0], R[:, 1]).T
        return q[:, :2] / q[:, 2:]

    def match(x: np.ndarray) -> None:  # ordre des coins observés (sens et départ) sur la projection courante
        for i, c in enumerate(obs):
            q = project(x, i)
            cands = [np.roll(g, k, 0) for g in (c, c[::-1]) for k in range(4)]
            obs[i] = cands[int(np.argmin([np.linalg.norm(g - q, axis=1).sum() for g in cands]))]

    def residuals(y: np.ndarray, i: int) -> np.ndarray:
        return (project(y, i) - obs[i]).ravel()

    x = np.zeros(8)
    for _ in range(2):  # l'appariement initial peut se tromper sur les vues éloignées de la vue de départ
        match(x)
        x, rms, keep = _robust_views(views, "card", residuals, x, params)
    C, R, w, h = unpack(x)
    (long, ax), (short, _) = sorted([(w, R[:, 0]), (h, R[:, 1])], key=lambda t: -t[0])
    dims = REFERENCES["card"]
    return {
        "center": C.tolist(),
        "normal": R[:, 2].tolist(),
        "axis_long": ax.tolist(),
        "size": [long, short],
        "mm_per_unit": [dims["width"] / long, dims["height"] / short],
        **_views_out(views, rms, keep),
    }


# --- Plan de table et repère monde ------------------------------------------------------------------------------
TABLE: dict[str, Any] = {
    "tol_rel": 0.003,  # seuil RANSAC : part de la distance médiane caméras → points (≈ 1 mm à 30 cm)
    "iters": 2000,
    "min_points": 30,  # en dessous : plan mal défini (échec)
    "object_ratio": 0.5,  # point « objet » : plus de cette part de ses observations dans le masque objet
}


def table_plane(
    xyz: np.ndarray, centers: np.ndarray, params: dict[str, Any] | None = None, seed: int = 0
) -> tuple[np.ndarray, float, np.ndarray]:
    """Plan de la table (n·X + d = 0, n unitaire vers les caméras) par RANSAC sur les points `xyz` (N, 3) hors
    objet, puis moindres carrés sur les points retenus. Plans candidats écartés s'ils passent entre les caméras
    `centers` (M, 3) : toutes les photos sont prises au-dessus de la table. Retourne n, d, masque des points."""
    p = {**TABLE, **(params or {})}
    if len(xyz) < 3:
        raise ValueError(f"trop peu de points pour le plan de table ({len(xyz)})")
    scene = float(np.median(np.linalg.norm(xyz[None] - centers[:, None], axis=2)))
    tol = p["tol_rel"] * scene
    rng = np.random.default_rng(seed)
    best, best_score = None, -1
    for _ in range(p["iters"]):
        a, b, c = xyz[rng.integers(len(xyz), size=3)]  # doublon : produit vectoriel nul, écarté
        n = np.cross(b - a, c - a)
        if (norm := np.linalg.norm(n)) < 1e-12:
            continue
        n /= norm
        side = (centers - a) @ n
        if (side > 0).any() and (side < 0).any():
            continue
        if (score := int((np.abs((xyz - a) @ n) < tol).sum())) > best_score:
            best, best_score = (n, a), score
    if best is None:
        raise ValueError("aucun plan de table laissant toutes les caméras du même côté")
    n, a = best
    for _ in range(3):  # moindres carrés sur les points retenus, puis nouveaux points retenus
        inl = np.abs((xyz - a) @ n) < tol
        a = xyz[inl].mean(0)
        n = np.linalg.svd(xyz[inl] - a)[2][2]
    if inl.sum() < p["min_points"]:
        raise ValueError(f"plan de table mal défini ({int(inl.sum())} points < {p['min_points']})")
    n = n if np.median((centers - a) @ n) > 0 else -n
    return n, float(-n @ a), np.abs((xyz - a) @ n) < tol


def plane_point(view: dict[str, Any], px: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """Point du plan (n, d) vu au pixel `px` (sans distorsion)."""
    o, r = _center(view), _rays(view, np.asarray(px, np.float64)[None])[0]
    return o - r * (n @ o + d) / (n @ r)


def view_up(view: dict[str, Any], X: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Direction de l'image (unitaire) de la normale n en X : « haut » de la table pour `fit_view`."""
    P = np.asarray(view["K"]) @ np.asarray(view["cam_from_world"], np.float64)
    q, dq = P @ [*X, 1.0], P[:, :3] @ n
    u = dq[:2] * q[2] - q[:2] * dq[2]  # dérivée de la projection le long de n
    return u / np.linalg.norm(u)


def world_frame(
    n: np.ndarray, d: float, center: np.ndarray, obj_xyz: np.ndarray, mm_per_unit: float
) -> np.ndarray:
    """Similitude (4, 4) unités `sparse/` → mm : Z = n (vers le haut), origine = `center` (référence) projeté sur
    la table, X = grand axe des points objet projetés sur la table, orienté vers l'objet."""
    O = center - (n @ center + d) * n
    x, _ = _basis(n)
    if len(obj_xyz) >= 2:
        q = obj_xyz - (obj_xyz - O) @ n[:, None] * n  # projection sur la table
        x = np.linalg.svd(q - q.mean(0))[2][0]
        x -= (x @ n) * n
        x /= np.linalg.norm(x)
        x = x if (q.mean(0) - O) @ x >= 0 else -x
    R = np.stack([x, np.cross(n, x), n])  # lignes : axes monde dans le repère `sparse/`
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = mm_per_unit * R, -mm_per_unit * R @ O
    return M


def sparse_points(rec: pycolmap.Reconstruction, masks: Path) -> tuple[np.ndarray, np.ndarray]:
    """Points 3D (N, 3) de `rec` et masque des points objet : plus de `object_ratio` de leurs observations dans
    le masque objet (`masks/<photo>.png`)."""
    votes: dict[int, list[bool]] = defaultdict(list)  # par point 3D : observation dans le masque objet ?
    for im in rec.images.values():
        if not (f := masks / f"{Path(im.name).stem}.png").is_file():
            continue
        mask = np.asarray(Image.open(f)) > 0
        for o in im.get_observation_points2D():
            x, y = np.minimum(np.asarray(o.xy, int), [mask.shape[1] - 1, mask.shape[0] - 1])
            votes[o.point3D_id].append(bool(mask[y, x]))
    ids = sorted(rec.points3D)
    xyz = np.array([rec.points3D[i].xyz for i in ids]).reshape(-1, 3)
    obj = np.array([bool(votes[i]) and np.mean(votes[i]) > TABLE["object_ratio"] for i in ids], bool)
    return xyz, obj


# --- Étape ------------------------------------------------------------------------------------------------------
STEP = "scale"
MIN_MASK_PX = 500  # masque de référence plus petit : référence absente de la photo


def run(project: Project, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Plan de table (points `sparse/` hors objet), arête haute de la référence par vue (« haut » donné par le
    plan), cercle ou rectangle 3D, puis repère monde. `reference_type` : par défaut celui de `segment`.
    Sortie : `scale.json` (mm par unité `sparse/`, similitude `sparse/` → monde en mm)."""
    p = {"reference_type": None, **(params or {})}
    root = project.root
    segment = project.data["steps"].get("segment", {}).get("params", {})
    kind = p["reference_type"] or segment.get("reference_type")
    if kind not in REFERENCES:
        raise ValueError(f"Référence inconnue : {kind} (choix : {', '.join(REFERENCES)})")
    p = {**p, "reference_type": kind, "view": PARAMS, "fit3d": FIT3D, "table": TABLE}
    versions = {"snap2cad": version("snap2cad"), "pycolmap": pycolmap.__version__}
    key = project.inputs_hash(["sparse", "masks", "images"], p, versions)
    if not project.needs_run(STEP, key):
        return project.data["steps"][STEP]

    t0 = time.perf_counter()
    rec = pycolmap.Reconstruction(root / "sparse")
    images = sorted(rec.images.values(), key=lambda im: im.name)
    xyz, obj = sparse_points(rec, root / "masks/object")
    n, d, inl = table_plane(xyz[~obj], np.array([im.projection_center() for im in images]))

    views, errors = [], {}
    for im in images:
        stem = Path(im.name).stem
        f = root / "masks/reference" / f"{stem}.png"
        mask = np.asarray(Image.open(f)) > 0 if f.is_file() else None
        if mask is None or mask.sum() < MIN_MASK_PX:
            continue
        cam = rec.cameras[im.camera_id]
        v = {"name": stem, "K": cam.calibration_matrix(), "cam_from_world": im.cam_from_world.matrix()}
        px = undistort(cam, np.argwhere(mask)[:, ::-1].mean(0, keepdims=True) + 0.5)[0]
        up = view_up(v, plane_point(v, px, n, d), n)
        rgb = np.asarray(Image.open(root / "images" / im.name).convert("RGB"))
        try:
            views.append({**v, **fit_view(rgb, mask, cam, kind, up)})
        except ValueError as e:
            errors[stem] = str(e)
    fit = fit_rectangle(views) if kind == "card" else fit_circle(views, kind)
    # Carte : grand côté seul (M4-fit3d : 0,04 % contre 0,29 % pour le petit côté, biaisé par le contour).
    s = float(np.atleast_1d(fit["mm_per_unit"])[0])
    C = np.asarray(fit["center"])
    dist = (xyz[~obj][inl] @ n + d) * s
    out = {
        "reference": {"type": kind, **REFERENCES[kind]},
        "mm_per_unit": s,
        "world_frame": "table_z_up",  # Z = normale de la table, origine = référence, X = grand axe de l'objet
        "world_from_sparse": world_frame(n, d, C, xyz[obj], s).round(9).tolist(),
        "table": {
            "normal": n.tolist(),
            "d": d,
            "n_points": int((~obj).sum()),
            "n_inliers": int(inl.sum()),
            "rms_mm": round(float(np.sqrt((dist**2).mean())), 4),
        },
        "reference_height_mm": round(float((n @ C + d) * s), 4),  # face haute : épaisseur attendue
        "reference_fit": fit,
        "view_errors": errors,
    }
    (root / "scale.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", "utf-8")
    project.record(
        STEP,
        key,
        p,
        ["scale.json"],
        code_version=f"snap2cad {versions['snap2cad']} + pycolmap {versions['pycolmap']}",
        duration_s=round(time.perf_counter() - t0, 2),
        result={"mm_per_unit": s, "n_views": len(fit["views"]), "table_inliers": int(inl.sum())},
    )
    return project.data["steps"][STEP]
