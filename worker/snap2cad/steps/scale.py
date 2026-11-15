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
Pixels en convention COLMAP (centre du pixel 0 en 0,5), comme `sparse/`.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pycolmap

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
