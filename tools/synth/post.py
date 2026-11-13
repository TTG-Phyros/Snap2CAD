"""Post-traitement « smartphone » des rendus (sans bpy) : distorsion radiale, flou de bougé, bruit, JPEG + EXIF.

Distorsion : modèle SIMPLE_RADIAL de COLMAP, x_d = x_u·(1 + k1·|x_u|²) en coordonnées normalisées.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ExifTags import IFD, Base


def undistort_normalized(xd: np.ndarray, k1: float, iters: int = 20) -> np.ndarray:
    """Inverse de x_d = x_u·(1 + k1·r_u²) par point fixe (converge pour |k1|·r² ≪ 1)."""
    xu = xd.copy()
    for _ in range(iters):
        xu = xd / (1 + k1 * (xu**2).sum(-1, keepdims=True))
    return xu


def project(X: np.ndarray, w2c: np.ndarray, K: np.ndarray, k1: float) -> tuple[np.ndarray, np.ndarray]:
    """Points monde (M, 3) → pixels (M, 2) dans l'image distordue, et profondeur z caméra."""
    Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
    x = Xc[:, :2] / Xc[:, 2:]
    x = x * (1 + k1 * (x**2).sum(1, keepdims=True))
    return x * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]], Xc[:, 2]


def distort(img: np.ndarray, K: np.ndarray, k1: float, nearest: bool = False) -> np.ndarray:
    """Image sans distorsion (rendu sténopé) → image avec distorsion k1, même K, même taille."""
    if k1 == 0:
        return img
    h, w = img.shape[:2]
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    f, c = np.array([K[0, 0], K[1, 1]], np.float32), np.array([K[0, 2], K[1, 2]], np.float32)
    # Centre du pixel (i, j) à (j + 0,5, i + 0,5) : convention COLMAP, comme K.
    xd = (np.stack([u + 0.5, v + 0.5], -1) - c) / f
    src = undistort_normalized(xd, k1) * f + c - 0.5
    if nearest:
        j = np.clip(np.rint(src[..., 0]).astype(int), 0, w - 1)
        i = np.clip(np.rint(src[..., 1]).astype(int), 0, h - 1)
        return img[i, j]
    x0 = np.clip(np.floor(src[..., 0]).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(src[..., 1]).astype(int), 0, h - 2)
    ax = np.clip(src[..., 0] - x0, 0, 1)[..., None]
    ay = np.clip(src[..., 1] - y0, 0, 1)[..., None]
    a = img.astype(np.float32)
    top = a[y0, x0] * (1 - ax) + a[y0, x0 + 1] * ax
    bot = a[y0 + 1, x0] * (1 - ax) + a[y0 + 1, x0 + 1] * ax
    return (top * (1 - ay) + bot * ay).astype(img.dtype)


def motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    """Flou de bougé linéaire de `length` pixels (moyenne de copies décalées)."""
    if length <= 1:
        return img
    t = np.radians(angle_deg)
    acc = np.zeros(img.shape, np.float32)
    for k in np.linspace(-(length - 1) / 2, (length - 1) / 2, length):
        acc += np.roll(img, (round(k * np.sin(t)), round(k * np.cos(t))), axis=(0, 1))
    return (acc / length).round().astype(img.dtype)


def save_jpeg(
    path: Path, rgb: np.ndarray, quality: int, focal_35mm: int, focal_mm: float, when: str, subsec: str
) -> None:
    exif = Image.Exif()
    exif[Base.Make] = "Snap2CAD"
    exif[Base.Model] = "Synthetic"
    ex = exif.get_ifd(IFD.Exif)
    ex[Base.FocalLength] = focal_mm
    ex[Base.FocalLengthIn35mmFilm] = focal_35mm
    ex[Base.DateTimeOriginal] = when
    ex[Base.SubsecTimeOriginal] = subsec
    Image.fromarray(rgb).save(path, quality=quality, exif=exif)


if __name__ == "__main__":  # auto-contrôle : distorsion de l'image cohérente avec `project`
    K = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
    k1 = 0.2
    xd = np.random.default_rng(0).uniform(-0.7, 0.7, (100, 2))
    xu = undistort_normalized(xd, k1, iters=100)
    np.testing.assert_allclose(xu * (1 + k1 * (xu**2).sum(1, keepdims=True)), xd, atol=1e-9)
    img = np.zeros((240, 320), np.uint8)
    X = np.array([[0.4, 0.3, 1.0]])  # près du coin : déplacement de plusieurs pixels
    u0, v0 = project(X, np.eye(4), K, 0.0)[0][0]
    img[int(v0), int(u0)] = 255
    u1, v1 = project(X, np.eye(4), K, k1)[0][0]
    i, j = np.argwhere(distort(img, K, k1, nearest=True) == 255).mean(0)
    assert abs(j + 0.5 - u1) < 1.5 and abs(i + 0.5 - v1) < 1.5 and abs(u1 - u0) > 5, (u0, v0, u1, v1, j, i)
    print("ok")
