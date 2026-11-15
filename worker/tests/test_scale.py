import cv2
import numpy as np
import pycolmap
import pytest

from snap2cad.steps import scale

SS = 4  # suréchantillonnage du rendu (anti-crénelage)


def render(inside_top, h=400, w=600, t=0.0, up=(0.0, -1.0)):
    """Référence vue de biais : face haute (fonction de pixels COLMAP → bool), tranche = face haute translatée
    jusqu'à t px le long de −up, sur une table texturée. Retourne l'image RGB et le masque (silhouette)."""
    y, x = (np.mgrid[0 : h * SS, 0 : w * SS] + 0.5) / SS
    top = inside_top(x, y)
    wall = np.zeros_like(top)
    for s in np.linspace(0, t, max(int(t * SS * 2), 1)):
        wall |= inside_top(x + s * up[0], y + s * up[1])
    wall &= ~top
    rng = np.random.default_rng(0)
    table = 150 + 10 * np.sin(x / 7.0)[..., None] + rng.normal(0, 2, (*x.shape, 1))
    img = np.where(top[..., None], [200.0, 190, 120], np.where(wall[..., None], [90.0, 85, 80], table))
    img = img.reshape(h, SS, w, SS, 3).mean((1, 3))
    return img.round().astype(np.uint8), (top | wall).reshape(h, SS, w, SS).mean((1, 3)) > 0.5


def camera(w=600, h=400, k1=0.0):
    return pycolmap.Camera(model="SIMPLE_RADIAL", width=w, height=h, params=[500.0, w / 2, h / 2, k1])


def test_distort_round_trip():
    cam = camera(k1=0.05)
    p = np.array([[10.0, 20.0], [300.0, 200.0], [590.0, 390.0]])
    np.testing.assert_allclose(scale.distort(cam, scale.undistort(cam, p)), p, atol=1e-6)


def test_coin_top_edge_not_silhouette():
    # Pièce de 2 € vue à 30° : demi-axes 120 × 60 px ; tranche de t = (épaisseur / rayon)·a·√(1 − (b/a)²).
    a, b, cx, cy = 120.0, 60.0, 300.3, 180.6
    t = (
        scale.REFERENCES["coin2"]["thickness"]
        / (scale.REFERENCES["coin2"]["diameter"] / 2)
        * a
        * np.sqrt(0.75)
    )
    rgb, mask = render(lambda x, y: ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 < 1, t=t)
    fit = scale.fit_view(rgb, mask, camera(), "coin2")
    ex, ey, fa, fb, _ = fit["ellipse"]
    assert abs(ex - cx) < 0.3 and abs(ey - cy) < 0.3, fit["ellipse"]  # la silhouette serait décalée de t/2
    np.testing.assert_allclose(sorted([fa, fb]), [b, a], atol=0.3)


def test_card_corners():
    # Carte en perspective (trapèze), tranche de 3 px côté caméra (0,76 mm à ≈ 4,7 px/mm).
    c = np.array([[150.3, 100.2], [450.7, 110.4], [500.1, 300.6], [100.4, 290.3]])

    def inside(x, y):
        ok = np.ones(x.shape, bool)
        for i in range(4):
            p0, p1 = c[i], c[(i + 1) % 4]
            ok &= (p1[0] - p0[0]) * (y - p0[1]) - (p1[1] - p0[1]) * (x - p0[0]) > 0
        return ok

    rgb, mask = render(inside, t=3.0)
    fit = scale.fit_view(rgb, mask, camera(), "card")
    got = np.array(fit["corners"])
    err = min(np.abs(np.roll(got, k, 0) - c).max() for k in range(4))
    assert err < 0.5, (got, c)


def test_empty_mask_raises():
    with pytest.raises(ValueError):
        scale.fit_view(np.zeros((50, 50, 3), np.uint8), np.zeros((50, 50), bool), camera(50, 50), "coin1")


def look_at(eye, target=(0.0, 0.0, 0.0)):
    """cam_from_world (3, 4) OpenCV : caméra en `eye` regardant `target`, z monde vers le haut."""
    z = np.subtract(target, eye) / np.linalg.norm(np.subtract(target, eye))
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    return np.c_[R, -R @ eye]


def views_around(n=12, unit=0.01, seed=1):
    """Caméras à 30–70° d'élévation, 300 mm, monde exprimé en unités de `unit` mm (comme les unités VGGT)."""
    rng = np.random.default_rng(seed)
    K = np.array([[3000.0, 0, 2016], [0, 3000, 1512], [0, 0, 1]])
    out = []
    for i in range(n):
        az, el = 2 * np.pi * i / n + rng.normal(0, 0.1), np.radians(rng.uniform(30, 70))
        eye = 300 * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]) + [20, -10, 0]
        out.append({"name": f"v{i}", "K": K, "cam_from_world": look_at(eye * unit, (0.0, 0.0, 0.0))})
    return out


def project(view, X):
    P = view["cam_from_world"]
    q = (X @ P[:, :3].T + P[:, 3]) @ view["K"].T
    return q[:, :2] / q[:, 2:]


def test_fit_circle_rejects_bad_views():
    unit, r_mm = 0.01, scale.REFERENCES["coin2"]["diameter"] / 2
    a = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    circle = np.c_[r_mm * np.cos(a) + 5, r_mm * np.sin(a) - 3, np.full(360, 2.2)] * unit
    views = views_around(unit=unit)
    rng = np.random.default_rng(2)
    for i, v in enumerate(views):
        (cx, cy), (w, h), t = cv2.fitEllipseDirect(project(v, circle).astype(np.float32))
        k = 1.03 if i in (2, 7) else 1.0  # deux vues ratées : ellipse 3 % trop grande
        v["ellipse"] = [cx + rng.normal(0, 0.1), cy + rng.normal(0, 0.1), k * w / 2, k * h / 2, t]
    fit = scale.fit_circle(views, "coin2")
    assert sorted(fit["rejected"]) == ["v2", "v7"], fit["rms_px"]
    assert abs(fit["mm_per_unit"] * unit - 1) < 5e-4, fit["mm_per_unit"]
    np.testing.assert_allclose(fit["center"], np.array([5, -3, 2.2]) * unit, atol=0.05 * unit)
    assert abs(fit["normal"][2]) > 0.9999


def test_fit_rectangle_corner_order_and_scale():
    unit, d = 0.01, scale.REFERENCES["card"]
    yaw = np.radians(25)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    local = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * [d["width"] / 2, d["height"] / 2]
    corners = np.c_[local @ R.T + [10, 4], np.full(4, 0.76)] * unit
    views = views_around(unit=unit, seed=3)
    rng = np.random.default_rng(4)
    for i, v in enumerate(views):
        c = project(v, corners) + rng.normal(0, 0.1, (4, 2))
        c = np.roll(c[:: 1 if i % 2 else -1], i, 0)  # ordre arbitraire, comme `quad_corners`
        v["corners"] = (c - c.mean(0)) * (1.04 if i == 5 else 1.0) + c.mean(0)
    fit = scale.fit_rectangle(views)
    assert fit["rejected"] == ["v5"], fit["rms_px"]
    np.testing.assert_allclose(np.array(fit["mm_per_unit"]) * unit, 1, atol=5e-4)


def test_fit_circle_fails_on_inconsistent_poses():
    unit, r_mm = 0.01, scale.REFERENCES["coin2"]["diameter"] / 2
    a = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    circle = np.c_[r_mm * np.cos(a), r_mm * np.sin(a), np.full(360, 2.2)] * unit
    views = views_around(unit=unit)
    for i, v in enumerate(views):
        (cx, cy), (w, h), t = cv2.fitEllipseDirect(project(v, circle).astype(np.float32))
        v["ellipse"] = [cx + 8 * (-1) ** i, cy, w / 2, h / 2, t]  # toutes décalées : aucune vue de référence
    with pytest.raises(ValueError, match="incohérentes"):
        scale.fit_circle(views, "coin2")
