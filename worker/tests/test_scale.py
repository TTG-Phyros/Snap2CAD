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
