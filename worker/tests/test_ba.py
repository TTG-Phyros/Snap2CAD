import numpy as np
import pycolmap
from PIL import Image

from snap2cad.steps import ba, poses_init


def test_tracker_frames_and_back_to_full_res(tmp_path):
    a = np.full((300, 400, 3), 255, np.uint8)
    a[148:153, 298:303] = 0  # tache centrée sur le pixel (300, 150) : (300,5 ; 150,5) en convention COLMAP
    Image.fromarray(a).save(tmp_path / "a.png")
    imgs, affine = poses_init.preprocess([tmp_path / "a.png"])
    frames, aff = ba.tracker_frames([tmp_path / "a.png"], affine, imgs.shape[1], 1036)
    assert frames.shape == (1, 1036, 1036, 3)
    g = frames[0].astype(float).sum(2)
    v, u = np.argwhere(g < g.min() + 30).mean(0)  # centre de la tache, centre du pixel 0 en 0
    np.testing.assert_allclose(ba.to_full_res(np.array([[[u, v]]]), aff)[0, 0], [300.5, 150.5], atol=0.3)


def ring(n=8):
    """Caméras en cercle regardant l'origine (monde → caméra)."""
    out = []
    for i in range(n):
        c = np.array([np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n), 0.4]) * 4
        z = -c / np.linalg.norm(c)
        x = np.cross(z, [0, 0, 1])
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])
        out.append(np.c_[R, -R @ c])
    return np.stack(out)


def test_adjust_recovers_focal_and_filters_errors():
    rng = np.random.default_rng(0)
    extr, xyz = ring(), rng.uniform(-1, 1, (300, 3))
    true_cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=640, height=480, params=[800, 320, 240, 0.02])
    xy = np.stack([true_cam.img_from_cam(xyz @ e[:, :3].T + e[:, 3]) for e in extr])
    mask = np.ones(xy.shape[:2], bool)
    xy[0, :5] += 40  # 5 observations aberrantes

    cam = pycolmap.Camera(
        model="SIMPLE_RADIAL", width=640, height=480, params=[880, 320, 240, 0], camera_id=1
    )
    names = [f"{i}.jpg" for i in range(len(extr))]
    rgb = np.zeros((len(xyz), 3), np.uint8)
    rec, ids = ba.build(
        names, [cam], [1] * len(extr), extr, xy, mask, xyz + rng.normal(0, 0.01, xyz.shape), rgb
    )
    assert len(ids) == len(xyz) and rec.num_reg_images() == len(extr)
    assert np.nanmedian(ba.errors(rec, ids, xy, mask)) > 1
    ba.adjust(rec)
    err = ba.errors(rec, ids, xy, mask)
    assert (err[0, :5] > 10).all()  # perte robuste : les aberrations restent visibles, donc filtrables
    assert np.median(err) < 1e-3
    np.testing.assert_allclose(rec.cameras[1].params[[0, 3]], [800, 0.02], rtol=1e-3, atol=1e-4)


def test_adjust_with_fixed_focal_keeps_it():
    extr, xyz = ring(), np.random.default_rng(1).uniform(-1, 1, (200, 3))
    true_cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=640, height=480, params=[800, 320, 240, 0])
    xy = np.stack([true_cam.img_from_cam(xyz @ e[:, :3].T + e[:, 3]) for e in extr])
    cam = pycolmap.Camera(
        model="SIMPLE_RADIAL", width=640, height=480, params=[800, 320, 240, 0], camera_id=1
    )
    names = [f"{i}.jpg" for i in range(len(extr))]
    rec, _ = ba.build(names, [cam], [1] * len(extr), extr, xy, np.ones(xy.shape[:2], bool), xyz * 1.01,
                      np.zeros((len(xyz), 3), np.uint8))  # fmt: skip
    ba.adjust(rec, refine_focal_length=False)
    assert rec.cameras[1].focal_length == 800


def test_rotation_gap_ignores_frame_change_and_finds_drifted_view():
    a = ring()
    q = pycolmap.Rotation3d(np.array([0.3, -0.2, 0.5])).matrix()  # changement de repère (axe-angle)
    b = a.copy()
    b[:, :, :3] = a[:, :, :3] @ q
    b[3, :, :3] = pycolmap.Rotation3d(np.radians([10.0, 0, 0])).matrix() @ b[3, :, :3]
    gap = ba.rotation_gap(a, b)
    assert abs(gap[3] - 10) < 1.5 and np.median(gap) < 1.5
