from pathlib import Path

import numpy as np
from PIL import Image

from tools.eval import datasets, metrics

rng = np.random.default_rng(0)


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def cameras(n=12):
    """Caméras en cercle autour de l'origine, regard vers le centre (monde → caméra)."""
    out = []
    for i in range(n):
        c = np.array([np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n), 0.5]) * 500
        z = -c / np.linalg.norm(c)
        x = np.cross(z, [0, 0, 1])
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])
        out.append(np.c_[R, -R @ c])
    return np.stack(out)


def test_decompose_projection_roundtrip():
    K = np.array([[2890.0, 0.3, 820], [0, 2880, 610], [0, 0, 1]])
    for rt, lam in zip(cameras(4), [1.0, -3.7, 0.02, 250.0], strict=True):
        Kd, rtd = datasets.decompose_projection(lam * K @ rt)
        np.testing.assert_allclose(Kd, K, atol=1e-6)
        np.testing.assert_allclose(rtd, rt, atol=1e-6)


def test_pose_errors_invariant_to_similarity():
    gt = cameras()
    s, R, t = 0.0123, rot([1, 2, 3], 70), np.array([3.0, -1, 2])
    noise = [rot(rng.normal(size=3), 0.5) for _ in gt]
    # monde estimé X_e = (R_sᵀ(X − t))/s : caméra R_e = R_gt·R_s avec bruit, centre C_e = R_sᵀ(C − t)/s
    est = []
    for (Rg, tg), Nz in zip(((m[:, :3], m[:, 3]) for m in gt), noise, strict=True):
        C = -Rg.T @ tg
        Re = Nz @ Rg @ R
        est.append(np.c_[Re, -Re @ (R.T @ (C - t) / s)])
    err = metrics.pose_errors(np.stack(est), gt)
    np.testing.assert_allclose(err["rotation_deg"], 0.5, atol=1e-6)
    np.testing.assert_allclose(err["center"], 0, atol=1e-6)


def test_mobilebrick_adapter_and_project(tmp_path):
    d = tmp_path / "mb/seq"
    for sub in ("image", "pose", "intrinsic", "mesh"):
        (d / sub).mkdir(parents=True)
    c2w = []
    for i, rt in enumerate(cameras(5)):
        m = np.linalg.inv(np.r_[rt, [[0, 0, 0, 1]]])
        m[:3, 3] /= 1000  # fichiers en mètres
        c2w.append(m)
        np.savetxt(d / f"pose/{i:06d}.txt", m)
        np.savetxt(d / f"intrinsic/{i:06d}.txt", [[1450, 0, 960], [0, 1450, 720], [0, 0, 1]])
        Image.new("RGB", (32, 24)).save(d / f"image/{i:06d}.jpg")
    scene = datasets.mobilebrick(tmp_path / "mb", "seq")
    np.testing.assert_allclose(scene.w2c[:, :3], cameras(5), atol=1e-9)  # inversé et converti en mm

    datasets.write_project(scene, tmp_path / "p.s2c", n=3)
    assert sorted(p.name for p in (tmp_path / "p.s2c/input").iterdir()) == [
        "000000.jpg",
        "000002.jpg",
        "000004.jpg",
    ]
    gt = datasets.read_gt(tmp_path / "p.s2c")
    assert list(gt["names"]) == ["000000.jpg", "000002.jpg", "000004.jpg"]
    np.testing.assert_allclose(gt["extrinsic"], cameras(5)[[0, 2, 4]], atol=1e-9)
    assert gt["K"].shape == (3, 3, 3) and float(gt["geometry_to_mm"]) == 1000.0


def test_synth_adapter(tmp_path):
    d = tmp_path / "synth/obj__base"
    (d / "images").mkdir(parents=True)
    names = [f"view_{i:03d}.jpg" for i in range(4)]
    for n in names:
        Image.new("RGB", (32, 24)).save(d / "images" / n)
    K = np.tile([[3000.0, 0, 2016], [0, 3000, 1512], [0, 0, 1]], (4, 1, 1))
    np.savez(d / "cameras.npz", names=np.asarray(names), extrinsic=cameras(4), K=K, k1=np.zeros(4),
             geometry="mesh.ply", geometry_to_mm=1.0)  # fmt: skip
    scene = datasets.synth(tmp_path / "synth", "obj__base")
    assert scene.name == "synth_obj__base" and scene.geometry == d / "mesh.ply"
    datasets.write_project(scene, tmp_path / "p.s2c")
    gt = datasets.read_gt(tmp_path / "p.s2c")
    assert list(gt["names"]) == names
    np.testing.assert_allclose(gt["extrinsic"], cameras(4))


DTU = Path(__file__).resolve().parents[2] / "datasets/dtu/SampleSet"


def test_dtu_adapter_projects_gt_points_into_images():
    import pytest
    import trimesh

    if not (DTU / "MVS Data/Points/stl/stl001_total.ply").is_file():
        pytest.skip("DTU absent")
    scene = datasets.dtu(DTU, 1)
    assert len(scene.images) == 49 and np.allclose(np.linalg.det(scene.w2c[:, :3, :3]), 1)
    pts = np.asarray(trimesh.load(scene.geometry).vertices)[::2000]
    for K, w2c in zip(scene.K, scene.w2c, strict=True):
        x = (K @ (w2c[:3, :3] @ pts.T + w2c[:3, 3:])).T
        assert (x[:, 2] > 0).all()  # objet devant chaque caméra
        uv = x[:, :2] / x[:, 2:]
        inside = ((uv >= 0) & (uv < [1600, 1200])).all(1).mean()
        assert inside > 0.8
