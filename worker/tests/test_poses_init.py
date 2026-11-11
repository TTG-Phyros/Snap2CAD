import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh
from PIL import Image

from snap2cad import device
from snap2cad.project import Project
from snap2cad.steps import poses_init


def dot_image(path, w, h, u, v):
    a = np.zeros((h, w, 3), np.uint8)
    a[v - 2 : v + 3, u - 2 : u + 3] = 255  # tache 5×5 centrée sur le pixel (u, v)
    Image.fromarray(a).save(path)


def test_preprocess_affine_maps_pixels(tmp_path):
    # paysage 4:3 (pas de recadrage) et portrait 3:4 (recadré à 518, puis l'autre complété en blanc)
    dot_image(tmp_path / "a.png", 800, 600, 600, 150)
    dot_image(tmp_path / "b.png", 600, 800, 150, 300)
    imgs, aff = poses_init.preprocess([tmp_path / "a.png", tmp_path / "b.png"])
    assert imgs.shape == (2, 518, 518, 3) and imgs.dtype == np.uint8
    for img, (sx, sy, tx, ty), (u, v) in zip(imgs, aff, [(600, 150), (150, 300)], strict=True):
        g = img[..., 1].astype(float) * (img.min(-1) < 250)  # ignore le complément blanc
        idx = np.indices(g.shape).reshape(2, -1)
        vv, uu = idx @ g.ravel() / g.sum()  # barycentre pondéré de la tache
        # centre du pixel (u, v) en coordonnées continues = u + 0,5
        assert abs(uu + 0.5 - (sx * (u + 0.5) + tx)) <= 0.5
        assert abs(vv + 0.5 - (sy * (v + 0.5) + ty)) <= 0.5


def test_full_res_intrinsics_inverts_affine():
    K = np.array([[3000.0, 0, 2016], [0, 3000, 1512], [0, 0, 1]])
    sx, sy, tx, ty = 518 / 4032, 392 / 3024, 0.0, 63.0
    Kv = np.array([[sx * 3000, 0, sx * 2016 + tx], [0, sy * 3000, sy * 1512 + ty], [0, 0, 1]])
    out = poses_init.full_res_intrinsics(Kv[None], np.array([[sx, sy, tx, ty]]))
    np.testing.assert_allclose(out[0], K)


class FakeVGGT(torch.nn.Module):
    point_head = track_head = None

    def forward(self, x):
        s, _, h, w = x.shape
        pose = torch.zeros(1, s, 9)
        pose[..., 6] = 1  # quaternion identité (xyzw)
        pose[..., 2] = torch.arange(s)  # translation z différente par vue
        pose[..., 7], pose[..., 8] = 2 * math.atan(h / 2 / 400), 2 * math.atan(w / 2 / 400)
        return {
            "pose_enc": pose,
            "depth": torch.ones(1, s, h, w, 1),
            "depth_conf": torch.rand(1, s, h, w, generator=torch.Generator().manual_seed(0)) + 1,
        }


def test_run_writes_outputs_and_caches(tmp_path, monkeypatch):
    (tmp_path / "images").mkdir()
    for i in range(3):
        Image.new("RGB", (80, 60), (i * 80, 0, 0)).save(tmp_path / "images" / f"{i}.jpg")
    calls = []
    monkeypatch.setattr(poses_init.vggt_model, "load", lambda info: calls.append(1) or FakeVGGT())
    info = device.DeviceInfo(torch.device("cpu"), torch.float32, "fake", 8.0, "full")

    rec = poses_init.run(Project(tmp_path), info, {"max_points": 1000})
    assert rec["status"] == "done" and rec["result"] == {"n_images": 3, "image_hw": [392, 518]}
    d = np.load(tmp_path / "vggt/predictions.npz")
    assert list(d["names"]) == ["0.jpg", "1.jpg", "2.jpg"]
    assert d["extrinsic"].shape == (3, 3, 4) and d["depth"].shape == (3, 392, 518)
    np.testing.assert_allclose(d["K_full"][0, 0, 0], 400 * 80 / 518, rtol=1e-4)
    np.testing.assert_allclose(d["K_full"][0, :2, 2], [40, 30], atol=0.1)

    ply = trimesh.load(tmp_path / "vggt/points.ply")
    assert (
        len(ply.vertices) == 1000 and "confidence" in ply.metadata["_ply_raw"]["vertex"]["data"].dtype.names
    )
    assert len(trimesh.load(tmp_path / "vggt/preview.glb").geometry) == 1

    poses_init.run(Project(tmp_path), info, {"max_points": 1000})
    assert calls == [1]  # deuxième appel servi par le cache


def test_too_many_images_refused(tmp_path):
    (tmp_path / "images").mkdir()
    for i in range(21):
        (tmp_path / "images" / f"{i:02d}.jpg").write_bytes(b"x")
    info = device.DeviceInfo(torch.device("cuda"), torch.bfloat16, "fake", 6.0, "degraded")
    with pytest.raises(RuntimeError, match="au plus 20"):
        poses_init.run(Project(tmp_path), info)


DTU = Path(__file__).resolve().parents[2] / "datasets/dtu/SampleSet/MVS Data/Rectified/scan1"


@pytest.mark.gpu
@pytest.mark.skipif(not DTU.is_dir(), reason="DTU absent")
def test_vggt_on_dtu(tmp_path):
    from snap2cad.steps import import_

    (tmp_path / "input").mkdir()
    for f in sorted(DTU.glob("*_3_r5000.png"))[:10]:
        os.link(f, tmp_path / "input" / f.name)
    import_.run(Project(tmp_path))
    rec = poses_init.run(Project(tmp_path))
    d = np.load(tmp_path / "vggt/predictions.npz")
    assert rec["result"]["n_images"] == 10 and np.isfinite(d["depth"]).all()
    # DTU : 1600×1200, focale réelle ≈ 2890 px
    assert 2300 < np.median(d["K_full"][:, 0, 0]) < 3500
