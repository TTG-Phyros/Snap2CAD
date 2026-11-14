import json
import os
from typing import ClassVar

import numpy as np
import pytest
import torch
from PIL import Image

from snap2cad import device
from snap2cad.project import Project
from snap2cad.steps import segment

INFO = device.DeviceInfo(torch.device("cpu"), torch.float32, "fake", 6.0, "degraded")


class FakePredictor:
    """Objet = moitié haute de chaque photo sauf la 3e (perdu), référence jamais trouvée ; masques 30×40."""

    image_size = 1024

    def __init__(self):
        self.clicks = []

    def init_state(self, path, **_):
        self.n = len(os.listdir(path))
        return {}

    def add_new_points_or_box(self, state, i, obj_id, points, labels, normalize_coords):
        assert not normalize_coords
        self.clicks.append((i, obj_id, points.tolist(), labels.tolist()))

    def propagate_in_video(self, state, start_frame_idx, reverse):
        for i in range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, self.n):
            logits = torch.full((2, 1, 30, 40), -1.0)
            logits[0, :, :15] = 1.0 if i != 2 else -1.0
            yield i, [1, 2], logits


class FakeImagePredictor:
    boxes: ClassVar[list] = []

    def __init__(self, model):
        pass

    def set_image(self, img):
        self.hw = img.shape[:2]

    def predict(self, box, multimask_output):
        self.boxes.append(box.tolist())
        m = np.zeros((1, *self.hw), bool)
        m[0, : self.hw[0] // 2] = True
        return m, np.ones(1), None


K_VGGT = np.array([[40.0, 0, 20], [0, 40, 15], [0, 0, 1]])  # photos 80×60 → VGGT 40×30
AFFINE = [0.5, 0.5, 0.0, 0.0]


def predictions(extrinsic, depth):
    n = len(extrinsic)
    return {
        "extrinsic": np.asarray(extrinsic, float),
        "intrinsic": np.tile(K_VGGT, (n, 1, 1)),
        "depth": np.asarray(depth, float),
        "depth_conf": np.ones((n, 30, 40)),
        "affine": np.tile(AFFINE, (n, 1)),
    }


def make_project(root, vggt=True):
    (root / "images").mkdir()
    images = []
    for name, dt, status in [
        ("a.jpg", "2026:01:01 10:00:02.5", "ok"),
        ("b.jpg", "2026:01:01 10:00:02.45", "ok"),  # même seconde, fraction plus petite : en premier
        ("c.jpg", None, "warning"),  # sans heure : à la fin
        ("d.jpg", "2026:01:01 09:00:00", "rejected"),
    ]:
        if status != "rejected":
            Image.new("RGB", (80, 60)).save(root / "images" / name)
        images.append({"source": f"input/{name}", "status": status, "datetime": dt, "file": f"images/{name}"})
        images[-1].update(width=80, height=60)
    (root / "exif.json").write_text(json.dumps({"images": images}), "utf-8")
    if vggt:  # caméras confondues, plan à profondeur 1
        (root / "vggt").mkdir()
        pred = predictions(np.tile(np.eye(3, 4), (3, 1, 1)), np.ones((3, 30, 40)))
        np.savez(root / "vggt/predictions.npz", names=np.array(["a.jpg", "b.jpg", "c.jpg"]), **pred)
    return Project(root)


def click(image, x, y, positive=True):
    return {"image": image, "x": x, "y": y, "positive": positive}


def test_run_orders_by_exif_and_writes_masks(tmp_path, monkeypatch):
    fake = FakePredictor()
    loads = []
    monkeypatch.setattr(segment.sam_model, "load", lambda name, info: loads.append(name) or fake)
    monkeypatch.setattr(segment, "SAM2ImagePredictor", FakeImagePredictor)
    FakeImagePredictor.boxes = []
    clicks = {
        "object": [click("b.jpg", 40, 30), click("b.jpg", 20, 50, False)],
        "reference": [click("a.jpg", 8, 6)],
    }

    rec = segment.run(make_project(tmp_path), INFO, {"clicks": clicks})
    assert loads == ["sam2.1-hiera-base-plus"]  # profil 6 Go
    assert rec["result"]["order"] == ["b.jpg", "a.jpg", "c.jpg"]
    assert [(i, obj, lab) for i, obj, _, lab in fake.clicks] == [(0, 1, [1, 0]), (1, 2, [1])]
    np.testing.assert_allclose(fake.clicks[0][2], [[0.5, 0.5], [0.25, 50 / 60]], rtol=1e-6)  # normalisés
    np.testing.assert_allclose(fake.clicks[1][2], [[0.1, 0.1]], rtol=1e-6)
    m = np.asarray(Image.open(tmp_path / "masks/object/c.png"))
    assert m.shape == (60, 80) and m[:30].min() == 255 and m[30:].max() == 0  # remis à la taille de la photo
    assert not np.asarray(Image.open(tmp_path / "masks/reference/a.png")).any()
    assert rec["result"]["area_ratio"]["object"] == [0.5] * 3  # c.jpg refaite par le repli 3D
    assert rec["result"]["fallback_3d"] == {"object": ["c.jpg"], "reference": []}
    (box,) = FakeImagePredictor.boxes  # moitié haute de b.jpg reprojetée, plus la marge
    np.testing.assert_allclose(box, [0, 0, 80, 31.5], atol=2)
    assert [c["code"] for c in rec["result"]["checks"]] == ["reference_few_views"]

    segment.run(Project(tmp_path), INFO, {"clicks": clicks})
    assert loads == ["sam2.1-hiera-base-plus"]  # deuxième appel servi par le cache


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"clicks": {}}, "objet"),
        ({"clicks": {"object": [click("d.jpg", 1, 1)]}}, "absente"),  # photo écartée par l'import
        ({"clicks": {"table": [click("a.jpg", 1, 1)]}}, "Cible inconnue"),
        ({"clicks": {"object": [click("a.jpg", 1, 1)]}, "reference_type": "coin5"}, "Référence inconnue"),
    ],
)
def test_invalid_params_refused(tmp_path, params, message):
    with pytest.raises(ValueError, match=message):
        segment.run(make_project(tmp_path), INFO, params)


def test_poses_init_required(tmp_path):
    with pytest.raises(RuntimeError, match="poses_init"):
        segment.run(make_project(tmp_path, vggt=False), INFO, {"clicks": {"object": [click("a.jpg", 1, 1)]}})


def test_lift_then_project_follows_camera_and_occlusion():
    # caméra 1 décalée de 0,25 vers +x : le plan (profondeur 1) glisse de 0,25·f = 10 px VGGT = 20 px photo
    shifted = np.c_[np.eye(3), [-0.25, 0, 0]]
    pred = predictions([np.eye(3, 4), shifted], np.ones((2, 30, 40)))
    mask = np.zeros((60, 80), bool)
    mask[20:40, 30:50] = True
    X = segment.lift(mask, pred, 0)
    p = {**segment.PARAMS, "box_margin": 0.0}
    np.testing.assert_allclose(segment.guide_box(X, pred, 0, (60, 80), p, True), [30, 20, 50, 40], atol=1.5)
    np.testing.assert_allclose(segment.guide_box(X, pred, 1, (60, 80), p, True), [10, 20, 30, 40], atol=1.5)

    pred["depth"][1] = 0.5  # quelque chose devant la cible dans la vue 1
    assert segment.guide_box(X, pred, 1, (60, 80), p, occlusion=True) is None
    assert segment.guide_box(X, pred, 1, (60, 80), p, occlusion=False) is not None
    assert segment.box_iou(segment.mask_box(mask), (30, 20, 50, 40)) == 1.0
    assert segment.box_iou(segment.mask_box(mask & False), (30, 20, 50, 40)) == 0.0
