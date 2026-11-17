import json
import os
from types import SimpleNamespace

import numpy as np
import pycolmap
import pytest
import torch
from PIL import Image

from snap2cad import device
from snap2cad.project import Project
from snap2cad.steps import segment

INFO = device.DeviceInfo(torch.device("cpu"), torch.float32, "fake", 6.0, "degraded")


class FakePredictor:
    """Objet = moitié haute de chaque photo, référence jamais trouvée ; masques en 30×40 (≠ photos)."""

    image_size = 1024

    def __init__(self):
        self.clicks = []

    def init_state(self, path, **_):
        self.n = len(os.listdir(path))
        return {}

    def reset_state(self, state):
        self.clicks = []

    def add_new_points_or_box(self, state, i, obj_id, points, labels, normalize_coords):
        assert not normalize_coords
        self.clicks.append((i, obj_id, points.tolist(), labels.tolist()))

    def propagate_in_video(self, state, start_frame_idx, reverse):
        for i in range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, self.n):
            logits = torch.full((2, 1, 30, 40), -1.0)
            logits[0, :, :15] = 1.0
            yield i, [1, 2], logits


def make_project(root):
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
    return Project(root)


def click(image, x, y, positive=True):
    return {"image": image, "x": x, "y": y, "positive": positive}


def test_run_orders_by_exif_and_writes_masks(tmp_path, monkeypatch):
    fake = FakePredictor()
    loads = []
    monkeypatch.setattr(segment.sam_model, "load", lambda name, info: loads.append(name) or fake)
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
    assert rec["result"]["area_ratio"]["object"] == [0.5] * 3
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


def test_guided_3d_finds_lost_view_and_box():
    """Caméra fixe (sténopé f = 100, 200 × 100 px), objet à z = 2 m, table derrière. Photo 0 cliquée,
    masque juste ; photo 1 perdue (masque sur la table) ; photo 2 juste mais non cliquée."""
    rng = np.random.default_rng(0)
    obj = np.c_[rng.uniform(-0.4, 0.4, 60), rng.uniform(-0.2, 0.2, 60), np.full(60, 2.0)]
    table = np.c_[rng.uniform(-1.8, 1.8, 200), rng.uniform(-0.9, 0.9, 200), np.full(200, 4.0)]
    table = table[(np.abs(table[:, 0]) > 1.2) | (np.abs(table[:, 1]) > 0.6)]  # hors de l'objet à l'image
    X = np.r_[obj, table]
    uv = X[:, :2] / X[:, 2:] * 100 + [100, 50]
    good = np.zeros((100, 200), bool)
    good[38:63, 78:123] = True  # projection de l'objet : x ∈ [80, 120], y ∈ [40, 60]
    wrong = np.zeros((100, 200), bool)
    wrong[:, :40] = True

    tracks = object.__new__(segment.Tracks)
    tracks.xyz = dict(enumerate(X))
    tracks.obs = {i: (np.arange(len(X)), uv) for i in range(3)}
    tracks.images = {i: SimpleNamespace(cam_from_world=pycolmap.Rigid3d()) for i in range(3)}
    cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=200, height=100, params=[100.0, 100.0, 50.0])
    tracks.cameras = dict.fromkeys(range(3), cam)

    ids, lost = segment.lost_views(tracks, {0: good, 1: wrong, 2: good}, {0}, segment.PARAMS)
    assert set(ids) == set(range(60))  # points de l'objet seuls
    assert list(lost) == [1] and lost[1] == 0.0
    box = tracks.box(1, ids, (100, 200))
    np.testing.assert_allclose(box, [80, 40, 120, 60], atol=1.5)
