import socket
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from snap2cad.models import manifest, tracks

ALIKED_URL = "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth"


def test_offline_serves_manifest_urls_and_refuses_others(tmp_path, monkeypatch):
    monkeypatch.setenv("SNAP2CAD_MODELS", str(tmp_path))
    assert manifest()["aliked-n16"]["files"][0]["url"] == ALIKED_URL  # adresse codée dans LightGlue
    (tmp_path / "aliked-n16").mkdir()
    torch.save({"w": torch.ones(2)}, tmp_path / "aliked-n16/model.pth")
    with tracks.offline():
        assert torch.hub.load_state_dict_from_url(ALIKED_URL, map_location="cpu")["w"].sum() == 2
        with pytest.raises(RuntimeError, match="hors ligne"):
            torch.hub.load_state_dict_from_url("https://example.com/other.pth")
        with pytest.raises(RuntimeError, match="non embarqué"):
            torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    assert torch.hub.load_state_dict_from_url is not tracks._load_state_dict  # restauré à la sortie


SYNTH = Path(__file__).resolve().parents[2] / "datasets/synth/Cole_Hardware_Mug_Classic_Blue__base"


@pytest.mark.gpu
@pytest.mark.skipif(not SYNTH.is_dir(), reason="jeu synthétique absent")
def test_predict_tracks_without_network(monkeypatch):
    from vggt.dependency.track_predict import predict_tracks

    def no_network(*_, **__):
        raise AssertionError("accès réseau")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    files = sorted((SYNTH / "images").glob("*.*"))[::8][:6]
    images = torch.stack(
        [torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((518, 518)))) for f in files]
    )
    images = images.permute(0, 3, 1, 2).float().cuda() / 255
    n = len(files)  # conf et points 3D viennent de VGGT dans le pipeline ; valeurs factices ici
    with tracks.offline(), torch.inference_mode():
        out = predict_tracks(
            images,
            conf=np.ones((n, 518, 518), np.float32),
            points_3d=np.zeros((n, 518, 518, 3), np.float32),
            max_query_pts=256,
            query_frame_num=2,
            fine_tracking=False,
        )
    tracks_xy, vis = out[0], out[1]
    assert tracks_xy.shape[0] == len(files) and tracks_xy.shape[-1] == 2
    assert np.isfinite(tracks_xy).all() and (vis > 0.5).mean() > 0.2
