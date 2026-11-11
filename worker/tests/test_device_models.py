import pytest

from snap2cad import device, models


@pytest.mark.parametrize(
    ("gib", "kind", "profile"),
    [
        (7.99, "cuda", "full"),
        (6.0, "cuda", "degraded"),
        (4.0, "cuda", "unsupported"),
        (64.0, "cpu", "unsupported"),
    ],
)
def test_profile_for(gib, kind, profile):
    assert device.profile_for(gib, kind) == profile


def test_manifest_well_formed():
    for m in models.manifest().values():
        for f in m["files"]:
            assert f["url"].startswith("https://") and len(f["sha256"]) == 64 and f["size"] > 0


def test_verify_reports_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("SNAP2CAD_MODELS", str(tmp_path))
    assert models.models_dir() == tmp_path
    assert all(p.startswith("manquant") for p in models.verify("vggt-1b"))


@pytest.mark.gpu
def test_vggt_bf16_runs():
    import torch

    from snap2cad.models import vggt

    info = device.detect()
    model = vggt.load(info)
    with torch.no_grad(), device.autocast(info):
        out = model(torch.rand(2, 3, 294, 518, device=info.device))
    assert out["pose_enc"].shape == (1, 2, 9) and torch.isfinite(out["depth"]).all()
