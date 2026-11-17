import numpy as np

from snap2cad.models import birefnet, manifest


def test_largest_component_skips_reference():
    m = np.zeros((10, 20), bool)
    m[1:4, 1:4] = True  # objet, petit à l'image
    m[5:9, 8:19] = True  # carte, plus grande
    card = np.zeros_like(m)
    card[5:9, 8:19] = True
    assert birefnet.largest_component(m).sum() == 44
    assert birefnet.largest_component(m, card).sum() == 9
    assert not birefnet.largest_component(np.zeros_like(m)).any()


def test_manifest_pins_code_and_weights():
    files = {f["path"].split("/")[-1]: f for f in manifest()[birefnet.NAME]["files"]}
    assert set(files) == {"birefnet.py", "BiRefNet_config.py", "model.safetensors"}
    assert all("/resolve/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4/" in f["url"] for f in files.values())
