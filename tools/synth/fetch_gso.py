"""Télécharge les objets Google Scanned Objects (CC-BY 4.0) du jeu synthétique depuis Gazebo Fuel.

Usage : uv run --project tools/synth python tools/synth/fetch_gso.py [nom ...]
Destination : datasets/gso/<nom>/ (meshes/model.obj, materials/textures/texture.png, model.sdf).
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "datasets/gso"
URL = "https://fuel.gazebosim.org/1.0/GoogleResearch/models/{}.zip"

# Premier lot (plan §9.3) : objets rigides de 5 à 30 cm, dont 2 noirs et 2 blancs peu texturés.
OBJECTS = [
    "Pony_C_Clamp_1440",
    "Cole_Hardware_Hammer_Black",
    "Cole_Hardware_Mug_Classic_Blue",
    "Threshold_Porcelain_Teapot_White",
    "Razer_Taipan_Black_Ambidextrous_Gaming_Mouse",
    "Logitech_Ultimate_Ears_Boom_Wireless_Speaker_Night_Black",
    "Markings_Desk_Caddy",
    "Ecoforms_Plant_Container_Quadra_Turquoise_QP12",
    "OXO_Soft_Works_Can_Opener_SnapLock",
    "Shurtape_Gaffers_Tape_Silver_2_x_60_yd",
]


def fetch(name: str) -> Path:
    out = DEST / name
    if (out / "meshes/model.obj").exists():
        return out
    with urllib.request.urlopen(URL.format(name), timeout=120) as r:
        data = r.read()
    out.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(out)  # extractall neutralise les chemins absolus et ".."
    if not (out / "meshes/model.obj").exists():
        raise RuntimeError(f"{name} : meshes/model.obj absent de l'archive")
    return out


if __name__ == "__main__":
    for n in sys.argv[1:] or OBJECTS:
        print(fetch(n))
