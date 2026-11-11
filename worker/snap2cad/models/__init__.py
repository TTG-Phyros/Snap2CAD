"""Magasin de modèles : emplacement par OS, manifeste et vérification (plan §3.1, §6.3)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from snap2cad.project import sha256_file

MANIFEST = Path(__file__).with_name("manifest.json")


def models_dir() -> Path:
    """`SNAP2CAD_MODELS` en dev (ex. `<dépôt>/models`), sinon le dossier utilisateur de l'OS."""
    if env := os.environ.get("SNAP2CAD_MODELS"):
        return Path(env)
    if sys.platform == "win32":
        return Path(os.environ["LOCALAPPDATA"]) / "Snap2CAD" / "models"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Snap2CAD" / "models"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "Snap2CAD" / "models"


def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text("utf-8"))["models"]


def verify(name: str, full_hash: bool = False) -> list[str]:
    """Problèmes du modèle `name` (vide = OK). Le SHA-256 complet est lent : réservé au téléchargement."""
    problems = []
    for f in manifest()[name]["files"]:
        p = models_dir() / f["path"]
        if not p.is_file():
            problems.append(f"manquant : {p}")
        elif p.stat().st_size != f["size"]:
            problems.append(f"taille inattendue : {p}")
        elif full_hash and sha256_file(p) != "sha256:" + f["sha256"]:
            problems.append(f"SHA-256 invalide : {p}")
    return problems
