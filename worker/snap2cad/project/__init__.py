"""Dossier projet : `project.json`, hash et décision de relance (plan §5.3, annexe A4)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _files(root: Path, rel: str) -> list[str]:
    p = root / rel
    if p.is_dir():
        return sorted(q.relative_to(root).as_posix() for q in p.rglob("*") if q.is_file())
    if p.is_file():
        return [rel]
    raise FileNotFoundError(p)


class Project:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.file = self.root / "project.json"
        self.data: dict[str, Any] = (
            json.loads(self.file.read_text("utf-8"))
            if self.file.exists()
            else {"schema_version": SCHEMA_VERSION}
        )
        self.data.setdefault("steps", {})

    def save(self) -> None:
        tmp = self.file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), "utf-8")
        os.replace(tmp, self.file)  # écriture atomique : un crash ne corrompt pas project.json

    def inputs_hash(self, inputs: list[str], params: dict[str, Any], versions: dict[str, str]) -> str:
        """Hash des fichiers d'entrée (dossiers développés), des paramètres et des versions code/modèle."""
        h = hashlib.sha256()
        for rel in sorted({f for i in inputs for f in _files(self.root, i)}):
            h.update(f"{rel}\0{sha256_file(self.root / rel)}\n".encode())
        h.update(json.dumps({"params": params, "versions": versions}, sort_keys=True).encode())
        return "sha256:" + h.hexdigest()

    def needs_run(self, step: str, key: str) -> bool:
        s = self.data["steps"].get(step)
        if not s or s.get("status") != "done" or s.get("inputs_hash") != key:
            return True
        return any(
            not (self.root / rel).is_file() or sha256_file(self.root / rel) != digest
            for rel, digest in s.get("outputs", {}).items()
        )

    def record(self, step: str, key: str, params: dict[str, Any], outputs: list[str], **extra: Any) -> None:
        self.data["steps"][step] = {
            "status": "done",
            "params": params,
            "inputs_hash": key,
            "outputs": {rel: sha256_file(self.root / rel) for rel in outputs},
            **extra,
        }
        self.save()
