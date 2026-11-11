"""PLY binaire avec attributs par sommet (plan §5.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_TYPES = {np.dtype(np.float32): "float", np.dtype(np.uint8): "uchar"}


def write_ply(path: Path, xyz: np.ndarray, **attrs: np.ndarray) -> None:
    """Nuage de points ; `attrs` : tableaux (N,) float32 ou uint8 (ex. red, green, blue, confidence)."""
    cols = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], **attrs}
    cols = {k: np.asarray(v, np.float32 if k in "xyz" else None) for k, v in cols.items()}
    arr = np.empty(len(xyz), [(k, v.dtype.newbyteorder("<")) for k, v in cols.items()])
    for k, v in cols.items():
        arr[k] = v
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
    header += "".join(f"property {_TYPES[v.dtype]} {k}\n" for k, v in cols.items()) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())
