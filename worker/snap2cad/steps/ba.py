"""Étape `ba` (plan §4.2 n° 4) : suivis de points en haute résolution, puis ajustement de faisceaux pycolmap.

Points ALIKED + SuperPoint suivis par le tracker VGGSfM dans des images T×T cadrées comme celles de VGGT
(`poses_init`), ramenés en pixels pleine résolution. Départ : poses et profondeurs VGGT, focale EXIF (sinon
VGGT). Une caméra par objectif. Filtrage des observations par erreur de reprojection entre deux ajustements.
Sorties : `sparse/` (COLMAP binaire, pixels pleine résolution, unités VGGT), `ba_report.json`.
"""

from __future__ import annotations

import json
import shutil
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
import torch
from PIL import Image
from vggt.dependency.track_predict import predict_tracks
from vggt.utils.geometry import unproject_depth_map_to_point_map

from snap2cad import device
from snap2cad.models import manifest, tracks
from snap2cad.project import Project
from snap2cad.steps.poses_init import SIZE

STEP = "ba"
PARAMS: dict[str, Any] = {
    "track_resolution": 1024,  # côté des images du tracker (VGGT : 1024)
    "max_query_pts": 4096,
    "query_frame_num": 8,
    "max_points_num": 4096,  # points suivis par lot (VGGT : 163840, déborde de 6 Go)
    "fine_tracking": True,
    "vis_thresh": 0.2,
    "camera_model": "SIMPLE_RADIAL",  # ou RADIAL
    # Smartphone : point principal à quelques px du centre ; libre, il dérive (MobileBrick : 0,37° → 0,52°).
    # À libérer pour des images recadrées ou rectifiées (DTU : décalé de 23 px, 0,66° → 0,17°).
    "refine_principal_point": False,
    "init_error_px": 8.0,  # rejet avant ajustement, en pixels de l'image du tracker (VGGT : 8)
    "filter_px": [4.0, 2.0],  # seuils successifs entre ajustements, en pixels de l'image du tracker
    "min_observations": 50,  # en dessous : vue mal contrainte (avertissement)
}


def tracker_frames(paths: list[Path], affine: np.ndarray, hv: int, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Images T×T cadrées comme VGGT (518 de large, complément centré jusqu'au carré), et leur
    transformation (S, 4) = (sx, sy, tx, ty) : u_T = sx·u + tx, v_T = sy·v + ty (coordonnées continues)."""
    k, pad = t / SIZE, (SIZE - hv) // 2
    out, aff = [], []
    for p, (sx, sy, tx, ty) in zip(paths, affine, strict=True):
        a = (k * sx, k * sy, k * tx, k * (ty + pad))
        with Image.open(p) as im:
            im = im.convert("RGB").transform(
                (t, t),
                Image.Transform.AFFINE,
                (1 / a[0], 0, -a[2] / a[0], 0, 1 / a[1], -a[3] / a[1]),
                Image.Resampling.BICUBIC,
                fillcolor=(255, 255, 255),
            )
        out.append(np.asarray(im))
        aff.append(a)
    return np.stack(out), np.asarray(aff)


def to_full_res(xy: np.ndarray, aff: np.ndarray) -> np.ndarray:
    """Suivis (S, P, 2) du tracker (centre du pixel 0 en 0) → pixels pleine résolution, convention COLMAP
    (centre du pixel 0 en 0,5)."""
    s, t = aff[:, None, :2], aff[:, None, 2:]
    return (xy + 0.5 - t) / s


def build(
    names: list[str],
    cams: list[pycolmap.Camera],
    cam_of: list[int],
    extr: np.ndarray,
    xy: np.ndarray,
    mask: np.ndarray,
    xyz: np.ndarray,
    rgb: np.ndarray,
) -> tuple[pycolmap.Reconstruction, dict[int, int]]:
    """Reconstruction COLMAP : image i = id i+1, point 2D j de chaque image = suivi j. Retourne aussi
    {suivi j : id du point 3D} pour les suivis vus au moins deux fois."""
    rec = pycolmap.Reconstruction()
    for c in cams:
        rec.add_camera(c)
    for i, name in enumerate(names):
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(extr[i, :, :3]), extr[i, :, 3])
        rec.add_image(
            pycolmap.Image(
                name=name,
                keypoints=xy[i].astype(np.float64),
                cam_from_world=pose,
                camera_id=cam_of[i],
                id=i + 1,
            )
        )
        rec.register_image(i + 1)
    ids = {}
    for j in np.flatnonzero(mask.sum(0) >= 2):
        track = pycolmap.Track([pycolmap.TrackElement(i + 1, j) for i in np.flatnonzero(mask[:, j])])
        ids[int(j)] = rec.add_point3D(xyz[j].astype(np.float64), track, rgb[j])
    return rec, ids


def errors(rec: pycolmap.Reconstruction, ids: dict[int, int], xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Erreur de reprojection (S, P) en pixels pleine résolution ; NaN hors observations."""
    err = np.full(mask.shape, np.nan)
    js = np.fromiter(ids, int)
    pts = np.array([rec.points3D[ids[j]].xyz for j in js]).reshape(-1, 3)
    for i in range(len(mask)):
        img = rec.images[i + 1]
        obs = mask[i, js]
        pc = pts[obs] @ img.cam_from_world.matrix()[:, :3].T + img.cam_from_world.matrix()[:, 3]
        uv = np.asarray(rec.cameras[img.camera_id].img_from_cam(pc), np.float64).reshape(-1, 2)
        e = np.linalg.norm(uv - xy[i, js[obs]], axis=1)
        err[i, js[obs]] = np.where(pc[:, 2] > 0, e, np.inf)
    return err


def adjust(rec: pycolmap.Reconstruction, refine_principal_point: bool = False) -> None:
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_principal_point = refine_principal_point
    opts.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    pycolmap.bundle_adjustment(rec, opts)


def stats(e: np.ndarray) -> dict[str, float]:
    e = e[np.isfinite(e)]
    return {k: round(float(f(e)), 3) for k, f in (("median", np.median), ("mean", np.mean))} | {
        "p90": round(float(np.percentile(e, 90)), 3)
    }


def run(
    project: Project, info: device.DeviceInfo | None = None, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    p = {**PARAMS, **(params or {})}
    root = project.root
    if p["camera_model"] not in ("SIMPLE_RADIAL", "RADIAL"):
        raise ValueError(f"Modèle de caméra non prévu : {p['camera_model']} (SIMPLE_RADIAL ou RADIAL)")
    shas = [manifest()[m]["files"][0]["sha256"] for m in tracks.MODELS]
    versions = {"snap2cad": version("snap2cad"), "vggt": version("vggt"), "pycolmap": pycolmap.__version__}
    key = project.inputs_hash(
        ["images", "exif.json", "vggt/predictions.npz"], p, {**versions, "models": shas}
    )
    if not project.needs_run(STEP, key):
        return project.data["steps"][STEP]

    info = info or device.detect()
    if info.profile == "unsupported":
        raise RuntimeError(f"GPU non pris en charge pour le suivi de points : {info.name}")
    t0 = time.perf_counter()
    with np.load(root / "vggt/predictions.npz") as d:
        pred = {
            k: d[k] for k in ("names", "extrinsic", "intrinsic", "K_full", "affine", "depth", "depth_conf")
        }
    names = [str(n) for n in pred["names"]]
    exif = {Path(e["file"]).name: e for e in json.loads((root / "exif.json").read_text("utf-8"))["images"]}
    s, hv = pred["depth"].shape[:2]
    t = p["track_resolution"]

    # Suivis : profondeur et confiance VGGT complétées jusqu'au carré, comme les images du tracker.
    pad = ((0, 0), ((SIZE - hv) // 2, SIZE - hv - (SIZE - hv) // 2), (0, 0))
    points = unproject_depth_map_to_point_map(pred["depth"][..., None], pred["extrinsic"], pred["intrinsic"])
    points = np.pad(points, (*pad, (0, 0)))
    conf = np.pad(pred["depth_conf"], pad)
    frames, aff = tracker_frames([root / "images" / n for n in names], pred["affine"], hv, t)
    images = torch.from_numpy(frames).to(info.device).permute(0, 3, 1, 2).float().div(255)
    device.reset_peak_memory(info)
    with tracks.offline(), torch.no_grad(), device.autocast(info):
        xy_t, vis, _, xyz, rgb = predict_tracks(
            images,
            conf=conf,
            points_3d=points,
            max_query_pts=p["max_query_pts"],
            query_frame_num=p["query_frame_num"],
            max_points_num=p["max_points_num"],
            keypoint_extractor="aliked+sp",
            fine_tracking=p["fine_tracking"],
        )
    peak = device.peak_memory_mb(info)
    t_tracks = time.perf_counter() - t0
    del images
    torch.cuda.empty_cache()
    xy = to_full_res(xy_t.astype(np.float64), aff)

    # Une caméra par objectif et taille d'image ; focale EXIF, sinon médiane VGGT du groupe.
    groups: dict[tuple[Any, ...], list[int]] = {}
    for i, n in enumerate(names):
        e = exif[n]
        groups.setdefault((e["lens_model"], e["focal_35mm"], e["width"], e["height"]), []).append(i)
    cams, cam_of, cam_info = [], [0] * s, []
    for cid, idx in enumerate(groups.values(), 1):
        e = exif[names[idx[0]]]
        exif_f = e["K"]["source"] == "exif_35mm"
        f = e["K"]["fx"] if exif_f else float(np.median(pred["K_full"][idx][:, [0, 1], [0, 1]]))
        extra = [0.0] * (1 if p["camera_model"] == "SIMPLE_RADIAL" else 2)
        cx, cy = e["width"] / 2, e["height"] / 2
        cams.append(pycolmap.Camera(model=p["camera_model"], width=e["width"], height=e["height"],
                                    params=[f, cx, cy, *extra], camera_id=cid))  # fmt: skip
        for i in idx:
            cam_of[i] = cid
        cam_info.append({"id": cid, "images": [names[i] for i in idx], "focal_init": round(f, 2),
                         "focal_source": "exif" if exif_f else "vggt"})  # fmt: skip

    # Rejet initial avec les poses VGGT, puis ajustements successifs avec filtrage.
    scale = np.array([max(exif[n]["width"], exif[n]["height"]) / t for n in names])[:, None]
    mask = vis > p["vis_thresh"]
    rec, ids = build(names, cams, cam_of, pred["extrinsic"].astype(np.float64), xy, mask, xyz, rgb)
    err = errors(rec, ids, xy, mask)
    mask &= np.nan_to_num(err, nan=np.inf) < p["init_error_px"] * scale
    rounds = [{"threshold_px": None, "observations": int(mask.sum()), "before": stats(err)}]
    for thr in [None, *p["filter_px"]]:
        if thr is not None:
            mask &= np.nan_to_num(err, nan=np.inf) < thr * scale
        extr = np.stack([rec.images[i + 1].cam_from_world.matrix() for i in range(s)])
        xyz_now = xyz.astype(np.float64).copy()
        for j, pid in ids.items():
            xyz_now[j] = rec.points3D[pid].xyz
        cams = [rec.cameras[c] for c in sorted(rec.cameras)]
        rec, ids = build(names, cams, cam_of, extr, xy, mask, xyz_now, rgb)
        adjust(rec, p["refine_principal_point"])
        err = errors(rec, ids, xy, mask)
        rounds.append({"threshold_px": thr, "observations": int(np.isfinite(err).sum()), "after": stats(err)})

    shutil.rmtree(root / "sparse", ignore_errors=True)
    (root / "sparse").mkdir()
    rec.write(root / "sparse")
    obs = np.isfinite(err).sum(1)
    per_image = [
        {
            "name": n,
            "observations": int(o),
            "reproj_median_px": round(float(np.nanmedian(e)), 3) if o else None,
        }
        for n, o, e in zip(names, obs, err, strict=True)
    ]
    for c in cam_info:
        c["params"] = [round(float(v), 6) for v in rec.cameras[c["id"]].params]
    checks = []
    if weak := [r["name"] for r in per_image if r["observations"] < p["min_observations"]]:
        checks.append({"level": "warning", "code": "weak_views", "images": weak,
                       "message": "Peu de points suivis sur certaines photos : poses peu fiables."})  # fmt: skip
    report = {
        "camera_model": p["camera_model"],
        "cameras": cam_info,
        "n_images": s,
        "n_tracks": int(xy.shape[1]),
        "n_points": len(ids),
        "n_observations": int(obs.sum()),
        "mean_track_length": round(float(obs.sum() / max(len(ids), 1)), 2),
        "reproj_px": stats(err),
        "rounds": rounds,
        "images": per_image,
        "checks": checks,
        "time_s": {"tracks": round(t_tracks, 1), "ba": round(time.perf_counter() - t0 - t_tracks, 1)},
    }
    (root / "ba_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    outputs = [*(f"sparse/{f}" for f in ("cameras.bin", "images.bin", "points3D.bin")), "ba_report.json"]
    project.record(
        STEP,
        key,
        p,
        outputs,
        code_version=f"snap2cad {versions['snap2cad']} + pycolmap {versions['pycolmap']}",
        device=info.name,
        duration_s=round(time.perf_counter() - t0, 2),
        peak_vram_mb=round(peak) if peak else None,
        result={"reproj_px": report["reproj_px"], "n_points": len(ids), "checks": checks},
    )
    return project.data["steps"][STEP]
