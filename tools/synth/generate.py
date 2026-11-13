"""Générateur Blender (Cycles) du jeu synthétique Snap2CAD (plan §9.3, jalon M1b).

Usage : uv run --project tools/synth python -m tools.synth.generate [série ...] [--samples 32] [--out DOSSIER]
Série : <objet GSO>__<variante>, variante ∈ base, noir, incomplete, flip, noref.
Sans argument : premier lot, 10 objets × 3 variantes (base, noir, puis incomplete / flip / noref en alternance).

Sortie `datasets/synth/<série>/` (monde en mm, z vers le haut, table en z = 0) :
- `images/view_XXX.jpg` : 4032 × 3024, EXIF (focale 35 mm, heure de prise de vue), distorsion k1, bruit, flou ;
- `labels/view_XXX.png` : 0 table / fond, 1 objet, 2 référence (pixels visibles) ;
- `depth/view_XXX.png` : profondeur z caméra en 1/20 mm (uint16), sur l'objet et la référence seulement ;
- `cameras.npz` : `names`, `extrinsic` (N, 3, 4) monde → caméra OpenCV en mm, `K` (N, 3, 3), `k1` (N,)
  (SIMPLE_RADIAL de COLMAP), `geometry` = `mesh.ply`, `geometry_to_mm` = 1 ;
- `mesh.ply` : mesh de l'objet en mm dans le repère monde de la série ;
- `scene.json` : objet (repère objet → monde, hors-tout), référence (type, cotes, pose, pixels visibles par vue),
  table, variante, paramètres de prise de vue par vue, licence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import tempfile
import zlib
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from PIL import Image

from tools.synth import post
from tools.synth.fetch_gso import OBJECTS, fetch

ROOT = Path(__file__).resolve().parents[2]
W, H = 4032, 3024
DIAG_35MM = math.hypot(36.0, 24.0)
DEPTH_SCALE = 20  # profondeur stockée en 1/20 mm
VARIANTS = ("base", "noir", "incomplete", "flip", "noref")
# Cotes officielles (mm). Diamètre du cœur des pièces : approché, purement visuel.
REFERENCES: dict[str, dict[str, float]] = {
    "coin2": {"diameter": 25.75, "thickness": 2.20, "core": 18.75},
    "coin1": {"diameter": 23.25, "thickness": 2.33, "core": 16.25},
    "card": {"width": 85.60, "height": 53.98, "thickness": 0.76, "corner_radius": 3.18},
}
TABLES = ("plain", "wood", "pattern")


# --- Scène -------------------------------------------------------------------------------------------


def principled(name: str, **inputs: Any) -> tuple[bpy.types.Material, Any, Any]:
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    for k, v in inputs.items():
        bsdf.inputs[k].default_value = v
    return m, nt, bsdf


def ramp(nt: Any, fac: Any, c0: tuple, c1: tuple) -> Any:
    r = nt.nodes.new("ShaderNodeValToRGB")
    r.color_ramp.elements[0].color, r.color_ramp.elements[1].color = (*c0, 1), (*c1, 1)
    nt.links.new(fac, r.inputs["Fac"])
    return r.outputs["Color"]


def coords(nt: Any) -> Any:
    return nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]


def add_table(kind: str, rng: np.random.Generator) -> None:
    bpy.ops.mesh.primitive_plane_add(size=8.0)
    m, nt, bsdf = principled("table", Roughness=0.5)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    nt.links.new(coords(nt), noise.inputs["Vector"])
    if kind == "plain":  # quasi uni : grain très faible
        noise.inputs["Scale"].default_value = 400.0
        base = rng.uniform(0.45, 0.75)
        col = ramp(nt, noise.outputs["Fac"], (base * 0.97,) * 3, (base,) * 3)
    elif kind == "wood":
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.inputs["Scale"].default_value = rng.uniform(4, 8)
        wave.inputs["Distortion"].default_value = 6.0
        wave.inputs["Detail"].default_value = 4.0
        nt.links.new(coords(nt), wave.inputs["Vector"])
        col = ramp(nt, wave.outputs["Fac"], (0.18, 0.09, 0.04), (0.45, 0.27, 0.14))
    else:  # motif : damier et taches
        checker = nt.nodes.new("ShaderNodeTexChecker")
        checker.inputs["Scale"].default_value = rng.uniform(15, 40)
        checker.inputs["Color1"].default_value = (0.7, 0.68, 0.6, 1)
        checker.inputs["Color2"].default_value = (0.25, 0.3, 0.38, 1)
        nt.links.new(coords(nt), checker.inputs["Vector"])
        noise.inputs["Scale"].default_value = 60.0
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
        mix.inputs["Factor"].default_value = 0.3
        nt.links.new(checker.outputs["Color"], mix.inputs[6])
        nt.links.new(noise.outputs["Color"], mix.inputs[7])
        col = mix.outputs[2]
    nt.links.new(col, bsdf.inputs["Base Color"])
    bpy.context.object.data.materials.append(m)


def add_object(name: str, noir: bool) -> bpy.types.Object:
    d = fetch(name)
    bpy.ops.wm.obj_import(filepath=str(d / "meshes/model.obj"), forward_axis="Y", up_axis="Z")
    obj = bpy.context.selected_objects[0]
    if noir:  # noir mat du plan §9.3
        m, _, _ = principled("noir", **{"Base Color": (0.04, 0.04, 0.04, 1), "Roughness": 0.7})
    else:  # le .mtl de GSO pointe vers une texture introuvable : matériau refait à la main
        m, nt, bsdf = principled("gso", Roughness=0.5)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(d / "materials/textures/texture.png"))
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.clear()
    obj.data.materials.append(m)
    obj.pass_index = 1
    return obj


def add_reference(kind: str) -> bpy.types.Object:
    r = {k: v / 1000 for k, v in REFERENCES[kind].items()}
    if kind == "card":
        w, h, rad = (
            r["width"] / 2 - r["corner_radius"],
            r["height"] / 2 - r["corner_radius"],
            r["corner_radius"],
        )
        outline = [
            (cx + rad * math.cos(a), cy + rad * math.sin(a))
            for q, (cx, cy) in enumerate([(w, h), (-w, h), (-w, -h), (w, -h)])
            for a in np.linspace(q * math.pi / 2, (q + 1) * math.pi / 2, 12)
        ]
        n = len(outline)
        verts = [(x, y, z) for z in (0, r["thickness"]) for x, y in outline]  # bas : 0..n-1, haut : n..2n-1
        sides = [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
        me = bpy.data.meshes.new("card")
        me.from_pydata(verts, [], [tuple(range(n, 2 * n)), tuple(range(n - 1, -1, -1)), *sides])
        ref = bpy.data.objects.new("reference", me)
        bpy.context.collection.objects.link(ref)
        # Carte imprimée : motif contrasté (le recto d'une carte d'identité ou bancaire est très texturé).
        m, nt, bsdf = principled("card", Roughness=0.35)
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.inputs["Scale"].default_value = 60.0
        nt.links.new(coords(nt), vor.inputs["Vector"])
        nt.links.new(
            ramp(nt, vor.outputs["Distance"], (0.9, 0.9, 0.95), (0.1, 0.3, 0.6)), bsdf.inputs["Base Color"]
        )
    else:
        # shortcut: tranche lisse (pas de stries ni de gravure en relief) ; à modéliser si M4 en a besoin.
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=256, radius=r["diameter"] / 2, depth=r["thickness"], location=(0, 0, r["thickness"] / 2)
        )
        ref = bpy.context.object
        bpy.ops.object.transform_apply(location=True)
        m, nt, bsdf = principled("coin", Metallic=1.0, Roughness=0.35)
        gold, silver = (0.85, 0.65, 0.3, 1), (0.8, 0.8, 0.78, 1)
        inner, outer = (silver, gold) if kind == "coin2" else (gold, silver)
        flat = nt.nodes.new("ShaderNodeVectorMath")  # rayon dans le plan de la pièce : |(x, y, 0)|
        flat.operation = "MULTIPLY"
        flat.inputs[1].default_value = (1, 1, 0)
        nt.links.new(coords(nt), flat.inputs[0])
        rad = nt.nodes.new("ShaderNodeVectorMath")
        rad.operation = "LENGTH"
        nt.links.new(flat.outputs[0], rad.inputs[0])
        lt = nt.nodes.new("ShaderNodeMath")
        lt.operation = "LESS_THAN"
        lt.inputs[1].default_value = r["core"] / 2
        nt.links.new(rad.outputs["Value"], lt.inputs[0])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs[6].default_value, mix.inputs[7].default_value = outer, inner
        nt.links.new(lt.outputs[0], mix.inputs["Factor"])
        nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
        bump = nt.nodes.new("ShaderNodeBump")  # relief de la frappe, pour donner des points d'intérêt
        bump.inputs["Strength"].default_value = 0.3
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 800.0
        nt.links.new(coords(nt), noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    ref.data.materials.append(m)
    ref.pass_index = 2
    return ref


def add_lights(rng: np.random.Generator) -> None:
    world = bpy.data.worlds.new("world")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = rng.uniform(0.1, 0.3)
    a = rng.uniform(0, 2 * np.pi)
    loc = (0.6 * math.cos(a), 0.6 * math.sin(a), rng.uniform(1.2, 1.8))
    bpy.ops.object.light_add(type="AREA", location=loc)
    light = bpy.context.object
    light.data.size = 1.2
    light.data.energy = rng.uniform(40, 110)
    light.rotation_euler = (0, 0, 0)  # la lampe éclaire vers −z


def setup_render(samples: int, tmp: Path) -> None:
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        if any(d.type == kind for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == kind
            sc.cycles.device = "GPU"
            sc.cycles.denoiser = "OPTIX" if kind == "OPTIX" else "OPENIMAGEDENOISE"
            sc.cycles.denoising_use_gpu = True  # OIDN sur processeur : ≈ 50 s par vue en 12 Mpx
            break
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.filepath = str(tmp / "rgb.png")
    vl = sc.view_layers[0]
    vl.use_pass_z = vl.use_pass_object_index = True
    sc.use_nodes = True
    nt = sc.node_tree
    rl = nt.nodes["Render Layers"]
    out = nt.nodes.new("CompositorNodeOutputFile")
    out.base_path = str(tmp)
    out.format.file_format, out.format.color_depth, out.format.color_mode = "OPEN_EXR", "32", "RGB"
    out.file_slots.clear()
    out.file_slots.new("depth")
    out.file_slots.new("index")
    nt.links.new(rl.outputs["Depth"], out.inputs["depth"])
    nt.links.new(rl.outputs["IndexOB"], out.inputs["index"])


def read_exr(path: Path) -> np.ndarray:
    img = bpy.data.images.load(str(path), check_existing=False)
    a = np.empty(W * H * img.channels, np.float32)
    img.pixels.foreach_get(a)
    bpy.data.images.remove(img)
    return a.reshape(H, W, -1)[::-1, :, 0]  # Blender : origine en bas à gauche


def world_vertices(obj: bpy.types.Object) -> np.ndarray:
    v = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", v)
    M = np.array(obj.matrix_world)
    return (v.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]) * 1000


# --- Caméras -----------------------------------------------------------------------------------------


def look_at(C: np.ndarray, T: np.ndarray, roll_deg: float) -> np.ndarray:
    """Monde → caméra OpenCV (3, 4) en mm : axe z vers la cible, y vers le bas de l'image."""
    z = (T - C) / np.linalg.norm(T - C)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.radians(roll_deg)
    x, y = math.cos(r) * x + math.sin(r) * y, -math.sin(r) * x + math.cos(r) * y
    R = np.stack([x, y, z])
    return np.c_[R, -R @ C]


def set_camera(cam: bpy.types.Object, w2c: np.ndarray) -> None:
    R, t = w2c[:, :3], w2c[:, 3]
    M = np.eye(4)
    M[:3, :3] = R.T @ np.diag([1.0, -1, -1])  # axes caméra Blender : y vers le haut, regard vers −z
    M[:3, 3] = -R.T @ t / 1000
    cam.matrix_world = M.T.tolist()  # mathutils attend des lignes


def plan_cameras(
    rng: np.random.Generator, variant: str, pts: np.ndarray, f_px: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Vues sur 2 ou 3 hauteurs autour de `pts` (coins des boîtes objet + référence, mm)."""
    c = (pts.min(0) + pts.max(0)) / 2
    R = np.linalg.norm(pts - c, axis=1).max()
    vfov = 2 * math.atan(H / 2 / f_px)
    n = int(rng.integers(26 if variant == "incomplete" else 15, 41))
    rings = [[20, 45], [15, 35, 60]][int(rng.integers(2))]
    a0 = rng.uniform(0, 360)
    views = []
    for k, elev in enumerate(rings):
        m = n // len(rings) + (k < n % len(rings))
        for i in range(m):
            az = a0 + 360 * i / m + rng.uniform(-0.3, 0.3) * 180 / m
            el = elev + rng.uniform(-4, 4)
            d = max(150.0, R / math.sin(rng.uniform(0.95, 1.2) * vfov / 2))
            dirv = np.array(
                [
                    math.cos(math.radians(el)) * math.cos(math.radians(az)),
                    math.cos(math.radians(el)) * math.sin(math.radians(az)),
                    math.sin(math.radians(el)),
                ]
            )
            views.append(
                {
                    "azimuth_deg": az % 360,
                    "elevation_deg": el,
                    "C": c + d * dirv,
                    "T": c + rng.normal(0, 0.08 * R, 3),
                    "roll_deg": rng.normal(0, 1.5),
                }
            )
    info: dict[str, Any] = {"rings_deg": rings}
    if variant == "incomplete":  # secteur de 100° jamais photographié, toutes hauteurs confondues
        s0 = rng.uniform(0, 360)
        views = [v for v in views if (v["azimuth_deg"] - s0) % 360 >= 100]
        info["missing_sector_deg"] = [s0, (s0 + 100) % 360]
    return views, info


def hide_reference(
    views: list[dict[str, Any]], ref_pts: np.ndarray, obj_c: np.ndarray, K: np.ndarray, n: int
):
    """Variante noref : décale la visée de `n` vues jusqu'à sortir la référence du cadre."""
    away = obj_c - ref_pts.mean(0)
    away[2] = 0
    away /= np.linalg.norm(away)
    chosen = sorted(range(0, len(views), max(1, len(views) // n))[:n])
    for i in chosen:
        v = views[i]
        for s in np.arange(0, 400, 5.0):
            T = v["T"] + s * away
            uv, z = post.project(ref_pts, look_at(v["C"], T, v["roll_deg"]), K, 0.0)
            if (z <= 0).all() or any(
                (uv[:, a] < 0).all() or (uv[:, a] > lim).all() for a, lim in ((0, W), (1, H))
            ):
                v["T"] = T
                break
    return chosen


# --- Série -------------------------------------------------------------------------------------------


def generate(series: str, out_root: Path, samples: int) -> Path:
    obj_name, variant = series.split("__")
    assert variant in VARIANTS, variant
    rng = np.random.default_rng(zlib.crc32(series.encode()))
    idx = OBJECTS.index(obj_name) if obj_name in OBJECTS else int(rng.integers(3))
    ref_kind = ("coin2", "coin1", "card")[idx % 3]
    table = TABLES[int(rng.integers(3))]
    out = out_root / series
    shutil.rmtree(out, ignore_errors=True)
    for sub in ("images", "labels", "depth"):
        (out / sub).mkdir(parents=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    tmp = Path(tempfile.mkdtemp(prefix="s2c_synth_"))
    setup_render(samples, tmp)
    add_table(table, rng)
    add_lights(rng)
    obj = add_object(obj_name, variant == "noir")
    # Pose : retournement éventuel (variante flip), lacet aléatoire, posé sur la table et centré.
    # shortcut: pas de simulation physique, l'objet retourné peut reposer sur un point ; suffisant pour la vision.
    obj.rotation_euler = (math.pi if variant == "flip" else 0.0, 0.0, rng.uniform(0, 2 * math.pi))
    bpy.context.view_layer.update()
    v = world_vertices(obj)
    lo, hi = v.min(0), v.max(0)
    obj.location = (-(lo[0] + hi[0]) / 2000, -(lo[1] + hi[1]) / 2000, -lo[2] / 1000)
    bpy.context.view_layer.update()
    verts = world_vertices(obj)
    radius_xy = np.linalg.norm(verts[:, :2], axis=1).max()

    ref = add_reference(ref_kind)
    ref_size = REFERENCES[ref_kind].get("diameter", REFERENCES[ref_kind].get("width", 0)) / 2
    gap = rng.uniform(15, 40) + ref_size
    if variant == "noref":
        gap += 40  # plus loin, pour pouvoir la sortir du cadre
    a = rng.uniform(0, 2 * math.pi)
    ref.location = ((radius_xy + gap) * math.cos(a) / 1000, (radius_xy + gap) * math.sin(a) / 1000, 0)
    ref.rotation_euler = (0, 0, rng.uniform(0, 2 * math.pi))
    bpy.context.view_layer.update()
    ref_pts = world_vertices(ref)
    ref_box = np.array(np.meshgrid(*zip(ref_pts.min(0), ref_pts.max(0)))).reshape(3, -1).T
    obj_box = np.array(np.meshgrid(*zip(verts.min(0), verts.max(0)))).reshape(3, -1).T

    f35 = int(rng.choice([24, 25, 26]))
    f_px = f35 * math.hypot(W, H) / DIAG_35MM
    K = np.array([[f_px, 0, W / 2], [0, f_px, H / 2], [0, 0, 1]])
    views, cam_info = plan_cameras(rng, variant, np.r_[obj_box, ref_box], f_px)
    if variant == "noref":
        cam_info["reference_hidden_views"] = hide_reference(views, ref_box, obj_box.mean(0), K, 6)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_fit, cam_data.sensor_width = "HORIZONTAL", 36.0
    cam_data.lens = f_px * 36.0 / W
    cam_data.clip_start, cam_data.clip_end = 0.01, 20.0
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    t0 = dt.datetime(2026, 1, 1, 10, 0, 0) + dt.timedelta(seconds=int(rng.integers(0, 6 * 3600)))
    names, extr, k1s, per_view = [], [], [], []
    for i, view in enumerate(views):
        name = f"view_{i:03d}"
        w2c = look_at(view["C"], view["T"], view["roll_deg"])
        set_camera(cam, w2c)
        bpy.ops.render.render(write_still=True)
        rgb = np.asarray(Image.open(tmp / "rgb.png").convert("RGB"))
        depth, index = read_exr(tmp / "depth0001.exr"), read_exr(tmp / "index0001.exr")
        labels = np.rint(index).clip(0, 2).astype(np.uint8)

        k1 = float(rng.uniform(0.0, 0.015))
        blur = int(rng.integers(3, 13)) if rng.random() < 0.15 else 0
        sigma = float(rng.uniform(1.0, 3.0))
        quality = int(rng.integers(85, 96))
        src = post.distortion_map(K, k1, H, W)  # une seule carte pour l'image, les labels et la profondeur
        rgb = post.motion_blur(post.remap(rgb, src), blur, rng.uniform(0, 180))
        rgb = np.clip(rgb + rng.normal(0, sigma, rgb.shape).astype(np.float32), 0, 255).astype(np.uint8)
        labels = post.remap(labels, src, nearest=True)
        depth = post.remap(depth, src, nearest=True)
        depth = np.where(labels > 0, np.clip(depth * 1000 * DEPTH_SCALE, 0, 65535), 0).astype(np.uint16)

        when = t0 + dt.timedelta(seconds=2 * i)
        post.save_jpeg(
            out / f"images/{name}.jpg", rgb, quality, f35, round(f35 / 4.6, 2),
            when.strftime("%Y:%m:%d %H:%M:%S"), f"{i % 100:02d}",
        )  # fmt: skip
        Image.fromarray(labels).save(out / f"labels/{name}.png")
        Image.fromarray(depth).save(out / f"depth/{name}.png")
        names.append(f"{name}.jpg")
        extr.append(w2c)
        k1s.append(k1)
        per_view.append(
            {
                "name": name,
                "azimuth_deg": round(view["azimuth_deg"], 2),
                "elevation_deg": round(view["elevation_deg"], 2),
                "k1": round(k1, 5),
                "motion_blur_px": blur,
                "noise_sigma": round(sigma, 2),
                "jpeg_quality": quality,
                "object_px": int((labels == 1).sum()),
                "reference_px": int((labels == 2).sum()),
            }
        )
        print(
            f"{series} {name} objet {per_view[-1]['object_px']} px, référence {per_view[-1]['reference_px']} px"
        )

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.ply_export(
        filepath=str(out / "mesh.ply"), export_selected_objects=True, global_scale=1000.0,
        forward_axis="Y", up_axis="Z", export_uv=False, export_normals=False, export_colors="NONE",
        export_triangulated_mesh=True,
    )  # fmt: skip
    np.savez(
        out / "cameras.npz",
        names=np.asarray(names),
        extrinsic=np.stack(extr),
        K=np.tile(K, (len(names), 1, 1)),
        k1=np.asarray(k1s),
        geometry="mesh.ply",
        geometry_to_mm=1.0,
    )
    M = np.array(obj.matrix_world)
    M[:3, 3] *= 1000
    local = (verts - M[:3, 3]) @ M[:3, :3]  # repère objet GSO, mm
    scene = {
        "series": series,
        "variant": variant,
        "object": {
            "name": obj_name,
            "source": "Google Scanned Objects (Google LLC), via Gazebo Fuel",
            "license": "CC-BY 4.0",
            "object_to_world_mm": M.round(6).tolist(),
            "overall_mm": (local.max(0) - local.min(0)).round(3).tolist(),  # hors-tout dans le repère objet
            "material": "noir_mat" if variant == "noir" else "texture",
        },
        "reference": {
            "type": ref_kind,
            "dims_mm": REFERENCES[ref_kind],
            "center_mm": (np.array(ref.location) * 1000).round(4).tolist(),
            "yaw_deg": round(math.degrees(ref.rotation_euler[2]), 4),
        },
        "table": {"kind": table, "plane": "z = 0"},
        "camera": {"width": W, "height": H, "focal_35mm": f35, "f_px": f_px, **cam_info},
        "render": {"engine": "CYCLES", "samples": samples, "blender": bpy.app.version_string},
        "depth_png_scale": DEPTH_SCALE,
        "views": per_view,
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=1, ensure_ascii=False), encoding="utf-8")
    check(out, verts)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def check(out: Path, verts: np.ndarray) -> None:
    """Auto-contrôle : les sommets projetés tombent sur le masque objet, la profondeur concorde."""
    cams = np.load(out / "cameras.npz")
    sample = verts[np.random.default_rng(0).choice(len(verts), min(3000, len(verts)), replace=False)]
    for name, w2c, K, k1 in zip(cams["names"], cams["extrinsic"], cams["K"], cams["k1"], strict=True):
        stem = Path(str(name)).stem
        labels = np.asarray(Image.open(out / f"labels/{stem}.png"))
        depth = np.asarray(Image.open(out / f"depth/{stem}.png")).astype(np.float64) / DEPTH_SCALE
        uv, z = post.project(sample, w2c, K, float(k1))
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        j, i = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
        on_obj = labels[i, j] == 1
        if ok.sum() < 50:
            continue
        assert on_obj.mean() > 0.9, f"{out.name}/{stem} : {on_obj.mean():.3f} des sommets sur le masque"
        dz = depth[i, j][on_obj] - z[ok][on_obj]
        # Seuils lâches (bords de silhouette, objets fins) : une erreur de convention (pose, profondeur
        # « distance au centre » au lieu de z caméra) met la moitié des sommets en défaut.
        front, surface = (dz > 1.5).mean(), (np.abs(dz) < 0.5).mean()
        assert front < 0.1, f"{out.name}/{stem} : {front:.3f} des sommets devant la surface visible"
        assert surface > 0.1, f"{out.name}/{stem} : {surface:.3f} des sommets sur la surface visible"


def first_batch() -> list[str]:
    third = ("incomplete", "flip", "noref")
    return [f"{o}__{v}" for i, o in enumerate(OBJECTS) for v in ("base", "noir", third[i % 3])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("series", nargs="*")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/synth")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    failed = []
    for s in a.series or first_batch():
        if a.skip_existing and (a.out / s / "scene.json").exists():
            continue
        try:
            print(generate(s, a.out, a.samples))
        except Exception as e:  # noqa: BLE001 — une série ratée n'arrête pas le lot ; relancer avec --skip-existing
            print(f"ÉCHEC {s} : {e!r}")
            failed.append(s)
    if failed:
        raise SystemExit(f"{len(failed)} série(s) en échec : {failed}")


if __name__ == "__main__":
    main()
