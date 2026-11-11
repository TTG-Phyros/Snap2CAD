import json
import math

import numpy as np
import pytest
from PIL import Image, ImageFilter
from PIL.ExifTags import IFD, Base

from snap2cad.project import Project
from snap2cad.steps import import_

RNG = np.random.default_rng(0)


def photo(path, f35=26, focal=5.7, lens="main 5.7mm", orientation=1, blur=0, size=(80, 60), fmt=None):
    img = Image.fromarray(RNG.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    exif = Image.Exif()
    exif[Base.Make], exif[Base.Model], exif[Base.Orientation] = "Apple", "iPhone 13 Pro", orientation
    ex = exif.get_ifd(IFD.Exif)
    if f35:
        ex[Base.FocalLengthIn35mmFilm] = f35
    ex[Base.FocalLength] = focal
    ex[Base.LensModel] = lens
    ex[Base.DateTimeOriginal] = "2026:11:11 10:00:00"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format=fmt, exif=exif.tobytes())


def series(root, n, **kw):
    for i in range(n):
        photo(root / "input" / f"IMG_{i:02d}.JPG", **kw)


def codes(r):
    return {c["code"] for c in r["checks"]}


def test_good_series_copies_jpeg_and_sets_intrinsics(tmp_path):
    series(tmp_path, 15)
    r = import_.run(Project(tmp_path))
    assert r["ok"] and r["n_images"] == 15 and r["checks"] == []
    e = r["images"][0]
    assert (tmp_path / "images/IMG_00.jpg").read_bytes() == (tmp_path / "input/IMG_00.JPG").read_bytes()
    assert e["K"]["fx"] == pytest.approx(26 * 100 / math.hypot(36, 24))  # diagonale 80×60 = 100 px
    assert (e["K"]["cx"], e["K"]["cy"], e["K"]["source"]) == (40, 30, "exif_35mm")
    assert e["lens_model"] == "main 5.7mm" and e["datetime"] == "2026:11:11 10:00:00"
    assert json.loads((tmp_path / "exif.json").read_text("utf-8")) == r


def test_exif_orientation_applied(tmp_path):
    series(tmp_path, 8, orientation=6)
    r = import_.run(Project(tmp_path))
    with Image.open(tmp_path / "images/IMG_00.jpg") as im:
        assert im.size == (60, 80)
    assert (r["images"][0]["width"], r["images"][0]["height"]) == (60, 80)


def test_heic_imported(tmp_path):
    series(tmp_path, 8)
    photo(tmp_path / "input/IMG_99.HEIC", fmt="HEIF")
    r = import_.run(Project(tmp_path))
    assert r["n_images"] == 9 and (tmp_path / "images/IMG_99.jpg").is_file()
    assert r["images"][-1]["focal_35mm"] == 26


def test_ultra_wide_rejected_macro_and_mixed_lenses_warned(tmp_path):
    series(tmp_path, 10)
    photo(tmp_path / "input/UW.JPG", f35=13, focal=1.57, lens="uw 1.57mm")
    photo(tmp_path / "input/MACRO.JPG", f35=26, focal=1.57, lens="uw 1.57mm")
    r = import_.run(Project(tmp_path))
    assert not (tmp_path / "images/UW.jpg").exists() and r["n_images"] == 11
    by = {c["code"]: c["images"] for c in r["checks"]}
    assert by["ultra_wide"] == ["UW.JPG"]
    assert by["cropped_ultra_wide"] == by["mixed_lenses"] == ["MACRO.JPG"]


def test_blur_and_photo_count(tmp_path):
    series(tmp_path, 9)
    photo(tmp_path / "input/BLUR.JPG", blur=3)
    r = import_.run(Project(tmp_path))
    by = {c["code"]: c["images"] for c in r["checks"]}
    assert by["blurry"] == ["BLUR.JPG"] and "few_photos" in by and r["ok"]

    series(tmp_path / "small", 5)
    r = import_.run(Project(tmp_path / "small"))
    assert not r["ok"] and "too_few_photos" in codes(r)


def test_missing_focal_and_duplicates(tmp_path):
    series(tmp_path, 8, f35=None)
    photo(tmp_path / "input/IMG_00.jpeg")
    r = import_.run(Project(tmp_path))
    assert {"no_focal", "duplicate_name"} <= codes(r) and r["n_images"] == 8
    assert r["images"][0]["K"]["fx"] == 1.2 * 80


def test_cached_until_input_changes(tmp_path):
    series(tmp_path, 8)
    import_.run(Project(tmp_path))
    (tmp_path / "images/IMG_00.jpg").touch()
    mtime = (tmp_path / "images/IMG_00.jpg").stat().st_mtime_ns
    import_.run(Project(tmp_path))
    assert (tmp_path / "images/IMG_00.jpg").stat().st_mtime_ns == mtime
    photo(tmp_path / "input/IMG_50.JPG")
    assert import_.run(Project(tmp_path))["n_images"] == 9
