from snap2cad.project import Project


def make(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"a")
    (tmp_path / "out.npz").write_bytes(b"out")
    p = Project(tmp_path)
    key = p.inputs_hash(["images"], {"res": 518}, {"vggt": "a288dd0"})
    p.record("poses_init", key, {"res": 518}, ["out.npz"], duration_s=1.0)
    return p, key


def test_cache_hit_after_record_and_reload(tmp_path):
    _, key = make(tmp_path)
    assert not Project(tmp_path).needs_run("poses_init", key)
    assert Project(tmp_path).data["steps"]["poses_init"]["duration_s"] == 1.0


def test_rerun_when_input_param_or_version_changes(tmp_path):
    p, key = make(tmp_path)
    assert p.inputs_hash(["images"], {"res": 1024}, {"vggt": "a288dd0"}) != key
    assert p.inputs_hash(["images"], {"res": 518}, {"vggt": "other"}) != key
    (tmp_path / "images" / "b.jpg").write_bytes(b"b")
    assert p.needs_run("poses_init", p.inputs_hash(["images"], {"res": 518}, {"vggt": "a288dd0"}))


def test_rerun_when_output_missing_or_modified(tmp_path):
    p, key = make(tmp_path)
    (tmp_path / "out.npz").write_bytes(b"tampered")
    assert p.needs_run("poses_init", key)
    (tmp_path / "out.npz").unlink()
    assert p.needs_run("poses_init", key)


def test_unknown_step_needs_run(tmp_path):
    assert Project(tmp_path).needs_run("scale", "sha256:x")
