import json
from pathlib import Path
import zipfile

import pytest

from morpheus_policy.scene import SceneValidationError, load_scene, verify_source_snapshot


def _write_bundle(tmp_path: Path, *, pos="0 0 -0.05", size="2 2 0.05", include="environment.xml") -> Path:
    scene = {
        "version": 1,
        "name": "test scene",
        "units": "m",
        "upAxis": "Z",
        "source": {"synthetic": True, "scaleConfirmed": False, "reviewed": False},
        "objects": [
            {
                "id": "floor",
                "type": "box",
                "position": [0, 0, -0.05],
                "size": [4, 4, 0.1],
                "quaternion": [1, 0, 0, 0],
                "role": "floor",
                "enabled": True,
            }
        ],
    }
    (tmp_path / "scene.json").write_text(json.dumps(scene))
    (tmp_path / "scene.xml").write_text(f'<mujoco><include file="{include}" /></mujoco>')
    (tmp_path / "environment.xml").write_text(
        f'<mujocoinclude><worldbody><geom name="env_floor" type="box" pos="{pos}" '
        f'size="{size}" quat="1 0 0 0" contype="1" conaffinity="1" /></worldbody></mujocoinclude>'
    )
    return tmp_path


def test_load_scene_preserves_v1_caveats_and_full_dimensions(tmp_path: Path) -> None:
    scene = load_scene(_write_bundle(tmp_path))

    assert scene.objects[0].size == (4.0, 4.0, 0.1)
    assert scene.bounds_xy == (-2.0, -2.0, 2.0, 2.0)
    assert scene.mjcf_path.name == "scene.xml"
    assert set(scene.hashes) == {"scene.json", "scene.xml", "environment.xml"}
    assert "metric scale is not confirmed" in scene.caveats


def test_scene_rejects_json_mjcf_half_extent_mismatch(tmp_path: Path) -> None:
    with pytest.raises(SceneValidationError, match="half-size"):
        load_scene(_write_bundle(tmp_path, size="1.9 2 0.05"))


def test_scene_rejects_remote_or_escaping_mjcf_include(tmp_path: Path) -> None:
    with pytest.raises(SceneValidationError, match="remote or absolute"):
        load_scene(_write_bundle(tmp_path, include="https://example.invalid/environment.xml"))


def test_scene_rejects_remote_xml_asset(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    (root / "environment.xml").write_text(
        '<mujocoinclude><asset><mesh name="x" file="https://example.invalid/x.obj" /></asset>'
        '<worldbody><geom name="env_floor" type="box" pos="0 0 -0.05" size="2 2 0.05" '
        'quat="1 0 0 0" contype="1" conaffinity="1" /></worldbody></mujocoinclude>'
    )
    with pytest.raises(SceneValidationError, match="remote or absolute MJCF asset"):
        load_scene(root)


def test_scene_rejects_zip_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../scene.json", "{}")
    with pytest.raises(SceneValidationError, match="unsafe archive member"):
        load_scene(archive)


def test_source_snapshot_rejects_changed_missing_and_symlinked_inputs(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    scene = load_scene(root)
    verify_source_snapshot(scene)

    environment = root / "environment.xml"
    environment.write_text("<mujocoinclude/>")
    with pytest.raises(SceneValidationError, match="source snapshot changed: environment.xml"):
        verify_source_snapshot(scene)

    environment.unlink()
    with pytest.raises(SceneValidationError, match="source snapshot file is missing: environment.xml"):
        verify_source_snapshot(scene)

    environment.symlink_to(root / "scene.xml")
    with pytest.raises(SceneValidationError, match="source snapshot contains a symlink: environment.xml"):
        verify_source_snapshot(scene)
