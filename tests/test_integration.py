"""Optional actual-input policy/model/export integration, not a proxy-robot demo."""

import hashlib
import json
import os
from pathlib import Path
import zipfile

import mujoco
import numpy as np
import pytest

from morpheus_policy.artifacts import sha256_file, write_artifacts
from morpheus_policy.cli import demonstration_task
from morpheus_policy.evaluation import compose_model, evaluate_episode
from morpheus_policy.registry import ROBOT_SUPPORT_BODIES, create_policy, robot_model_path
from morpheus_policy.scene import load_scene


def test_real_supplied_scene_policy_and_portable_replay(tmp_path):
    path = os.environ.get("MORPHEUS_SCENE_BUNDLE")
    assets = Path(".morpheus/assets/g1").resolve()
    if not path or not assets.exists():
        pytest.skip("Set MORPHEUS_SCENE_BUNDLE and fetch pinned G1 assets for real integration")
    scene = load_scene(path)
    before = dict(scene.hashes)
    policy = create_policy("unitree_g1_12dof", assets)
    assert policy.validate_assets(assets)["valid"]
    model, spec, model_hash = compose_model(scene, robot_model_path("unitree_g1_12dof", assets))
    assert policy.validate(model)["valid"]
    from morpheus_policy.policies.g1 import DEFAULT_ANGLES
    report, replay = evaluate_episode(scene, model, policy, demonstration_task(scene), model_hash=model_hash,
                                      initial_joint_positions=DEFAULT_ANGLES, robot_id="unitree_g1_12dof",
                                      allowed_support_bodies=ROBOT_SUPPORT_BODIES["unitree_g1_12dof"])
    assert report["execution"] == "completed"
    assert report["readiness"] == "PREVIEW_ONLY"
    assert report["metrics"]["forbidden_contacts"] == 0
    assert report["metrics"]["max_joint_force_limit_overrun"] == 0
    output = write_artifacts(tmp_path / "evidence", report, replay, spec=spec, model=model,
                             third_party_license_path=assets / "LICENSE")
    restored = mujoco.MjSpec.from_zip(str(output / "model.mjz")).compile()
    data = mujoco.MjData(restored)
    data.qpos[:] = replay["samples"][-1]["qpos"]
    mujoco.mj_forward(restored, data)
    for name, pose in replay["samples"][-1]["body_poses"].items():
        body = mujoco.mj_name2id(restored, mujoco.mjtObj.mjOBJ_BODY, name)
        np.testing.assert_allclose(data.xpos[body], pose[:3], atol=1e-12)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model_mjz_sha256"] == sha256_file(output / "model.mjz")
    assert report["hashes"]["model_mjz_sha256"] == replay["hashes"]["model_mjz_sha256"] == manifest["model_mjz_sha256"]
    with zipfile.ZipFile(output / "model.mjz") as archive:
        assert archive.read("THIRD_PARTY_LICENSE.txt") == (assets / "LICENSE").read_bytes()
        xml = next(name for name in archive.namelist() if name.endswith(".xml"))
        assert hashlib.sha256(archive.read(xml)).hexdigest() == replay["hashes"]["mjcf_sha256"]
    assert dict(load_scene(path).hashes) == before
