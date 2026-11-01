"""Integration evidence must bind the displayed learner and native episode."""

from dataclasses import replace
import hashlib
import json

import pytest
import numpy as np

from morpheus_policy.learned_navigation import load_navigation_artifact, LearnedNavigationController
from morpheus_policy.learning import HomeNavConfig, PPOTrainConfig, train_scene_specific_ppo
from morpheus_policy.profiles import navigation_profile
from morpheus_policy.scene import Box, Scene
from morpheus_policy.viewer_export import _attach_native, export_viewer_run
from morpheus_policy.navigation import NavigationConfig


@pytest.fixture
def fixture_scene(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    text = '{"fixture":"upstream supplied test geometry"}\n'
    (path / "scene.json").write_text(text)
    floor = Box("floor", (0, 0, -.05), (6, 6, .1), (1, 0, 0, 0), "floor")
    return Scene("adapter fixture", (floor,), {"synthetic": True}, path, path / "scene.xml",
                 {"scene.json": hashlib.sha256(text.encode()).hexdigest()}, ("Fixture only",))


def test_fixed_arm_export_keeps_base_stationary_and_source_bytes(fixture_scene, tmp_path):
    target = tmp_path / "viewer" / "soarm.json"
    run = export_viewer_run(fixture_scene, navigation_profile("soarm"), (0, 0), (1, 0), target, demo_cleaning=True)
    assert run["training"] is None and not run["routes"]
    assert run["replay"]["frames"][0]["position"] == run["replay"]["frames"][-1]["position"]
    assert len(run["cleaning"]) == 2
    assert all(intent["status"] == "planned_tool_unverified" for intent in run["cleaning"])
    assert (target.parent / "scene.json").read_bytes() == (fixture_scene.root / "scene.json").read_bytes()
    assert run["readiness"] == "PREVIEW_ONLY"


def test_loaded_navigation_rejects_changed_profile_scene_and_checkpoint(fixture_scene, tmp_path):
    profile = navigation_profile("g1")
    directory = tmp_path / "training"
    train_scene_specific_ppo(fixture_scene, profile, directory,
        env_config=HomeNavConfig(horizon_steps=20, ray_count=4),
        train_config=PPOTrainConfig(n_envs=1, rollout_steps=8, updates=1, epochs=1, hidden_size=8,
                                   minibatch_size=8, train_scenarios=2, heldout_scenarios=2, imitation_epochs=1))
    load_navigation_artifact(directory, fixture_scene, profile)
    with pytest.raises(ValueError, match="another robot"):
        load_navigation_artifact(directory, fixture_scene, replace(profile, radius_m=.36))
    with pytest.raises(ValueError, match="another scene"):
        load_navigation_artifact(directory, replace(fixture_scene, name="changed"), profile)
    checkpoint = directory / "reduced_order_ppo.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_navigation_artifact(directory, fixture_scene, profile)


def test_floor_audit_is_bound_to_source_and_preserves_sampled_scope(fixture_scene, tmp_path):
    audit_path = tmp_path / "audit.json"
    audit = {"authority": "input-floor-support-audit", "inputs": {"scene_sha256": "wrong"},
             "classification": "no_sampled_mesh_support_with_positive_envelope_clearance"}
    audit_path.write_text(json.dumps(audit))
    args = (fixture_scene, navigation_profile("soarm"), (0, 0), (1, 0), tmp_path / "viewer" / "soarm.json")
    with pytest.raises(ValueError, match="does not match"):
        export_viewer_run(*args, floor_audit=audit_path)
    audit["inputs"]["scene_sha256"] = fixture_scene.hashes["scene.json"]
    audit_path.write_text(json.dumps(audit))
    run = export_viewer_run(*args, floor_audit=audit_path)
    assert run["scene"]["supportStatus"] == "sampled_no_overlap"
    assert "sampled" in run["scene"]["supportNote"]
    assert run["readiness"] == "PREVIEW_ONLY"


def _native_fixture(directory, *, learned):
    directory.mkdir()
    report = {"hashes": {"scene_json_sha256": "a" * 64, "navigation_policy_sha256": "b" * 64},
              "navigation_policy": {"trained": learned, "checkpoint_sha256": "b" * 64},
              "execution": "completed", "metrics": {"forbidden_contacts": 0}}
    (directory / "report.json").write_text(json.dumps(report))
    (directory / "replay.json").write_text("{}")
    manifest = {f"{name}_sha256": hashlib.sha256((directory / f"{name}.json").read_bytes()).hexdigest()
                for name in ("report", "replay")}
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_learned_wrapper_turns_before_translating_and_predictively_brakes(fixture_scene):
    class ForwardPolicy:
        def predict(self, observation):
            return np.array([1.0, 0.0, .5])
    profile = navigation_profile("g1")
    cfg = NavigationConfig()
    controller = LearnedNavigationController(fixture_scene, profile, ForwardPolicy(), HomeNavConfig(), (1, 0), cfg)
    xy = np.array([0.0, 0.0])
    command = controller.command(xy, np.pi/2, 1.1, .02)
    assert command[0] == 0 and command[2] > 0
    assert np.array_equal(xy, [0, 0])
    controller = LearnedNavigationController(fixture_scene, profile, ForwardPolicy(), HomeNavConfig(), (0, 0), cfg)
    controller.current[0] = .3
    command = controller.command(np.array([2.5, 0]), 0, 1.1, .02)
    assert command[0] == 0  # A gradual speed ramp cannot override the safety brake.


def test_native_transfer_requires_same_highlevel_checkpoint_and_video_is_optional(tmp_path):
    native = tmp_path / "native"
    _native_fixture(native, learned=True)
    run = {"scene": {"sha256": "a" * 64}, "training": {"checkpointSha256": "c" * 64}}
    with pytest.raises(ValueError, match="loaded navigation checkpoint"):
        _attach_native(run, tmp_path, native)
    run["training"] = None
    with pytest.raises(ValueError, match="loaded navigation checkpoint"):
        _attach_native(run, tmp_path, native)
    run["training"] = {"checkpointSha256": "b" * 64}
    _attach_native(run, tmp_path, native)
    assert run["validation"]["completed"] and "nativeVideoUrl" not in run
    (native / "demo-1080p.mp4").write_bytes(b"test-only-video-presence")
    _attach_native(run, tmp_path, native)
    assert run["nativeVideoUrl"] == "/policies/g1-native/demo.mp4"
