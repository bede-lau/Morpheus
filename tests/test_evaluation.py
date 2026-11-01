"""Mandatory gates exercised through an alternate injected policy and real physics."""

from dataclasses import replace
import hashlib
import json

import mujoco
import numpy as np
import pytest

from morpheus_policy.evaluation import Task, Thresholds, evaluate_episode
from morpheus_policy.scene import Box, Scene
from morpheus_policy.qualification import qualification_tasks, summarize_qualification


class TestPolicy:
    __test__ = False

    def __init__(self, value=0.0):
        self.value, self.calls = value, 0

    def validate(self, model):
        return {"valid": True, "policy_sha256": "a" * 64, "test_fixture_only": True}

    def reset(self, model, data):
        self.calls = 0

    def step(self, model, data, command):
        self.calls += 1
        return np.full(model.nu, self.value)


@pytest.fixture
def setup_scene(tmp_path):
    # Tiny contract fixture, not a generated home or supported robot.
    source = tmp_path / "scene.json"
    source.write_text(json.dumps({"test_fixture": True}))
    floor = Box("floor", (0, 0, -0.05), (6, 6, 0.1), (1, 0, 0, 0), "floor")
    scene = Scene("contract fixture", (floor,), {}, tmp_path, tmp_path / "scene.xml",
                  {"scene.json": hashlib.sha256(source.read_bytes()).hexdigest()}, ("Test fixture only",))
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002" gravity="0 0 0"/>
      <worldbody><geom name="env_floor" type="box" size="3 3 .05" pos="0 0 -.05"/>
        <body name="test_body" pos="0 0 .8"><freejoint name="root"/><geom name="test_geom" type="sphere" size=".1" mass="1"/></body>
      </worldbody><actuator><general joint="root" gear="1 0 0 0 0 0"/></actuator>
    </mujoco>''')
    return scene, model


def test_alternate_policy_uses_actual_steps_and_cannot_upgrade_preview(setup_scene):
    scene, model = setup_scene
    policy = TestPolicy()
    report, replay = evaluate_episode(scene, model, policy, Task((0, 0), (0, 0)),
                                      model_hash="b" * 64, robot_id="contract_fixture")
    assert report["execution"] == "completed"
    assert report["readiness"] == "PREVIEW_ONLY"
    assert report["hardware_readiness"] == "UNVERIFIED"
    assert policy.calls == report["metrics"]["physics_steps"] > 500
    assert replay["robot"]["id"] == "contract_fixture"
    assert replay["samples"][0]["time_s"] == 0
    assert replay["hashes"]["policy_sha256"] == "a" * 64
    assert all(b["time_s"] > a["time_s"] for a, b in zip(replay["samples"], replay["samples"][1:]))


def test_invalid_actuation_stops_before_a_step(setup_scene):
    scene, model = setup_scene
    report, _ = evaluate_episode(scene, model, TestPolicy(float("nan")), Task((0, 0), (0, 0)), model_hash="b" * 64)
    assert report["execution"] == "failed"
    assert report["readiness"] == "BLOCKED"
    assert report["metrics"]["physics_steps"] == 0
    assert "invalid actuator controls" in report["failures"][0]


def test_no_teleporting_to_goal(setup_scene):
    scene, model = setup_scene
    report, replay = evaluate_episode(scene, model, TestPolicy(), Task((0, 0), (1, 0)),
                                      model_hash="b" * 64, thresholds=Thresholds(max_seconds=2))
    assert report["execution"] == "failed"
    assert report["metrics"]["goal_error_m"] == pytest.approx(1.0)
    assert replay["samples"][-1]["qpos"][0] == pytest.approx(0.0)


def test_collision_masks_cannot_disable_the_evidence(setup_scene):
    scene, model = setup_scene
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    with pytest.raises(ValueError, match="collidable robot"):
        evaluate_episode(scene, model, TestPolicy(), Task((0, 0), (0, 0)), model_hash="b" * 64)


def test_actual_forbidden_contact_stops_and_is_replayable(setup_scene):
    scene, model = setup_scene
    model.opt.gravity[:] = [0, 0, -9.81]
    report, replay = evaluate_episode(scene, model, TestPolicy(), Task((0, 0), (0, 0)), model_hash="b" * 64,
                                      thresholds=Thresholds(min_base_height_m=0.05))
    assert report["execution"] == "failed"
    assert report["metrics"]["forbidden_contacts"] > 0
    assert replay["events"][0]["kind"] == "contact"
    assert replay["events"][0]["object_ids"] == ["floor"]


def test_divergence_keeps_last_finite_replay_state(setup_scene, monkeypatch):
    scene, model = setup_scene
    real_step = mujoco.mj_step

    def diverge(current_model, data):
        real_step(current_model, data)
        data.qpos[0] = float("nan")

    monkeypatch.setattr(mujoco, "mj_step", diverge)
    report, replay = evaluate_episode(scene, model, TestPolicy(), Task((0, 0), (0, 0)), model_hash="b" * 64)
    assert report["readiness"] == "BLOCKED"
    assert replay["events"][0]["kind"] == "non_finite"
    assert len(replay["samples"]) == 1
    json.dumps({"report": report, "replay": replay}, allow_nan=False)


def test_compose_rejects_upstream_dynamic_body(setup_scene, tmp_path):
    from morpheus_policy.evaluation import compose_model
    scene, _ = setup_scene
    path = tmp_path / "scene.xml"
    path.write_text('<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>')
    hashes = dict(scene.hashes)
    hashes["scene.xml"] = hashlib.sha256(path.read_bytes()).hexdigest()
    scene = replace(scene, hashes=hashes)
    with pytest.raises(ValueError, match="static upstream"):
        compose_model(scene, tmp_path / "unused_robot.xml")


def test_qualification_keeps_upstream_uncertainty_and_all_failures(setup_scene):
    scene, _ = setup_scene
    task = Task((0, 0), (1, 0))
    tasks = list(qualification_tasks(task))
    assert len(tasks) == 20 and len({t.start_xy for t in tasks}) == 20
    reports = [{"execution": "completed", "readiness": "PREVIEW_ONLY", "failures": [],
                "metrics": {"duration_s": 1.0, "goal_error_m": 0.1, "forbidden_contacts": 0, "max_tilt_rad": 0.1},
                "hashes": {"scene_json_sha256": scene.hashes["scene.json"], "mjcf_sha256": "b" * 64, "policy_sha256": "a" * 64},
                "controller_config": {"speed_m_s": 0.3}} for _ in range(20)]
    summary = summarize_qualification(scene, task, reports, seed=1)
    assert summary["readiness"] == "PREVIEW_ONLY"
    reports[3] = {"execution": "failed", "failures": ["fall"]}
    summary = summarize_qualification(scene, task, reports, seed=1)
    assert summary["readiness"] == "BLOCKED" and summary["coverage"]["successes"] == 19
    assert summary["episodes"][3]["failures"] == ["fall"]
    with pytest.raises(ValueError, match="20 to 100"):
        list(qualification_tasks(task, episodes=1))
