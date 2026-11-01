from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from morpheus_policy.policies.g1 import (
    ACTION_SCALE,
    CHECKPOINT_BYTES,
    CHECKPOINT_SHA256,
    CONTROL_DECIMATION,
    DEFAULT_ANGLES,
    G1Policy,
    JOINT_NAMES,
    JOINT_FORCE_LIMITS,
    KD,
    KP,
    NUM_ACTIONS,
    UPSTREAM_COMMIT,
    manifest,
)


def _model_xml(timestep: float = 0.002, prefix: str = "", force_limits: bool = True) -> str:
    links = "\n".join(
        (
            f'<body name="{prefix}link_{index}"><joint name="{prefix}{name}" type="hinge" '
            + (
                f'actuatorfrclimited="true" actuatorfrcrange="{-JOINT_FORCE_LIMITS[index]} {JOINT_FORCE_LIMITS[index]}" '
                if force_limits
                else ""
            )
            + '/><geom type="sphere" size="0.01" mass="0.01"/>'
        )
        for index, name in enumerate(JOINT_NAMES)
    )
    closes = "</body>" * len(JOINT_NAMES)
    motors = "\n".join(f'<motor name="{prefix}motor_{index}" joint="{prefix}{name}"/>' for index, name in enumerate(JOINT_NAMES))
    return f"""
    <mujoco model="g1_12dof_contract_fixture">
      <option timestep="{timestep}"/>
      <worldbody>
        <body name="{prefix}pelvis" pos="0 0 0.8">
          <freejoint name="{prefix}floating_base_joint"/>
          <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
          {links}{closes}
        </body>
      </worldbody>
      <actuator>{motors}</actuator>
    </mujoco>
    """


def _model_data(timestep: float = 0.002, prefix: str = "", force_limits: bool = True):
    model = mujoco.MjModel.from_xml_string(_model_xml(timestep, prefix, force_limits))
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    data.qpos[7:] = DEFAULT_ANGLES
    mujoco.mj_forward(model, data)
    return model, data


def _unloaded_policy(model, prefix: str = ""):
    """Make a contract-valid test double without weakening artifact verification."""
    policy = object.__new__(G1Policy)
    policy._torch = None
    policy._actor = None
    policy.name_prefix = prefix
    policy._physics_steps = 0
    policy._previous_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy._target_positions = DEFAULT_ANGLES.copy()
    policy._validated_model_id = id(model)
    return policy


def test_manifest_pins_exact_unitree_artifact():
    value = manifest()
    assert value["commit"] == UPSTREAM_COMMIT
    assert value["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert value["checkpoint_bytes"] == CHECKPOINT_BYTES
    assert value["joints"] == list(JOINT_NAMES)


def test_validate_accepts_only_the_exact_model_contract():
    model, _ = _model_data()
    policy = _unloaded_policy(model)
    contract = policy.validate(model)
    assert contract["valid"] is True
    assert contract["errors"] == []
    assert contract["expected_joint_force_limits"] == JOINT_FORCE_LIMITS.tolist()

    wrong_timestep_model, _ = _model_data(0.003)
    wrong_contract = policy.validate(wrong_timestep_model)
    assert wrong_contract["valid"] is False
    assert any("timestep" in error for error in wrong_contract["errors"])


def test_validate_rejects_disabled_or_wrong_joint_force_limits():
    model, _ = _model_data(force_limits=False)
    policy = _unloaded_policy(model)
    contract = policy.validate(model)
    assert contract["valid"] is False
    assert any("force limits enabled" in error for error in contract["errors"])


def test_prefixed_attachment_contract_validates_and_reset_never_writes_base_state():
    model, data = _model_data(prefix="robot/")
    policy = _unloaded_policy(model, prefix="robot/")
    reset_calls: list[bool] = []

    class Actor:
        def reset_memory(self):
            reset_calls.append(True)

    policy._actor = Actor()
    policy._load_actor = lambda: None
    qpos_before = data.qpos.copy()
    qvel_before = data.qvel.copy()
    contract = policy.validate(model)
    assert contract["valid"], contract["errors"]
    assert contract["joint_names_found"] == [f"robot/{name}" for name in JOINT_NAMES]
    policy.reset(model, data)
    assert reset_calls == [True]
    np.testing.assert_array_equal(data.qpos, qpos_before)
    np.testing.assert_array_equal(data.qvel, qvel_before)


def test_step_writes_only_pd_control_and_never_teleports_base():
    model, data = _model_data()
    policy = _unloaded_policy(model)
    qpos_before = data.qpos.copy()
    qvel_before = data.qvel.copy()

    torque = policy.step(model, data, [0.0, 0.0, 0.0])

    np.testing.assert_array_equal(data.qpos, qpos_before)
    np.testing.assert_array_equal(data.qvel, qvel_before)
    np.testing.assert_allclose(torque, np.zeros(NUM_ACTIONS), atol=1e-7)
    np.testing.assert_allclose(data.ctrl, torque)

    data.qpos[7] += 0.1
    torque = policy.step(model, data, [0.0, 0.0, 0.0])
    expected = (DEFAULT_ANGLES - data.qpos[7:]) * KP - data.qvel[6:] * KD
    np.testing.assert_allclose(torque, expected)


def test_actor_update_is_every_tenth_tick_and_target_is_retained():
    model, data = _model_data()
    policy = _unloaded_policy(model)
    actor_calls: list[np.ndarray] = []
    action = np.linspace(-0.2, 0.2, NUM_ACTIONS, dtype=np.float32)

    def infer(observation: np.ndarray) -> np.ndarray:
        actor_calls.append(observation.copy())
        return action

    policy._infer = infer
    for _ in range(CONTROL_DECIMATION - 1):
        policy.step(model, data, [0.1, 0.0, 0.2])
    assert actor_calls == []
    torque = policy.step(model, data, [0.1, 0.0, 0.2])
    assert len(actor_calls) == 1
    np.testing.assert_allclose(policy._target_positions, DEFAULT_ANGLES + ACTION_SCALE * action)
    np.testing.assert_allclose(torque, ACTION_SCALE * action * KP, rtol=1e-6, atol=1e-6)
    assert actor_calls[0].shape == (47,)


def test_rejects_invalid_command_before_state_mutation():
    model, data = _model_data()
    policy = _unloaded_policy(model)
    ctrl_before = data.ctrl.copy()
    with pytest.raises(ValueError, match="three finite"):
        policy.step(model, data, [np.nan, 0.0, 0.0])
    np.testing.assert_array_equal(data.ctrl, ctrl_before)


def test_checkpoint_mismatch_fails_closed(tmp_path: Path):
    incorrect = tmp_path / "motion.pt"
    incorrect.write_bytes(b"not-the-pinned-checkpoint")
    with pytest.raises(ValueError, match="byte size"):
        G1Policy(incorrect)


def test_asset_gate_rejects_missing_or_unprovenanced_bundle(tmp_path: Path):
    contract = G1Policy.validate_assets(tmp_path)
    assert contract["valid"] is False
    assert "missing asset motion.pt" in contract["errors"]
    assert "missing asset provenance.json" in contract["errors"]


def test_real_checkpoint_resets_recurrent_memory_deterministically_when_available():
    asset_root = Path(".morpheus/assets/g1")
    checkpoint = asset_root / "motion.pt"
    scene = asset_root / "robot/scene.xml"
    if not checkpoint.is_file() or not scene.is_file():
        pytest.skip("pinned G1 artifact has not been fetched")

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    data.qpos[7:] = DEFAULT_ANGLES
    mujoco.mj_forward(model, data)
    policy = G1Policy(checkpoint)
    policy.reset(model, data)
    for _ in range(CONTROL_DECIMATION):
        first_torque = policy.step(model, data, [0.1, 0.0, 0.0])

    mujoco.mj_resetData(model, data)
    data.qpos[3] = 1.0
    data.qpos[7:] = DEFAULT_ANGLES
    mujoco.mj_forward(model, data)
    policy.reset(model, data)
    for _ in range(CONTROL_DECIMATION):
        second_torque = policy.step(model, data, [0.1, 0.0, 0.0])

    np.testing.assert_allclose(first_torque, second_torque, rtol=0.0, atol=1e-7)


def test_fetched_asset_bundle_passes_source_integrity_gate_when_available():
    asset_root = Path(".morpheus/assets/g1")
    if not asset_root.is_dir():
        pytest.skip("pinned G1 artifact has not been fetched")
    contract = G1Policy.validate_assets(asset_root)
    assert contract["valid"], contract["errors"]


def test_asset_gate_rejects_provenance_escape_and_symlink_paths_when_available(tmp_path: Path):
    source = Path(".morpheus/assets/g1")
    if not source.is_dir():
        pytest.skip("pinned G1 artifact has not been fetched")
    clone = tmp_path / "g1"
    shutil.copytree(source, clone)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (clone / "robot/meshes/escape.STL").symlink_to(outside)
    provenance_path = clone / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    inventory = provenance["asset_file_sha256"]
    inventory["../outside.txt"] = "00"
    inventory["robot/meshes/escape.STL"] = "00"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    contract = G1Policy.validate_assets(clone)
    assert contract["valid"] is False
    assert any("../outside.txt" in error for error in contract["errors"])
    assert any("robot/meshes/escape.STL" in error for error in contract["errors"])
