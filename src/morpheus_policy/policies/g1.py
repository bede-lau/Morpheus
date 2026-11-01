"""Exact, stateful adapter for Unitree's pinned 12-DoF G1 TorchScript policy.

The adapter deliberately owns no scene state.  It reads MuJoCo's current state,
updates only ``data.ctrl``, and leaves stepping to the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


UPSTREAM_REPOSITORY = "https://github.com/unitreerobotics/unitree_rl_gym"
UPSTREAM_COMMIT = "276801e46c5d433564f24658bac64f254b7d2d4b"
CHECKPOINT_RELATIVE_PATH = "deploy/pre_train/g1/motion.pt"
CHECKPOINT_SHA256 = "cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d"
CHECKPOINT_BYTES = 145_745
MODEL_XML_SHA256 = "747ede40aa726b7352bae8353e95d0d0f908cec2257a27cbd78bc6e5a2d5a314"
SCENE_XML_SHA256 = "482d49902ca2b9fdc49d84ef5d8779fa69d66bc53ef440b4bf50079d1bac9995"
LICENSE_SHA256 = "aef6394ba1597725a68308167324e675f562e6606027404deb1b9da254c2b9c1"
POLICY_ID = "unitree_g1_12dof_lstm_v1"

PHYSICS_DT_S = 0.002
CONTROL_DECIMATION = 10
ACTOR_DT_S = PHYSICS_DT_S * CONTROL_DECIMATION
NUM_OBSERVATIONS = 47
NUM_ACTIONS = 12
GAIT_PERIOD_S = 0.8

JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
DEFAULT_ANGLES = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    dtype=np.float32,
)
KP = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=np.float32)
KD = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=np.float32)
JOINT_FORCE_LIMITS = np.array([88, 139, 88, 139, 50, 50] * 2, dtype=np.float64)
ACTION_SCALE = 0.25
ANGULAR_VELOCITY_SCALE = 0.25
JOINT_POSITION_SCALE = 1.0
JOINT_VELOCITY_SCALE = 0.05
COMMAND_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)


@dataclass(frozen=True)
class G1Manifest:
    policy_id: str = POLICY_ID
    repository: str = UPSTREAM_REPOSITORY
    commit: str = UPSTREAM_COMMIT
    checkpoint_relative_path: str = CHECKPOINT_RELATIVE_PATH
    checkpoint_sha256: str = CHECKPOINT_SHA256
    checkpoint_bytes: int = CHECKPOINT_BYTES
    physics_dt_s: float = PHYSICS_DT_S
    actor_dt_s: float = ACTOR_DT_S
    observations: int = NUM_OBSERVATIONS
    actions: int = NUM_ACTIONS
    joints: tuple[str, ...] = JOINT_NAMES

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "repository": self.repository,
            "commit": self.commit,
            "checkpoint_relative_path": self.checkpoint_relative_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_bytes": self.checkpoint_bytes,
            "model_xml_sha256": MODEL_XML_SHA256,
            "scene_xml_sha256": SCENE_XML_SHA256,
            "physics_dt_s": self.physics_dt_s,
            "actor_dt_s": self.actor_dt_s,
            "observations": self.observations,
            "actions": self.actions,
            "joints": list(self.joints),
        }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _projected_gravity(quaternion: np.ndarray) -> np.ndarray:
    """Match Unitree's exported MuJoCo deploy loop exactly."""
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


class G1Policy:
    """A CPU-only policy which writes actuator torques but never moves the base."""

    manifest = G1Manifest()

    def __init__(self, checkpoint_path: str | Path, name_prefix: str = "") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.name_prefix = name_prefix
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"G1 checkpoint not found: {self.checkpoint_path}")
        if self.checkpoint_path.stat().st_size != CHECKPOINT_BYTES:
            raise ValueError("G1 checkpoint byte size does not match the pinned artifact")
        actual_hash = _sha256_file(self.checkpoint_path)
        if actual_hash != CHECKPOINT_SHA256:
            raise ValueError("G1 checkpoint SHA-256 does not match the pinned artifact")

        self._torch: Any | None = None
        self._actor: Any | None = None
        self._physics_steps = 0
        self._previous_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self._target_positions = DEFAULT_ANGLES.copy()
        self._validated_model_id: int | None = None

    def validate(self, model: Any) -> dict[str, Any]:
        """Return the exact model-contract result without modifying MuJoCo state."""
        errors: list[str] = []
        details: dict[str, Any] = {
            **self.manifest.as_dict(),
            "policy_id": POLICY_ID,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "expected_joint_names": [self._name(name) for name in JOINT_NAMES],
            "expected_dt_s": PHYSICS_DT_S,
            "expected_actions": NUM_ACTIONS,
            "expected_observations": NUM_OBSERVATIONS,
        }

        for attribute, expected in (("nq", 19), ("nv", 18), ("nu", NUM_ACTIONS)):
            observed = getattr(model, attribute, None)
            details[attribute] = observed
            if observed != expected:
                errors.append(f"{attribute}={observed!r}, expected {expected}")

        timestep = getattr(getattr(model, "opt", None), "timestep", None)
        details["timestep_s"] = timestep
        if timestep is None or not np.isclose(timestep, PHYSICS_DT_S, rtol=0.0, atol=1e-12):
            errors.append(f"timestep={timestep!r}, expected {PHYSICS_DT_S}")

        try:
            import mujoco

            free_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self._name("floating_base_joint"))
            if free_joint < 0:
                errors.append(f"missing free joint {self._name('floating_base_joint')}")
            else:
                details["free_joint_qpos_address"] = int(model.jnt_qposadr[free_joint])
                details["free_joint_dof_address"] = int(model.jnt_dofadr[free_joint])
                if int(model.jnt_qposadr[free_joint]) != 0 or int(model.jnt_dofadr[free_joint]) != 0:
                    errors.append("floating_base_joint must be first in qpos and qvel")
                if int(model.jnt_type[free_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
                    errors.append("floating_base_joint must be a MuJoCo free joint")

            found_joints: list[str | None] = []
            for index, name in enumerate(JOINT_NAMES):
                resolved_name = self._name(name)
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, resolved_name)
                found_joints.append(resolved_name if joint_id >= 0 else None)
                if joint_id < 0:
                    errors.append(f"missing joint {resolved_name}")
                    continue
                if int(model.jnt_qposadr[joint_id]) != 7 + index:
                    errors.append(f"joint {resolved_name} has unexpected qpos address")
                if int(model.jnt_dofadr[joint_id]) != 6 + index:
                    errors.append(f"joint {resolved_name} has unexpected qvel address")
                if index < int(getattr(model, "nu", 0)):
                    actuator_joint = int(model.actuator_trnid[index, 0])
                    if actuator_joint != joint_id:
                        errors.append(f"actuator {index} is not mapped to {resolved_name}")
                    expected_limit = JOINT_FORCE_LIMITS[index]
                    force_range = np.asarray(model.jnt_actfrcrange[joint_id], dtype=np.float64)
                    if int(model.jnt_actfrclimited[joint_id]) != 1:
                        errors.append(f"joint {resolved_name} must have actuator-force limits enabled")
                    elif not np.allclose(force_range, (-expected_limit, expected_limit), rtol=0.0, atol=1e-12):
                        errors.append(f"joint {resolved_name} has an unexpected actuator-force range")
                    if int(model.actuator_forcelimited[index]) != 0:
                        errors.append(f"actuator {index} must not override the pinned joint-force limits")
                    if int(model.actuator_ctrllimited[index]) != 0:
                        errors.append(f"actuator {index} must not add a control-range limit")
            details["joint_names_found"] = found_joints
            details["expected_joint_force_limits"] = JOINT_FORCE_LIMITS.tolist()
        except ImportError:
            errors.append("MuJoCo is required to validate the G1 model")
        except (AttributeError, IndexError, TypeError) as error:
            errors.append(f"invalid MuJoCo model contract: {error}")

        details["valid"] = not errors
        details["errors"] = errors
        return details

    @staticmethod
    def validate_assets(asset_root: str | Path) -> dict[str, Any]:
        """Validate the fetched robot bundle before composing it with a scene.

        MuJoCo's compiled model has no reliable source-path hash.  The evaluator
        therefore calls this explicit asset gate before XML composition, while
        ``validate(model)`` checks the resulting physics contract.
        """
        root = Path(asset_root).expanduser().resolve()
        required_hashes = {
            "motion.pt": CHECKPOINT_SHA256,
            "robot/g1_12dof.xml": MODEL_XML_SHA256,
            "robot/scene.xml": SCENE_XML_SHA256,
            "LICENSE": LICENSE_SHA256,
        }
        errors: list[str] = []
        actual_hashes: dict[str, str] = {}
        for relative, expected in required_hashes.items():
            path = root / relative
            if not path.is_file():
                errors.append(f"missing asset {relative}")
                continue
            actual = _sha256_file(path)
            actual_hashes[relative] = actual
            if actual != expected:
                errors.append(f"asset hash mismatch: {relative}")

        provenance_path = root / "provenance.json"
        provenance: dict[str, Any] = {}
        if not provenance_path.is_file():
            errors.append("missing asset provenance.json")
        else:
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                if provenance.get("commit") != UPSTREAM_COMMIT:
                    errors.append("asset provenance commit does not match pinned Unitree commit")
                inventory = provenance.get("asset_file_sha256")
                if not isinstance(inventory, dict):
                    errors.append("asset provenance has no complete file-hash inventory")
                else:
                    for relative, expected in inventory.items():
                        candidate = _safe_inventory_file(root, relative)
                        if not isinstance(expected, str) or candidate is None:
                            errors.append(f"asset provenance references missing file: {relative}")
                        elif _sha256_file(candidate) != expected:
                            errors.append(f"asset provenance hash mismatch: {relative}")
                    mesh_entries = [name for name in inventory if isinstance(name, str) and name.startswith("robot/meshes/")]
                    if not mesh_entries:
                        errors.append("asset provenance contains no G1 mesh files")
            except (OSError, json.JSONDecodeError, TypeError) as error:
                errors.append(f"invalid asset provenance: {error}")

        return {
            "valid": not errors,
            "errors": errors,
            "asset_root": str(root),
            "actual_anchor_hashes": actual_hashes,
            "commit": provenance.get("commit"),
        }

    def _name(self, name: str) -> str:
        return f"{self.name_prefix}{name}"

    def reset(self, model: Any, data: Any) -> None:
        """Reset recurrent/controller state; deliberately do not write qpos, qvel, or base pose."""
        contract = self.validate(model)
        if not contract["valid"]:
            raise ValueError("incompatible G1 MuJoCo model: " + "; ".join(contract["errors"]))
        self._load_actor()
        assert self._actor is not None
        self._actor.reset_memory()
        self._physics_steps = 0
        self._previous_action.fill(0.0)
        self._target_positions = DEFAULT_ANGLES.copy()
        self._validated_model_id = id(model)
        self._assert_data_contract(data)

    def step(self, model: Any, data: Any, command: Any) -> np.ndarray:
        """Write PD torque to ``data.ctrl`` and return a detached copy of that torque.

        Call once before every ``mujoco.mj_step``. The actor updates every tenth
        call; the retained target receives PD control on every call.
        """
        if self._validated_model_id != id(model):
            self.reset(model, data)
        self._assert_data_contract(data)
        command_array = np.asarray(command, dtype=np.float32)
        if command_array.shape != (3,) or not np.isfinite(command_array).all():
            raise ValueError("G1 command must be three finite body-frame values [vx, vy, yaw_rate]")

        self._physics_steps += 1
        if self._physics_steps % CONTROL_DECIMATION == 0:
            observation = self._observation(data, command_array)
            action = self._infer(observation)
            if action.shape != (NUM_ACTIONS,) or not np.isfinite(action).all():
                raise RuntimeError("G1 actor returned an invalid action")
            self._previous_action = action.astype(np.float32, copy=True)
            self._target_positions = DEFAULT_ANGLES + ACTION_SCALE * self._previous_action

        qpos = np.asarray(data.qpos[7 : 7 + NUM_ACTIONS], dtype=np.float32)
        qvel = np.asarray(data.qvel[6 : 6 + NUM_ACTIONS], dtype=np.float32)
        torque = (self._target_positions - qpos) * KP - qvel * KD
        if not np.isfinite(torque).all():
            raise RuntimeError("G1 PD controller produced non-finite torque")
        data.ctrl[:] = torque
        return torque.copy()

    def _load_actor(self) -> None:
        if self._actor is not None:
            return
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Torch is required to load the G1 TorchScript policy") from error
        self._torch = torch
        self._actor = torch.jit.load(str(self.checkpoint_path), map_location="cpu")
        self._actor.eval()

    def _infer(self, observation: np.ndarray) -> np.ndarray:
        self._load_actor()
        assert self._torch is not None and self._actor is not None
        tensor = self._torch.from_numpy(observation).unsqueeze(0)
        with self._torch.no_grad():
            action = self._actor(tensor).detach().cpu().numpy().squeeze(0)
        return np.asarray(action, dtype=np.float32)

    def _observation(self, data: Any, command: np.ndarray) -> np.ndarray:
        observation = np.zeros(NUM_OBSERVATIONS, dtype=np.float32)
        qpos = np.asarray(data.qpos[7 : 7 + NUM_ACTIONS], dtype=np.float32)
        qvel = np.asarray(data.qvel[6 : 6 + NUM_ACTIONS], dtype=np.float32)
        quaternion = np.asarray(data.qpos[3:7], dtype=np.float32)
        angular_velocity = np.asarray(data.qvel[3:6], dtype=np.float32)
        if not np.isfinite(np.concatenate((qpos, qvel, quaternion, angular_velocity))).all():
            raise ValueError("G1 state must be finite")

        phase = ((self._physics_steps * PHYSICS_DT_S) % GAIT_PERIOD_S) / GAIT_PERIOD_S
        observation[0:3] = angular_velocity * ANGULAR_VELOCITY_SCALE
        observation[3:6] = _projected_gravity(quaternion)
        observation[6:9] = command * COMMAND_SCALE
        observation[9:21] = (qpos - DEFAULT_ANGLES) * JOINT_POSITION_SCALE
        observation[21:33] = qvel * JOINT_VELOCITY_SCALE
        observation[33:45] = self._previous_action
        observation[45:47] = (np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase))
        return observation

    @staticmethod
    def _assert_data_contract(data: Any) -> None:
        qpos = getattr(data, "qpos", None)
        qvel = getattr(data, "qvel", None)
        ctrl = getattr(data, "ctrl", None)
        if qpos is None or qvel is None or ctrl is None:
            raise ValueError("MuJoCo data must provide qpos, qvel, and ctrl")
        if len(qpos) != 19 or len(qvel) != 18 or len(ctrl) != NUM_ACTIONS:
            raise ValueError("MuJoCo data dimensions do not match G1 12-DoF contract")


def manifest() -> dict[str, Any]:
    """Return JSON-ready, pinned policy provenance for reports and registry wiring."""
    return G1Manifest().as_dict()


def _safe_inventory_file(root: Path, relative: Any) -> Path | None:
    """Resolve an untrusted provenance path only when it stays inside ``root``."""
    if not isinstance(relative, str):
        return None
    path = PurePosixPath(relative)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        return None
    candidate = root.joinpath(*path.parts)
    try:
        candidate.relative_to(root)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            return None
    return resolved if resolved.is_file() else None
