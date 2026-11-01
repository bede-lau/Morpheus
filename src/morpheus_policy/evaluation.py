"""Actual actuated physics evaluation against upstream-owned environment geometry."""

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from .navigation import NavigationConfig, RouteController, yaw_from_wxyz
from .planning import clearance, plan_route
from .artifacts import sha256_file
from .scene import verify_source_snapshot


@dataclass(frozen=True)
class Task:
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    start_yaw: float = 0.0
    id: str = "explicit_navigation_target"
    synthetic: bool = False

    def __post_init__(self):
        vals = (*self.start_xy, *self.goal_xy, self.start_yaw)
        if len(self.start_xy) != 2 or len(self.goal_xy) != 2 or not all(math.isfinite(v) for v in vals):
            raise ValueError("Task requires finite start XY, goal XY, and yaw")


@dataclass(frozen=True)
class Thresholds:
    radius_m: float = 0.35
    body_height_m: float = 1.40
    clearance_m: float = 0.10
    max_tilt_rad: float = 0.75
    min_base_height_m: float = 0.45
    max_joint_limit_overrun_rad: float = 0.04
    max_joint_speed_rad_s: float = 35.0
    max_seconds: float = 35.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Evaluation thresholds must be finite and positive")
        if self.max_seconds > 120:
            raise ValueError("Episode limit is 120 simulated seconds")


def compose_model(scene, robot_xml_path):
    """Attach a registered robot to unchanged input geometry using native MjSpec."""
    verify_source_snapshot(scene)
    parent = mujoco.MjSpec.from_file(str(scene.mjcf_path))
    if parent.joints:
        raise ValueError("Expected a static upstream environment without joints; supply robot separately")
    robot = mujoco.MjSpec.from_file(str(robot_xml_path))
    frame = parent.worldbody.add_frame()
    parent.attach(robot, frame=frame, prefix="robot/", suffix="")
    model = parent.compile()
    verify_source_snapshot(scene)
    # Serialization is hashed for model identity, not written back to input files.
    model_hash = hashlib.sha256(parent.to_xml().encode()).hexdigest()
    return model, parent, model_hash


def initialize(model, data, task, initial_joint_positions=None):
    """Explicit episode reset is the only place the floating-base pose is assigned."""
    mujoco.mj_resetData(model, data)
    if model.nq < 7 or int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("Navigation requires a registered floating-base model")
    data.qpos[:2] = task.start_xy
    data.qpos[3:7] = [math.cos(task.start_yaw / 2), 0, 0, math.sin(task.start_yaw / 2)]
    # Initial posture is injected at composition, not selected by robot type here.
    if initial_joint_positions is not None:
        posture = np.asarray(initial_joint_positions, dtype=float)
        if posture.shape != data.qpos[7:].shape or not np.isfinite(posture).all():
            raise ValueError("Initial posture does not match the floating-base model")
        data.qpos[7:] = posture
    mujoco.mj_forward(model, data)


def _names(model, kind, count):
    return [mujoco.mj_id2name(model, kind, i) or f"unnamed_{i}" for i in range(count)]


def _readiness(execution, caveats, qualified):
    if execution != "completed":
        return "BLOCKED"
    if caveats or not qualified:
        return "PREVIEW_ONLY"
    return "SIM_READY"


def evaluate_episode(scene, model, policy, task, *, model_hash, config=None, thresholds=None,
                     initial_joint_positions=None, robot_id="unspecified", allowed_support_bodies=(), qualified=False):
    """Inject any contract-compatible policy; all failure gates stay in this loop."""
    cfg, limits = config or NavigationConfig(), thresholds or Thresholds()
    route = plan_route(scene, task.start_xy, task.goal_xy, radius_m=limits.radius_m,
                       margin_m=limits.clearance_m, body_height_m=limits.body_height_m)
    manifest = policy.validate(model)
    if not manifest.get("valid", True):
        raise ValueError(f"Policy/model contract failed: {manifest.get('errors', [])}")
    data = mujoco.MjData(model)
    initialize(model, data, task, initial_joint_positions)
    policy.reset(model, data)
    controller = RouteController(route, cfg)
    geom_names = _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
    body_names = _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    floor_names = {f"env_{box.id}" for box in scene.floors}
    env_names = {f"env_{box.id}" for box in scene.objects}
    env_ids = {i for i, name in enumerate(geom_names) if name in env_names}
    if len(env_ids) != len(env_names):
        raise ValueError("Composed model is missing supplied environment geoms")
    # A robot geom must participate in collision filtering. Visual-only geoms are excluded.
    robot_ids = {i for i in range(model.ngeom) if i not in env_ids and (model.geom_contype[i] or model.geom_conaffinity[i])}
    if not robot_ids:
        raise ValueError("No collidable robot geometry")
    for env in env_ids:
        if not any((model.geom_contype[r] & model.geom_conaffinity[env]) or (model.geom_contype[env] & model.geom_conaffinity[r]) for r in robot_ids):
            raise ValueError(f"Collision mask disables robot/environment testing: {geom_names[env]}")
    failures, events, samples = [], [], []
    command = np.zeros(3)
    metrics = {"duration_s": 0.0, "goal_error_m": float(np.linalg.norm(np.subtract(task.goal_xy, task.start_xy))),
               "forbidden_contacts": 0, "min_clearance_m": float("inf"), "max_tilt_rad": 0.0,
               "min_base_height_m": float("inf"), "max_joint_speed_rad_s": 0.0,
               "max_joint_limit_overrun_rad": 0.0, "max_abs_ctrl": 0.0, "physics_steps": 0}
    metrics.update(max_abs_applied_joint_force=0.0, max_joint_force_limit_overrun=0.0)
    dt = float(model.opt.timestep)
    actor_stride = max(1, round(0.02 / dt))
    start_wall = time.monotonic()
    arrival_since = None
    execution = "failed"

    def fail(kind, detail):
        failures.append(detail)
        event_time = float(data.time) if math.isfinite(data.time) else samples[-1]["time_s"]
        events.append({"start_s": event_time, "end_s": event_time, "kind": kind,
                       "severity": "fail", "object_ids": [], "detail": detail})

    def sample():
        # Preserve the last finite state when the simulator diverges; never emit NaN JSON.
        if not math.isfinite(data.time) or not all(np.isfinite(v).all() for v in (data.qpos, data.qvel, data.ctrl)):
            return
        # After mj_step, mj_forward synchronizes body poses to current qpos for replay.
        mujoco.mj_forward(model, data)
        samples.append({
            "time_s": float(data.time), "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
            "ctrl": data.ctrl.tolist(), "command": command.tolist(),
            "body_poses": {name: [*data.xpos[i].tolist(), *data.xquat[i].tolist()] for i, name in enumerate(body_names) if i},
            "metrics": {k: v for k, v in metrics.items() if k not in ("physics_steps",)},
        })

    # Initial clearance is calculated before moving; no input confidence is upgraded.
    metrics["min_clearance_m"] = float(clearance(scene, data.qpos[:2], radius_m=limits.radius_m, body_height_m=limits.body_height_m))
    metrics["min_base_height_m"] = float(data.qpos[2])
    sample()
    for step in range(math.ceil(limits.max_seconds / dt)):
        if step % actor_stride == 0:
            command = controller.command(data.qpos[:2], yaw_from_wxyz(data.qpos[3:7]), float(data.time), 0.02)
        controls = np.asarray(policy.step(model, data, command), dtype=float)
        if controls.shape != (model.nu,) or not np.isfinite(controls).all():
            fail("contract_failure", "Policy returned invalid actuator controls")
            break
        data.ctrl[:] = controls
        mujoco.mj_step(model, data)
        metrics["physics_steps"] += 1
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            fail("non_finite", "Physics state became non-finite")
            break
        w, x, y, z = data.qpos[3:7]
        tilt = math.acos(float(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
        height = float(data.qpos[2])
        margin = float(clearance(scene, data.qpos[:2], radius_m=limits.radius_m, body_height_m=limits.body_height_m))
        error = float(np.linalg.norm(data.qpos[:2] - task.goal_xy))
        speed = float(np.max(np.abs(data.qvel[6:]), initial=0))
        overrun = 0.0
        force_overrun = 0.0
        applied_force = 0.0
        for joint in range(model.njnt):
            if model.jnt_limited[joint]:
                position = float(data.qpos[model.jnt_qposadr[joint]])
                lo, hi = model.jnt_range[joint]
                overrun = max(overrun, float(lo - position), float(position - hi))
            if int(model.jnt_type[joint]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                force = float(data.qfrc_actuator[model.jnt_dofadr[joint]])
                applied_force = max(applied_force, abs(force))
                if model.jnt_actfrclimited[joint]:
                    lo, hi = model.jnt_actfrcrange[joint]
                    force_overrun = max(force_overrun, float(lo - force), float(force - hi))
        metrics.update(duration_s=float(data.time), goal_error_m=error,
                       min_clearance_m=min(metrics["min_clearance_m"], margin),
                       max_tilt_rad=max(metrics["max_tilt_rad"], tilt),
                       min_base_height_m=min(metrics["min_base_height_m"], height),
                       max_joint_speed_rad_s=max(metrics["max_joint_speed_rad_s"], speed),
                       max_joint_limit_overrun_rad=max(metrics["max_joint_limit_overrun_rad"], overrun),
                       max_abs_applied_joint_force=max(metrics["max_abs_applied_joint_force"], applied_force),
                       max_joint_force_limit_overrun=max(metrics["max_joint_force_limit_overrun"], force_overrun),
                       max_abs_ctrl=max(metrics["max_abs_ctrl"], float(np.max(np.abs(controls), initial=0))))
        for contact in data.contact[:data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            pair = {a, b}
            involves_robot = bool(pair & robot_ids)
            if not involves_robot:
                continue
            names = [geom_names[a], geom_names[b]]
            support = any(name in floor_names for name in names)
            # Only foot/ankle support is permitted, never torso-on-floor.
            other = b if geom_names[a] in floor_names else a
            foot = body_names[int(model.geom_bodyid[other])] in allowed_support_bodies
            if not (support and foot):
                metrics["forbidden_contacts"] += 1
                fail("contact", f"Forbidden contact: {names[0]} / {names[1]}")
                events[-1]["object_ids"] = [name[4:] for name in names if name.startswith("env_")]
                break
        if failures:
            break
        if any(warning.number for warning in data.warning):
            fail("simulation_warning", "MuJoCo issued a warning; run is invalid")
        elif tilt > limits.max_tilt_rad or height < limits.min_base_height_m:
            fail("fall", f"Base tilt/height limit exceeded: {tilt:.3f} rad, {height:.3f} m")
        elif margin < limits.clearance_m - 1e-6:
            fail("low_clearance", f"Envelope clearance {margin:.3f} m below {limits.clearance_m:.3f} m")
        elif overrun > limits.max_joint_limit_overrun_rad:
            fail("joint_limit", f"Joint limit overrun {overrun:.3f} rad")
        elif speed > limits.max_joint_speed_rad_s:
            fail("joint_speed", f"Joint speed {speed:.3f} rad/s exceeds limit")
        elif not np.isfinite(data.qfrc_actuator).all() or force_overrun > 1e-6:
            fail("actuator_limit", "Applied joint force is non-finite or exceeds the model's force limits")
        if step % actor_stride == actor_stride - 1 or failures:
            sample()
        if failures:
            break
        if error <= cfg.goal_tolerance_m and data.time >= cfg.settle_s:
            arrival_since = float(data.time) if arrival_since is None else arrival_since
            if data.time - arrival_since >= 0.5:
                execution = "completed"
                break
        else:
            arrival_since = None
    if execution != "completed" and not failures:
        fail("timeout", "Goal not reached within the declared simulation time budget")
    if float(data.time) > samples[-1]["time_s"]:
        sample()
    metrics["wall_time_s"] = time.monotonic() - start_wall
    hashes = dict(scene.hashes)
    # Stable keys required by the browser bridge; extras retain full source provenance.
    hashes.update(scene_json_sha256=scene.hashes.get("scene.json", sha256_file(scene.root / "scene.json")),
                  mjcf_sha256=model_hash,
                  policy_sha256=manifest.get("policy_sha256", manifest.get("checkpoint_sha256", manifest.get("sha256", "0" * 64))),
                  evaluator_version=f"mujoco-{mujoco.__version__}/morpheus-0.1.0")
    caveats = list(scene.caveats)
    if task.synthetic:
        caveats.append("Task start/goal were selected for a numerical demonstration, not a labelled kitchen application.")
    caveats.append("Single-episode report; 20 held-out qualification episodes have not been established by this result.")
    report = {
        "schema_version": "1.0", "execution": execution,
        "readiness": _readiness(execution, caveats, qualified), "hardware_readiness": "UNVERIFIED",
        "task": asdict(task), "controller_config": asdict(cfg), "thresholds": asdict(limits),
        "hashes": hashes, "policy": manifest, "metrics": metrics, "failures": failures, "caveats": caveats,
        "coverage": {"episodes": 1, "qualification_episodes_required": 20, "qualification_complete": False},
    }
    replay = {
        "schema_version": "1.0", "authority": "python-mujoco", "hashes": hashes,
        "robot": {"id": robot_id, "joint_names": joint_names, "body_names": body_names, "nq": model.nq},
        "frame": {"units": "m", "up_axis": "Z", "quaternion_order": "wxyz"},
        "route": {"points_m": [[float(p[0]), float(p[1]), 0.0] for p in route], "target_id": task.id},
        "samples": samples, "events": events,
    }
    return report, replay
