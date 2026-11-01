"""Explicit kinematic reach calculation for a supplied fixed-arm MJCF.

This does not command hardware, train manipulation, or certify collision-free reach.
"""

import numpy as np
import mujoco


def solve_reach(model, data, joint_names, site_name, target_m, *, max_iterations=300, tolerance_m=0.01, damping=0.05):
    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError("Supply unique ordered arm joint names")
    if not 1 <= max_iterations <= 2000 or not 0 < tolerance_m <= 0.1 or not damping > 0:
        raise ValueError("Invalid bounded IK parameters")
    target = np.asarray(target_m, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Target must be a finite XYZ point in metres")
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site < 0:
        raise ValueError(f"Unknown end-effector site: {site_name}")
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    if any(j < 0 for j in joints):
        raise ValueError("Unknown arm joint")
    if any(int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)) for j in joints):
        raise ValueError("IK accepts only scalar hinge/slide joints")
    qadr = np.array([model.jnt_qposadr[j] for j in joints])
    dadr = np.array([model.jnt_dofadr[j] for j in joints])
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    jac = np.zeros((3, model.nv))
    jacrot = np.zeros_like(jac)
    for iteration in range(max_iterations):
        mujoco.mj_forward(model, scratch)
        error = target - scratch.site_xpos[site]
        if np.linalg.norm(error) <= tolerance_m:
            break
        mujoco.mj_jacSite(model, scratch, jac, jacrot, site)
        j = jac[:, dadr]
        delta = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(3), error)
        scratch.qpos[qadr] += np.clip(delta, -0.08, 0.08)
        for joint, adr in zip(joints, qadr):
            if model.jnt_limited[joint]:
                scratch.qpos[adr] = np.clip(scratch.qpos[adr], *model.jnt_range[joint])
    mujoco.mj_forward(model, scratch)
    error_m = float(np.linalg.norm(target - scratch.site_xpos[site]))
    return {
        "calculation": "kinematic_reach_only", "reached": error_m <= tolerance_m,
        "target_error_m": error_m, "iterations": iteration + 1,
        "joint_names": list(joint_names), "joint_positions_rad_or_m": scratch.qpos[qadr].tolist(),
        "collision_verified": False, "readiness": "PREVIEW_ONLY", "hardware_readiness": "UNVERIFIED",
    }
