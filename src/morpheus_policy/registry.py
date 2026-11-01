"""Explicit capability registry; unverified robot controllers stay unavailable."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotInfo:
    id: str
    embodiment: str
    controller: str
    availability: str
    capabilities: tuple[str, ...]
    limitation: str


ROBOTS = {
    "unitree_g1_12dof": RobotInfo(
        "unitree_g1_12dof", "humanoid", "unitree_ppo_lstm_47x12", "download_required",
        ("flat_ground_velocity_tracking", "scene_conditioned_navigation"),
        "Exact official 12-leg-DoF model only; no arm/manipulation or hardware certification.",
    ),
    "unitree_go2": RobotInfo(
        "unitree_go2", "quadruped", "robot_specific_velocity_actor", "unsupported",
        (), "A compatible Go2 checkpoint and observation/action manifest have not been supplied.",
    ),
    "fixed_arm": RobotInfo(
        "fixed_arm", "arm", "damped_least_squares_ik", "model_configuration_required",
        ("kinematic_reach_calculation",),
        "Requires explicit joint/site mapping; IK is not grasping or physics-validated manipulation.",
    ),
}

ROBOT_MODEL_PATHS = {"unitree_g1_12dof": "robot/g1_12dof.xml"}
ROBOT_SUPPORT_BODIES = {"unitree_g1_12dof": ("robot/left_ankle_roll_link", "robot/right_ankle_roll_link")}


def robot_model_path(robot_id, assets):
    if robot_id not in ROBOT_MODEL_PATHS:
        raise ValueError(f"No verified locomotion model registered for {robot_id}")
    return Path(assets) / ROBOT_MODEL_PATHS[robot_id]


def list_robots():
    return [asdict(value) for value in ROBOTS.values()]


def create_policy(robot_id, assets: Path):
    info = ROBOTS.get(robot_id)
    if info is None:
        raise ValueError(f"Unknown robot: {robot_id}")
    if robot_id != "unitree_g1_12dof":
        raise ValueError(f"UNSUPPORTED for locomotion: {robot_id}: {info.limitation}")
    from .policies.g1 import G1Policy
    return G1Policy(assets / "motion.pt", name_prefix="robot/")
