"""Load a bound high-level PPO policy and adapt its commands to native evaluation."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .learning import FootprintProfile, HomeNavConfig, ReducedOrderNavEnv, load_reduced_order_ppo, _scene_fingerprint
from .planning import plan_route
from .navigation import wrap_angle

HEADING_LIMIT_RAD = .35
BRAKE_BUFFER_M = .06
PREDICTIVE_HORIZON_S = .5
GOAL_SLOWDOWN_DISTANCE_M = .6


def load_navigation_artifact(directory, scene, profile):
    directory = Path(directory)
    metadata = json.loads((directory / "reduced_order_ppo.json").read_text())
    expected = FootprintProfile.from_spec(profile)
    if metadata["scene_fingerprint"] != _scene_fingerprint(scene):
        raise ValueError("Navigation checkpoint belongs to another scene")
    if metadata["profile"] != asdict(expected):
        raise ValueError("Navigation checkpoint belongs to another robot footprint/command contract")
    path = directory / "reduced_order_ppo.pt"
    checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Navigation checkpoint hash is missing or does not match metadata")
    policy = load_reduced_order_ppo(path)
    config = HomeNavConfig(**metadata["environment"])
    return policy, config, metadata, {"algorithm": "PPO", "training_model": metadata["abstraction"],
        "checkpoint_sha256": checkpoint_hash, "scene_fingerprint": metadata["scene_fingerprint"],
        "profile": metadata["profile"], "trained": True,
        "architecture": "A_star_global_waypoints + learned_local_PPO + frozen_locomotion",
        "safety_filter": "turn_alignment + proximity_speed + predictive_footprint_braking",
        "safety_parameters": {"heading_limit_rad": HEADING_LIMIT_RAD, "brake_buffer_m": BRAKE_BUFFER_M,
                              "predictive_horizon_s": PREDICTIVE_HORIZON_S, "goal_slowdown_distance_m": GOAL_SLOWDOWN_DISTANCE_M},
        "validation_model": "native_MuJoCo_with_frozen_locomotion"}


class LearnedNavigationController:
    """Observe actual simulator pose; never assign simulator state or use a route fallback."""

    def __init__(self, scene, profile, policy, env_config, goal_xy, navigation_config):
        self.env = ReducedOrderNavEnv(scene, profile, env_config, seed=0)
        self.goal = np.asarray(goal_xy, dtype=float)
        self.env.goal = self.goal.copy()
        self.scene = scene
        self.route = None
        self.waypoint = 1
        self.policy = policy
        self.config = navigation_config
        self.current = np.zeros(3)
        self.target = np.zeros(3)
        self.next_inference = 0.0

    def command(self, xy, yaw, time_s, dt):
        if time_s < self.config.settle_s or np.linalg.norm(self.goal - xy) <= self.config.goal_tolerance_m:
            self.current[:] = 0
            self.target[:] = 0
            return self.current.copy()
        self.env.position = np.asarray(xy, dtype=float).copy()
        if self.route is None:
            self.route = np.asarray(plan_route(self.scene, xy, self.goal, radius_m=self.env.profile.radius_m,
                margin_m=self.env.config.clearance_margin_m, body_height_m=self.env.profile.body_height_m))
        while self.waypoint < len(self.route) - 1 and np.linalg.norm(self.route[self.waypoint] - xy) < self.config.lookahead_m:
            self.waypoint += 1
        self.env.goal = self.route[self.waypoint].copy()
        self.env.yaw = float(yaw)
        self.env.command = self.current.copy()
        if time_s >= self.next_inference:
            action = np.asarray(self.policy.predict(self.env.observation()), dtype=float)
            profile = self.env.profile
            self.target = action * [profile.max_linear_speed_m_s, profile.max_lateral_speed_m_s, profile.max_yaw_rate_rad_s]
            delta = self.env.goal - self.env.position
            heading_error = wrap_angle(np.arctan2(delta[1], delta[0]) - yaw)
            if abs(heading_error) > HEADING_LIMIT_RAD:
                self.target[:2] = 0
            self.target[:2] *= min(1.0, float(np.linalg.norm(self.goal - xy)) / GOAL_SLOWDOWN_DISTANCE_M)
            norm = np.linalg.norm(self.target[:2])
            if norm > self.config.speed_m_s:
                self.target[:2] *= self.config.speed_m_s / norm
            self.target[2] = np.clip(self.target[2], -self.config.max_yaw_rate, self.config.max_yaw_rate)
            self.next_inference = time_s + self.env.config.dt_s
        self.current[:2] += np.clip(self.target[:2] - self.current[:2], -self.config.acceleration_m_s2 * dt,
                                   self.config.acceleration_m_s2 * dt)
        self.current[2] = self.target[2]
        # Braking can suppress translation but never substitutes an A* velocity
        # or joint action. Learned yaw remains subject to the evaluator bounds.
        cosine, sine = np.cos(yaw), np.sin(yaw)
        velocity = np.array([cosine*self.current[0]-sine*self.current[1], sine*self.current[0]+cosine*self.current[1]])
        sweep = self.env.position + np.linspace(.05, PREDICTIVE_HORIZON_S, 10)[:, None] * velocity
        if np.any(self.env._field.clearance(sweep, self.env.profile.radius_m) < self.env.config.clearance_margin_m + BRAKE_BUFFER_M):
            self.current[:2] = 0
        return self.current.copy()
