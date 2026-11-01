"""Scene-specific PPO over a reduced-order static-footprint navigation model.

This module deliberately learns only high-level body-frame velocity commands.
It does not learn gait, contacts, or hardware behaviour; a frozen embodiment
controller must validate selected routes in MuJoCo separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .planning import PlanningError, clearance, plan_route
from .scene import Scene


LEARNING_SCHEMA_VERSION = "1.0"
OBSERVATION_CONTRACT = "goal_body_xy,sin_yaw,cos_yaw,command_body,radial_clearance"
ACTION_CONTRACT = "normalized_body_vx_vy_yaw_rate"


@dataclass(frozen=True)
class FootprintProfile:
    """Embodiment-neutral high-level command and envelope contract."""

    id: str
    radius_m: float
    body_height_m: float
    max_linear_speed_m_s: float
    max_lateral_speed_m_s: float
    max_yaw_rate_rad_s: float

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("footprint profile requires an id")
        values = (self.radius_m, self.body_height_m, self.max_linear_speed_m_s,
                  self.max_lateral_speed_m_s, self.max_yaw_rate_rad_s)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("footprint profile values must be finite and positive")

    @classmethod
    def from_spec(cls, spec: "FootprintProfile | Mapping[str, Any] | Any") -> "FootprintProfile":
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, Mapping):
            values: Mapping[str, Any] = spec
        else:
            fields = ("id", "radius_m", "body_height_m", "max_linear_speed_m_s", "max_speed_m_s",
                      "max_lateral_speed_m_s", "max_yaw_rate_rad_s", "max_yaw_rate")
            values = {field: getattr(spec, field) for field in fields if hasattr(spec, field)}
        if not values:
            raise ValueError("profile specification must provide footprint fields")

        def required(*names: str) -> Any:
            for name in names:
                if name in values:
                    return values[name]
            raise ValueError(f"profile specification is missing one of: {', '.join(names)}")

        return cls(
            id=str(required("id")),
            radius_m=float(required("radius_m")),
            body_height_m=float(required("body_height_m")),
            max_linear_speed_m_s=float(required("max_linear_speed_m_s", "max_speed_m_s")),
            # Differential-drive profiles simply expose no lateral degree of
            # freedom, while holonomic profiles can opt in explicitly.
            max_lateral_speed_m_s=float(values.get("max_lateral_speed_m_s", 1e-6)),
            max_yaw_rate_rad_s=float(required("max_yaw_rate_rad_s", "max_yaw_rate")),
        )


@dataclass(frozen=True)
class HomeNavConfig:
    dt_s: float = 0.10
    horizon_steps: int = 160
    goal_tolerance_m: float = 0.25
    clearance_margin_m: float = 0.10
    ray_count: int = 16
    ray_range_m: float = 2.0
    ray_step_m: float = 0.10
    pose_noise_m: float = 0.02
    yaw_noise_rad: float = 0.04
    radius_jitter_m: float = 0.02
    max_sample_attempts: int = 2_000

    def __post_init__(self) -> None:
        if self.horizon_steps < 2 or self.ray_count < 4 or self.max_sample_attempts < 1:
            raise ValueError("home navigation configuration is too small")
        values = (self.dt_s, self.goal_tolerance_m, self.clearance_margin_m, self.ray_range_m, self.ray_step_m)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("home navigation configuration must be finite and positive")
        if any(not math.isfinite(value) or value < 0 for value in (self.pose_noise_m, self.yaw_noise_rad, self.radius_jitter_m)):
            raise ValueError("noise values must be finite and non-negative")


@dataclass(frozen=True)
class PPOTrainConfig:
    seed: int = 7
    n_envs: int = 8
    rollout_steps: int = 64
    updates: int = 16
    learning_rate: float = 1e-4
    imitation_learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 128
    hidden_size: int = 64
    train_scenarios: int = 48
    heldout_scenarios: int = 16
    curriculum_start_goal_max_m: float = 0.90
    curriculum_end_goal_max_m: float = 2.00
    imitation_epochs: int = 128
    heldout_seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_envs < 1 or self.rollout_steps < 2 or self.updates < 1 or self.epochs < 1 or self.imitation_epochs < 0:
            raise ValueError("PPO dimensions must be positive")
        if self.minibatch_size < 2 or self.hidden_size < 8 or self.train_scenarios < 1 or self.heldout_scenarios < 1:
            raise ValueError("PPO configuration is too small")
        values = (self.learning_rate, self.imitation_learning_rate, self.gamma, self.gae_lambda, self.clip_ratio,
                  self.entropy_coefficient, self.value_coefficient, self.max_grad_norm)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("PPO hyperparameters must be finite and positive")
        if not (math.isfinite(self.curriculum_start_goal_max_m) and math.isfinite(self.curriculum_end_goal_max_m)
                and 0 < self.curriculum_start_goal_max_m <= self.curriculum_end_goal_max_m):
            raise ValueError("curriculum goal distances must be finite, positive, and increasing")
        if self.heldout_seed is not None and not isinstance(self.heldout_seed, int):
            raise ValueError("heldout seed must be an integer or null")


@dataclass(frozen=True)
class NavScenario:
    start_xy: tuple[float, float]
    start_yaw: float
    goal_xy: tuple[float, float]


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    successes: int
    collisions: int
    timeouts: int
    mean_return: float
    mean_steps: float
    rollouts: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "success_rate": self.successes / self.episodes if self.episodes else 0.0,
                "rollouts": list(self.rollouts)}


@dataclass(frozen=True)
class TrainingResult:
    artifact_path: Path
    metadata_path: Path
    rollout_path: Path
    untrained: EvaluationResult
    warm_started: EvaluationResult
    trained: EvaluationResult
    astar: EvaluationResult
    update_losses: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "metadata_path": str(self.metadata_path),
            "rollout_path": str(self.rollout_path),
            "untrained": self.untrained.as_dict(),
            "warm_started": self.warm_started.as_dict(),
            "trained": self.trained.as_dict(),
            "astar": self.astar.as_dict(),
            "update_losses": list(self.update_losses),
        }


class _ClearanceField:
    """Precomputed conservative XY projections with batched clearance queries."""

    def __init__(self, scene: Scene, body_height_m: float) -> None:
        floor = scene.floors[0]
        w, x, y, z = floor.quaternion
        if abs(x) > 1e-6 or abs(y) > 1e-6:
            raise PlanningError("a tilted floor cannot provide a finite navigation support plane")
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.floor_center = np.asarray(floor.position[:2], dtype=np.float64)
        self.floor_cosine, self.floor_sine = math.cos(yaw), math.sin(yaw)
        self.floor_half = np.asarray(floor.size[:2], dtype=np.float64) / 2
        floor_top = floor.position[2] + floor.size[2] / 2
        bounds = [_projected_bounds(box) for box in scene.obstacles
                  if box.position[2] - box.size[2] / 2 < floor_top + body_height_m]
        self.bounds = np.asarray(bounds, dtype=np.float64).reshape((-1, 4)) if bounds else np.empty((0, 4))

    def clearance(self, positions: np.ndarray, radius_m: float) -> np.ndarray:
        points = np.asarray(positions, dtype=np.float64)
        if points.shape[-1] != 2:
            raise ValueError("clearance positions must end in xy")
        shape, flat = points.shape[:-1], points.reshape(-1, 2)
        offset = flat - self.floor_center
        local_x = self.floor_cosine * offset[:, 0] + self.floor_sine * offset[:, 1]
        local_y = -self.floor_sine * offset[:, 0] + self.floor_cosine * offset[:, 1]
        outside_x = np.maximum(np.abs(local_x) - self.floor_half[0], 0.0)
        outside_y = np.maximum(np.abs(local_y) - self.floor_half[1], 0.0)
        inside = np.minimum(self.floor_half[0] - np.abs(local_x), self.floor_half[1] - np.abs(local_y))
        available = np.where((outside_x > 0) | (outside_y > 0), -np.hypot(outside_x, outside_y), inside) - radius_m
        if len(self.bounds):
            lower, upper = self.bounds[:, :2], self.bounds[:, 2:]
            outside = np.maximum(np.maximum(lower[None, :, :] - flat[:, None, :], 0),
                                 np.maximum(flat[:, None, :] - upper[None, :, :], 0))
            euclidean = np.hypot(outside[:, :, 0], outside[:, :, 1])
            inside_distance = np.minimum.reduce((flat[:, None, 0] - lower[None, :, 0], upper[None, :, 0] - flat[:, None, 0],
                                                 flat[:, None, 1] - lower[None, :, 1], upper[None, :, 1] - flat[:, None, 1]))
            signed = np.where((outside[:, :, 0] > 0) | (outside[:, :, 1] > 0), euclidean, -inside_distance) - radius_m
            available = np.minimum(available, signed.min(axis=1))
        return available.reshape(shape)


class ReducedOrderNavEnv:
    """Static supplied-scene footprint dynamics for high-level PPO only."""

    def __init__(self, scene: Scene, profile: FootprintProfile | Mapping[str, Any], config: HomeNavConfig | None = None,
                 seed: int = 0) -> None:
        self.scene = scene
        self.profile = FootprintProfile.from_spec(profile)
        self.config = config or HomeNavConfig()
        self.rng = np.random.default_rng(seed)
        self.position = np.zeros(2, dtype=np.float64)
        self.yaw = 0.0
        self.goal = np.zeros(2, dtype=np.float64)
        self.command = np.zeros(3, dtype=np.float64)
        self.steps = 0
        self._radius = self.profile.radius_m
        self._field = _ClearanceField(scene, self.profile.body_height_m)

    @property
    def observation_dim(self) -> int:
        return 7 + self.config.ray_count

    @property
    def action_dim(self) -> int:
        return 3

    def reset(self, scenario: NavScenario | None = None, *, training: bool = False) -> np.ndarray:
        self._radius = self.profile.radius_m + (self.rng.uniform(-self.config.radius_jitter_m, self.config.radius_jitter_m) if training else 0.0)
        self._radius = max(1e-4, self._radius)
        if scenario is None:
            scenario = sample_scenarios(self.scene, self.profile, self.config, 1, int(self.rng.integers(0, 2**31 - 1)))[0]
        self.position = np.asarray(scenario.start_xy, dtype=np.float64).copy()
        self.yaw = float(scenario.start_yaw)
        self.goal = np.asarray(scenario.goal_xy, dtype=np.float64).copy()
        if training:
            self.position += self.rng.normal(0.0, self.config.pose_noise_m, size=2)
            self.yaw += float(self.rng.normal(0.0, self.config.yaw_noise_rad))
            if self._clearance(self.position) <= self.config.clearance_margin_m:
                self.position = np.asarray(scenario.start_xy, dtype=np.float64).copy()
        self.command.fill(0.0)
        self.steps = 0
        return self.observation()

    def observation(self) -> np.ndarray:
        delta_world = self.goal - self.position
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        goal_body = np.array((cosine * delta_world[0] + sine * delta_world[1],
                              -sine * delta_world[0] + cosine * delta_world[1]), dtype=np.float32)
        scale = max(self.config.ray_range_m, 1e-6)
        command_scale = np.array((self.profile.max_linear_speed_m_s, self.profile.max_lateral_speed_m_s,
                                  self.profile.max_yaw_rate_rad_s), dtype=np.float32)
        values = np.concatenate((
            np.clip(goal_body / scale, -2.0, 2.0),
            np.array((math.sin(self.yaw), math.cos(self.yaw)), dtype=np.float32),
            np.asarray(self.command, dtype=np.float32) / command_scale,
            self._rays(),
        ))
        return values.astype(np.float32, copy=False)

    def step(self, normalized_action: Iterable[float]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        action = np.asarray(tuple(normalized_action), dtype=np.float64)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("reduced-order PPO action must be three finite values")
        action = np.clip(action, -1.0, 1.0)
        command = action * np.array((self.profile.max_linear_speed_m_s, self.profile.max_lateral_speed_m_s,
                                     self.profile.max_yaw_rate_rad_s))
        previous_distance = float(np.linalg.norm(self.goal - self.position))
        self.yaw = _wrap_angle(self.yaw + command[2] * self.config.dt_s)
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        world_velocity = np.array((cosine * command[0] - sine * command[1],
                                   sine * command[0] + cosine * command[1]))
        proposed = self.position + world_velocity * self.config.dt_s
        self.command = command
        self.steps += 1
        # Test the swept disc, not merely its endpoint: an action must not
        # tunnel through an obstacle thinner than speed * dt.
        travel = float(np.linalg.norm(proposed - self.position))
        samples = max(1, math.ceil(travel / max(self.config.ray_step_m / 2.0, 1e-3)))
        sweep = self.position[None, :] + np.linspace(1.0 / samples, 1.0, samples)[:, None] * (proposed - self.position)
        collision = bool(np.any(self._field.clearance(sweep, self._radius) <= self.config.clearance_margin_m))
        if not collision:
            self.position = proposed
        distance = float(np.linalg.norm(self.goal - self.position))
        success = distance <= self.config.goal_tolerance_m
        timeout = self.steps >= self.config.horizon_steps
        done = collision or success or timeout
        margin = self._clearance(self.position)
        progress = previous_distance - distance
        clearance_penalty = max(0.0, self.config.clearance_margin_m - margin)
        reward = 4.0 * progress - 0.01 - 2.0 * clearance_penalty - 0.002 * float(np.square(action).sum())
        if collision:
            reward -= 5.0
        if success:
            reward += 8.0
        return self.observation(), float(reward), done, {
            "success": success,
            "collision": collision,
            "timeout": timeout and not success,
            "distance_m": distance,
            "clearance_m": margin,
            "position": self.position.tolist(),
            "yaw_rad": self.yaw,
            "command": self.command.tolist(),
        }

    def _clearance(self, position: np.ndarray) -> float:
        return float(self._field.clearance(np.asarray(position, dtype=np.float64), self._radius))

    def _rays(self) -> np.ndarray:
        candidate = np.arange(self.config.ray_step_m, self.config.ray_range_m + 1e-9, self.config.ray_step_m)
        angles = self.yaw + 2.0 * math.pi * np.arange(self.config.ray_count) / self.config.ray_count
        direction = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        points = self.position[None, None, :] + direction[:, None, :] * candidate[None, :, None]
        blocked = self._field.clearance(points, self._radius) <= self.config.clearance_margin_m
        first = blocked.argmax(axis=1)
        distance = np.where(blocked.any(axis=1), candidate[first], self.config.ray_range_m)
        return (distance / self.config.ray_range_m).astype(np.float32)


class ObservationNormalizer:
    def __init__(self, dimension: int) -> None:
        self.count = 1e-4
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.variance = np.ones(dimension, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.mean.shape[0]:
            raise ValueError("observation batch has an incompatible dimension")
        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_variance = batch.var(axis=0)
        delta = batch_mean - self.mean
        combined = self.count + batch_count
        self.mean += delta * batch_count / combined
        aggregate = self.variance * self.count + batch_variance * batch_count + np.square(delta) * self.count * batch_count / combined
        self.variance = aggregate / combined
        self.count = combined

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(values, dtype=np.float32) - self.mean) / np.sqrt(self.variance + 1e-8), -10.0, 10.0).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"count": self.count, "mean": self.mean.tolist(), "variance": self.variance.tolist()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ObservationNormalizer":
        normalizer = cls(len(state["mean"]))
        normalizer.count = float(state["count"])
        normalizer.mean = np.asarray(state["mean"], dtype=np.float64)
        normalizer.variance = np.asarray(state["variance"], dtype=np.float64)
        return normalizer


class PPOActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(observation_dim, hidden_size), nn.Tanh(),
                                      nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, observation: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        hidden = self.backbone(observation)
        return torch.distributions.Normal(self.actor_mean(hidden), self.log_std.exp()), self.critic(hidden).squeeze(-1)


class ReducedOrderPPOPolicy:
    def __init__(self, network: PPOActorCritic, normalizer: ObservationNormalizer) -> None:
        self.network = network.eval()
        self.normalizer = normalizer

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        normalized = self.normalizer.normalize(np.asarray(observation, dtype=np.float32)[None, :])
        with torch.no_grad():
            distribution, _ = self.network(torch.from_numpy(normalized))
            action = distribution.mean if deterministic else distribution.sample()
        return np.clip(action.numpy()[0], -1.0, 1.0).astype(np.float32)


class ReducedOrderNavigationController:
    """Turn a validated PPO observation into bounded velocity commands.

    This adapter does not set MuJoCo state or perform contact control.  A
    caller must feed the resulting command to its embodiment controller and
    retain independent collision/fall evaluation.
    """

    def __init__(self, scene: Scene, profile: FootprintProfile | Mapping[str, Any], policy: ReducedOrderPPOPolicy,
                 goal_xy: Sequence[float], config: HomeNavConfig | None = None,
                 *, max_linear_accel_m_s2: float = 0.60, max_yaw_accel_rad_s2: float = 1.20) -> None:
        self.profile = FootprintProfile.from_spec(profile)
        self.policy = policy
        self.environment = ReducedOrderNavEnv(scene, self.profile, config, seed=0)
        goal = np.asarray(goal_xy, dtype=np.float64)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("controller goal must be a finite xy pair")
        if not (math.isfinite(max_linear_accel_m_s2) and max_linear_accel_m_s2 > 0
                and math.isfinite(max_yaw_accel_rad_s2) and max_yaw_accel_rad_s2 > 0):
            raise ValueError("controller accelerations must be finite and positive")
        self.environment.goal = goal
        self.max_linear_accel_m_s2 = max_linear_accel_m_s2
        self.max_yaw_accel_rad_s2 = max_yaw_accel_rad_s2
        self._last_command = np.zeros(3, dtype=np.float64)

    def set_goal(self, goal_xy: Sequence[float]) -> None:
        goal = np.asarray(goal_xy, dtype=np.float64)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("controller goal must be a finite xy pair")
        self.environment.goal = goal

    def command(self, xy: Sequence[float], yaw_rad: float, time_s: float, dt_s: float) -> np.ndarray:
        position = np.asarray(xy, dtype=np.float64)
        if position.shape != (2,) or not np.isfinite(position).all() or not math.isfinite(yaw_rad):
            raise ValueError("controller pose must be finite")
        if not (math.isfinite(time_s) and math.isfinite(dt_s) and dt_s > 0):
            raise ValueError("controller time and dt must be finite, with dt positive")
        self.environment.position = position
        self.environment.yaw = _wrap_angle(float(yaw_rad))
        self.environment.command = self._last_command.copy()
        action = self.policy.predict(self.environment.observation())
        target = np.asarray(action, dtype=np.float64) * np.array((self.profile.max_linear_speed_m_s,
                                                                   self.profile.max_lateral_speed_m_s,
                                                                   self.profile.max_yaw_rate_rad_s))
        speed = float(np.linalg.norm(target[:2]))
        if speed > self.profile.max_linear_speed_m_s:
            target[:2] *= self.profile.max_linear_speed_m_s / speed
        linear_delta = target[:2] - self._last_command[:2]
        maximum_delta = self.max_linear_accel_m_s2 * dt_s
        delta_norm = float(np.linalg.norm(linear_delta))
        if delta_norm > maximum_delta:
            linear_delta *= maximum_delta / delta_norm
        yaw_delta = float(np.clip(target[2] - self._last_command[2], -self.max_yaw_accel_rad_s2 * dt_s,
                                  self.max_yaw_accel_rad_s2 * dt_s))
        self._last_command = self._last_command + np.array((linear_delta[0], linear_delta[1], yaw_delta))
        return self._last_command.astype(np.float64, copy=True)


def sample_scenarios(scene: Scene, profile: FootprintProfile | Mapping[str, Any], config: HomeNavConfig,
                     count: int, seed: int, *, min_goal_distance_m: float | None = None,
                     max_goal_distance_m: float | None = None) -> tuple[NavScenario, ...]:
    profile = FootprintProfile.from_spec(profile)
    if count < 1:
        raise ValueError("scenario count must be positive")
    minimum_distance = max(0.75, 3 * config.goal_tolerance_m) if min_goal_distance_m is None else float(min_goal_distance_m)
    if not math.isfinite(minimum_distance) or minimum_distance <= 0:
        raise ValueError("minimum goal distance must be finite and positive")
    if max_goal_distance_m is not None and (not math.isfinite(max_goal_distance_m) or max_goal_distance_m < minimum_distance):
        raise ValueError("maximum goal distance must be finite and no smaller than the minimum")
    rng = np.random.default_rng(seed)
    minimum_x, minimum_y, maximum_x, maximum_y = scene.bounds_xy
    scenarios: list[NavScenario] = []
    for _ in range(count):
        for _attempt in range(config.max_sample_attempts):
            start = (float(rng.uniform(minimum_x, maximum_x)), float(rng.uniform(minimum_y, maximum_y)))
            goal = (float(rng.uniform(minimum_x, maximum_x)), float(rng.uniform(minimum_y, maximum_y)))
            distance = math.dist(start, goal)
            if distance < minimum_distance or (max_goal_distance_m is not None and distance > max_goal_distance_m):
                continue
            try:
                plan_route(scene, start, goal, radius_m=profile.radius_m, margin_m=config.clearance_margin_m,
                           body_height_m=profile.body_height_m)
            except PlanningError:
                continue
            # Both train and held-out samples begin approximately goal-facing.
            # This supports bounded local-goal navigation evaluation, not
            # arbitrary-heading or recovery claims.
            heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
            scenarios.append(NavScenario(start, _wrap_angle(heading + float(rng.normal(0.0, 0.35))), goal))
            break
        else:
            raise RuntimeError("could not sample a free, connected training scenario from the supplied scene")
    return tuple(scenarios)


def train_scene_specific_ppo(scene: Scene, profile: FootprintProfile | Mapping[str, Any], output_dir: str | Path,
                             *, env_config: HomeNavConfig | None = None,
                             train_config: PPOTrainConfig | None = None) -> TrainingResult:
    """Train and serialize only a reduced-order PPO navigation policy."""
    profile = FootprintProfile.from_spec(profile)
    env_config = env_config or HomeNavConfig()
    train_config = train_config or PPOTrainConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    train_scenarios = sample_scenarios(scene, profile, env_config, train_config.train_scenarios, train_config.seed,
                                       max_goal_distance_m=train_config.curriculum_end_goal_max_m)
    heldout_seed = train_config.seed + 1 if train_config.heldout_seed is None else train_config.heldout_seed
    heldout_scenarios = sample_scenarios(scene, profile, env_config, train_config.heldout_scenarios, heldout_seed,
                                         max_goal_distance_m=train_config.curriculum_end_goal_max_m)
    while {_scenario_key(item) for item in train_scenarios} & {_scenario_key(item) for item in heldout_scenarios}:
        heldout_seed += 1
        heldout_scenarios = sample_scenarios(scene, profile, env_config, train_config.heldout_scenarios, heldout_seed,
                                             max_goal_distance_m=train_config.curriculum_end_goal_max_m)
    environments = [ReducedOrderNavEnv(scene, profile, env_config, train_config.seed + 100 + index) for index in range(train_config.n_envs)]
    initial_scenarios = _curriculum_scenarios(train_scenarios, train_config, 0)
    observations = np.stack([env.reset(initial_scenarios[index % len(initial_scenarios)], training=True) for index, env in enumerate(environments)])
    normalizer = ObservationNormalizer(environments[0].observation_dim)
    network = PPOActorCritic(environments[0].observation_dim, environments[0].action_dim, train_config.hidden_size)
    policy = ReducedOrderPPOPolicy(network, normalizer)
    untrained = evaluate_policy(scene, profile, policy, heldout_scenarios, env_config)
    teacher_observations, teacher_actions = _teacher_dataset(scene, profile, env_config, train_scenarios)
    normalizer.update(teacher_observations)
    imitation_optimizer = torch.optim.Adam(network.parameters(), lr=train_config.imitation_learning_rate)
    imitation_loss = _warm_start_from_routes(network, imitation_optimizer, normalizer, teacher_observations, teacher_actions, train_config)
    warm_started = evaluate_policy(scene, profile, ReducedOrderPPOPolicy(network, normalizer), heldout_scenarios, env_config)
    optimizer = torch.optim.Adam(network.parameters(), lr=train_config.learning_rate)
    update_losses: list[float] = []
    generator = np.random.default_rng(train_config.seed + 2)

    for _update in range(train_config.updates):
        # Rollout collection and its PPO update must use the same snapshot.
        # Incorporate freshly reset observations only on the next update.
        normalizer.update(observations)
        active_scenarios = _curriculum_scenarios(train_scenarios, train_config, _update)
        rollout = _collect_rollout(network, normalizer, environments, observations, active_scenarios, env_config, generator,
                                   train_config.rollout_steps)
        observations = rollout.pop("next_observations")
        update_losses.append(_ppo_update(network, optimizer, rollout, normalizer, train_config, generator))

    trained = evaluate_policy(scene, profile, ReducedOrderPPOPolicy(network, normalizer), heldout_scenarios, env_config)
    astar = evaluate_astar_baseline(scene, profile, heldout_scenarios, env_config)
    artifact_path = output / "reduced_order_ppo.pt"
    metadata_path = output / "reduced_order_ppo.json"
    rollout_path = output / "reduced_order_ppo_rollouts.json"
    payload = {"schema_version": LEARNING_SCHEMA_VERSION, "network_state": network.state_dict(),
               "normalizer": normalizer.state_dict(), "observation_dim": environments[0].observation_dim,
               "action_dim": environments[0].action_dim, "hidden_size": train_config.hidden_size}
    torch.save(payload, artifact_path)
    metadata = _metadata(scene, profile, env_config, train_config, train_scenarios, heldout_scenarios, untrained, warm_started,
                         trained, astar, update_losses, _sha256_file(artifact_path), imitation_loss, len(teacher_observations), heldout_seed)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rollout_path.write_text(json.dumps({"schema_version": LEARNING_SCHEMA_VERSION, "abstraction": "reduced_order_static_footprint",
                                        "trained": trained.as_dict(), "warm_started": warm_started.as_dict(), "untrained": untrained.as_dict(),
                                        "astar": astar.as_dict()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TrainingResult(artifact_path, metadata_path, rollout_path, untrained, warm_started, trained, astar, tuple(update_losses))


def load_reduced_order_ppo(path: str | Path, *, expected_scene_fingerprint: str | None = None,
                           expected_profile: FootprintProfile | Mapping[str, Any] | None = None) -> ReducedOrderPPOPolicy:
    """Load a locally-created artifact after binding it to its JSON contract."""
    artifact = Path(path)
    metadata_path = artifact.with_suffix(".json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("reduced-order PPO metadata is required beside the checkpoint") from error
    if metadata.get("schema_version") != LEARNING_SCHEMA_VERSION or metadata.get("checkpoint_sha256") != _sha256_file(artifact):
        raise ValueError("checkpoint does not match its reduced-order PPO metadata")
    if expected_scene_fingerprint is not None and metadata.get("scene_fingerprint") != expected_scene_fingerprint:
        raise ValueError("checkpoint was trained for a different scene fingerprint")
    if expected_profile is not None and metadata.get("profile") != asdict(FootprintProfile.from_spec(expected_profile)):
        raise ValueError("checkpoint was trained for a different footprint profile")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != LEARNING_SCHEMA_VERSION or payload.get("action_dim") != 3:
        raise ValueError("unsupported reduced-order PPO artifact")
    network = PPOActorCritic(int(payload["observation_dim"]), int(payload["action_dim"]), int(payload["hidden_size"]))
    network.load_state_dict(payload["network_state"])
    return ReducedOrderPPOPolicy(network, ObservationNormalizer.from_state_dict(payload["normalizer"]))


def evaluate_policy(scene: Scene, profile: FootprintProfile | Mapping[str, Any], policy: ReducedOrderPPOPolicy,
                    scenarios: Sequence[NavScenario], config: HomeNavConfig | None = None) -> EvaluationResult:
    env = ReducedOrderNavEnv(scene, profile, config, seed=0)
    rollouts: list[dict[str, Any]] = []
    for scenario in scenarios:
        observation, total = env.reset(scenario), 0.0
        for _ in range(env.config.horizon_steps):
            observation, reward, done, info = env.step(policy.predict(observation))
            total += reward
            if done:
                break
        rollouts.append({"scenario": asdict(scenario), "return": total, "steps": env.steps, **info})
    return _evaluation_result(rollouts)


def evaluate_astar_baseline(scene: Scene, profile: FootprintProfile | Mapping[str, Any], scenarios: Sequence[NavScenario],
                            config: HomeNavConfig | None = None) -> EvaluationResult:
    profile = FootprintProfile.from_spec(profile)
    env = ReducedOrderNavEnv(scene, profile, config, seed=0)
    rollouts: list[dict[str, Any]] = []
    for scenario in scenarios:
        route = plan_route(scene, scenario.start_xy, scenario.goal_xy, radius_m=profile.radius_m,
                           margin_m=env.config.clearance_margin_m, body_height_m=profile.body_height_m)
        observation, total, index = env.reset(scenario), 0.0, 1
        for _ in range(env.config.horizon_steps):
            while index < len(route) - 1 and math.dist(env.position, route[index]) < env.config.goal_tolerance_m:
                index += 1
            action = _route_action(env, np.asarray(route[index], dtype=np.float64))
            observation, reward, done, info = env.step(action)
            total += reward
            if done:
                break
        rollouts.append({"scenario": asdict(scenario), "return": total, "steps": env.steps, **info})
    return _evaluation_result(rollouts)


def _route_action(environment: ReducedOrderNavEnv, target: np.ndarray) -> np.ndarray:
    delta = target - environment.position
    heading = math.atan2(delta[1], delta[0])
    yaw_error = _wrap_angle(heading - environment.yaw)
    return np.array((max(0.0, math.cos(yaw_error)), 0.0,
                     np.clip(1.5 * yaw_error / environment.profile.max_yaw_rate_rad_s, -1, 1)), dtype=np.float32)


def _collect_rollout(network: PPOActorCritic, normalizer: ObservationNormalizer, environments: list[ReducedOrderNavEnv],
                     observations: np.ndarray, scenarios: Sequence[NavScenario], config: HomeNavConfig,
                     rng: np.random.Generator, steps: int) -> dict[str, np.ndarray]:
    raw_observations: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    log_probs: list[np.ndarray] = []
    values: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    dones: list[np.ndarray] = []
    current = observations
    for _ in range(steps):
        normalized = normalizer.normalize(current)
        with torch.no_grad():
            distribution, value = network(torch.from_numpy(normalized))
            raw_action = distribution.sample()
            action = torch.tanh(raw_action)
            log_probability = distribution.log_prob(raw_action).sum(-1) - torch.log(1 - action.square() + 1e-6).sum(-1)
        next_observations, step_rewards, step_dones = [], [], []
        for index, env in enumerate(environments):
            next_observation, reward, done, _info = env.step(action.numpy()[index])
            if done:
                next_observation = env.reset(scenarios[int(rng.integers(len(scenarios)))], training=True)
            next_observations.append(next_observation)
            step_rewards.append(reward)
            step_dones.append(done)
        raw_observations.append(current.copy())
        raw_actions.append(raw_action.numpy())
        actions.append(action.numpy())
        log_probs.append(log_probability.numpy())
        values.append(value.numpy())
        rewards.append(np.asarray(step_rewards, dtype=np.float32))
        dones.append(np.asarray(step_dones, dtype=np.float32))
        current = np.stack(next_observations)
    with torch.no_grad():
        _, last_values = network(torch.from_numpy(normalizer.normalize(current)))
    return {"raw_observations": np.stack(raw_observations), "raw_actions": np.stack(raw_actions), "actions": np.stack(actions), "log_probs": np.stack(log_probs),
            "values": np.stack(values), "rewards": np.stack(rewards), "dones": np.stack(dones),
            "last_values": last_values.numpy(), "next_observations": current}


def _teacher_dataset(scene: Scene, profile: FootprintProfile, config: HomeNavConfig,
                     scenarios: Sequence[NavScenario]) -> tuple[np.ndarray, np.ndarray]:
    """Replay bounded A* paths as local velocity-action labels for warm start."""
    environment = ReducedOrderNavEnv(scene, profile, config, seed=0)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for scenario in scenarios:
        route = plan_route(scene, scenario.start_xy, scenario.goal_xy, radius_m=profile.radius_m,
                           margin_m=config.clearance_margin_m, body_height_m=profile.body_height_m)
        observation, route_index = environment.reset(scenario), 1
        for _ in range(config.horizon_steps):
            while route_index < len(route) - 1 and math.dist(environment.position, route[route_index]) < config.goal_tolerance_m:
                route_index += 1
            action = _route_action(environment, np.asarray(route[route_index], dtype=np.float64))
            observations.append(observation)
            actions.append(action)
            observation, _reward, done, _info = environment.step(action)
            if done:
                break
    if not observations:
        raise RuntimeError("A* produced no valid warm-start trajectories")
    return np.stack(observations), np.stack(actions)


def _warm_start_from_routes(network: PPOActorCritic, optimizer: torch.optim.Optimizer, normalizer: ObservationNormalizer,
                            observations: np.ndarray, actions: np.ndarray, config: PPOTrainConfig) -> float | None:
    if config.imitation_epochs == 0:
        return None
    normalized = torch.from_numpy(normalizer.normalize(observations))
    targets = torch.from_numpy(actions.astype(np.float32))
    losses: list[float] = []
    network.train()
    for _ in range(config.imitation_epochs):
        _distribution, _value = network(normalized)
        prediction = torch.tanh(_distribution.loc)
        loss = torch.nn.functional.mse_loss(prediction, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach()))
    network.eval()
    return float(np.mean(losses))


def _ppo_update(network: PPOActorCritic, optimizer: torch.optim.Optimizer, rollout: Mapping[str, np.ndarray],
                normalizer: ObservationNormalizer, config: PPOTrainConfig, rng: np.random.Generator) -> float:
    rewards, values, dones = rollout["rewards"], rollout["values"], rollout["dones"]
    advantages = np.zeros_like(rewards)
    last_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    next_values = rollout["last_values"]
    for index in range(rewards.shape[0] - 1, -1, -1):
        mask = 1.0 - dones[index]
        delta = rewards[index] + config.gamma * next_values * mask - values[index]
        last_advantage = delta + config.gamma * config.gae_lambda * mask * last_advantage
        advantages[index] = last_advantage
        next_values = values[index]
    returns = advantages + values
    observation = normalizer.normalize(rollout["raw_observations"].reshape(-1, rollout["raw_observations"].shape[-1]))
    actions = rollout["actions"].reshape(-1, 3)
    raw_actions = rollout["raw_actions"].reshape(-1, 3)
    old_log_probs = rollout["log_probs"].reshape(-1)
    advantages = advantages.reshape(-1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = returns.reshape(-1)
    indices = np.arange(len(returns))
    losses: list[float] = []
    network.train()
    for _ in range(config.epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), config.minibatch_size):
            batch = indices[start:start + config.minibatch_size]
            distribution, value = network(torch.from_numpy(observation[batch]))
            bounded = torch.from_numpy(actions[batch])
            raw = torch.from_numpy(raw_actions[batch])
            new_log_prob = distribution.log_prob(raw).sum(-1) - torch.log(1 - bounded.square() + 1e-6).sum(-1)
            ratio = torch.exp(new_log_prob - torch.from_numpy(old_log_probs[batch]))
            advantage = torch.from_numpy(advantages[batch])
            policy_loss = -torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * advantage).mean()
            value_loss = torch.nn.functional.mse_loss(value, torch.from_numpy(returns[batch]))
            entropy = distribution.entropy().sum(-1).mean()
            loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
    network.eval()
    return float(np.mean(losses))


def _evaluation_result(rollouts: Sequence[dict[str, Any]]) -> EvaluationResult:
    return EvaluationResult(
        episodes=len(rollouts), successes=sum(bool(item["success"]) for item in rollouts),
        collisions=sum(bool(item["collision"]) for item in rollouts), timeouts=sum(bool(item["timeout"]) for item in rollouts),
        mean_return=float(np.mean([item["return"] for item in rollouts])), mean_steps=float(np.mean([item["steps"] for item in rollouts])),
        rollouts=tuple(rollouts),
    )


def _metadata(scene: Scene, profile: FootprintProfile, env_config: HomeNavConfig, train_config: PPOTrainConfig,
              train_scenarios: Sequence[NavScenario], heldout_scenarios: Sequence[NavScenario], untrained: EvaluationResult,
              warm_started: EvaluationResult, trained: EvaluationResult, astar: EvaluationResult, losses: Sequence[float],
              checkpoint_sha256: str, imitation_loss: float | None, imitation_samples: int, heldout_seed: int) -> dict[str, Any]:
    return {
        "schema_version": LEARNING_SCHEMA_VERSION,
        "algorithm": "PPO",
        "checkpoint_sha256": checkpoint_sha256,
        "abstraction": "reduced_order_static_footprint",
        "limitation": "Not a gait, contact, MuJoCo-physics, sim-to-real, or hardware-readiness policy.",
        "input_caveats": list(scene.caveats),
        "evidence_scope": "Collision-proxy navigation only; this does not establish correspondence to a visual splat or supported physical floor.",
        "scene_fingerprint": _scene_fingerprint(scene), "profile": asdict(profile), "environment": asdict(env_config),
        "training": asdict(train_config), "heldout_seed": heldout_seed,
        "scenario_distribution": {
            "same_scene": True,
            "goal_distance_min_m": max(0.75, 3 * env_config.goal_tolerance_m),
            "goal_distance_max_m": train_config.curriculum_end_goal_max_m,
            "initial_heading_noise_std_rad": 0.35,
            "untested": ["arbitrary headings", "new rooms", "dynamic obstacles", "obstacle-detour learning"],
        },
        "observation_contract": {"name": OBSERVATION_CONTRACT, "dimension": 7 + env_config.ray_count},
        "warm_start": {"method": "A*-route local velocity imitation before PPO", "epochs": train_config.imitation_epochs,
                       "samples": imitation_samples, "mean_mse": imitation_loss},
        "action_contract": {"name": ACTION_CONTRACT, "dimension": 3, "bounds": [-1.0, 1.0]},
        "splits": {"train_scenarios": [asdict(item) for item in train_scenarios], "heldout_scenarios": [asdict(item) for item in heldout_scenarios]},
        "evaluation": {"untrained": untrained.as_dict(), "warm_started": warm_started.as_dict(),
                       "trained": trained.as_dict(), "astar_baseline": astar.as_dict()},
        "update_losses": list(losses),
    }


def _scene_fingerprint(scene: Scene) -> str:
    payload = {"name": scene.name, "hashes": dict(scene.hashes), "objects": [asdict(item) for item in scene.objects]}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _projected_bounds(box: Any) -> tuple[float, float, float, float]:
    """XY AABB of every rotated 3-D box corner, matching planner conservatism."""
    w, x, y, z = box.quaternion
    rotation = np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                         (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                         (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))), dtype=np.float64)
    half = np.asarray(box.size, dtype=np.float64) / 2
    corners = np.array([[sx, sy, sz] for sx in (-half[0], half[0]) for sy in (-half[1], half[1]) for sz in (-half[2], half[2])])
    projected = corners @ rotation[:2, :].T + np.asarray(box.position[:2], dtype=np.float64)
    return float(projected[:, 0].min()), float(projected[:, 1].min()), float(projected[:, 0].max()), float(projected[:, 1].max())


def _curriculum_scenarios(scenarios: Sequence[NavScenario], config: PPOTrainConfig, update: int) -> tuple[NavScenario, ...]:
    fraction = update / max(config.updates - 1, 1)
    maximum = config.curriculum_start_goal_max_m + fraction * (config.curriculum_end_goal_max_m - config.curriculum_start_goal_max_m)
    active = tuple(item for item in scenarios if math.dist(item.start_xy, item.goal_xy) <= maximum)
    # Sampling ensures at least one entry normally.  The fallback maintains a
    # bounded, deterministic training run for unusually sparse scenes.
    return active or (min(scenarios, key=lambda item: math.dist(item.start_xy, item.goal_xy)),)


def _scenario_key(item: NavScenario) -> tuple[float, float, float, float, float]:
    return tuple(round(value, 8) for value in (*item.start_xy, item.start_yaw, *item.goal_xy))


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))
