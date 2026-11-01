"""Bounded route tracking: route geometry becomes body-frame velocity commands."""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class NavigationConfig:
    speed_m_s: float = 0.30
    yaw_gain: float = 1.2
    max_yaw_rate: float = 0.60
    lookahead_m: float = 0.35
    acceleration_m_s2: float = 0.25
    goal_tolerance_m: float = 0.25
    settle_s: float = 1.0

    def __post_init__(self):
        vals = asdict(self)
        if any(not math.isfinite(v) or v <= 0 for v in vals.values()):
            raise ValueError("Navigation parameters must be finite and positive")
        if self.speed_m_s > 0.6 or self.max_yaw_rate > 1.0:
            raise ValueError("Navigation command exceeds this demo's evaluated envelope")


def yaw_from_wxyz(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class RouteController:
    """State is per episode; slowing/arrival is separate from robot locomotion."""

    def __init__(self, route, config=None):
        self.route = np.asarray(route, dtype=float)
        if self.route.ndim != 2 or self.route.shape[1] != 2 or len(self.route) < 2:
            raise ValueError("A route requires at least two XY points")
        if not np.isfinite(self.route).all():
            raise ValueError("Route contains non-finite coordinates")
        self.config = config or NavigationConfig()
        self.index = 1
        self.speed = 0.0

    def command(self, xy, yaw, time_s, dt):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or not math.isfinite(yaw):
            raise ValueError("Robot pose must be finite XY/yaw")
        error = float(np.linalg.norm(self.route[-1] - xy))
        cfg = self.config
        if time_s < cfg.settle_s or error <= cfg.goal_tolerance_m:
            self.speed = 0.0
            return np.zeros(3)
        while self.index < len(self.route) - 1 and np.linalg.norm(self.route[self.index] - xy) < cfg.lookahead_m:
            self.index += 1
        delta = self.route[self.index] - xy
        heading_error = wrap_angle(math.atan2(delta[1], delta[0]) - yaw)
        # Turn before moving into a corridor. No lateral command for the v1 actor.
        desired_speed = min(cfg.speed_m_s, max(0.05, error * 0.5)) * max(0.0, math.cos(heading_error)) ** 2
        if abs(heading_error) > 0.8:
            desired_speed = 0.0
        self.speed += float(np.clip(desired_speed - self.speed, -cfg.acceleration_m_s2 * dt, cfg.acceleration_m_s2 * dt))
        return np.array([self.speed, 0.0, np.clip(cfg.yaw_gain * heading_error, -cfg.max_yaw_rate, cfg.max_yaw_rate)])
