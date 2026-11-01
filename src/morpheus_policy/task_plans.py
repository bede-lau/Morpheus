"""Bounded numeric task plans over an upstream-owned static scene.

This module deliberately separates safe base-route calculation from cleaning
intent.  A route is checked by :mod:`planning`; a cleaning sweep is an
explicit user-target overlay and never a manipulation or collision-execution
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

from .planning import PlanningError, clearance, plan_route
from .scene import Scene


MAX_ROUTE_OPTIONS = 3
MAX_VIA_CANDIDATES = 12
MAX_SWEEP_LANES = 16

RouteStatus = Literal["planned"]
CleaningStatus = Literal["not_requested", "planned_tool_unverified", "unsupported"]


@dataclass(frozen=True)
class RouteOption:
    """A collision-checked numeric route, not a locomotion execution."""

    id: str
    label: str
    kind: Literal["planned"]
    points_m: tuple[tuple[float, float, float], ...]
    length_m: float
    clearance_m: float
    status: RouteStatus
    task: Literal["inspect_passage"] = "inspect_passage"
    note: str = "Numeric route planning only; locomotion and task execution are not implied."


@dataclass(frozen=True)
class CleaningPatch:
    """An explicit planar user target for an unverified cleaning preview."""

    center_xy_m: tuple[float, float]
    size_xy_m: tuple[float, float]
    height_m: float
    label: str = "user cleaning patch"


@dataclass(frozen=True)
class CleaningIntent:
    """A capability-labelled cleaning intent; never a command to hardware."""

    task: Literal["inspect_passage", "cleaning_preview"]
    robot_id: str
    action: str
    capability: str | None
    status: CleaningStatus
    base_motion: bool
    sweep_points_m: tuple[tuple[float, float, float], ...]
    synthetic_numerical_preview: bool
    note: str


_CLEANING_CAPABILITIES = {
    "stretch": frozenset({"wipe", "sweep"}),
    "soarm": frozenset({"wipe", "sweep"}),
    "go2": frozenset(),
    "unitree_go2": frozenset(),
    "g1": frozenset(),
    "unitree_g1_12dof": frozenset(),
}


def plan_navigation_options(
    scene: Scene,
    start_xy: Iterable[float],
    goal_xy: Iterable[float],
    *,
    radius_m: float,
    body_height_m: float,
    margin_m: float = 0.10,
    resolution_m: float = 0.10,
    max_routes: int = MAX_ROUTE_OPTIONS,
) -> tuple[RouteOption, ...]:
    """Return at most three distinct, collision-checked routes to one goal.

    The primary is the existing conservative A* route.  Alternatives must
    pass through one bounded set of clear lateral waypoints and each leg is
    independently checked by the same planner.  Returning fewer than three
    routes means no other sufficiently distinct safe numeric route was found.
    """
    if not isinstance(max_routes, int) or not 1 <= max_routes <= MAX_ROUTE_OPTIONS:
        raise ValueError(f"max_routes must be an integer from 1 to {MAX_ROUTE_OPTIONS}")
    start = _xy(start_xy, "start_xy")
    goal = _xy(goal_xy, "goal_xy")
    radius = _positive(radius_m, "radius_m")
    height = _positive(body_height_m, "body_height_m")
    margin = _nonnegative(margin_m, "margin_m")
    resolution = _positive(resolution_m, "resolution_m")

    primary = plan_route(scene, start, goal, radius, margin, resolution, height)
    options = [_route_option("route-1", "Primary route", scene, primary, radius, height)]
    if max_routes == 1 or start == goal:
        return tuple(options)

    envelope = radius + margin
    separation = max(resolution * 2, 0.15)
    for via in _via_candidates(start, goal):
        if len(options) >= max_routes:
            break
        if clearance(scene, via, envelope, height) <= 0:
            continue
        try:
            first = plan_route(scene, start, via, radius, margin, resolution, height)
            second = plan_route(scene, via, goal, radius, margin, resolution, height)
        except PlanningError:
            continue
        candidate = _deduplicate_xy([*first, *second])
        if len(candidate) < 2 or not _is_distinct(candidate, [option.points_m for option in options], separation):
            continue
        route_id = f"route-{len(options) + 1}"
        options.append(_route_option(route_id, f"Alternative {len(options)}", scene, candidate, radius, height))
    return tuple(options)


def plan_cleaning_intent(
    robot_id: str,
    patch: CleaningPatch | None = None,
    *,
    action: str = "wipe",
    capability: str | None = None,
    synthetic_numerical_preview: bool = False,
) -> CleaningIntent:
    """Describe an explicit-target cleaning preview without inventing a task.

    No target means the default task remains ``inspect_passage``.  Go2 and G1
    have no cleaning capability in this package.  Stretch and SO-ARM100 can
    receive a serpentine *preview* only for a supplied patch; a tool, arm
    kinematics, contact and execution remain unverified.  SO-ARM100 never
    receives a base-motion request.
    """
    normalized_robot = robot_id.strip().lower()
    normalized_action = action.strip().lower()
    if not normalized_robot:
        raise ValueError("robot_id must be non-empty")
    if not normalized_action:
        raise ValueError("action must be non-empty")
    if patch is None:
        return CleaningIntent(
            task="inspect_passage", robot_id=robot_id, action=normalized_action, capability=capability,
            status="not_requested", base_motion=False, sweep_points_m=(),
            synthetic_numerical_preview=synthetic_numerical_preview,
            note="No explicit cleaning target was supplied; default task is inspect passage.",
        )

    center, size, height = _validated_patch(patch)
    supported_actions = _CLEANING_CAPABILITIES.get(normalized_robot, frozenset())
    requested_capability = (capability or normalized_action).strip().lower()
    if normalized_action not in supported_actions or requested_capability not in supported_actions:
        return CleaningIntent(
            task="cleaning_preview", robot_id=robot_id, action=normalized_action, capability=capability,
            status="unsupported", base_motion=False, sweep_points_m=(),
            synthetic_numerical_preview=synthetic_numerical_preview,
            note="This robot/profile has no verified cleaning-tool capability for the requested action.",
        )

    base_motion = normalized_robot == "stretch"
    label = "synthetic numerical preview" if synthetic_numerical_preview else f"explicit patch '{patch.label}'"
    return CleaningIntent(
        task="cleaning_preview", robot_id=robot_id, action=normalized_action, capability=capability,
        status="planned_tool_unverified", base_motion=base_motion,
        sweep_points_m=_serpentine(center, size, height),
        synthetic_numerical_preview=synthetic_numerical_preview,
        note=(f"Sweep generated for {label}; tool, end-effector collision, contact, and execution are unverified."),
    )


def _route_option(
    route_id: str,
    label: str,
    scene: Scene,
    points_xy: Iterable[Iterable[float]],
    radius_m: float,
    body_height_m: float,
) -> RouteOption:
    floor = scene.floors[0]
    floor_top = floor.position[2] + floor.size[2] / 2
    points = tuple((float(point[0]), float(point[1]), floor_top) for point in points_xy)  # type: ignore[index]
    return RouteOption(
        id=route_id,
        label=label,
        kind="planned",
        points_m=points,
        length_m=sum(math.dist(left, right) for left, right in zip(points, points[1:])),
        clearance_m=min(clearance(scene, point[:2], radius_m, body_height_m) for point in points),
        status="planned",
    )


def _via_candidates(start: tuple[float, float], goal: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance == 0:
        return ()
    lateral = (-dy / distance, dx / distance)
    offsets = (max(0.40, distance * 0.25), max(0.75, distance * 0.50))
    candidates: list[tuple[float, float]] = []
    for along in (0.35, 0.50, 0.65):
        center = (start[0] + along * dx, start[1] + along * dy)
        for offset in offsets:
            for side in (1.0, -1.0):
                candidates.append((center[0] + side * lateral[0] * offset, center[1] + side * lateral[1] * offset))
    return tuple(candidates[:MAX_VIA_CANDIDATES])


def _is_distinct(
    candidate: list[list[float]],
    existing: Iterable[tuple[tuple[float, float, float], ...]],
    required_separation: float,
) -> bool:
    candidate_xy = [(point[0], point[1]) for point in candidate]
    for route in existing:
        existing_xy = [(point[0], point[1]) for point in route]
        if _max_distance_to_route(candidate_xy, existing_xy) < required_separation:
            return False
    return True


def _max_distance_to_route(points: list[tuple[float, float]], route: list[tuple[float, float]]) -> float:
    if len(route) == 1:
        return max(math.dist(point, route[0]) for point in points)
    return max(
        min(_point_segment_distance(point, left, right) for left, right in zip(route, route[1:]))
        for point in points
    )


def _point_segment_distance(point: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> float:
    dx, dy = right[0] - left[0], right[1] - left[1]
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.dist(point, left)
    ratio = min(1.0, max(0.0, ((point[0] - left[0]) * dx + (point[1] - left[1]) * dy) / denominator))
    return math.dist(point, (left[0] + ratio * dx, left[1] + ratio * dy))


def _deduplicate_xy(points: Iterable[Iterable[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for point in points:
        xy = [float(point[0]), float(point[1])]  # type: ignore[index]
        if not result or xy != result[-1]:
            result.append(xy)
    return result


def _validated_patch(patch: CleaningPatch) -> tuple[tuple[float, float], tuple[float, float], float]:
    center = _xy(patch.center_xy_m, "patch.center_xy_m")
    size = _xy(patch.size_xy_m, "patch.size_xy_m")
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("patch.size_xy_m must contain positive dimensions")
    height = _positive(patch.height_m, "patch.height_m")
    return center, size, height


def _serpentine(center: tuple[float, float], size: tuple[float, float], height: float) -> tuple[tuple[float, float, float], ...]:
    lanes = min(MAX_SWEEP_LANES, max(2, math.ceil(size[1] / max(0.10, min(size) / 4))))
    low_x, high_x = center[0] - size[0] / 2, center[0] + size[0] / 2
    low_y, high_y = center[1] - size[1] / 2, center[1] + size[1] / 2
    points: list[tuple[float, float, float]] = []
    for index in range(lanes):
        y = low_y + (high_y - low_y) * index / (lanes - 1)
        endpoints = ((low_x, y), (high_x, y)) if index % 2 == 0 else ((high_x, y), (low_x, y))
        points.extend((x, y_value, height) for x, y_value in endpoints)
    return tuple(points)


def _xy(value: Iterable[float], label: str) -> tuple[float, float]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a numeric two-vector") from error
    if len(result) != 2 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be a finite two-vector")
    return result[0], result[1]


def _positive(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _nonnegative(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result
