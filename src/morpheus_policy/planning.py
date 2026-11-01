"""Conservative, bounded planar navigation over supplied static geometry."""

from __future__ import annotations

import heapq
import math
from typing import Iterable

from .scene import Box, Scene, SceneValidationError, _yaw


MAX_GRID_CELLS = 1_000_000
# Conservative upright envelope for the supported G1 preview.  Callers can
# supply another finite positive height for a different embodiment/profile.
NAVIGATION_BODY_HEIGHT_M = 1.4


class PlanningError(ValueError):
    """Raised when a request is unsafe, out of bounds, or unreachable."""


def clearance(
    scene: Scene,
    xy: Iterable[float],
    radius_m: float = 0.35,
    body_height_m: float = NAVIGATION_BODY_HEIGHT_M,
) -> float:
    """Return signed free clearance for a circular envelope in metres.

    Positive values are free clearance; zero touches an obstacle or the finite
    floor boundary; negative values overlap it.  The conservative obstacle
    projection encloses each rotated 3-D box's XY footprint.
    """
    point = _xy(xy, "xy")
    radius = _nonnegative_finite(radius_m, "radius_m")
    body_height = _positive_finite(body_height_m, "body_height_m")
    if len(scene.floors) != 1:
        raise SceneValidationError("exactly one floor is required for clearance")

    available = _signed_distance_inside_floor(point, scene.floors[0]) - radius
    for obstacle in scene.obstacles:
        if _blocks_navigation(obstacle, scene.floors[0], body_height):
            available = min(available, _signed_distance_to_projection(point, obstacle) - radius)
    return available


def plan_route(
    scene: Scene,
    start_xy: Iterable[float],
    goal_xy: Iterable[float],
    radius_m: float = 0.35,
    margin_m: float = 0.10,
    resolution_m: float = 0.10,
    body_height_m: float = NAVIGATION_BODY_HEIGHT_M,
) -> list[list[float]]:
    """Find a conservative cardinal A* route without moving either endpoint."""
    start = _xy(start_xy, "start_xy")
    goal = _xy(goal_xy, "goal_xy")
    radius = _nonnegative_finite(radius_m, "radius_m")
    margin = _nonnegative_finite(margin_m, "margin_m")
    resolution = _positive_finite(resolution_m, "resolution_m")
    body_height = _positive_finite(body_height_m, "body_height_m")
    envelope = radius + margin

    if clearance(scene, start, envelope, body_height) <= 0:
        raise PlanningError("start is obstructed or its envelope leaves the supplied floor")
    if clearance(scene, goal, envelope, body_height) <= 0:
        raise PlanningError("goal is obstructed or its envelope leaves the supplied floor")
    if _segment_safe(scene, start, goal, envelope, resolution, body_height):
        return [[start[0], start[1]], [goal[0], goal[1]]]

    min_x, min_y, max_x, max_y = scene.bounds_xy
    # Padding makes a free cell conservative for every point in that cell.
    cell_padding = math.sqrt(2) * resolution / 2
    min_x += envelope + cell_padding
    min_y += envelope + cell_padding
    max_x -= envelope + cell_padding
    max_y -= envelope + cell_padding
    if min_x >= max_x or min_y >= max_y:
        raise PlanningError("robot envelope does not fit inside the supplied floor")
    width = int(math.floor((max_x - min_x) / resolution)) + 1
    height = int(math.floor((max_y - min_y) / resolution)) + 1
    if width * height > MAX_GRID_CELLS:
        raise PlanningError("planning grid exceeds bounded-cell limit")

    def point_for(cell: tuple[int, int]) -> tuple[float, float]:
        return (min_x + cell[0] * resolution, min_y + cell[1] * resolution)

    def cell_for(point: tuple[float, float]) -> tuple[int, int]:
        return (
            min(max(int(round((point[0] - min_x) / resolution)), 0), width - 1),
            min(max(int(round((point[1] - min_y) / resolution)), 0), height - 1),
        )

    start_cell, goal_cell = cell_for(start), cell_for(goal)
    start_anchor, goal_anchor = point_for(start_cell), point_for(goal_cell)
    if not _cell_free(scene, start_anchor, envelope, cell_padding, body_height):
        raise PlanningError("no conservative grid connection from start")
    if not _cell_free(scene, goal_anchor, envelope, cell_padding, body_height):
        raise PlanningError("no conservative grid connection to goal")
    if not _segment_safe(scene, start, start_anchor, envelope, resolution, body_height):
        raise PlanningError("start cannot connect to a free route without clearance loss")
    if not _segment_safe(scene, goal_anchor, goal, envelope, resolution, body_height):
        raise PlanningError("goal cannot connect to a free route without clearance loss")

    path_cells = _astar(
        scene, start_cell, goal_cell, point_for, width, height, envelope, cell_padding, body_height
    )
    if path_cells is None:
        raise PlanningError("no route through supplied static geometry")
    points = [start, *(point_for(cell) for cell in path_cells), goal]
    route = _deduplicate(points)
    if any(
        not _segment_safe(scene, left, right, envelope, resolution, body_height)
        for left, right in zip(route, route[1:])
    ):
        raise PlanningError("route segment failed conservative clearance check")
    return [[point[0], point[1]] for point in route]


def _astar(
    scene: Scene,
    start: tuple[int, int],
    goal: tuple[int, int],
    point_for: object,
    width: int,
    height: int,
    envelope: float,
    cell_padding: float,
    body_height: float,
) -> list[tuple[int, int]] | None:
    # Four-neighbour motion prevents diagonal corner cutting by construction.
    queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
    cost = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current_cost != cost.get(current):
            continue
        if current == goal:
            return _reconstruct(parent, goal)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbour[0] < width and 0 <= neighbour[1] < height):
                continue
            point = point_for(neighbour)  # type: ignore[operator]
            if not _cell_free(scene, point, envelope, cell_padding, body_height):
                continue
            candidate = current_cost + 1.0
            if candidate >= cost.get(neighbour, math.inf):
                continue
            cost[neighbour] = candidate
            parent[neighbour] = current
            heuristic = abs(goal[0] - neighbour[0]) + abs(goal[1] - neighbour[1])
            heapq.heappush(queue, (candidate + heuristic, candidate, neighbour))
    return None


def _cell_free(
    scene: Scene,
    point: tuple[float, float],
    envelope: float,
    padding: float,
    body_height: float,
) -> bool:
    return clearance(scene, point, envelope + padding, body_height) > 0


def _segment_safe(
    scene: Scene,
    start: tuple[float, float],
    goal: tuple[float, float],
    envelope: float,
    resolution: float,
    body_height: float,
) -> bool:
    distance = math.dist(start, goal)
    steps = max(1, math.ceil(distance / max(resolution / 2, 1e-9)))
    for index in range(steps + 1):
        ratio = index / steps
        point = (start[0] + (goal[0] - start[0]) * ratio, start[1] + (goal[1] - start[1]) * ratio)
        if clearance(scene, point, envelope, body_height) <= 0:
            return False
    return True


def _reconstruct(
    parent: dict[tuple[int, int], tuple[int, int] | None], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    route: list[tuple[int, int]] = []
    current: tuple[int, int] | None = goal
    while current is not None:
        route.append(current)
        current = parent[current]
    route.reverse()
    return route


def _deduplicate(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    route: list[tuple[float, float]] = []
    for point in points:
        if not route or point != route[-1]:
            route.append(point)
    return route


def _blocks_navigation(box: Box, floor: Box, body_height: float) -> bool:
    floor_top = floor.position[2] + floor.size[2] / 2
    box_bottom = box.position[2] - box.size[2] / 2
    return box_bottom < floor_top + body_height


def _signed_distance_to_projection(point: tuple[float, float], box: Box) -> float:
    # Project every 3-D oriented-box corner to an XY AABB.  The AABB is larger
    # than the projection, so it remains safe for yawed or tilted source boxes.
    corners = _projected_xy_corners(box)
    minimum_x, maximum_x = min(x for x, _ in corners), max(x for x, _ in corners)
    minimum_y, maximum_y = min(y for _, y in corners), max(y for _, y in corners)
    return _signed_distance_to_axis_box(point, minimum_x, minimum_y, maximum_x, maximum_y)


def _signed_distance_inside_floor(point: tuple[float, float], floor: Box) -> float:
    w, x, y, z = floor.quaternion
    if abs(x) > _QUAT_TILT_EPSILON or abs(y) > _QUAT_TILT_EPSILON:
        raise PlanningError("a tilted floor cannot provide a finite navigation support plane")
    yaw = _yaw(floor.quaternion)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    dx, dy = point[0] - floor.position[0], point[1] - floor.position[1]
    local_x, local_y = cosine * dx + sine * dy, -sine * dx + cosine * dy
    half_x, half_y = floor.size[0] / 2, floor.size[1] / 2
    outside_x, outside_y = max(abs(local_x) - half_x, 0.0), max(abs(local_y) - half_y, 0.0)
    if outside_x or outside_y:
        return -math.hypot(outside_x, outside_y)
    return min(half_x - abs(local_x), half_y - abs(local_y))


def _signed_distance_to_axis_box(
    point: tuple[float, float], minimum_x: float, minimum_y: float, maximum_x: float, maximum_y: float
) -> float:
    dx = max(minimum_x - point[0], 0.0, point[0] - maximum_x)
    dy = max(minimum_y - point[1], 0.0, point[1] - maximum_y)
    if dx or dy:
        return math.hypot(dx, dy)
    return -min(point[0] - minimum_x, maximum_x - point[0], point[1] - minimum_y, maximum_y - point[1])


def _projected_xy_corners(box: Box) -> tuple[tuple[float, float], ...]:
    matrix = _rotation_matrix(box.quaternion)
    half = tuple(value / 2 for value in box.size)
    points: list[tuple[float, float]] = []
    for sx in (-half[0], half[0]):
        for sy in (-half[1], half[1]):
            for sz in (-half[2], half[2]):
                x = box.position[0] + matrix[0][0] * sx + matrix[0][1] * sy + matrix[0][2] * sz
                y = box.position[1] + matrix[1][0] * sx + matrix[1][1] * sy + matrix[1][2] * sz
                points.append((x, y))
    return tuple(points)


def _rotation_matrix(quaternion: tuple[float, float, float, float]) -> tuple[tuple[float, float, float], ...]:
    w, x, y, z = quaternion
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _xy(value: Iterable[float], label: str) -> tuple[float, float]:
    try:
        point = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise PlanningError(f"{label} must be a numeric two-vector") from error
    if len(point) != 2 or not all(math.isfinite(item) for item in point):
        raise PlanningError(f"{label} must be a finite two-vector")
    return point[0], point[1]


def _nonnegative_finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise PlanningError(f"{label} must be finite and non-negative")
    return result


def _positive_finite(value: float, label: str) -> float:
    result = _nonnegative_finite(value, label)
    if result == 0:
        raise PlanningError(f"{label} must be positive")
    return result


_QUAT_TILT_EPSILON = 1e-6
