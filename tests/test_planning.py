from pathlib import Path
import math

import pytest

from morpheus_policy.planning import PlanningError, clearance, plan_route
from morpheus_policy.scene import Box, Scene


def _scene(*objects: Box) -> Scene:
    return Scene(
        name="planner fixture",
        objects=objects,
        source={},
        root=Path("."),
        mjcf_path=Path("scene.xml"),
        hashes={},
        caveats=(),
    )


def _floor(size: float = 10) -> Box:
    return Box("floor", (0, 0, -0.05), (size, size, 0.1), (1, 0, 0, 0), "floor")


def _obstacle(identifier: str, position: tuple[float, float, float], size=(1, 1, 1), quaternion=(1, 0, 0, 0)) -> Box:
    return Box(identifier, position, size, quaternion, "obstacle")


def test_clearance_respects_obstacles_and_finite_floor() -> None:
    scene = _scene(_floor(4), _obstacle("table", (0, 0, 0.5)))
    assert clearance(scene, (0, 0), 0.35) < 0
    assert clearance(scene, (1.9, 0), 0.35) < 0
    assert clearance(scene, (1.0, 1.0), 0.1) > 0


def test_plan_route_keeps_exact_endpoints_and_avoids_obstacle() -> None:
    scene = _scene(_floor(8), _obstacle("island", (0, 0, 0.5), (1, 1, 1)))
    route = plan_route(scene, (-2.13, -0.17), (2.07, 0.19), radius_m=0.2, margin_m=0.1, resolution_m=0.2)
    assert route[0] == [-2.13, -0.17]
    assert route[-1] == [2.07, 0.19]
    assert len(route) > 2
    assert all(clearance(scene, point, 0.3) > 0 for point in route)


def test_plan_rejects_obstructed_endpoints_and_unreachable_floor() -> None:
    scene = _scene(_floor(2), _obstacle("center", (0, 0, 0.5), (1, 1, 1)))
    with pytest.raises(PlanningError, match="start is obstructed"):
        plan_route(scene, (0, 0), (0.8, 0.8), radius_m=0.1, margin_m=0.0)
    with pytest.raises(PlanningError, match="goal is obstructed"):
        plan_route(scene, (-0.8, -0.8), (0, 0), radius_m=0.1, margin_m=0.0)


def test_high_obstacle_does_not_block_ground_navigation() -> None:
    scene = _scene(_floor(6), _obstacle("ceiling", (0, 0, 2), (2, 2, 0.2)))
    assert clearance(scene, (0, 0), 0.2) > 0
    assert plan_route(scene, (-1, 0), (1, 0), radius_m=0.2, margin_m=0.0) == [[-1.0, 0.0], [1.0, 0.0]]


def test_body_height_is_a_conservative_explicit_planning_profile() -> None:
    low_overhead = _obstacle("low_overhead", (0, 0, 1.15), (2, 2, 0.1))
    above_g1 = _obstacle("above_g1", (0, 0, 1.55), (2, 2, 0.1))
    low_scene = _scene(_floor(6), low_overhead)
    high_scene = _scene(_floor(6), above_g1)

    # Its bottom at 1.10 m blocks the 1.4 m default G1 envelope, but a 1.0 m
    # profile can correctly plan beneath it without a robot-name switch.
    assert clearance(low_scene, (0, 0), 0.2) <= 0
    assert clearance(low_scene, (0, 0), 0.2, body_height_m=1.0) > 0
    assert plan_route(
        low_scene, (-2, 0), (2, 0), radius_m=0.2, margin_m=0.0, body_height_m=1.0
    ) == [[-2.0, 0.0], [2.0, 0.0]]
    assert len(plan_route(low_scene, (-2, 0), (2, 0), radius_m=0.2, margin_m=0.0)) > 2

    # Its bottom at 1.50 m remains above the supported conservative envelope.
    assert clearance(high_scene, (0, 0), 0.2) > 0
    assert plan_route(high_scene, (-1, 0), (1, 0), radius_m=0.2, margin_m=0.0) == [[-1.0, 0.0], [1.0, 0.0]]

    with pytest.raises(PlanningError, match="body_height_m must be positive"):
        clearance(low_scene, (0, 0), body_height_m=0)
    with pytest.raises(PlanningError, match="body_height_m must be finite"):
        plan_route(low_scene, (-1, 0), (1, 0), body_height_m=math.nan)


def test_rotated_box_projection_is_conservative() -> None:
    quarter_yaw = 0.3826834323650898
    scene = _scene(_floor(8), _obstacle("yawed", (0, 0, 0.5), (2, 0.4, 1), (0.9238795325112867, 0, 0, quarter_yaw)))
    # This lies inside the box's conservative XY AABB, but outside its exact
    # yawed rectangle; treating it as blocked confirms no rotated footprint is missed.
    assert clearance(scene, (0.75, 0.75), 0.0) <= 0
