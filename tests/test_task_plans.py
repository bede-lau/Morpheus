from pathlib import Path

import pytest

from morpheus_policy.scene import Box, Scene
from morpheus_policy.task_plans import CleaningPatch, plan_cleaning_intent, plan_navigation_options


def _scene(*objects: Box) -> Scene:
    return Scene("task plans", objects, {}, Path("."), Path("scene.xml"), {}, ())


def _floor(size: float = 10) -> Box:
    return Box("floor", (0, 0, -0.05), (size, size, 0.1), (1, 0, 0, 0), "floor")


def _obstacle(identifier: str, position: tuple[float, float, float], size: tuple[float, float, float]) -> Box:
    return Box(identifier, position, size, (1, 0, 0, 0), "obstacle")


def test_navigation_options_are_bounded_distinct_and_safe() -> None:
    scene = _scene(_floor(10), _obstacle("island", (0, 0, 0.5), (1.2, 1.2, 1)))
    routes = plan_navigation_options(
        scene, (-3, 0), (3, 0), radius_m=0.25, body_height_m=1.4, margin_m=0.10, resolution_m=0.1
    )
    assert 1 <= len(routes) <= 3
    assert routes[0].id == "route-1"
    assert all(route.kind == "planned" and route.status == "planned" for route in routes)
    assert all(route.points_m[0][:2] == (-3.0, 0.0) and route.points_m[-1][:2] == (3.0, 0.0) for route in routes)
    assert all(route.clearance_m > 0.10 and route.length_m > 0 for route in routes)
    assert len({tuple(route.points_m) for route in routes}) == len(routes)
    assert all(point[2] == 0 for route in routes for point in route.points_m)


def test_navigation_returns_fewer_routes_when_alternatives_are_not_safe() -> None:
    scene = _scene(_floor(2), _obstacle("corridor", (0, 0.45, 0.5), (2, 0.2, 1)), _obstacle("corridor2", (0, -0.45, 0.5), (2, 0.2, 1)))
    routes = plan_navigation_options(
        scene, (-0.7, 0), (0.7, 0), radius_m=0.1, body_height_m=1.4, margin_m=0.02, resolution_m=0.05
    )
    assert len(routes) == 1
    assert routes[0].points_m[0][:2] == (-0.7, 0)
    with pytest.raises(ValueError, match="max_routes"):
        plan_navigation_options(scene, (-0.7, 0), (0.7, 0), radius_m=0.1, body_height_m=1.4, max_routes=4)


def test_cleaning_intent_needs_explicit_target_and_honours_capabilities() -> None:
    no_target = plan_cleaning_intent("stretch")
    assert no_target.task == "inspect_passage"
    assert no_target.status == "not_requested"
    assert not no_target.sweep_points_m

    patch = CleaningPatch((1, 2), (0.6, 0.4), 0.9, "counter spill")
    stretch = plan_cleaning_intent("stretch", patch, action="wipe")
    assert stretch.status == "planned_tool_unverified"
    assert stretch.base_motion is True
    assert len(stretch.sweep_points_m) >= 4
    assert {point[2] for point in stretch.sweep_points_m} == {0.9}

    arm = plan_cleaning_intent("soarm", patch, action="sweep", synthetic_numerical_preview=True)
    assert arm.status == "planned_tool_unverified"
    assert arm.base_motion is False
    assert arm.synthetic_numerical_preview is True

    for robot in ("go2", "unitree_g1_12dof"):
        unsupported = plan_cleaning_intent(robot, patch)
        assert unsupported.status == "unsupported"
        assert unsupported.sweep_points_m == ()

    with pytest.raises(ValueError, match="positive dimensions"):
        plan_cleaning_intent("stretch", CleaningPatch((0, 0), (0, 1), 1))
