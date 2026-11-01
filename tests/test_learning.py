from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from morpheus_policy.learning import (
    FootprintProfile,
    HomeNavConfig,
    NavScenario,
    PPOTrainConfig,
    ReducedOrderNavEnv,
    _ClearanceField,
    load_reduced_order_ppo,
    sample_scenarios,
    train_scene_specific_ppo,
)
from morpheus_policy.planning import clearance
from morpheus_policy.scene import Box, Scene


PROFILE_SPEC = {
    "id": "test_mobile",
    "radius_m": 0.20,
    "body_height_m": 1.0,
    "max_speed_m_s": 0.60,
    "max_yaw_rate": 1.2,
}


def _scene(tmp_path: Path, *, obstacle: bool = False) -> Scene:
    floor = Box("floor", (0.0, 0.0, -0.05), (6.0, 6.0, 0.10), (1.0, 0.0, 0.0, 0.0), "floor")
    objects = [floor]
    if obstacle:
        objects.append(Box("counter", (0.0, 0.0, 0.50), (0.60, 0.60, 1.0), (1.0, 0.0, 0.0, 0.0), "obstacle"))
    return Scene("learning_fixture", tuple(objects), {}, tmp_path, tmp_path / "scene.xml", {"scene.json": "fixture"}, ())


def test_profile_accepts_registry_style_speed_aliases() -> None:
    profile = FootprintProfile.from_spec(PROFILE_SPEC)
    assert profile.max_linear_speed_m_s == 0.60
    assert profile.max_lateral_speed_m_s < 1e-5
    assert profile.max_yaw_rate_rad_s == 1.2


def test_reduced_order_environment_clips_and_terminates_before_obstacle(tmp_path: Path) -> None:
    environment = ReducedOrderNavEnv(
        _scene(tmp_path, obstacle=True), PROFILE_SPEC,
        HomeNavConfig(dt_s=0.10, horizon_steps=50, ray_count=8, ray_range_m=1.5, ray_step_m=0.10),
    )
    observation = environment.reset(NavScenario((-2.0, 0.0), 0.0, (2.0, 0.0)))
    assert observation.shape == (15,)
    for _ in range(50):
        observation, _reward, done, info = environment.step((99.0, 0.0, 0.0))
        assert np.isfinite(observation).all()
        if done:
            break
    assert done
    assert info["collision"]
    assert environment.position[0] < -0.35
    assert abs(info["command"][0]) <= 0.60


def test_reduced_order_environment_does_not_tunnel_through_thin_wall(tmp_path: Path) -> None:
    floor = Box("floor", (0.0, 0.0, -0.05), (6.0, 6.0, 0.10), (1.0, 0.0, 0.0, 0.0), "floor")
    wall = Box("thin_wall", (0.0, 0.0, 0.50), (0.04, 2.0, 1.0), (1.0, 0.0, 0.0, 0.0), "obstacle")
    scene = Scene("thin_wall", (floor, wall), {}, tmp_path, tmp_path / "scene.xml", {}, ())
    environment = ReducedOrderNavEnv(scene, PROFILE_SPEC, HomeNavConfig(dt_s=1.0, horizon_steps=3, ray_count=8, ray_step_m=0.10))
    environment.reset(NavScenario((-0.50, 0.0), 0.0, (1.0, 0.0)))
    _observation, _reward, done, info = environment.step((1.0, 0.0, 0.0))
    assert done and info["collision"]
    assert environment.position[0] == -0.50


def test_vectorized_clearance_matches_planner_conservative_projection(tmp_path: Path) -> None:
    floor = Box("floor", (0.0, 0.0, -0.05), (6.0, 6.0, 0.10), (1.0, 0.0, 0.0, 0.0), "floor")
    half_yaw = np.pi / 8
    rotated = Box("rotated", (0.4, -0.2, 0.5), (1.0, 0.4, 1.0), (np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)), "obstacle")
    scene = Scene("projection", (floor, rotated), {}, tmp_path, tmp_path / "scene.xml", {}, ())
    points = np.array(((-2.0, 0.0), (-0.2, -0.2), (0.4, -0.2), (2.8, 0.0), (3.1, 0.0)))
    expected = np.array([clearance(scene, point, 0.2, 1.0) for point in points])
    actual = _ClearanceField(scene, 1.0).clearance(points, 0.2)
    assert np.allclose(actual, expected, atol=1e-12)


def test_cpu_ppo_smoke_serializes_artifact_and_heldout_metrics(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    environment = HomeNavConfig(
        dt_s=0.10, horizon_steps=40, ray_count=8, ray_range_m=1.5, ray_step_m=0.10,
        pose_noise_m=0.0, yaw_noise_rad=0.0, radius_jitter_m=0.0,
    )
    training = PPOTrainConfig(
        seed=13, n_envs=2, rollout_steps=16, updates=3, epochs=1, minibatch_size=16, hidden_size=16,
        train_scenarios=6, heldout_scenarios=3, imitation_epochs=2, heldout_seed=10007,
    )
    scenarios = sample_scenarios(scene, PROFILE_SPEC, environment, 3, training.seed)
    assert scenarios == sample_scenarios(scene, PROFILE_SPEC, environment, 3, training.seed)
    result = train_scene_specific_ppo(scene, PROFILE_SPEC, tmp_path / "artifact", env_config=environment, train_config=training)

    assert result.artifact_path.is_file()
    assert result.metadata_path.is_file()
    assert result.rollout_path.is_file()
    assert len(result.update_losses) == training.updates
    assert np.isfinite(result.update_losses).all()
    assert result.trained.episodes == training.heldout_scenarios
    assert result.warm_started.episodes == training.heldout_scenarios
    assert result.astar.episodes == training.heldout_scenarios
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["abstraction"] == "reduced_order_static_footprint"
    assert metadata["scene_fingerprint"]
    assert metadata["heldout_seed"] == 10007
    assert metadata["evaluation"]["warm_started"]["episodes"] == training.heldout_scenarios
    assert metadata["splits"]["train_scenarios"] != metadata["splits"]["heldout_scenarios"]
    policy = load_reduced_order_ppo(result.artifact_path)
    action = policy.predict(ReducedOrderNavEnv(scene, PROFILE_SPEC, environment).reset(scenarios[0]))
    assert action.shape == (3,)
    assert np.isfinite(action).all()
    assert np.all(np.abs(action) <= 1.0)
