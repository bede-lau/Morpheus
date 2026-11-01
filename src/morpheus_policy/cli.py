"""Local policy-calculation workflow for supplied scenes; no environment builder."""

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

from .artifacts import write_artifacts
from .evaluation import Task, Thresholds, compose_model, evaluate_episode
from .navigation import NavigationConfig
from .planning import clearance, plan_route
from .registry import create_policy, list_robots, robot_model_path, ROBOT_SUPPORT_BODIES
from .scene import load_scene


def _xy(value):
    try:
        result = tuple(float(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use x,y in metres") from error
    if len(result) != 2 or not all(math.isfinite(v) for v in result):
        raise argparse.ArgumentTypeError("Use finite x,y in metres")
    return result


def demonstration_task(scene):
    """Choose numerical free-space endpoints; do not invent kitchen semantics."""
    min_x, min_y, max_x, max_y = scene.bounds_xy
    candidates = [(float(x), float(y)) for x in np.arange(min_x + 0.5, max_x - 0.5, 0.25)
                  for y in np.arange(min_y + 0.5, max_y - 0.5, 0.25)]
    candidates.sort(key=lambda p: clearance(scene, p), reverse=True)
    for start in candidates[:30]:
        for distance in (1.0, 0.8, 0.6):
            for angle in np.arange(0, 2 * math.pi, math.pi / 8):
                goal = (float(start[0] + distance * math.cos(angle)), float(start[1] + distance * math.sin(angle)))
                try:
                    route = plan_route(scene, start, goal)
                except ValueError:
                    continue
                if len(route) == 2:
                    return Task(start, goal, float(angle), "numerical_free_space_demo", True)
    raise ValueError("No conservative numerical demonstration route fits this robot envelope")


def _task(args, scene):
    if args.demo_task:
        if args.start is not None or args.goal is not None:
            raise ValueError("Use explicit start/goal OR --demo-task")
        return demonstration_task(scene)
    if args.start is None or args.goal is None:
        raise ValueError("Supply --start=x,y --goal=x,y, or explicitly choose --demo-task")
    yaw = args.yaw if args.yaw is not None else math.atan2(args.goal[1] - args.start[1], args.goal[0] - args.start[0])
    return Task(args.start, args.goal, yaw)


def _json(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("robots", help="List actual policy availability and limitations")
    inspect = sub.add_parser("inspect", help="Validate and compile the unchanged supplied environment")
    inspect.add_argument("scene", type=Path)
    arm = sub.add_parser("reach", help="Kinematic IK only for an explicitly mapped supplied arm model")
    arm.add_argument("mjcf", type=Path)
    arm.add_argument("--joints", required=True, help="Ordered comma-separated scalar joint names")
    arm.add_argument("--site", required=True)
    arm.add_argument("--target", required=True, help="x,y,z in metres")
    for command in ("plan", "run", "qualify", "optimize"):
        item = sub.add_parser(command)
        item.add_argument("scene", type=Path)
        item.add_argument("--start", type=_xy)
        item.add_argument("--goal", type=_xy)
        item.add_argument("--yaw", type=float)
        item.add_argument("--demo-task", action="store_true", help="Explicitly select a synthetic numerical target")
        item.add_argument("--robot", default="unitree_g1_12dof")
        item.add_argument("--assets", type=Path, default=Path(".morpheus/assets/g1"))
        item.add_argument("--duration", type=float, default=35.0)
        item.add_argument("--speed", type=float, default=0.30)
        item.add_argument("--output", type=Path, default=Path("outputs") / command)
        if command == "qualify":
            item.add_argument("--episodes", type=int, default=20)
            item.add_argument("--seed", type=int, default=20261101)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "robots":
            _json(list_robots())
            return 0
        if args.command == "reach":
            from .arm import solve_reach
            target = tuple(float(v) for v in args.target.split(","))
            model = mujoco.MjModel.from_xml_path(str(args.mjcf))
            _json(solve_reach(model, mujoco.MjData(model), args.joints.split(","), args.site, target))
            return 0
        scene = load_scene(args.scene)
        if args.command == "inspect":
            model = mujoco.MjModel.from_xml_path(str(scene.mjcf_path))
            _json({"scene": scene.name, "objects": len(scene.objects), "bounds_xy_m": scene.bounds_xy,
                   "engine": f"MuJoCo {mujoco.__version__}", "compiled": True,
                   "nq": model.nq, "nu": model.nu, "ngeom": model.ngeom,
                   "source": dict(scene.source), "hashes": dict(scene.hashes),
                   "caveats": scene.caveats, "readiness": "PREVIEW_ONLY", "hardware_readiness": "UNVERIFIED"})
            return 0
        task = _task(args, scene)
        route = plan_route(scene, task.start_xy, task.goal_xy)
        if args.command == "plan":
            _json({"calculation": "geometric_route_only", "task": asdict(task), "route_xy_m": route,
                   "readiness": "PREVIEW_ONLY", "hardware_readiness": "UNVERIFIED", "caveats": scene.caveats})
            return 0
        if args.output.exists():
            raise ValueError(f"Output already exists; choose a new directory: {args.output}")
        assets = args.assets.expanduser().resolve()
        policy = create_policy(args.robot, assets)
        asset_gate = policy.validate_assets(assets)
        if not asset_gate["valid"]:
            raise ValueError(f"Policy asset integrity failed: {asset_gate['errors']}")
        model, spec, model_hash = compose_model(scene, robot_model_path(args.robot, assets))
        if not policy.validate_assets(assets)["valid"]:
            raise ValueError("Policy assets changed during model composition")
        from .policies.g1 import DEFAULT_ANGLES
        config = NavigationConfig(speed_m_s=args.speed)
        limits = Thresholds(max_seconds=args.duration)

        def evaluate(current_task, current_config=config):
            return evaluate_episode(scene, model, policy, current_task, model_hash=model_hash,
                                    config=current_config, thresholds=limits,
                                    initial_joint_positions=DEFAULT_ANGLES, robot_id=args.robot,
                                    allowed_support_bodies=ROBOT_SUPPORT_BODIES[args.robot])

        if args.command == "qualify":
            from .qualification import qualification_tasks, summarize_qualification
            reports = []
            for index, current_task in enumerate(qualification_tasks(task, seed=args.seed, episodes=args.episodes)):
                print(f"Evaluating held-out episode {index + 1}/{args.episodes}", file=sys.stderr, flush=True)
                try:
                    report, _ = evaluate(current_task)
                except ValueError as error:
                    report = {"execution": "failed", "failures": [str(error)], "task": asdict(current_task)}
                reports.append(report)
            result = summarize_qualification(scene, task, reports, seed=args.seed)
            args.output.mkdir(parents=True)
            (args.output / "qualification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            _json({k: v for k, v in result.items() if k != "episodes"})
            return 0 if result["execution"] == "completed" else 2
        if args.command == "optimize":
            from .optimization import optimize
            def callback(current_config, seed):
                rng = np.random.default_rng(seed)
                current_task = replace(task, start_xy=tuple(np.asarray(task.start_xy) + rng.uniform(-0.01, 0.01, 2)))
                print(f"Evaluating controller speed={current_config.speed_m_s}, seed={seed}", file=sys.stderr, flush=True)
                try:
                    report, _ = evaluate(current_task, current_config)
                    return report
                except ValueError as error:
                    return {"execution": "failed", "metrics": {}, "failures": [str(error)]}
            result = optimize(callback, config, [replace(config, speed_m_s=0.20), replace(config, speed_m_s=0.40)], [101, 102], [1001, 1002])
            args.output.mkdir(parents=True)
            (args.output / "optimization.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            _json(result)
            return 0
        report, replay = evaluate(task)
        output = write_artifacts(args.output, report, replay, spec=spec, model=model,
                                 third_party_license_path=assets / "LICENSE")
        _json({"output": str(output.resolve()), "execution": report["execution"], "readiness": report["readiness"],
               "hardware_readiness": report["hardware_readiness"], "metrics": report["metrics"],
               "failures": report["failures"], "caveats": report["caveats"]})
        return 0 if report["execution"] == "completed" else 2
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"execution": "unavailable", "readiness": "BLOCKED", "error": str(error),
                          "hardware_readiness": "UNVERIFIED"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
