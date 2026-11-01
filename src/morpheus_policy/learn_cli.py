"""Train scene-bound high-level PPO, export viewer runs and test G1 transfer."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

from .cli import _xy, demonstration_task
from .profiles import NAVIGATION_PROFILES, navigation_profile
from .scene import load_scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "export", "validate-g1"))
    parser.add_argument("scene", type=Path)
    parser.add_argument("--profiles", default="g1,stretch,go2,soarm")
    parser.add_argument("--output", type=Path, default=Path("outputs/learning"))
    parser.add_argument("--viewer-output", type=Path, default=Path("public/policies"))
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--heldout-seed", type=int, default=10007)
    parser.add_argument("--imitation-epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--imitation-learning-rate", type=float, default=1e-3)
    parser.add_argument("--start", type=_xy)
    parser.add_argument("--goal", type=_xy)
    parser.add_argument("--demo-cleaning", action="store_true", help="Explicit opt-in numerical tool paths; no surface/cleaning claim")
    parser.add_argument("--assets", type=Path, default=Path(".morpheus/assets/g1"))
    parser.add_argument("--native-artifacts", type=Path, help="Attach separately verified G1 native report/replay")
    parser.add_argument("--floor-audit", type=Path, help="Attach a source-hash-bound mesh floor support audit")
    parser.add_argument("--episodes", type=int, default=1, help="Native validation: 1 episode, or 20..100 perturbed starts")
    parser.add_argument("--native-output", type=Path)
    args = parser.parse_args(argv)
    try:
        scene = load_scene(args.scene)
        if (args.start is None) != (args.goal is None):
            raise ValueError("Supply both --start and --goal")
        if args.start is None:
            task = demonstration_task(scene)
        else:
            import math
            from .evaluation import Task
            task = Task(args.start, args.goal, math.atan2(args.goal[1]-args.start[1], args.goal[0]-args.start[0]))
        if args.command == "validate-g1":
            return _validate_g1(args, scene, task)
        from .learning import train_scene_specific_ppo, PPOTrainConfig
        from .viewer_export import export_viewer_run
        results = []
        for name in args.profiles.split(","):
            profile = navigation_profile(name.strip())
            directory = args.output / profile.id
            if args.command == "train" and profile.mobile:
                if (directory / "reduced_order_ppo.pt").exists():
                    raise ValueError(f"Checkpoint exists; use export or a new --output: {directory}")
                print(f"Training {profile.id}: actual reduced-order PPO on CPU", file=sys.stderr, flush=True)
                start_time = time.monotonic()
                trained = train_scene_specific_ppo(scene, profile, directory,
                            train_config=PPOTrainConfig(seed=args.seed, updates=args.updates,
                                heldout_seed=args.heldout_seed, imitation_epochs=args.imitation_epochs,
                                learning_rate=args.learning_rate, imitation_learning_rate=args.imitation_learning_rate))
                print(json.dumps({"profile": profile.id, "wall_seconds": time.monotonic()-start_time,
                                  "trained_successes": trained.trained.successes,
                                  "untrained_successes": trained.untrained.successes,
                                  "heldout_episodes": trained.trained.episodes}), flush=True)
            checkpoint_dir = directory if (directory / "reduced_order_ppo.pt").exists() else None
            result = export_viewer_run(scene, profile, task.start_xy, task.goal_xy,
                         args.viewer_output / f"{profile.id}.json", training_dir=checkpoint_dir, demo_cleaning=args.demo_cleaning,
                         native_artifacts=args.native_artifacts, floor_audit=args.floor_audit)
            results.append({"robot": profile.id, "trained": result["training"] is not None,
                            "routes": len(result["routes"]), "cleaning_intents": len(result["cleaning"]),
                            "readiness": result["readiness"]})
        print(json.dumps(results, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"error": str(error), "hardware_readiness": "UNVERIFIED"}), file=sys.stderr)
        return 2


def _validate_g1(args, scene, task):
    from .artifacts import write_artifacts
    from .evaluation import compose_model, evaluate_episode, Thresholds
    from .learned_navigation import load_navigation_artifact, LearnedNavigationController
    from .navigation import NavigationConfig
    from .policies.g1 import DEFAULT_ANGLES
    from .registry import create_policy, robot_model_path, ROBOT_SUPPORT_BODIES
    from .qualification import qualification_tasks, summarize_qualification
    out = args.native_output or args.output / "g1-native"
    if out.exists():
        raise ValueError(f"Native output exists: {out}")
    tasks = [task] if args.episodes == 1 else list(qualification_tasks(task, seed=args.heldout_seed, episodes=args.episodes))
    profile = navigation_profile("g1")
    actor, env_config, _, nav_manifest = load_navigation_artifact(args.output / "g1", scene, profile)
    robot_id = profile.physics_robot_id
    locomotion = create_policy(robot_id, args.assets)
    if not locomotion.validate_assets(args.assets)["valid"]:
        raise ValueError("G1 assets failed integrity validation")
    model, spec, model_hash = compose_model(scene, robot_model_path(robot_id, args.assets))
    if not locomotion.validate_assets(args.assets)["valid"]:
        raise ValueError("G1 assets changed during composition")
    config = NavigationConfig()
    reports, first_replay = [], None
    for index, current_task in enumerate(tasks):
        print(f"Native learned-policy validation {index + 1}/{len(tasks)}", file=sys.stderr, flush=True)
        controller = LearnedNavigationController(scene, profile, actor, env_config, current_task.goal_xy, config)
        report, replay = evaluate_episode(scene, model, locomotion, current_task, model_hash=model_hash,
            config=config, thresholds=Thresholds(), initial_joint_positions=DEFAULT_ANGLES,
            robot_id=robot_id, allowed_support_bodies=ROBOT_SUPPORT_BODIES[robot_id],
            navigation_controller=controller, navigation_manifest=nav_manifest)
        reports.append(report)
        if first_replay is None:
            first_replay = replay
    qualification = summarize_qualification(scene, task, reports, seed=args.heldout_seed) if len(tasks) >= 20 else None
    report, replay = reports[0], first_replay
    # Signature is shared with the standard evaluator CLI.
    write_artifacts(out, deepcopy(report), deepcopy(replay), model=model, spec=spec, third_party_license_path=args.assets / "LICENSE")
    if qualification:
        (out / "qualification.json").write_text(json.dumps(qualification, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"execution": report["execution"], "metrics": report["metrics"], "failures": report["failures"]}, indent=2))
    return 0 if (qualification or report)["execution"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
