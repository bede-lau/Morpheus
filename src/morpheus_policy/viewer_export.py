"""Data adapter for the existing Three/Spark workspace; never creates a scene."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil

from .learning import NavScenario, ReducedOrderNavEnv
from .learned_navigation import load_navigation_artifact
from .planning import clearance
from .task_plans import CleaningPatch, plan_cleaning_intent, plan_navigation_options
from .artifacts import sha256_file


def export_viewer_run(scene, profile, start, goal, output_path, *, training_dir=None, demo_cleaning=False, native_artifacts=None, floor_audit=None):
    """Export measured reduced-order trajectory and separately calculated choices."""
    floor = scene.floors[0]
    floor_z = floor.position[2] + floor.size[2] / 2
    yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
    routes, cleaning, metadata, training, final_info = [], [], None, None, None
    def frame(time, xy, heading):
        return {"time": float(time), "position": [float(xy[0]), float(xy[1]), float(floor_z)],
                "yaw": float(heading - math.pi / 2), "armExtension": 0.0}
    frames = [frame(0, start, yaw), frame(1, start, yaw)]
    if profile.mobile:
        for option in plan_navigation_options(scene, start, goal, radius_m=profile.radius_m, body_height_m=profile.body_height_m):
            routes.append({"id": option.id, "label": option.label, "kind": option.kind,
                           "points": option.points_m, "lengthM": option.length_m,
                           "clearanceM": option.clearance_m, "status": "Footprint checked · geometric plan"})
        if training_dir is not None:
            policy, config, metadata, _ = load_navigation_artifact(training_dir, scene, profile)
            env = ReducedOrderNavEnv(scene, profile, config, seed=0)
            obs = env.reset(NavScenario(tuple(start), yaw, tuple(goal)))
            frames = [frame(0, env.position, env.yaw)]
            min_clearance = clearance(scene, start, profile.radius_m, profile.body_height_m)
            for index in range(config.horizon_steps):
                obs, _, done, final_info = env.step(policy.predict(obs))
                frames.append(frame((index + 1) * config.dt_s, env.position, env.yaw))
                min_clearance = min(min_clearance, final_info["clearance_m"])
                if done:
                    break
            points = [f["position"] for f in frames]
            routes.append({"id": "learned-ppo", "label": "PPO rollout", "kind": "learned", "points": points,
                           "lengthM": sum(math.dist(a, b) for a, b in zip(points, points[1:])),
                           "clearanceM": min_clearance,
                           "status": "Reached target" if final_info["success"] else "Collision stop" if final_info["collision"] else "Time limit"})
            cfg = metadata["training"]
            training = {"algorithm": "A* imitation + PPO" if metadata.get("warm_start", {}).get("epochs", 0) else "PPO", "model": metadata["abstraction"],
                        "steps": cfg["updates"] * cfg["rollout_steps"] * cfg["n_envs"], "seed": cfg["seed"],
                        "heldoutSuccessRate": metadata["evaluation"]["trained"]["success_rate"],
                        "baselineSuccessRate": metadata["evaluation"]["untrained"]["success_rate"],
                        "astarSuccessRate": metadata["evaluation"]["astar_baseline"]["success_rate"],
                        "warmStartedSuccessRate": metadata["evaluation"].get("warm_started", {}).get("success_rate"),
                        "heldoutEpisodes": metadata["evaluation"]["trained"]["episodes"],
                        "evaluationScope": "Same static proxy; 0.75–2 m goals, initial heading noise 0.35 rad. This evaluation did not establish detour navigation.",
                        "checkpointSha256": metadata["checkpoint_sha256"]}
    else:
        # Fixed arms never receive a learned base trajectory.
        frames = [frame(0, start, 0), frame(4, start, 0)]
    if demo_cleaning:
        # Explicit opt-in numerical overlay: no inferred counter/surface identity.
        patch = CleaningPatch(tuple(goal), (.45, .35), max(.05, floor_z + .8), "Numerical demonstration patch")
        for action in ("wipe", "sweep"):
            intent = plan_cleaning_intent(profile.id, patch, action=action, synthetic_numerical_preview=True)
            if len(intent.sweep_points_m) >= 2:
                cleaning.append({"id": f"intent-{action}", "label": "Wipe demonstration patch" if action == "wipe" else "Sweep demonstration patch",
                                 "kind": action, "points": intent.sweep_points_m,
                                 "status": intent.status, "reason": intent.note + " Target is numerical, not a verified household surface."})
    capabilities = {
        "navigation": "Scene-trained footprint PPO; physical transfer unverified" if training else "Stationary base; navigation not applicable" if not profile.mobile else "Geometric routes; no learned artifact loaded",
        "locomotion": "Matched frozen G1 PPO/LSTM available; separate native evaluation required" if profile.physics_robot_id else "No compatible physics controller supplied",
        "cleaning": "Tool-path intent only; tool, reach, contact and cleaning effect unverified" if cleaning else "No cleaning tool/controller configured",
    }
    result = {"version": 1, "robot": profile.id, "readiness": "PREVIEW_ONLY", "training": training,
              "scene": {"sha256": scene.hashes["scene.json"], "manifestUrl": "/policies/scene.json", "scaleConfirmed": scene.source.get("scaleConfirmed") is True,
                        "reviewed": scene.source.get("reviewed") is True, "synthetic": scene.source.get("synthetic") is True},
              "capability": capabilities, "replay": {"version": 1, "source": "backend", "robot": profile.id,
                        "duration": frames[-1]["time"], "frames": frames},
              "routes": routes, "cleaning": cleaning, "profile": profile.contract(),
              "rolloutResult": final_info,
              "caveats": [*scene.caveats, "Navigation profiles are assumed conservative footprints, not calibrated robot models.",
                          "Procedural robot motion visualizes a reduced-order trajectory, not gait or contact physics.",
                          "Per-scene training is adaptation; hardware remains UNVERIFIED."]}
    if demo_cleaning:
        result["caveats"].append("Cleaning targets are explicit numerical demo overlays; no cleaning action was physically executed.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scene.root / "scene.json", path.parent / "scene.json")
    if floor_audit is not None:
        audit = json.loads(Path(floor_audit).read_text())
        if audit.get("authority") != "input-floor-support-audit" or audit.get("inputs", {}).get("scene_sha256") != scene.hashes["scene.json"]:
            raise ValueError("Floor-support audit does not match the source scene")
        if audit.get("classification") == "no_sampled_mesh_support_with_positive_envelope_clearance":
            run_note = "Proxy navigation only: no sampled mesh floor point also fits the audited G1 clearance envelope. Review the supplied collision geometry."
            result["scene"].update(supportStatus="sampled_no_overlap", supportNote=run_note)
            result["caveats"].append(run_note)
        shutil.copyfile(floor_audit, path.parent / "floor-audit.json")
        result["scene"]["floorAuditUrl"] = "/policies/floor-audit.json"
    if native_artifacts is not None and profile.physics_robot_id is not None:
        _attach_native(result, path.parent, Path(native_artifacts))
    path.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    return result


def _attach_native(run, directory, artifacts):
    manifest = json.loads((artifacts / "manifest.json").read_text())
    report = json.loads((artifacts / "report.json").read_text())
    for name in ("report", "replay"):
        if sha256_file(artifacts / f"{name}.json") != manifest[f"{name}_sha256"]:
            raise ValueError(f"Native {name} hash mismatch")
    if report["hashes"]["scene_json_sha256"] != run["scene"]["sha256"]:
        raise ValueError("Native evidence belongs to another scene")
    learned = report.get("navigation_policy", {}).get("trained", False)
    if learned:
        expected = (run.get("training") or {}).get("checkpointSha256")
        recorded = report.get("navigation_policy", {}).get("checkpoint_sha256")
        if not expected or recorded != expected or report["hashes"].get("navigation_policy_sha256") != expected:
            raise ValueError("Native learned-navigation evidence does not match the loaded navigation checkpoint")
    name = "g1-native" if learned else "g1-native-baseline"
    destination = directory / name
    destination.mkdir(exist_ok=True)
    for filename in ("report.json", "replay.json", "manifest.json", "THIRD_PARTY_LICENSE.txt"):
        if (artifacts / filename).exists():
            shutil.copyfile(artifacts / filename, destination / filename)
    video = artifacts / "demo-1080p.mp4"
    if video.exists():
        shutil.copyfile(video, destination / "demo.mp4")
        run["nativeVideoUrl"] = f"/policies/{name}/demo.mp4"
    run["nativeReplayUrl"] = f"/policies/{name}/replay.json"
    run["nativeManifestUrl"] = f"/policies/{name}/manifest.json"
    run["validation"] = {"model": "Native MuJoCo + frozen G1 PPO/LSTM",
        "status": ("Matched learned-policy transfer test; separate episode" if learned else "Geometric-controller baseline; separate task") + ": " + report["execution"],
        "completed": report["execution"] == "completed", "forbiddenContacts": report["metrics"]["forbidden_contacts"],
        "reportUrl": f"/policies/{name}/report.json"}
    qualification_path = artifacts / "qualification.json"
    if qualification_path.exists():
        qualification = json.loads(qualification_path.read_text())
        provenance = qualification.get("provenance", {}).get("hashes", {})
        for key in ("scene_json_sha256", "navigation_policy_sha256", "navigation_controller_sha256"):
            if provenance.get(key) != report["hashes"].get(key):
                raise ValueError("Native qualification does not match the displayed controller/scene")
        coverage = qualification["coverage"]
        shutil.copyfile(qualification_path, destination / "qualification.json")
        run["validation"].update(qualificationUrl=f"/policies/{name}/qualification.json",
            qualificationEpisodes=coverage["episodes"], qualificationSuccesses=coverage["successes"],
            qualificationComplete=coverage["qualification_complete"],
            qualificationScope=coverage["tested_variation"])
