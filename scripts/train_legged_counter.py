"""Fit contact control parameters on a kitchen proxy, then test held-out starts."""
import argparse
import copy
import json
import math
from pathlib import Path
from xml.etree.ElementTree import ElementTree

import mujoco
import numpy as np

from pipeline.room_mesh import cell_index
from pipeline.task_planner import ROBOTS, World, plan_task
from scripts.kitchen_manipulator_models import GO2_ARM_JOINTS, LegActuation, g1_arm, go2_arm
from scripts.run_legged_aisle import Go2Policy
from scripts.train_stretch_aisle import FRAME_INTERVAL, MENAGERIE_COMMIT, angle_error, load_model, yaw_of

MAX_FORCE = 45.
MIN_CONTACT = .25


class CounterTrial:
    def __init__(self, robot, xml, assets, checkpoint):
        self.robot = robot
        self.world = World()
        if robot == "g1":
            from morpheus_policy.policies.g1 import DEFAULT_ANGLES, G1Policy, JOINT_NAMES, UPSTREAM_COMMIT

            G1Policy.validate_assets(assets)
            ref = load_model(assets / "robot/g1_12dof.xml", self.world)
            actor = G1Policy(assets / "motion.pt")
            tree = g1_arm(xml, JOINT_NAMES)
            self.revision = UPSTREAM_COMMIT
            self.leg_names, self.angles = JOINT_NAMES, DEFAULT_ANGLES
            self.visual_model = "g1-hands"
            self.key, self.height, self.fall_height = "stand", .79, .52
            self.goal = np.array([1.865, 1.785])
            self.center, self.half_extents = (1.493, 1.945, .904), (.010, .100, .033)
            self.reach_start, self.release, self.end = 5.4, 9.25, 12.
        else:
            tree = go2_arm(xml)
            ref = load_model(xml, self.world)
            actor = Go2Policy(checkpoint)
            self.revision = Go2Policy.revision
            self.leg_names = [ref.joint(i).name for i in range(1, ref.njnt)]
            self.angles = Go2Policy.angles
            self.visual_model = "go2-arm"
            self.key, self.height, self.fall_height = "home", .27, .17
            self.goal = np.array([1.95, 1.75])
            self.center, self.half_extents = (1.443, 1.748, .898), (.010, .095, .032)
            self.reach_start, self.release, self.end = 6., 12.8, 15.
        self.robot_tree = tree
        # load_model appends the kitchen to the tree in place.
        marker = dict(name="counter_marker", type="box", pos=" ".join(map(str, self.center)),
                      size=" ".join(map(str, self.half_extents)))
        model = self.model = load_model(xml, self.world, [marker], robot_tree=copy.deepcopy(tree))
        self.data = mujoco.MjData(model)
        self.legs = LegActuation(actor, ref, model, self.leg_names)
        self.joints = {model.joint(i).name: int(model.jnt_qposadr[i]) for i in range(1, model.njnt)}
        self.marker, self.floor = model.geom("counter_marker").id, model.geom("aisle_floor").id
        if robot == "g1":
            names = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
            arm = [f"right_{name}_joint" for name in names]
            self.arm_pos = [model.joint(n).qposadr[0] for n in arm]
            self.arm_vel = [model.joint(n).dofadr[0] for n in arm]
            self.arm_controls = [model.actuator(n).id for n in arm]
            self.arm_limits = np.array([model.joint(n).range for n in arm])
            self.tip_site = model.site("reach_tip").id
            self.jacobian = np.zeros((3, model.nv))
        else:
            self.arm_controls = [model.actuator(name).id for name in GO2_ARM_JOINTS]

    def control(self, step, settings):
        data = self.data
        t = data.time
        delta = self.goal - data.qpos[:2]
        distance = np.linalg.norm(delta)
        heading = yaw_of(data.qpos[3:7])
        forward = np.array([math.cos(heading), math.sin(heading)])
        side = np.array([-forward[1], forward[0]])
        approach_error = angle_error(math.atan2(*delta[::-1]), heading)
        if self.robot == "g1":
            gain, target_x = settings
            turn = approach_error if t < 2 and distance > .07 else angle_error(math.pi, heading)
            command = [np.clip(forward @ delta * 1.3, -.10, .25), np.clip(side @ delta * .8, -.1, .1),
                       np.clip(turn * 1.5, -.75, .75)]
            if step % 10 == 0 and t > self.reach_start:
                target = np.array([target_x, self.center[1], .901])
                if t >= self.release:
                    target += [.18, 0, .08]
                mujoco.mj_jacSite(self.model, data, self.jacobian, None, self.tip_site)
                jac = self.jacobian[:, self.arm_vel]
                update = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(3) * .007, target - data.site_xpos[self.tip_site])
                values = data.qpos[self.arm_pos] + np.clip(gain * update, -.024, .024)
                data.ctrl[self.arm_controls] = np.clip(values, self.arm_limits[:, 0] + .01, self.arm_limits[:, 1] - .01)
        else:
            lift, extension = settings
            if t < 3.5 and distance > .08:
                command = [min(.2, distance * 1.15) * max(0, math.cos(approach_error)), 0,
                           np.clip(approach_error * 2, -.65, .65)]
            else:
                face = angle_error(math.atan2(*(np.array([1.45, self.center[1]]) - data.qpos[:2])[::-1]), heading)
                command = [np.clip(forward @ delta * 1.1, -.09, .09), np.clip(side @ delta * 1.05, -.12, .12),
                           np.clip(face * 1.6, -.55, .55)]
            data.ctrl[self.arm_controls] = [lift if self.reach_start < t < 13.8 else 0,
                                           min(extension, max(0, (t - 7.9) * .055)) if t < self.release else 0,
                                           .020, .020]
        self.legs.step(data, command)

    def measure(self, wrench):
        model, data = self.model, self.data
        force = 0.
        for i, contact in enumerate(data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            if self.marker in (a, b):
                other = model.geom(b if a == self.marker else a)
                hand = model.body(int(other.bodyid[0])).name
                allowed = (hand.startswith("right_hand_") or other.name == "reach_pad") if self.robot == "g1" else other.name in {"kit_left_pad", "kit_right_pad"}
                if not allowed:
                    return force, f"forbidden marker contact: {other.name}"
                mujoco.mj_contactForce(model, data, i, wrench)
                force += wrench[0]
            elif self.floor in (a, b):
                other = model.geom(b if a == self.floor else a)
                foot = model.body(int(other.bodyid[0])).name.endswith("ankle_roll_link") if self.robot == "g1" else other.name in {"FL", "FR", "RL", "RR"}
                if not foot:
                    return force, f"non-foot ground contact: {other.name}"
            elif 0 in (model.geom_bodyid[a], model.geom_bodyid[b]):
                return force, f"obstacle contact: {model.geom(a).name}, {model.geom(b).name}"
        return float(force), "marker force limit" if force > MAX_FORCE else None

    def run(self, settings, offset, record=False):
        model, data = self.model, self.data
        mujoco.mj_resetDataKeyframe(model, data, model.key(self.key).id)
        data.qpos[:2] = self.world.spawn + offset[:2]
        data.qpos[2] = self.height + offset[3]
        heading = math.atan2(*(self.goal - data.qpos[:2])[::-1]) + offset[2]
        data.qpos[3:7] = [math.cos(heading / 2), 0, 0, math.sin(heading / 2)]
        data.qpos[self.legs.positions] = self.angles
        mujoco.mj_forward(model, data)
        self.legs.reset(data)
        dt = model.opt.timestep
        sample_period = round(FRAME_INTERVAL / dt)
        wrench = np.zeros(6)
        frames = []
        first_touch = failure = None
        force = peak = longest = uninterrupted = 0.
        minimum = math.inf
        ticks = round(self.end / dt)
        for step in range(ticks + 1):
            if step % sample_period == 0:
                pos = data.qpos[:2].copy()
                row, col, valid = cell_index(self.world.mesh, pos)
                clearance = min(self.world.clearance([pos])[0], self.world.mesh_clear[row[0], col[0]]) - ROBOTS[self.robot]["radius"]
                minimum = min(minimum, float(clearance))
                if not (valid[0] and self.world.mesh["interior"][row[0], col[0]]) or clearance <= 0:
                    failure = "base clearance"
                    break
                if record:
                    frames.append({"time": round(data.time, 3), "position": data.qpos[:3].round(6).tolist(),
                                   "yaw": round(yaw_of(data.qpos[3:7]) - math.pi / 2, 6), "armExtension": 0,
                                   "physicsPose": {"quaternion": data.qpos[3:7].round(8).tolist(),
                                                   "joints": [round(float(data.qpos[j]), 6) for j in self.joints.values()]},
                                   "contactForce": round(force, 3)})
            if step == ticks:
                break
            self.control(step, settings)
            mujoco.mj_step(model, data)
            force, failure = self.measure(wrench)
            peak = max(peak, force)
            if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning) or data.qpos[2] < self.fall_height:
                failure = "fall or invalid physics state"
            if failure:
                break
            if self.reach_start <= data.time < self.release and force > 1.:
                if first_touch is None:
                    first_touch = float(data.time)
                uninterrupted += dt
                longest = max(longest, uninterrupted)
            else:
                uninterrupted = 0.
        return {"success": failure is None and longest >= MIN_CONTACT and force < 1., "failure": failure,
                "dwell": longest, "firstTouch": first_touch, "peakForce": peak, "minimumClearance": minimum,
                "goalError": float(np.linalg.norm(data.qpos[:2] - self.goal)), "frames": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=("g1", "go2"))
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.robot == "g1" and args.asset_root is None or args.robot == "go2" and args.checkpoint is None:
        parser.error("Supply --asset-root for G1 or --checkpoint for Go2")
    trial = CounterTrial(args.robot, args.robot_xml.resolve(), args.asset_root, args.checkpoint)
    search = ((.11, .125, .14), (1.508, 1.512)) if args.robot == "g1" else ((.40, .42), (.219, .23, .241))
    limit = np.array([.009, .012, .016, .010])
    training = np.random.default_rng(736).uniform(-limit, limit, (4, len(limit)))
    scores = []
    for first in search[0]:
        for second in search[1]:
            settings = (first, second)
            runs = [trial.run(settings, offset) for offset in training]
            print(args.robot, settings, [(r["success"], r["failure"], round(r["dwell"], 2)) for r in runs], flush=True)
            scores.append(((sum(r["success"] for r in runs), min(r["dwell"] for r in runs),
                            -max(r["peakForce"] for r in runs)), settings))
    _, settings = max(scores)
    held = np.random.default_rng(2736).uniform(-limit, limit, (12, len(limit)))
    results = [trial.run(settings, offset) for offset in held]
    actual = trial.run(settings, np.zeros(len(limit)), record=True)
    frames = actual.pop("frames")
    for result in results:
        result.pop("frames")
    passes = sum(result["success"] for result in results)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps({"settings": settings, "nominal": actual, "heldOut": results}, indent=2) + "\n")
    print("Nominal", actual, "held-out", passes, flush=True)
    if not actual["success"] or passes != len(held):
        return 2
    envelope = plan_task("Counter reach", args.robot, trial.world, use_openai=False)
    replay = envelope["replay"]
    replay["frames"], replay["duration"] = frames, frames[-1]["time"]
    touch = round(math.ceil(actual["firstTouch"] / FRAME_INTERVAL) * FRAME_INTERVAL, 3)
    replay["contact"] = {"point": [trial.center[0] + trial.half_extents[0], *trial.center[1:]], "travel": [0, 0, 0],
                         "padSize": [2 * x for x in trial.half_extents], "kind": "counter", "start": trial.reach_start,
                         "contact": touch, "release": trial.release, "end": replay["duration"]}
    replay["events"] = [{"time": 0, "label": "Walking to the counter under MuJoCo leg actuation"},
                        {"time": trial.reach_start, "label": "Moving the physical arm toward the counter marker"},
                        {"time": touch, "label": "Hand contact measured in MuJoCo"},
                        {"time": trial.release, "label": "Retracting the arm"},
                        {"time": replay["duration"], "label": "Counter reach complete · contact and release measured"}]
    replay["physics"] = {"engine": f"MuJoCo {mujoco.__version__}", "modelRevision": f"{MENAGERIE_COMMIT}/{trial.revision}",
                         "visualModel": trial.visual_model, "jointNames": list(trial.joints),
                         "goalError": actual["goalError"], "minimumClearance": actual["minimumClearance"],
                         "heldOutPassed": passes, "heldOutTotal": len(held), "contactDwellSeconds": actual["dwell"],
                         "peakContactForceNewtons": actual["peakForce"]}
    replay["verdict"]["planner"] = "MuJoCo contact parameter search with a pretrained walking policy"
    replay["verdict"]["summary"] = f"{ROBOTS[args.robot]['name']} contacted and released the fixture with {actual['dwell']:.2f} s of continuous contact."
    replay["verdict"]["steps"][0]["note"] = f"{actual['minimumClearance']:.2f} m minimum sampled base clearance."
    replay["verdict"]["steps"][1]["note"] = f"{actual['peakForce']:.1f} N peak normal force. {passes}/{len(held)} held-out starts completed."
    (args.output / f"{args.robot}.json").write_text(json.dumps(envelope, separators=(",", ":"), allow_nan=False) + "\n")
    ElementTree(trial.robot_tree).write(args.output / f"{trial.visual_model}.xml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
