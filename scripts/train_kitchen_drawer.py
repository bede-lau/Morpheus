"""Tune drawer approach and grasp against a passive rail in MuJoCo."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.room_mesh import cell_index
from pipeline.task_planner import ROBOTS, World, plan_task
from scripts.kitchen_manipulator_models import GO2_ARM_JOINTS, LegActuation, g1_arm, go2_arm
from scripts.run_legged_aisle import Go2Policy
from scripts.train_stretch_aisle import FRAME_INTERVAL, MENAGERIE_COMMIT, angle_error, load_model, yaw_of

HANDLE = (1.761, -.763, .855)
DRAWER_GEOMETRY = (
    ("handle", (0, 0, 0), (.018, .023, .016), .09),
    ("mount", (-.051, 0, -.071), (.034, .010, .074), .12),
    ("front", (-.078, 0, -.220), (.014, .145, .038), .65),
    ("tray", (-.206, 0, -.253), (.126, .143, .009), .85),
)
PHASES = (("drive", 55.), ("align", 4.), ("lift", 2.3), ("orient", 1.5),
          ("reach", 1.4), ("grasp", 1.), ("pull", 5.6), ("release", 2.))
GO2_PHASES = (("drive", 15.), ("align", 6.), ("lift", 2.), ("reach", 5.),
              ("grasp", 2.), ("pull", 6.2), ("release", 3.8))
G1_PHASES = (("drive", 19.), ("align", 5.), ("lift", 3.), ("reach", 5.),
             ("grasp", 2.), ("pull", 6.5), ("release", 4.5))


def vector(values):
    return " ".join(map(str, values))


def drawer_tree(tree):
    """Attach a mass-bearing rail and handle to the scene via a passive slide joint."""
    for key in tree.findall("./keyframe/key"):
        for attr in ("qpos", "qvel"):
            if key.get(attr):
                key.set(attr, key.get(attr) + " 0")
    drawer = ET.Element("body", name="drawer", pos=vector(HANDLE))
    ET.SubElement(drawer, "joint", name="drawer_rail", type="slide", axis="1 0 0", range="0 .18",
                  damping="1.5", frictionloss=".40")
    for name, pos, size, mass in DRAWER_GEOMETRY:
        ET.SubElement(drawer, "geom", name=f"drawer_{name}", type="box", pos=vector(pos), size=vector(size),
                      mass=str(mass), friction="1.3 .02 .003", condim="4")
    return tree, drawer


class DrawerTrial:
    def __init__(self, xml, checkpoint=None, g1_assets=None):
        self.world = World()
        self.legs = None
        self.robot = "g1" if g1_assets else "go2" if checkpoint else "stretch"
        self.phases = {"g1": G1_PHASES, "go2": GO2_PHASES, "stretch": PHASES}[self.robot]
        self.goal = np.array({"g1": [2.135, -.94], "go2": [2.23, -.762], "stretch": [2.257, -.762]}[self.robot])
        self.revision = MENAGERIE_COMMIT
        if g1_assets:
            from morpheus_policy.policies.g1 import DEFAULT_ANGLES, G1Policy, JOINT_NAMES, UPSTREAM_COMMIT

            G1Policy.validate_assets(g1_assets)
            observation = load_model(g1_assets / "robot/g1_12dof.xml", self.world)
            actor, names, self.angles = G1Policy(g1_assets / "motion.pt"), JOINT_NAMES, DEFAULT_ANGLES
            self.revision += f"/{UPSTREAM_COMMIT}"
            tree = g1_arm(xml, names)
            wrist = tree.find(".//body[@name='right_wrist_yaw_link']")
            ET.SubElement(wrist, "site", name="pinch_center", pos=".142 .043 .003")
        elif checkpoint:
            tree = go2_arm(xml)
            observation = load_model(xml, self.world)
            actor, self.angles = Go2Policy(checkpoint), Go2Policy.angles
            names = [observation.joint(i).name for i in range(1, observation.njnt)]
            self.revision += f"/{Go2Policy.revision}"
        else:
            if hashlib.sha256(xml.read_bytes()).hexdigest() != "154088989df3b3d01a184b8e2df7cb82481efcd532f2498719acf1d77f001a6d":
                raise ValueError("Expected the pinned Stretch Menagerie model")
            tree = ET.parse(xml).getroot()
            tree.find("./actuator/position[@name='gripper']").set("forcerange", "-20 20")
        tree, drawer = drawer_tree(tree)
        table = [dict(name="practice_table_top", type="box", pos="1.48 -.76 .687", size=".21 .325 .013")]
        for i, (x, y) in enumerate((x, y) for x in (1.305, 1.655) for y in (-1.05, -.47)):
            table.append(dict(name=f"practice_table_leg_{i}", type="box", pos=f"{x} {y} .337", size=".012 .012 .337"))
        model = self.model = load_model(xml, self.world, table, robot_tree=tree, bodies=[drawer])
        self.data = mujoco.MjData(model)
        if self.robot != "stretch":
            self.legs = LegActuation(actor, observation, model, names)
        if checkpoint:
            self.arm = [model.actuator(n).id for n in GO2_ARM_JOINTS]
            self.fingers = {model.body(f"kit_{side}_finger").id: side for side in ("left", "right")}
        elif g1_assets:
            arm = [f"right_{name}_joint" for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
            self.arm = [model.actuator(n).id for n in arm]
            self.arm_pos = [model.joint(n).qposadr[0] for n in arm]
            self.arm_vel = [model.joint(n).dofadr[0] for n in arm]
            self.arm_limits = np.array([model.joint(n).range for n in arm])
            self.finger_controls = [model.actuator(f"right_hand_{n}_joint").id for n in
                                    ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
            self.fingers = {i: model.body(i).name for i in range(model.nbody) if model.body(i).name.startswith("right_hand_")}
            self.wrist_body = model.body("right_wrist_yaw_link").id
            self.tip = model.site("pinch_center").id
            self.linear = np.zeros((3, model.nv))
            self.angular = np.zeros_like(self.linear)
        else:
            self.wheels = [model.actuator(n).id for n in ("left_wheel_vel", "right_wheel_vel")]
            self.arm = [model.actuator(n).id for n in ("lift", "arm", "wrist_yaw", "wrist_pitch", "gripper")]
            self.fingers = {model.body(f"{part}_{side}").id: side for part in ("rubber_tip", "link_gripper_finger")
                            for side in ("left", "right")}
        self.drawer = model.joint("drawer_rail").qposadr[0]
        self.drawer_body = model.body("drawer").id
        self.joints = {model.joint(i).name: int(model.jnt_qposadr[i]) for i in range(1, model.njnt)
                       if model.joint(i).name != "drawer_rail"}
        self.handle = model.geom("drawer_handle").id
        self.floor = model.geom("aisle_floor").id

    def control_g1(self, phase, step, settings, yaw):
        roll, lateral, withdrawal = settings
        data = self.data
        elapsed = step * self.model.opt.timestep - 24.
        delta = self.goal - data.qpos[:2]
        distance = np.linalg.norm(delta)
        front = np.array([math.cos(yaw), math.sin(yaw)])
        side = np.array([-front[1], front[0]])
        if phase == "drive" and distance > .12:
            heading = angle_error(math.atan2(*delta[::-1]), yaw)
            command = [min(.33, distance * .9) * max(0, math.cos(heading)), 0, np.clip(heading * 2, -.58, .58)]
        else:
            command = [np.clip(front @ delta * 1.4, -.12, .12), np.clip(side @ delta * 1.1, -.10, .10),
                       np.clip(1.55 * angle_error(math.pi, yaw), -.48, .48)]
        self.legs.step(data, command)
        if step % 10 == 0 and elapsed > 1:
            extension = .10 if elapsed < 3 else max(0., min(.195, (elapsed - 10) * .035))
            reach = np.array(HANDLE) + [.008 + extension, lateral, .004]
            if elapsed > 16.5:
                reach += np.array([.04, -.023, withdrawal]) * min(1., (elapsed - 16.5) * .75)
            rotation = Rotation.from_euler("z", math.pi).as_matrix() @ Rotation.from_euler("x", roll).as_matrix()
            delta_rotation = Rotation.from_matrix(rotation @ data.site_xmat[self.tip].reshape(3, 3).T).as_rotvec()
            target = np.r_[reach - data.site_xpos[self.tip], delta_rotation * .12]
            mujoco.mj_jacSite(self.model, data, self.linear, self.angular, self.tip)
            jac = np.r_[self.linear[:, self.arm_vel], self.angular[:, self.arm_vel] * .12]
            update = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * .007, target)
            data.ctrl[self.arm] = np.clip(data.qpos[self.arm_pos] + np.clip(update * .16, -.03, .03),
                                         self.arm_limits[:, 0] + .008, self.arm_limits[:, 1] - .008)
        data.ctrl[self.finger_controls] = ([.16, -.8, -1.05, .8, .95, .8, .95] if 8 <= elapsed <= 19. else
                                           [0., -.10, 0., 0., 0., 0., 0.])

    def control_go2(self, phase, tick, settings, yaw):
        lift, reach, _ = settings
        data = self.data
        displacement = self.goal - data.qpos[:2]
        distance = np.linalg.norm(displacement)
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        side = np.array([-forward[1], forward[0]])
        error = angle_error(math.atan2(*displacement[::-1]), yaw)
        if phase == "drive" and distance > .115:
            command = [min(.30, distance * 1.25) * max(0, math.cos(error)), 0, np.clip(error * 2, -.60, .60)]
        else:
            command = [np.clip(forward @ displacement * 1.25, -.10, .10), np.clip(side @ displacement * 1.1, -.11, .11),
                       np.clip(angle_error(math.pi, yaw) * 1.8, -.55, .55)]
        self.legs.step(data, command)
        elapsed = tick * self.model.opt.timestep
        extension = reach if phase == "grasp" else 0.
        if phase == "reach":
            extension = min(reach, elapsed * .045)
        elif phase == "pull":
            extension = max(0., reach - elapsed * .032)
        gripping = .028 if phase in ("grasp", "pull") else 0.
        elevated = phase not in ("drive", "align") and data.time < 37.6
        data.ctrl[self.arm] = [lift if elevated else 0., extension, gripping, gripping]

    def run(self, settings, offset, frames=None):
        lift, reach, wrist = settings
        model, data = self.model, self.data
        key, height = {"g1": ("stand", .79), "go2": ("home", .27), "stretch": ("stow", .005)}[self.robot]
        mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
        data.qpos[:2] = self.world.spawn + offset[:2]
        heading = math.atan2(*(self.goal - data.qpos[:2])[::-1]) if self.legs else -math.pi / 2
        initial_yaw = heading + offset[2]
        data.qpos[2:7] = [height, math.cos(initial_yaw / 2), 0, 0, math.sin(initial_yaw / 2)]
        if self.legs:
            data.qpos[self.legs.positions] = self.angles
        else:
            for name, value in (("joint_lift", .23), ("joint_wrist_yaw", 3.14), ("joint_wrist_pitch", -.4)):
                data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        if self.legs:
            self.legs.reset(data)
        dt = model.opt.timestep
        sample_period = round(FRAME_INTERVAL / dt)
        peak = grip_time = longest_grip = pull_with_grip = max_displacement = 0.
        force = 0.
        first_grasp = None
        minimum_clearance = math.inf
        previous_slide = 0.
        wrench = np.zeros(6)
        failure = None
        elapsed_steps = 0

        def capture():
            frames.append({"time": round(data.time, 3), "position": data.qpos[:3].round(6).tolist(),
                           "yaw": round(yaw_of(data.qpos[3:7]) - math.pi / 2, 6), "armExtension": 0,
                           "physicsPose": {"quaternion": data.qpos[3:7].round(8).tolist(),
                                           "joints": [round(float(data.qpos[j]), 6) for j in self.joints.values()]},
                           "contactForce": round(force, 3), "drawerDisplacement": round(float(data.qpos[self.drawer]), 6)})

        for phase, duration in self.phases:
            for tick in range(round(duration / dt)):
                x, y = data.qpos[:2]
                yaw = yaw_of(data.qpos[3:7])
                if tick % sample_period == 0:
                    if frames is not None:
                        capture()
                    pos = (x, y)
                    row, col, valid = cell_index(self.world.mesh, pos)
                    clearance = min(self.world.clearance([pos])[0], self.world.mesh_clear[row[0], col[0]]) - ROBOTS[self.robot]["radius"]
                    minimum_clearance = min(minimum_clearance, float(clearance))
                    if not (valid[0] and self.world.mesh["interior"][row[0], col[0]]) or clearance <= 0:
                        failure = "base clearance"
                        break
                if self.robot == "g1":
                    self.control_g1(phase, elapsed_steps + tick, settings, yaw)
                elif self.legs:
                    self.control_go2(phase, tick, settings, yaw)
                else:
                    heading_error = angle_error(-math.pi / 2, yaw)
                    speed, turn = 0., np.clip(1.1 * heading_error, -.028, .028)
                    if phase == "drive":
                        ahead_y = max(-.765, y - .52)
                        progress = np.clip((self.world.spawn[1] - ahead_y) / 2.2, 0, 1)
                        lane = self.world.spawn[0] + .382 * progress ** 2 * (3 - 2 * progress)
                        error = angle_error(math.atan2(ahead_y - y, lane - x), yaw) if y + .765 > .16 else heading_error
                        speed = min(.079, .55 * max(0, y + .762)) * max(0, math.cos(error))
                        turn = np.clip(1.6 * error, -.033, .033)
                    if phase not in ("drive", "align"):
                        extension = reach if phase in ("reach", "grasp") else 0.
                        if phase == "pull":
                            extension = max(0, reach - tick * dt * .038)
                        data.ctrl[self.arm] = [lift, extension, 3.14 if phase == "lift" else wrist,
                                               -.4 if phase == "lift" else 0., -.016 if phase in ("grasp", "pull") else .02]
                    wheel_targets = np.clip(np.array([speed - .17035 * turn, speed + .17035 * turn]) * 3 / .0508, -6, 6)
                    data.ctrl[self.wheels] += np.clip(wheel_targets - data.ctrl[self.wheels], -.004, .004)
                mujoco.mj_step(model, data)
                touched = set()
                force = 0.
                for index, contact in enumerate(data.contact):
                    a, b = int(contact.geom1), int(contact.geom2)
                    if self.floor in (a, b):
                        if self.legs:
                            foot = model.geom(b if a == self.floor else a)
                            upright = model.body(int(foot.bodyid[0])).name.endswith("ankle_roll_link") if self.robot == "g1" else foot.name in {"FL", "FR", "RL", "RR"}
                            if not upright:
                                failure = "non-foot ground contact"
                        continue
                    body_a, body_b = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
                    if 0 in (body_a, body_b) and self.drawer_body not in (body_a, body_b):
                        failure = "obstacle contact"
                        break
                    if self.handle in (a, b):
                        other = body_b if a == self.handle else body_a
                        if self.robot == "g1" and other not in self.fingers and other != self.wrist_body:
                            failure = "non-hand contact with handle"
                            break
                        mujoco.mj_contactForce(model, data, index, wrench)
                        force += wrench[0]
                        if other in self.fingers and wrench[0] > .2:
                            touched.add(self.fingers[other])
                peak = max(peak, float(force))
                displacement = float(data.qpos[self.drawer])
                engaged = bool(touched) if self.robot == "g1" else len(touched) == 2
                if first_grasp is None and phase in ("grasp", "pull") and engaged:
                    first_grasp = float(data.time)
                if phase == "pull" and engaged:
                    grip_time += dt
                    longest_grip = max(longest_grip, grip_time)
                    pull_with_grip += max(0, displacement - previous_slide)
                else:
                    grip_time = 0.
                previous_slide = displacement
                max_displacement = max(max_displacement, displacement)
                min_height = .52 if self.robot == "g1" else .17
                if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning) or self.legs and data.qpos[2] < min_height:
                    failure = "invalid physics state"
                if failure:
                    break
            if failure:
                break
            elapsed_steps += round(duration / dt)
        if frames is not None:
            capture()
        final_slide = float(data.qpos[self.drawer])
        return {"success": (failure is None and final_slide >= .12 and pull_with_grip >= .10
                            and longest_grip >= .25 and force < .2),
                "failure": failure, "peakDrawerTravelMeters": max_displacement, "finalDrawerTravelMeters": final_slide,
                "travelUnderGraspMeters": pull_with_grip, "continuousGraspSeconds": longest_grip,
                "peakHandleForceNewtons": peak, "finalHandleForceNewtons": force, "firstGrasp": first_grasp,
                "minimumClearanceMeters": minimum_clearance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-xml", type=Path, required=True)
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--go2-checkpoint", type=Path, help="Run Go2 with its simulated arm and pinned walking actor")
    models.add_argument("--g1-assets", type=Path, help="Run G1 with articulated hands and the pinned walking actor")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trial = DrawerTrial(args.robot_xml.resolve(), args.go2_checkpoint, args.g1_assets)
    limit = np.array([.008, .012, .015])
    candidates = []
    offsets = np.random.default_rng(284).uniform(-limit, limit, (3, len(limit)))
    if trial.robot == "g1":
        grid = [(roll, lateral, withdrawal) for roll in (1.22, 1.35) for lateral in (-.015, -.006) for withdrawal in (.10, .185)]
        offsets = np.vstack([np.zeros(len(limit)), offsets])
    elif trial.robot == "go2":
        grid = [(lift, reach, 0.) for lift in (.365, .370, .380) for reach in (.174, .184, .197)]
    else:
        grid = [(lift, .139, wrist) for lift in (.75, .76, .77) for wrist in (.10, .15, .20)]
    for settings in grid:
        runs = [trial.run(settings, offset) for offset in offsets]
        score = (sum(run["success"] for run in runs), min(run["travelUnderGraspMeters"] for run in runs),
                 min(run["finalDrawerTravelMeters"] for run in runs))
        candidates.append((score, settings))
        print(settings, score, flush=True)
    _, settings = max(candidates)
    frames = []
    actual = trial.run(settings, np.zeros(len(limit)), frames=frames)
    held = np.random.default_rng(1284).uniform(-limit, limit, (12, len(limit)))
    results = [trial.run(settings, offset) for offset in held]
    report = {"settings": settings, "nominal": actual, "heldOut": results}
    with args.output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    passed = sum(run["success"] for run in results)
    print("Nominal", actual, "held-out", passed, flush=True)
    if not actual["success"] or passed != len(results):
        return 2
    envelope = plan_task("Drawer opening", trial.robot, trial.world, use_openai=False)
    replay = envelope["replay"]
    replay["frames"], replay["duration"] = frames, frames[-1]["time"]
    start = round(sum(seconds for _, seconds in trial.phases[:2]), 3)
    pull = round(sum(seconds for _, seconds in trial.phases[:-2]), 3)
    release = round(sum(seconds for _, seconds in trial.phases[:-1]), 3)
    grasp = round(math.ceil(actual["firstGrasp"] / FRAME_INTERVAL) * FRAME_INTERVAL, 3)
    travel = actual["finalDrawerTravelMeters"]
    replay["contact"] = {"point": HANDLE, "travel": [travel, 0, 0], "kind": "drawer", "start": start,
                         "contact": grasp, "release": release, "end": replay["duration"],
                         "drawerParts": [{"position": pos, "size": [2 * x for x in size]} for _, pos, size, _ in DRAWER_GEOMETRY]}
    replay["events"] = [{"time": 0, "label": "Navigating to the practice table"},
                        {"time": start, "label": "Positioning the gripper around the drawer handle"},
                        {"time": grasp, "label": "Grasping the handle with the fingers" if trial.robot == "g1" else "Grasping the handle with both fingers"},
                        {"time": pull, "label": "Pulling the drawer along its physical rails"},
                        {"time": release, "label": "Releasing the drawer handle"},
                        {"time": replay["duration"], "label": "Drawer open after release · travel measured in MuJoCo"}]
    replay["events"].sort(key=lambda event: event["time"])
    replay["physics"] = {"engine": f"MuJoCo {mujoco.__version__}", "modelRevision": trial.revision,
                         "jointNames": list(trial.joints), "goalError": float(np.linalg.norm(np.array(frames[-1]["position"][:2]) - trial.goal)),
                         "minimumClearance": actual["minimumClearanceMeters"], "heldOutPassed": passed,
                         "heldOutTotal": len(results), "contactDwellSeconds": actual["continuousGraspSeconds"],
                         "peakContactForceNewtons": actual["peakHandleForceNewtons"]}
    replay["verdict"]["planner"] = "MuJoCo drawer parameter search"
    if trial.robot == "g1":
        replay["physics"]["visualModel"] = "g1-hands"
    if trial.robot == "go2":
        replay["physics"]["visualModel"] = "go2-arm"
    replay["verdict"]["summary"] = f"{ROBOTS[trial.robot]['name']} grasped the handle and left the drawer {travel * 100:.1f} cm open after release."
    replay["verdict"]["steps"][0]["note"] = f"{actual['minimumClearanceMeters']:.2f} m minimum sampled base clearance."
    replay["verdict"]["steps"][1]["note"] = f"{actual['continuousGraspSeconds']:.2f} s continuous grasp. {passed}/{len(results)} held-out starts completed."
    with args.output.with_name(args.output.stem + f"-{trial.robot}.json").open("x") as file:
        json.dump(envelope, file, separators=(",", ":"), allow_nan=False)
        file.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
