"""Train a differential-drive feedback controller by CEM in the Stretch 3 MuJoCo model.

Inputs are the reviewed kitchen fixture and the pinned Menagerie robot model.
Export measured qpos and wheel commands. The reference route supplies only a
start and goal. Every subsequent state is produced by mj_step.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from pipeline.room_mesh import STEP, cell_index
from pipeline.task_planner import ROBOTS, World, plan_task

MENAGERIE_COMMIT = "eea4c7431b63da8278b0ff31638e0b0e941cec0e"
FRAME_INTERVAL = .05
GOAL_TOLERANCE = .12
PROXY_HEIGHT = 1.65
ROOT = Path(__file__).resolve().parents[1]


def load_model(robot_xml, world):
    tree = ET.parse(robot_xml).getroot()
    compiler = tree.find("compiler")
    for key in ("assetdir","meshdir","texturedir"):
        if compiler.get(key):
            compiler.set(key,str(robot_xml.parent / compiler.get(key)))
    body = tree.find("worldbody")
    ET.SubElement(body, "geom", type="plane", size="0 0 .1", name="aisle_floor")
    grids = world.mesh
    if grids is None:
        raise ValueError("The kitchen companion collider is required")
    xs, ys = grids["xs"], grids["ys"]
    for i, row in enumerate(grids["obstacle"]):
        # Merge neighbouring obstacle cells into a single contact geom per run.
        indices = np.flatnonzero(np.diff(np.r_[False, row, False]))
        for start, stop in indices.reshape(-1, 2):
            ET.SubElement(body, "geom", name=f"aisle_grid_{i}_{start}", type="box",
                          pos=f"{(xs[start]+xs[stop-1])/2} {ys[i]} {PROXY_HEIGHT/2}",
                          size=f"{(stop-start)*STEP/2} {STEP/2} {PROXY_HEIGHT/2}")
    for i, (left, bottom, right, top) in enumerate(world.rects):
        ET.SubElement(body, "geom", name=f"aisle_box_{i}", type="box",
                      pos=f"{(left+right)/2} {(bottom+top)/2} {PROXY_HEIGHT/2}",
                      size=f"{(right-left)/2} {(top-bottom)/2} {PROXY_HEIGHT/2}")
    return mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def angle_error(target, current):
    delta = target-current
    return math.atan2(math.sin(delta), math.cos(delta))


class Aisle:
    def __init__(self, robot_xml):
        self.world = World()
        self.source = plan_task("Aisle clearance","stretch",self.world,use_openai=False)
        route = self.source["replay"]["frames"]
        self.start = np.array(route[0]["position"][:2])
        self.goal = np.array(route[-1]["position"][:2])
        self.direction = self.goal-self.start
        self.length = np.linalg.norm(self.direction)
        self.direction /= self.length
        self.heading = math.atan2(self.direction[1], self.direction[0])
        self.model = load_model(robot_xml, self.world)
        self.data = mujoco.MjData(self.model)
        self.home = self.model.key("stow").id
        self.floor = self.model.geom("aisle_floor").id
        self.wheels = [self.model.actuator(name).id for name in ("left_wheel_vel", "right_wheel_vel")]
        self.joints = {self.model.joint(i).name:int(self.model.jnt_qposadr[i]) for i in range(1,self.model.njnt)}
        self.xml_sha = hashlib.sha256(robot_xml.read_bytes()).hexdigest()
        if self.xml_sha!="154088989df3b3d01a184b8e2df7cb82481efcd532f2498719acf1d77f001a6d":
            raise ValueError("Stretch XML checksum differs from the pinned Menagerie model")

    def episode(self, gains, offsets, record=False):
        cruise, steering, lookahead, brake = gains
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self.home)
        for name,value in (("joint_lift",.23),("joint_wrist_yaw",3.14),("joint_wrist_pitch",-.4)):
            d.qpos[self.joints[name]] = value
        d.qpos[:2] = self.start+offsets[:2]
        yaw = self.heading+offsets[2]
        d.qpos[2:7] = [.005, math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
        mujoco.mj_forward(m,d)
        trajectory, collision = [], None
        min_clearance = math.inf
        finish = False
        period = round(FRAME_INTERVAL/m.opt.timestep)
        for step in range(960):
            pos = d.qpos[:2].copy()
            distance = float(np.linalg.norm(self.goal-pos))
            row, col, valid = cell_index(self.world.mesh, pos)
            clearance = float(min(self.world.clearance([pos])[0],self.world.mesh_clear[row[0],col[0]]))-ROBOTS["stretch"]["radius"]
            min_clearance = min(min_clearance,clearance)
            if not (valid[0] and self.world.mesh["interior"][row[0],col[0]]) or clearance<=0:
                collision = "footprint clearance"
            if record:
                trajectory.append({"time":round(step*FRAME_INTERVAL,3),"position":d.qpos[:3].round(6).tolist(),
                                   "yaw":round(yaw_of(d.qpos[3:7])-math.pi/2,6), "armExtension":0,
                                   "physicsPose":{"quaternion":d.qpos[3:7].round(8).tolist(),
                                                  "joints":[round(float(d.qpos[i]),6) for i in self.joints.values()],
                                                  "wheelTargets":d.ctrl[self.wheels].round(6).tolist()}})
            if collision:
                break
            if distance < GOAL_TOLERANCE and np.linalg.norm(d.qvel[:2])<.065:
                finish = True
                break
            progress = np.dot(pos-self.start,self.direction)
            target = self.start+min(self.length,progress+lookahead)*self.direction
            error = angle_error(math.atan2(*(target-pos)[::-1]),yaw_of(d.qpos[3:7]))
            linear = min(cruise,brake*max(0,distance-.065))*max(0,math.cos(error))
            omega = np.clip(steering*error,-1.3,1.3)
            # Menagerie's velocity control input is measured after the 3:1 actuator gear.
            d.ctrl[self.wheels] = np.clip(np.array([linear-.17035*omega,linear+.17035*omega])*3/.0508,-6,6)
            for _ in range(period):
                mujoco.mj_step(m,d)
                for contact in d.contact:
                    geoms = [int(contact.geom1),int(contact.geom2)]
                    if self.floor not in geoms and any(m.geom_bodyid[g]==0 for g in geoms):
                        collision = "obstacle contact"
                        break
                if collision:
                    break
            if collision or any(w.number for w in d.warning):
                collision = collision or "MuJoCo warning"
                break
        seconds = float(d.time)
        return {"success":finish and not collision,"collision":collision,"seconds":seconds,
                "goalError":distance,"minimumClearance":min_clearance,"frames":trajectory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=61, type=int)
    parser.add_argument("--generations", default=5, type=int)
    args = parser.parse_args()
    aisle = Aisle(args.robot_xml.resolve())
    rng = np.random.default_rng(args.seed)
    low, high = np.array([.055,.65,.25,.45]), np.array([.105,5.,1.,2.])
    mean, spread = (low+high)/2, (high-low)/4
    # Leave initial perturbation cases disjoint from acceptance cases.
    training = rng.uniform([-.018,-.018,-.04],[.018,.018,.04],size=(3,3))
    best = None
    for generation in range(args.generations):
        candidates = rng.normal(mean,spread,size=(8,4)).clip(low,high)
        candidates[0] = mean
        def score(gains):
            episodes = [aisle.episode(gains,initial) for initial in training]
            return sum((0 if run["success"] else 60)+run["seconds"]+4*run["goalError"] for run in episodes)
        ranked = sorted((score(gains),tuple(gains)) for gains in candidates)
        if best is None or ranked[0][0]<best[0]:
            best = ranked[0]
        elite = np.array([row[1] for row in ranked[:3]])
        mean, spread = elite.mean(0), np.maximum(elite.std(0),(high-low)*.035)
        print("generation", generation+1, "best score", round(best[0],3), best[1], flush=True)
    held_rng = np.random.default_rng(args.seed+10000)
    held_out = [aisle.episode(best[1],initial) for initial in held_rng.uniform([-.022,-.022,-.05],[.022,.022,.05],size=(12,3))]
    run = aisle.episode(best[1],np.zeros(3),record=True)
    frames = run.pop("frames")
    report = {"controller":"differential-drive feedback, cross-entropy optimization",
              "model":"Hello Robot Stretch 3","modelRevision":MENAGERIE_COMMIT,"modelXmlSha256":aisle.xml_sha,
              "trainingSeed":args.seed,"acceptanceSeed":args.seed+10000,"generations":args.generations,
              "gains":best[1],"goal":aisle.goal.tolist(),"goalTolerance":GOAL_TOLERANCE,"episode":run,
              "heldOut":[{k:v for k,v in trial.items() if k!="frames"} for trial in held_out]}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    if not (run["success"] and all(trial["success"] for trial in held_out)):
        print("Acceptance failed. See", args.output/"report.json",flush=True)
        return 2
    built = aisle.source
    replay = built["replay"]
    replay["frames"],replay["duration"] = frames,frames[-1]["time"]
    replay["physics"] = {"engine":f"MuJoCo {mujoco.__version__}","modelRevision":MENAGERIE_COMMIT,
                         "jointNames":list(aisle.joints),"goalError":run["goalError"],"minimumClearance":run["minimumClearance"],
                         "heldOutPassed":len(held_out),"heldOutTotal":len(held_out)}
    replay["events"] = [{"time":0,"label":"Driving through the aisle under MuJoCo wheel actuation"},
                        {"time":replay["duration"],"label":"Reached aisle checkpoint · physics run complete"}]
    replay["verdict"]["planner"] = "MuJoCo CEM"
    replay["verdict"]["summary"] = f"Stretch reached the aisle checkpoint with {run['goalError']:.2f} m goal error and no obstacle contacts."
    replay["verdict"]["steps"][0]["note"] = f"{run['minimumClearance']:.2f} m minimum sampled footprint margin. 12/12 held-out starts completed."
    (args.output/"stretch.json").write_text(json.dumps(built,separators=(",",":"),allow_nan=False)+"\n")
    print("Qualified 12/12 held-out episodes and the recorded run.",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
