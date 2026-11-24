"""Tune Stretch counter contact against a physical target in the kitchen proxy."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from pipeline.room_mesh import cell_index
from pipeline.task_planner import ROBOTS, World, plan_task
from scripts.train_stretch_aisle import FRAME_INTERVAL, MENAGERIE_COMMIT, angle_error, load_model, yaw_of

BUTTON_CENTER = [1.441, 1.776, .894]
BUTTON_HALF_EXTENTS = [.009, .079, .027]
APPROACH_Y = 1.7341
CONTACT_FORCE = 1.
OVERLOAD_FORCE = 50.
CONTACT_SEC = .25
DRIVE_END = 230
PRESS_START = 285
RETRACT_START = 355
WRIST_STOW_START = 388
EPISODE_END = 433


class CounterTrial:
    def __init__(self, robot_xml):
        self.world = World()
        marker = {"name":"counter_button","type":"box","pos":" ".join(map(str,BUTTON_CENTER)),
                  "size":" ".join(map(str,BUTTON_HALF_EXTENTS))}
        self.model = load_model(robot_xml,self.world,[marker])
        self.data = mujoco.MjData(self.model)
        model = self.model
        self.wheels = [model.actuator(name).id for name in ("left_wheel_vel","right_wheel_vel")]
        self.joints = {model.joint(i).name:int(model.jnt_qposadr[i]) for i in range(1,model.njnt)}
        self.floor = model.geom("aisle_floor").id
        self.marker = model.geom("counter_button").id
        self.fingertips = {model.body(name).id for name in ("rubber_tip_left","rubber_tip_right")}
        self.arm = model.actuator("arm").id
        self.lift = model.actuator("lift").id
        self.wrist = model.actuator("wrist_yaw").id
        self.wrist_pitch = model.actuator("wrist_pitch").id
        self.goal = np.array([self.world.spawn[0], APPROACH_Y])

    def run(self, settings, offset, record=False):
        lift, extension = settings
        model,data = self.model,self.data
        mujoco.mj_resetDataKeyframe(model,data,model.key("stow").id)
        data.qpos[:2] = self.world.spawn+offset[:2]
        yaw = -math.pi/2+offset[2]
        data.qpos[2:7] = [.005,math.cos(yaw/2),0,0,math.sin(yaw/2)]
        for name,value in (("joint_lift",.23),("joint_wrist_yaw",3.14),("joint_wrist_pitch",-.4)):
            data.qpos[self.joints[name]] = value
        mujoco.mj_forward(model,data)
        period = round(FRAME_INTERVAL/model.opt.timestep)
        first_touch = None
        consecutive = max_consecutive = 0
        failure = None
        peak_force = force = 0.
        margin = math.inf
        frames = []
        for tick in range(EPISODE_END+1):
            now = float(data.time)
            pos = data.qpos[:2].copy()
            row,col,valid = cell_index(self.world.mesh,pos)
            clearance = float(min(self.world.clearance([pos])[0],self.world.mesh_clear[row[0],col[0]]))-ROBOTS["stretch"]["radius"]
            margin = min(clearance,margin)
            if not (valid[0] and self.world.mesh["interior"][row[0],col[0]]) or clearance <= 0:
                failure = "base clearance"
                break
            if record:
                frames.append({"time":round(now,3),"position":data.qpos[:3].round(6).tolist(),
                               "yaw":round(yaw_of(data.qpos[3:7])-math.pi/2,6),"armExtension":0,
                               "physicsPose":{"quaternion":data.qpos[3:7].round(8).tolist(),
                                              "joints":[round(float(data.qpos[j]),6) for j in self.joints.values()]},
                               "contactForce":round(force,3)})
            if tick==EPISODE_END:
                break
            if tick<DRIVE_END:
                delta = self.goal-pos
                distance = np.linalg.norm(delta)
                error = angle_error(math.atan2(*delta[::-1]),yaw_of(data.qpos[3:7]))
                speed = min(.090,1.1*max(0,distance-.035))*max(0,math.cos(error))
                turn = np.clip(2.2*error,-.25,.25) if distance>.043 else 0
                data.ctrl[self.wheels] = np.clip(np.array([speed-.17035*turn,speed+.17035*turn])*3/.0508,-6,6)
            else:
                data.ctrl[self.wheels] = 0
                data.ctrl[self.lift] = lift
                data.ctrl[self.wrist] = 0 if tick<WRIST_STOW_START else 3.14
                data.ctrl[self.wrist_pitch] = 0 if tick<WRIST_STOW_START else -.4
                data.ctrl[self.arm] = extension if PRESS_START<=tick<RETRACT_START else 0
            for _ in range(period):
                mujoco.mj_step(model,data)
                force = 0.
                for index,contact in enumerate(data.contact):
                    a,b = int(contact.geom1),int(contact.geom2)
                    if self.marker in (a,b):
                        tip = b if a==self.marker else a
                        if model.geom_bodyid[tip] not in self.fingertips:
                            failure = "non-fingertip marker contact"
                            break
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(model,data,index,wrench)
                        force += wrench[0]
                    elif self.floor not in (a,b) and 0 in (model.geom_bodyid[a],model.geom_bodyid[b]):
                        failure = "obstacle contact"
                        break
                if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
                    failure = "invalid physics state"
                if force>OVERLOAD_FORCE:
                    failure = "marker force limit"
                if failure:
                    break
                if PRESS_START<=tick<RETRACT_START and force>CONTACT_FORCE:
                    if first_touch is None:
                        first_touch = float(data.time)
                    consecutive += 1
                    max_consecutive = max(consecutive,max_consecutive)
                    peak_force = max(peak_force,float(force))
                else:
                    consecutive = 0
            if failure:
                break
        dwell = max_consecutive*model.opt.timestep
        return {"success":failure is None and dwell>=CONTACT_SEC and force<CONTACT_FORCE,
                "failure":failure,"dwell":dwell,"firstTouch":first_touch,
                "peakForce":peak_force,"minimumClearance":margin,"frames":frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-xml",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    xml = args.robot_xml.resolve()
    if hashlib.sha256(xml.read_bytes()).hexdigest()!="154088989df3b3d01a184b8e2df7cb82481efcd532f2498719acf1d77f001a6d":
        raise ValueError("Stretch XML checksum differs from the pinned Menagerie model")
    trial = CounterTrial(xml)
    train = np.random.default_rng(369).uniform([-.008,-.012,-.015],[.008,.012,.015],size=(4,3))
    candidates = []
    for lift in (.776,.785,.794,.803):
        for extension in (.025,.035,.045,.058,.075):
            runs = [trial.run((lift,extension),offset) for offset in train]
            successes = sum(row["success"] for row in runs)
            score = (successes,min(row["dwell"] for row in runs),-max(row["peakForce"] for row in runs))
            candidates.append((score,(lift,extension)))
    best,settings = max(candidates)
    print("Training",best,settings,flush=True)
    held = np.random.default_rng(10369).uniform([-.01,-.014,-.02],[.01,.014,.02],size=(12,3))
    reports = [trial.run(settings,offset) for offset in held]
    actual = trial.run(settings,np.zeros(3),record=True)
    frames = actual.pop("frames")
    for row in reports:
        row.pop("frames")
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"report.json").write_text(json.dumps({"settings":settings,"nominal":actual,"heldOut":reports},indent=2)+"\n")
    passes = sum(row["success"] for row in reports)
    print("Nominal",actual,"held-out",passes,flush=True)
    if not actual["success"] or passes!=len(reports):
        return 2
    envelope = plan_task("Counter reach","stretch",trial.world,use_openai=False)
    replay = envelope["replay"]
    replay["frames"],replay["duration"] = frames,frames[-1]["time"]
    touch = math.ceil(actual["firstTouch"]*20)/20
    replay["contact"] = {"point":[BUTTON_CENTER[0]+BUTTON_HALF_EXTENTS[0],*BUTTON_CENTER[1:]],
                         "travel":[0,0,0],"padSize":[2*x for x in BUTTON_HALF_EXTENTS],"kind":"counter","start":DRIVE_END*FRAME_INTERVAL,
                         "contact":touch,"release":RETRACT_START*FRAME_INTERVAL,"end":replay["duration"]}
    replay["events"] = [{"time":0,"label":"Driving to the counter under MuJoCo wheel actuation"},
                        {"time":DRIVE_END*FRAME_INTERVAL,"label":"Extending the physical arm toward the counter marker"},
                        {"time":touch,"label":"Fingertip contact measured in MuJoCo"},
                        {"time":RETRACT_START*FRAME_INTERVAL,"label":"Retracting the gripper"},
                        {"time":replay["duration"],"label":"Counter reach complete · contact and release measured"}]
    replay["physics"] = {"engine":f"MuJoCo {mujoco.__version__}","modelRevision":MENAGERIE_COMMIT,
                         "jointNames":list(trial.joints),"goalError":float(np.linalg.norm(np.array(frames[-1]["position"][:2])-trial.goal)),"minimumClearance":actual["minimumClearance"],
                         "heldOutPassed":passes,"heldOutTotal":len(reports),
                         "contactDwellSeconds":actual["dwell"],"peakContactForceNewtons":actual["peakForce"]}
    replay["verdict"]["planner"] = "MuJoCo contact parameter search"
    replay["verdict"]["summary"] = f"Stretch pressed and released the fixture with {actual['dwell']:.2f} s of fingertip contact."
    replay["verdict"]["steps"][1]["note"] = f"{actual['peakForce']:.1f} N peak normal force. {passes}/{len(reports)} held-out starts completed."
    (args.output/"stretch.json").write_text(json.dumps(envelope,separators=(",",":"),allow_nan=False)+"\n")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
