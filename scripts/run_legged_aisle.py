"""Qualify matched walking policies against the Morpheus aisle.

Use the existing morpheus_policy G1 adapter from the policy workspace on
PYTHONPATH. This runs its pinned checkpoint without fine-tuning its weights.
"""
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

GOAL_TOLERANCE = .22


def run(robot, model_path, checkpoint, policy, angles, output, policy_label, revision):
    world = World()
    model = load_model(model_path, world)
    data = mujoco.MjData(model)
    envelope = plan_task("Aisle clearance",robot,world,use_openai=False)
    replay = envelope["replay"]
    start,goal = (np.array(frame["position"][:2]) for frame in (replay["frames"][0],replay["frames"][-1]))
    direction = goal-start
    length = np.linalg.norm(direction)
    direction /= length
    heading = math.atan2(*direction[::-1])
    floor = model.geom("aisle_floor").id
    joints = [model.joint(i).name for i in range(1, model.njnt)]

    standing_height = .27 if robot=="go2" else model.qpos0[2]
    min_height = .16 if robot=="go2" else .56
    feet = {geom for geom in range(model.ngeom)
            if model.body(int(model.geom_bodyid[geom])).name.endswith("ankle_roll_link")
            or model.geom(geom).name in {"FL","FR","RL","RR"}}

    def episode(offset, record=False):
        mujoco.mj_resetData(model,data)
        data.qpos[:2] = start+offset[:2]
        yaw = heading+offset[2]
        data.qpos[3:7] = [math.cos(yaw/2),0,0,math.sin(yaw/2)]
        data.qpos[2] = standing_height
        data.qpos[7:] = angles
        mujoco.mj_forward(model,data)
        policy.reset(model,data)
        frames, failure = [], None
        margin = math.inf
        period = round(FRAME_INTERVAL/model.opt.timestep)
        settled = 0
        for tick in range(round(20/FRAME_INTERVAL)):
            pos = data.qpos[:2].copy()
            distance = float(np.linalg.norm(goal-pos))
            row,col,valid = cell_index(world.mesh,pos)
            local_clearance = min(world.clearance([pos])[0],world.mesh_clear[row[0],col[0]])-ROBOTS[robot]["radius"]
            margin = min(margin,float(local_clearance))
            clearance_failed = not (valid[0] and world.mesh["interior"][row[0],col[0]] and local_clearance>0)
            if clearance_failed:
                failure = "footprint clearance"
            if record:
                frames.append({"time":round(tick*FRAME_INTERVAL,3),"position":data.qpos[:3].round(6).tolist(),
                               "yaw":round(yaw_of(data.qpos[3:7])-math.pi/2,6),"armExtension":0,
                               "physicsPose":{"quaternion":data.qpos[3:7].round(8).tolist(),
                                              "joints":data.qpos[7:].round(6).tolist()}})
            settled = settled+1 if distance<GOAL_TOLERANCE and np.linalg.norm(data.qvel[:2])<.24 else 0
            if clearance_failed or settled==8:
                break
            progress = np.dot(pos-start,direction)
            waypoint = start+min(length,progress+.65)*direction
            turn = angle_error(math.atan2(*(waypoint-pos)[::-1]),yaw_of(data.qpos[3:7]))
            velocity = min(.32,max(0,distance-.10)*1.6)
            command = [velocity,0,np.clip(1.65*turn,-.55,.55)]
            for _ in range(period):
                policy.step(model,data,command)
                mujoco.mj_step(model,data)
                if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
                    failure = "invalid physics state"
                    break
                if data.qpos[2]<min_height or np.linalg.norm(data.qpos[4:6])>.23:
                    failure = "fall threshold"
                    break
                for contact in data.contact:
                    a,b = int(contact.geom1),int(contact.geom2)
                    if floor in (a,b):
                        robot_geom = b if a==floor else a
                        if robot_geom in feet:
                            continue
                    elif model.geom_bodyid[a]!=0 and model.geom_bodyid[b]!=0:
                        continue
                    failure = "forbidden contact"
                    break
                if failure:
                    break
            if failure:
                break
        return {"success":settled==8 and failure is None,"seconds":float(data.time),
                "goalError":distance,"minimumClearance":margin,"failure":failure,"frames":frames}

    rng = np.random.default_rng(1841)
    held = [episode(delta) for delta in rng.uniform([-.02,-.02,-.04],[.02,.02,.04],size=(12,3))]
    actual = episode(np.zeros(3),record=True)
    frames = actual.pop("frames")
    output.mkdir(parents=True,exist_ok=False)
    (output/"report.json").write_text(json.dumps({"checkpointSha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
         "goal":goal.tolist(),"goalTolerance":GOAL_TOLERANCE,"episode":actual,"heldOut":held}, indent=2)+"\n")
    success = sum(result["success"] for result in held)
    print(f"Nominal completion {actual['success']}; held out {success}/{len(held)}")
    if not actual["success"] or success!=len(held):
        return 2
    replay["frames"],replay["duration"] = frames,frames[-1]["time"]
    replay["physics"] = {"engine":f"MuJoCo {mujoco.__version__}","visualModel":"g1-12dof" if robot=="g1" else robot,
                         "modelRevision":revision if robot=="g1" else MENAGERIE_COMMIT,
                         "controllerRevision":revision,"checkpointSha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"jointNames":joints,"goalError":actual["goalError"],
                         "minimumClearance":actual["minimumClearance"],"heldOutPassed":success,"heldOutTotal":len(held)}
    replay["events"] = [{"time":0,"label":"Walking through the aisle under MuJoCo joint actuation"},
                        {"time":replay["duration"],"label":"Reached aisle checkpoint · physics run complete"}]
    replay["verdict"]["planner"] = policy_label
    replay["verdict"]["summary"] = f"{ROBOTS[robot]['name']} reached the aisle checkpoint with {actual['goalError']:.2f} m goal error and no forbidden contacts."
    replay["verdict"]["steps"][0]["note"] = f"{actual['minimumClearance']:.2f} m minimum sampled footprint margin. {success}/{len(held)} held-out starts completed."
    (output/f"{robot}.json").write_text(json.dumps(envelope,separators=(",",":"),allow_nan=False)+"\n")
    return 0


class Go2Policy:
    """Replay the pinned ASIG-X observation contract in MuJoCo."""
    revision = "87b4b89bd646819aebbb13d548830b1a4347a028"
    digest = "5d48960be68239920255c05ab7e4f61e11ec77054e61b80d044b9758330198dc"
    angles = np.array([.1,.9,-1.8,-.1,.9,-1.8,.1,.9,-1.8,-.1,.9,-1.8])

    def __init__(self, checkpoint):
        import onnxruntime
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=self.digest:
            raise ValueError("Go2 checkpoint checksum differs from pinned checkpoint")
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        self.actor = onnxruntime.InferenceSession(str(checkpoint),sess_options=opts,providers=["CPUExecutionProvider"])

    def reset(self, model, _):
        expected = [f"{leg}_{joint}_joint" for leg in ("FL","FR","RL","RR") for joint in ("hip","thigh","calf")]
        if (model.nq,model.nv,model.nu)!=(19,18,12) or not np.isclose(model.opt.timestep,.002):
            raise ValueError("Unexpected Go2 MuJoCo state contract")
        if [model.joint(i).name for i in range(1,model.njnt)]!=expected:
            raise ValueError("Unexpected Go2 joint order")
        self.history = np.zeros((10,45),dtype=np.float32)
        self.action = np.zeros(12)
        self.step_count = 0

    def step(self, model, data, command):
        if self.step_count%10==0:
            gravity = data.xmat[1].reshape(3,3).T @ np.array([0,0,-1])
            obs = np.r_[data.qvel[3:6],gravity,command,data.qpos[7:]-self.angles,data.qvel[6:],self.action]
            if not self.step_count:
                self.history[:] = obs
            else:
                self.history[:-1] = self.history[1:]
                self.history[-1] = obs
            self.action = self.actor.run(["actions"],{"obs":self.history.reshape(1,-1).clip(-100,100)})[0].reshape(-1).clip(-1,1)
        self.step_count += 1
        target = self.angles+.5*self.action
        data.ctrl[:] = np.clip(35*(target-data.qpos[7:])-.5*data.qvel[6:],
                               model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot",choices=("g1","go2"))
    parser.add_argument("--asset-root",type=Path)
    parser.add_argument("--robot-xml",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.robot=="g1":
        from morpheus_policy.policies.g1 import DEFAULT_ANGLES, G1Policy, UPSTREAM_COMMIT
        root = args.asset_root.resolve()
        G1Policy.validate_assets(root)
        checkpoint = root/"motion.pt"
        result = run("g1",root/"robot/g1_12dof.xml",checkpoint,G1Policy(checkpoint),
                     DEFAULT_ANGLES,args.output,"Unitree G1 PPO/LSTM",UPSTREAM_COMMIT)
    else:
        checkpoint = args.checkpoint.resolve()
        model = args.robot_xml.resolve()
        if hashlib.sha256(model.read_bytes()).hexdigest()!="50adb09a4365293e2acdaf2010ae35a82b0f09fea18ae51806fef91e310ed04a":
            raise ValueError("Go2 XML checksum differs from the pinned Menagerie model")
        result = run("go2",model,checkpoint,Go2Policy(checkpoint),
                     Go2Policy.angles,args.output,"ASIG-X Go2 policy",Go2Policy.revision)
    raise SystemExit(result)
