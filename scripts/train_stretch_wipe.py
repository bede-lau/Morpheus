"""Train a physical Stretch wipe against a deformable cloth in the sample kitchen."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from pipeline.room_mesh import cell_index
from pipeline.task_planner import ROBOTS, World, plan_task
from scripts.train_stretch_aisle import MENAGERIE_COMMIT, angle_error, load_model, yaw_of

ROWS, COLS = 8, 7
APPROACH = 59.
WIPE = 16.
END = 25.
GRID = .025
# A local stain on the practice surface, aligned to the physical contact sampling grid.
PATCH = [(x, y) for x in (64, 65, 66, 67) for y in (-32, -31, -30, -29, -28, -27)]


def setup(xml):
    if hashlib.sha256(xml.read_bytes()).hexdigest() != "154088989df3b3d01a184b8e2df7cb82481efcd532f2498719acf1d77f001a6d":
        raise ValueError("Expected the pinned Stretch Menagerie model")
    tree = ET.parse(xml).getroot()
    option = tree.find("option")
    option.attrib.update(timestep=".002", integrator="discrete", noslip_iterations="0")
    cloth = ET.Element("flexcomp", name="cloth", type="grid", dim="2", count=f"{ROWS} {COLS} 1",
                       spacing=".024 .021 .021", pos="1.635 -.75 .708", radius=".003", mass=".039", rgba=".88 .45 .20 1")
    ET.SubElement(cloth, "contact", selfcollide="auto", friction="1.0 .02 .001", condim="3")
    ET.SubElement(cloth, "elasticity", young="220000", poisson=".30", damping=".009", thickness=".002", elastic2d="both")
    tab = ET.Element("body", name="cloth_tab", pos="1.719 -.75 .708")
    ET.SubElement(tab, "freejoint", name="cloth_tab_joint")
    ET.SubElement(tab, "geom", name="cloth_tab_handle", type="box", size=".014 .028 .010", mass=".008",
                  friction="1.6 .018 .002", condim="4")
    for vertex in (51, 53):
        ET.SubElement(tree.find("equality"), "connect", name=f"cloth_seam_{vertex}", body1=f"cloth_{vertex}",
                      body2="cloth_tab", anchor="0 0 0")
    for key in tree.findall("./keyframe/key"):
        for attr in ("qpos", "qvel"):
            if key.get(attr):
                tab_state = " 1.719 -.75 .708 1 0 0 0" if attr == "qpos" else " 0" * 6
                key.set(attr, key.get(attr) + " 0" * (ROWS * COLS * 3) + tab_state)
    world = World()
    table = dict(name="practice_table_top", type="box", pos="1.48 -.76 .687", size=".21 .325 .013")
    return world, load_model(xml, world, [table], robot_tree=tree, bodies=[cloth, tab])


def run(xml, settings, offset, record=False):
    height, sweep, press_height = settings
    world, model = setup(xml)
    data = mujoco.MjData(model)
    arm = [model.actuator(n).id for n in ("lift", "arm", "wrist_yaw", "wrist_pitch", "gripper")]
    wheels = [model.actuator(n).id for n in ("left_wheel_vel", "right_wheel_vel")]
    tips = {model.body(n).id for n in ("rubber_tip_left", "rubber_tip_right")}
    table = model.geom("practice_table_top").id
    tab = model.geom("cloth_tab_handle").id
    tab_body = model.body("cloth_tab").id
    floor = model.geom("aisle_floor").id
    joints = {model.joint(i).name: int(model.jnt_qposadr[i]) for i in range(1, model.njnt)
              if not model.body(int(model.jnt_bodyid[i])).name.startswith("cloth_")}
    mujoco.mj_resetDataKeyframe(model, data, model.key("stow").id)
    heading = -math.pi / 2 + offset[2]
    data.qpos[:7] = [*(world.spawn + offset[:2]), .005, math.cos(heading / 2), 0., 0., math.sin(heading / 2)]
    for name, value in (("joint_lift", .23), ("joint_wrist_yaw", 3.14), ("joint_wrist_pitch", -.4)):
        data.qpos[joints[name]] = value
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    lift_count = longest_lift = 0
    first_lift = None
    dirt = set(PATCH)
    frames, cloth_frames = [], []
    wrench = np.zeros(6)
    failure = None
    margin = math.inf
    ticks = round((APPROACH + END) / dt)
    for step in range(ticks + 1):
        t = step * dt - APPROACH
        if step % round(.1 / dt) == 0:
            pos = data.qpos[:2].copy()
            row, col, valid = cell_index(world.mesh, pos)
            clearance = min(world.clearance([pos])[0], world.mesh_clear[row[0], col[0]]) - ROBOTS["stretch"]["radius"]
            margin = min(margin, float(clearance))
            if clearance <= 0 or not (valid[0] and world.mesh["interior"][row[0], col[0]]):
                failure = "base clearance"
                break
        if record and step % round(.1 / dt) == 0:
            frames.append({"time": round(data.time, 3), "position": data.qpos[:3].round(6).tolist(),
                           "yaw": round(yaw_of(data.qpos[3:7]) - math.pi / 2, 6), "armExtension": 0,
                           "physicsPose": {"quaternion": data.qpos[3:7].round(8).tolist(),
                                           "joints": [round(float(data.qpos[j]), 6) for j in joints.values()]}})
            if data.time < 5.1 or t >= 0:
                cloth_frames.append({"time": frames[-1]["time"], "points": data.flexvert_xpos.round(4).reshape(-1).tolist(),
                                     "tabPosition": data.xpos[tab_body].round(4).tolist(),
                                     "tabQuaternion": data.xquat[tab_body].round(6).tolist(),
                                     "wiped": [i for i, cell in enumerate(PATCH) if cell not in dirt]})
        if step == ticks:
            break
        if t < 0:
            x, y = data.qpos[:2]
            yaw = yaw_of(data.qpos[3:7])
            heading = angle_error(-math.pi / 2, yaw)
            speed, turn = 0., np.clip(1.1 * heading, -.028, .028)
            if t < -4:
                ahead_y = max(-.774, y - .52)
                progress = np.clip((world.spawn[1] - ahead_y) / 2.2, 0, 1)
                line_x = world.spawn[0] + .365 * progress ** 2 * (3 - 2 * progress)
                error = angle_error(math.atan2(ahead_y - y, line_x - x), yaw) if y + .774 > .16 else heading
                speed = min(.079, .55 * max(0., y + .771)) * max(0., math.cos(error))
                turn = np.clip(1.6 * error, -.033, .033)
            target = np.clip(np.array([speed - .17035 * turn, speed + .17035 * turn]) * 3 / .0508, -6, 6)
            data.ctrl[wheels] += np.clip(target - data.ctrl[wheels], -.004, .004)
        else:
            data.ctrl[wheels] = 0.
            lift = .29 * min(1., (t - 9.6) * .85) if 9.6 <= t < 13.8 else max(press_height, .29 - (t - 13.8) * .139) if t >= 13.8 else 0.
            extension = min(.164, max(0., (t - 5.) * .08))
            extension -= .077 * (np.clip((t - 8.3) * .8, 0., 1.) - np.clip((t - 12.4) * .875, 0., 1.))
            wrist = 3.14 if t < 3. else sweep * math.sin((t - WIPE) * 2.35) if t >= WIPE else 0.
            data.ctrl[arm] = [height + lift, extension + (.015 if t > WIPE else 0.), wrist,
                              -.4 if t < 3. else 0., -.02 if t >= 7.7 else .024]
        before = data.flexvert_xpos.copy() if t >= WIPE else None
        mujoco.mj_step(model, data)
        touched = set()
        for index, contact in enumerate(data.contact):
            a, b = map(int, contact.geom)
            if a < 0 or b < 0:
                other = max(a, b)
                if other >= 0:
                    if model.geom_bodyid[other] in tips:
                        touched.add(int(model.geom_bodyid[other]))
                    if other == table and before is not None:
                        cell = tuple(np.round(contact.pos[:2] / GRID).astype(int))
                        if cell in dirt:
                            mujoco.mj_contactForce(model, data, index, wrench)
                            closest = np.argmin(np.linalg.norm(data.flexvert_xpos - contact.pos, axis=1))
                            speed = np.linalg.norm(data.flexvert_xpos[closest, :2] - before[closest, :2]) / dt
                            if wrench[0] > .012 and speed > .018:
                                dirt.remove(cell)
                continue
            if tab in (a, b):
                tip = int(model.geom_bodyid[b if a == tab else a])
                if tip in tips:
                    touched.add(tip)
            elif floor not in (a, b) and 0 in (model.geom_bodyid[a], model.geom_bodyid[b]):
                mujoco.mj_contactForce(model, data, index, wrench)
                robot_geom = b if model.geom_bodyid[a] == 0 else a
                failure = f"{model.body(int(model.geom_bodyid[robot_geom])).name} contact with {model.geom(a if robot_geom == b else b).name} at {t:.3f}s ({wrench[0]:.2f} N)"
                break
        fully_lifted = 9.6 <= t < 13.8 and data.flexvert_xpos[:, 2].min() > .73 and len(touched) == 2
        if fully_lifted and first_lift is None:
            first_lift = float(data.time)
        lift_count = lift_count + 1 if fully_lifted else 0
        longest_lift = max(longest_lift, lift_count)
        if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
            failure = "invalid physics state"
        if failure:
            break
    cleared = len(PATCH) - len(dirt)
    measured = {"success": failure is None and longest_lift * dt >= .2 and cleared >= 8,
                "failure": failure, "clothLiftSeconds": longest_lift * dt, "wipedCells": cleared, "firstLift": first_lift,
                "minimumClearance": margin, "goalError": float(np.linalg.norm(data.qpos[:2] - [2.24, -.771]))}
    if record:
        triangles = model.flex_elem.reshape(-1, 3).tolist()
        measured.update(frames=frames, clothFrames=cloth_frames, triangles=triangles, jointNames=list(joints))
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    xml = args.robot_xml.resolve()
    limit = np.array([.006, .009, .010])
    train = np.random.default_rng(962).uniform(-limit, limit, (2, len(limit)))
    candidates = ((.550, .27, .079), (.550, .25, .091), (.550, .31, .089), (.547, .30, .086))
    with ProcessPoolExecutor(max_workers=4) as pool:
        batches = {settings: [pool.submit(run, xml, settings, offset) for offset in train] for settings in candidates}
        scores = []
        for settings, futures in batches.items():
            rows = [future.result() for future in futures]
            score = sum(row["success"] for row in rows), min(row["wipedCells"] for row in rows)
            print(settings, score, rows, flush=True)
            scores.append((score, settings))
        _, settings = max(scores)
        held = np.random.default_rng(1962).uniform(-limit, limit, (12, len(limit)))
        futures = [pool.submit(run, xml, settings, offset) for offset in held]
        nominal = pool.submit(run, xml, settings, np.zeros(len(limit)), True)
        reports = [future.result() for future in futures]
        actual = nominal.result()
    frames, cloth_frames, triangles, joints = (actual.pop(key) for key in ("frames", "clothFrames", "triangles", "jointNames"))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps({"settings": settings, "nominal": actual, "heldOut": reports}, indent=2) + "\n")
    passed = sum(report["success"] for report in reports)
    print("Nominal", actual, "held-out", passed, flush=True)
    if not actual["success"] or passed != len(held):
        return 2
    envelope = plan_task("wipe the table", "stretch", World(), use_openai=False)
    replay = envelope["replay"]
    replay.pop("wipes", None)
    replay["frames"], replay["duration"] = frames, frames[-1]["time"]
    replay["cloth"] = {"faces": triangles, "frames": cloth_frames, "marks": [[x * GRID, y * GRID, .702] for x, y in PATCH]}
    replay["events"] = [{"time": 0, "label": "Navigating to the practice table"},
                        {"time": APPROACH, "label": "Approaching the cloth’s pickup tab"},
                        {"time": round(actual["firstLift"], 3), "label": "Cloth lifted clear of the table"},
                        {"time": APPROACH + WIPE, "label": "Wiping the surface with the grasped cloth"},
                        {"time": replay["duration"], "label": "Table wipe complete · contact measured in MuJoCo"}]
    replay["physics"] = {"engine": f"MuJoCo {mujoco.__version__}", "modelRevision": MENAGERIE_COMMIT,
                         "jointNames": joints, "goalError": actual["goalError"], "minimumClearance": actual["minimumClearance"],
                         "heldOutPassed": passed, "heldOutTotal": len(held)}
    replay["verdict"]["planner"] = "MuJoCo cloth contact parameter search"
    replay["verdict"]["summary"] = f"Stretch lifted the cloth and wiped {actual['wipedCells']} marked cells with measured sliding contact."
    replay["verdict"]["steps"][0]["note"] = "Drove to the practice table under wheel actuation."
    replay["verdict"]["steps"][1]["note"] = f"{actual['clothLiftSeconds']:.2f} s clear of the surface. {passed}/{len(held)} held-out starts completed."
    (args.output / "stretch.json").write_text(json.dumps(envelope, separators=(",", ":"), allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
