"""Export a MuJoCo robot (home pose) to a hierarchical GLB + sidecar JSON for the Morpheus viewer.
Usage: export_robot.py <robot id> <mjcf path> <out dir> [joint=value ...]
Node names = MJCF body names. Root is rotated so MuJoCo +X forward becomes viewer +Y; lowest visual point sits at z=0.
"""
import sys, json, pathlib, numpy as np, mujoco, trimesh
robot, xml, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
overrides = dict(a.split("=") for a in sys.argv[4:])
TARGET_FACES = 90000
m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
if m.nkey: d.qpos[:] = m.key_qpos[0]
for name, value in overrides.items():
    j = m.joint(name); d.qpos[m.jnt_qposadr[j.id]] = float(value)
mujoco.mj_forward(m, d)
def T(pos, quat_or_mat):
    t = np.eye(4); t[:3, 3] = pos
    if len(quat_or_mat) == 4:
        r = np.zeros(9); mujoco.mju_quat2Mat(r, quat_or_mat); t[:3, :3] = r.reshape(3, 3)
    else: t[:3, :3] = np.asarray(quat_or_mat).reshape(3, 3)
    return t
scene = trimesh.Scene()
ROOT = "robot"
# world->root: rotate +X forward to +Y, lift later so min z = 0.
root_T = T([0, 0, 0], [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
scene.graph.update(frame_to=ROOT, frame_from=scene.graph.base_frame, matrix=root_T)
body_world = {0: np.eye(4)}
bodies = {}
# face budget: proportional decimation of the visual meshes
visual = [g for g in range(m.ngeom) if m.geom_group[g] < 3 and (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0))]
total_faces = sum(int(m.mesh_facenum[m.geom_dataid[g]]) for g in visual if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
ratio = min(1.0, TARGET_FACES / max(1, total_faces))
mesh_cache = {}
def mesh_for(g):
    mid = int(m.geom_dataid[g])
    if mid in mesh_cache: return mesh_cache[mid]
    va, vn, fa, fn = m.mesh_vertadr[mid], m.mesh_vertnum[mid], m.mesh_faceadr[mid], m.mesh_facenum[mid]
    tm = trimesh.Trimesh(m.mesh_vert[va:va + vn].copy(), m.mesh_face[fa:fa + fn].copy(), process=True)
    tm.merge_vertices()
    target = int(max(400, fn * ratio))
    if fn > target and fn > 500:
        try: tm = tm.simplify_quadric_decimation(face_count=target)
        except Exception as e: print("decimation skipped", mid, e)
    mesh_cache[mid] = tm; return tm
mat_cache = {}
def material(rgba, metal=0.35, rough=0.55):
    key = tuple(np.round(rgba, 3))
    if key not in mat_cache: mat_cache[key] = trimesh.visual.material.PBRMaterial(baseColorFactor=[float(x) for x in rgba], metallicFactor=metal, roughnessFactor=rough, name="m%d" % len(mat_cache))
    return mat_cache[key]
minz = np.inf; faces = 0
for b in range(1, m.nbody):
    name = m.body(b).name or f"body{b}"; parent = int(m.body_parentid[b])
    world = T(d.xpos[b], d.xmat[b]); body_world[b] = world
    parent_T = body_world[parent]; local = np.linalg.inv(parent_T) @ world
    pname = ROOT if parent == 0 else (m.body(parent).name or f"body{parent}")
    scene.graph.update(frame_to=name, frame_from=pname, matrix=local)
    info = {"parent": pname}
    if m.body_jntnum[b]:
        j = int(m.body_jntadr[b]); jt = int(m.jnt_type[j])
        if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            info["joint"] = {"name": m.joint(j).name, "type": "hinge" if jt == mujoco.mjtJoint.mjJNT_HINGE else "slide", "axis": m.jnt_axis[j].tolist(), "anchor": m.jnt_pos[j].tolist(), "home": float(d.qpos[m.jnt_qposadr[j]]), "range": m.jnt_range[j].tolist() if m.jnt_limited[j] else None}
        elif jt == mujoco.mjtJoint.mjJNT_FREE: info["free"] = True
    bodies[name] = info
    for g in range(int(m.body_geomadr[b]), int(m.body_geomadr[b] + m.body_geomnum[b])):
        if g not in visual: continue
        gt = int(m.geom_type[g]); size = m.geom_size[g]
        if gt == mujoco.mjtGeom.mjGEOM_MESH: geom = mesh_for(g).copy()
        elif gt == mujoco.mjtGeom.mjGEOM_BOX: geom = trimesh.creation.box(extents=size * 2)
        elif gt == mujoco.mjtGeom.mjGEOM_SPHERE: geom = trimesh.creation.icosphere(2, size[0])
        elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER: geom = trimesh.creation.cylinder(radius=size[0], height=size[1] * 2)
        elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE: geom = trimesh.creation.capsule(radius=size[0], height=size[1] * 2)
        else: continue
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        if rgba[3] < 0.05: continue
        geom.visual = trimesh.visual.TextureVisuals(material=material(rgba))
        gT = T(m.geom_pos[g], m.geom_quat[g])
        pts = trimesh.transform_points(geom.vertices, root_T @ world @ gT); minz = min(minz, pts[:, 2].min()); faces += len(geom.faces)
        scene.add_geometry(geom, node_name=f"{name}/geom{g}", geom_name=f"g{g}", parent_node_name=name, transform=gT)
root_T[2, 3] = -minz
scene.graph.update(frame_to=ROOT, frame_from=scene.graph.base_frame, matrix=root_T)
out.mkdir(parents=True, exist_ok=True)
glb = out / f"{robot}.glb"; glb.write_bytes(scene.export(file_type="glb", include_normals=False))
bounds = scene.bounds
cams = {m.camera(c).name: (root_T @ body_world[int(m.cam_bodyid[c])] @ T(m.cam_pos[c], m.cam_quat[c]))[:3, 3].round(4).tolist() for c in range(m.ncam)}
for b in range(1, m.nbody):
    bodies[m.body(b).name or f"body{b}"]["position"] = (root_T @ body_world[b])[:3, 3].round(4).tolist()
meta = {"robot": robot, "source": "google-deepmind/mujoco_menagerie", "model": pathlib.Path(xml).name, "forward": "+y", "units": "m", "liftZ": float(-minz), "bounds": {"min": bounds[0].round(4).tolist(), "max": bounds[1].round(4).tolist()}, "faces": faces, "cameras": cams, "bodies": bodies}
(out / f"{robot}.json").write_text(json.dumps(meta, indent=1))
print(robot, "faces", faces, "size", glb.stat().st_size // 1024, "KB", "bounds", np.round(bounds, 3).tolist(), "cams", cams)
