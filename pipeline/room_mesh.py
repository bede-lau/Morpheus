"""Room interior and obstacle grids from the companion collider mesh.

The 0.22 m collision columns in output/kitchen/scene.json are too sparse to
define walls, so the task planner uses public/samples/rustic-kitchen-collider.glb
as the source of truth for where the room interior is.

Coordinate convention (same as scripts/prepare-sample.mjs and src/viewer.ts):
raw GLB vertices are Y-down; scene.source.transform.matrix (column-major)
maps raw [x, y, z] -> [x, z, -y] and then applies the centering / floor
alignment translation. The matrix is applied to the raw primitive positions
(GLB node transforms are identity and ignored, as in prepare-sample.mjs).

Heights are measured above a plane fitted to the scanned floor: the scan's
floor is tilted by ~0.15 m across the galley, so fixed world-z bands would cut
off half of the floor.

Grids (0.05 m, same cell centres as landmarks.free_grid over the floor box):
  floor     cells holding floor-band points  (-0.06 < h < 0.08)
  obstacle  cells holding obstacle-band points (0.12 < h < 1.9)
  surface   per-cell max height of near-horizontal points in the obstacle band
  top       per-cell max height of any obstacle-band point
  interior  floor & ~obstacle, closed, largest 8-connected component

Cached in output/kitchen/mesh-grid.npz keyed by the GLB sha256 + parameters.
Run as a module to print a summary: python -m pipeline.room_mesh
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
GLB = REPO / "public" / "samples" / "rustic-kitchen-collider.glb"
SCENE = REPO / "output" / "kitchen" / "scene.json"
CACHE = REPO / "output" / "kitchen" / "mesh-grid.npz"
STEP = 0.05
SAMPLES = 400_000
SEED = 7
FLOOR_BAND = (-0.06, 0.08)
OBSTACLE_BAND = (0.12, 1.9)
HORIZONTAL_NZ = 0.8
MIN_OBSTACLE_POINTS = 2
CACHE_VERSION = 3
# Raw [x,y,z] -> [x,z,-y]: the rotation every consumer of the sample applies.
RAW_TO_WORLD_ROTATION = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)


def source_matrix(scene):
    t = scene["source"]["transform"]
    m = np.array(t["matrix"], float).reshape(4, 4)
    m = m.T if t.get("matrixOrder", "column-major") == "column-major" else m
    if not np.allclose(m[:3, :3], RAW_TO_WORLD_ROTATION, atol=1e-6):
        raise ValueError("scene source transform is not the [x,z,-y] raw-to-world convention")
    return m


def floor_rect(scene):
    for obj in scene["objects"]:
        if obj.get("role") == "floor" or obj.get("id") == "floor":
            (x, y, _), (sx, sy, _) = obj["position"], obj["size"]
            return np.array([x - sx / 2, y - sy / 2, x + sx / 2, y + sy / 2])
    raise ValueError("scene has no floor box")


def axes(floor, step=STEP):
    return np.arange(floor[0] + step / 2, floor[2], step), np.arange(floor[1] + step / 2, floor[3], step)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw_mesh(glb=GLB):
    import trimesh
    loaded = trimesh.load(str(glb), process=False)
    meshes = list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def sample_world(mesh, matrix, count=SAMPLES, seed=SEED):
    import trimesh
    pts, face = trimesh.sample.sample_surface(mesh, count, seed=seed)
    world = pts @ matrix[:3, :3].T + matrix[:3, 3]
    nz = np.abs((mesh.face_normals[face] @ matrix[:3, :3].T)[:, 2])
    return world, nz


def fit_floor_plane(world, nz):
    """Robust z = a x + b y + c through the lowest horizontal surface (the floor)."""
    cand = (nz > 0.9) & (world[:, 2] > -0.35) & (world[:, 2] < 0.2)
    pts = world[cand]
    # Lowest point per 0.1 m cell, so tabletops and clutter do not pull the fit up.
    key = np.floor(pts[:, :2] / 0.1).astype(np.int64)
    order = np.lexsort((pts[:, 2], key[:, 1], key[:, 0]))
    key, pts = key[order], pts[order]
    first = np.ones(len(pts), bool)
    first[1:] = np.any(key[1:] != key[:-1], axis=1)
    low = pts[first]
    keep = np.ones(len(low), bool)
    coef = np.array([0.0, 0.0, 0.0])
    for tol in (0.12, 0.06, 0.03, 0.03):
        A = np.c_[low[keep, 0], low[keep, 1], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, low[keep, 2], rcond=None)
        resid = low[:, 2] - (low[:, 0] * coef[0] + low[:, 1] * coef[1] + coef[2])
        keep = np.abs(resid) < tol
    return coef, float(np.sqrt(np.mean(resid[keep] ** 2))), int(keep.sum())


def build(scene, glb=GLB, count=SAMPLES, seed=SEED):
    matrix = source_matrix(scene)
    floor = floor_rect(scene)
    xs, ys = axes(floor)
    mesh = load_raw_mesh(glb)
    world, nz = sample_world(mesh, matrix, count, seed)
    plane, rms, inliers = fit_floor_plane(world, nz)
    h = world[:, 2] - (world[:, 0] * plane[0] + world[:, 1] * plane[1] + plane[2])
    j = np.floor((world[:, 0] - floor[0]) / STEP).astype(int)
    i = np.floor((world[:, 1] - floor[1]) / STEP).astype(int)
    inside = (i >= 0) & (i < len(ys)) & (j >= 0) & (j < len(xs))
    shape = (len(ys), len(xs))
    fsel = inside & (h > FLOOR_BAND[0]) & (h < FLOOR_BAND[1])
    osel = inside & (h > OBSTACLE_BAND[0]) & (h < OBSTACLE_BAND[1])
    floor_n = np.zeros(shape, int)
    obst_n = np.zeros(shape, int)
    np.add.at(floor_n, (i[fsel], j[fsel]), 1)
    np.add.at(obst_n, (i[osel], j[osel]), 1)
    top = np.zeros(shape)
    np.maximum.at(top, (i[osel], j[osel]), h[osel])
    surface = np.zeros(shape)
    ssel = osel & (nz > HORIZONTAL_NZ)
    np.maximum.at(surface, (i[ssel], j[ssel]), h[ssel])
    floor_mask = floor_n > 0
    obstacle = obst_n >= MIN_OBSTACLE_POINTS
    # Interior: unobstructed floor, small closing to bridge sampling holes, largest component.
    open_floor = ndimage.binary_closing(floor_mask & ~obstacle, structure=np.ones((3, 3)), iterations=1) & ~obstacle
    labels, n = ndimage.label(open_floor, structure=np.ones((3, 3)))
    interior = np.zeros(shape, bool)
    if n:
        sizes = ndimage.sum(open_floor, labels, range(1, n + 1))
        interior = labels == (int(np.argmax(sizes)) + 1)
    rel = mesh.vertices @ matrix[:3, :3].T + matrix[:3, 3]
    return {"xs": xs, "ys": ys, "floor": floor_mask, "obstacle": obstacle, "interior": interior,
            "top": top, "surface": surface, "plane": np.asarray(plane), "planeRms": rms, "planeInliers": inliers,
            "meshMin": rel.min(0), "meshMax": rel.max(0), "floorRect": floor}


def cache_key(glb, scene):
    params = json.dumps([CACHE_VERSION, STEP, SAMPLES, SEED, FLOOR_BAND, OBSTACLE_BAND, HORIZONTAL_NZ, MIN_OBSTACLE_POINTS,
                         scene["source"]["transform"]["matrix"], floor_rect(scene).round(6).tolist()])
    return sha256(glb) + ":" + hashlib.sha256(params.encode()).hexdigest()[:16]


def load_grids(scene=None, glb=GLB, cache=CACHE):
    """Return the grid dict, using the npz cache when its key matches. None when the GLB is missing."""
    glb = Path(glb)
    if not glb.exists():
        return None
    scene = json.loads(SCENE.read_text()) if scene is None else scene
    key = cache_key(glb, scene)
    cache = Path(cache) if cache else None
    if cache and cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as data:
                if str(data["key"]) == key:
                    out = {k: data[k] for k in data.files if k != "key"}
                    for k in ("planeRms",):
                        out[k] = float(out[k])
                    out["planeInliers"] = int(out["planeInliers"])
                    for k in ("floor", "obstacle", "interior"):
                        out[k] = out[k].astype(bool)
                    out["key"] = key
                    return out
        except Exception:
            pass
    out = build(scene, glb)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + ".tmp.npz")
        np.savez_compressed(tmp, key=np.array(key), **out)
        tmp.replace(cache)
    out["key"] = key
    return out


def obstacle_distance(grids):
    """Metres from each cell centre to the nearest mesh obstacle cell centre."""
    return ndimage.distance_transform_edt(~grids["obstacle"]) * STEP


def cell_index(grids, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    xs, ys = grids["xs"], grids["ys"]
    j = np.round((pts[:, 0] - xs[0]) / STEP).astype(int)
    i = np.round((pts[:, 1] - ys[0]) / STEP).astype(int)
    ok = (i >= 0) & (i < len(ys)) & (j >= 0) & (j < len(xs))
    return np.clip(i, 0, len(ys) - 1), np.clip(j, 0, len(xs) - 1), ok


def main():
    g = load_grids()
    if g is None:
        print("GLB missing:", GLB)
        return
    print(f"mesh world bounds min {np.round(g['meshMin'], 3).tolist()} max {np.round(g['meshMax'], 3).tolist()}")
    print(f"mesh XY extent {g['meshMax'][0] - g['meshMin'][0]:.2f} x {g['meshMax'][1] - g['meshMin'][1]:.2f} m; "
          f"floor box {g['floorRect'][2] - g['floorRect'][0]:.2f} x {g['floorRect'][3] - g['floorRect'][1]:.2f} m")
    a, b, c = g["plane"]
    print(f"floor plane z = {a:.4f} x + {b:.4f} y + {c:.4f}  (rms {g['planeRms']:.3f} m, {g['planeInliers']} cells)")
    xs, ys = g["xs"], g["ys"]
    ii, jj = np.nonzero(g["interior"])
    print(f"interior {g['interior'].sum() * STEP * STEP:.2f} m2, x {xs[jj].min():.2f}..{xs[jj].max():.2f}, "
          f"y {ys[ii].min():.2f}..{ys[ii].max():.2f}; obstacle cells {int(g['obstacle'].sum())}")


if __name__ == "__main__":
    main()
