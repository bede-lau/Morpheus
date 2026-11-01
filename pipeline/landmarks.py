"""Label unlabeled collision columns in output/kitchen/scene.json as task landmarks.

Heuristic only: boxes are occupied columns, not semantic objects. Labels are
geometry guesses (table / counter / sink / tall / door) for the task planner.
Run as a module to print a summary: python -m pipeline.landmarks
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCENE = REPO / "output" / "kitchen" / "scene.json"
CELL = 0.22            # column size used by the reconstruction
GRID = 0.05            # occupancy grid used for door detection / planning
DOOR_INSET = 0.45      # robots start this far inside the door
DOOR_CLEARANCE = 0.47  # largest mobile footprint (go2 0.42) + 0.05 margin
SURFACE_BAND = (0.6, 1.25)   # column tops treated as work surfaces
TALL_MIN = 1.6


def load_scene(path=SCENE):
    return json.loads(Path(path).read_text())


def split_boxes(scene):
    """Return (floor rect [x0,y0,x1,y1], floor top z, obstacle list)."""
    floor, boxes = None, []
    for obj in scene["objects"]:
        if obj.get("enabled", True) is False or obj.get("type") != "box":
            continue
        (x, y, z), (sx, sy, sz) = obj["position"], obj["size"]
        if obj.get("role") == "floor" or obj.get("id") == "floor":
            floor = (np.array([x - sx / 2, y - sy / 2, x + sx / 2, y + sy / 2]), z + sz / 2)
            continue
        boxes.append({"id": obj["id"], "rect": (x - sx / 2, y - sy / 2, x + sx / 2, y + sy / 2),
                      "bottom": z - sz / 2, "top": z + sz / 2})
    if floor is None:
        raise ValueError("scene has no floor box")
    return floor[0], floor[1], boxes


def rect_distance(points, rects):
    """Distance from each point (N,2) to the nearest rect (M,4). Returns (N,)."""
    points = np.asarray(points, float).reshape(-1, 2)
    rects = np.asarray(rects, float).reshape(-1, 4)
    if not len(rects):
        return np.full(len(points), np.inf)
    px, py = points[:, :1], points[:, 1:2]
    dx = np.maximum(np.maximum(rects[None, :, 0] - px, 0), px - rects[None, :, 2])
    dy = np.maximum(np.maximum(rects[None, :, 1] - py, 0), py - rects[None, :, 3])
    return np.sqrt(dx * dx + dy * dy).min(axis=1)


def free_grid(floor, rects, clearance, step=GRID, floor_margin=None):
    """Boolean free grid (ny, nx) and its cell-centre coordinate axes."""
    xs = np.arange(floor[0] + step / 2, floor[2], step)
    ys = np.arange(floor[1] + step / 2, floor[3], step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    free = rect_distance(pts, rects) >= clearance
    margin = clearance if floor_margin is None else floor_margin
    inside = ((pts[:, 0] >= floor[0] + margin) & (pts[:, 0] <= floor[2] - margin)
              & (pts[:, 1] >= floor[1] + margin) & (pts[:, 1] <= floor[3] - margin))
    return (free & inside).reshape(len(ys), len(xs)), xs, ys


def components(mask, eight=True):
    """Label connected True cells. Returns (labels, sizes) with label 0 = background."""
    labels = np.zeros(mask.shape, int)
    sizes = [0]
    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if eight:
        neigh += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    h, w = mask.shape
    for start in zip(*np.nonzero(mask)):
        if labels[start]:
            continue
        label = len(sizes)
        labels[start] = label
        stack, count = [start], 0
        while stack:
            i, j = stack.pop()
            count += 1
            for di, dj in neigh:
                a, b = i + di, j + dj
                if 0 <= a < h and 0 <= b < w and mask[a, b] and not labels[a, b]:
                    labels[a, b] = label
                    stack.append((a, b))
        sizes.append(count)
    return labels, sizes


def find_door(floor, rects):
    """Largest obstacle-free run along one floor side whose inset point opens onto the main free region."""
    free, xs, ys = free_grid(floor, rects, DOOR_CLEARANCE, floor_margin=DOOR_INSET - 0.02)
    labels, sizes = components(free)
    main = int(np.argmax(sizes))

    def opens(p):
        i, j = int(round((p[1] - ys[0]) / GRID)), int(round((p[0] - xs[0]) / GRID))
        return 0 <= i < len(ys) and 0 <= j < len(xs) and labels[i, j] == main

    x0, y0, x1, y1 = floor
    sides = [  # name, start, direction along side, inward normal, length
        ("west", (x0, y0), (0, 1), (1, 0), y1 - y0),
        ("east", (x1, y0), (0, 1), (-1, 0), y1 - y0),
        ("south", (x0, y0), (1, 0), (0, 1), x1 - x0),
        ("north", (x0, y1), (1, 0), (0, -1), x1 - x0),
    ]
    best = None
    for name, start, along, normal, length in sides:
        run = []
        ts = np.arange(0, length + 1e-9, GRID)
        for t in list(ts) + [None]:
            ok = False
            if t is not None:
                b = np.array(start) + t * np.array(along)
                inset = b + DOOR_INSET * np.array(normal)
                # The strip between the wall and the inset point must be box-free too.
                strip = np.stack([b + s * np.array(normal) for s in (0.02, 0.2, DOOR_INSET)])
                ok = opens(inset) and rect_distance(strip, rects).min() >= 0.3
            if ok:
                run.append(t)
            elif run:
                width = run[-1] - run[0]
                if best is None or width > best[0]:
                    best = (width, name, start, along, normal, (run[0] + run[-1]) / 2)
                run = []
    if best is None:
        raise ValueError("no door-like gap found on the floor perimeter")
    width, name, start, along, normal, mid = best
    centre = np.array(start) + mid * np.array(along)
    spawn = centre + DOOR_INSET * np.array(normal)
    yaw = math.atan2(-normal[0], normal[1])  # yaw 0 = +Y, forward = (-sin, cos)
    return {"side": name, "centre": centre, "width": width, "along": along, "normal": normal,
            "spawn": spawn, "yaw": yaw}


def column_grid(boxes):
    """Rasterize boxes into the reconstruction's 0.22 m columns: dict (i,j) -> top height."""
    ox = min(b["rect"][0] for b in boxes)
    oy = min(b["rect"][1] for b in boxes)
    cells = {}
    for b in boxes:
        x0, y0, x1, y1 = b["rect"]
        for i in range(int(round((x0 - ox) / CELL)), max(int(round((x1 - ox) / CELL)), int(round((x0 - ox) / CELL)) + 1)):
            for j in range(int(round((y0 - oy) / CELL)), max(int(round((y1 - oy) / CELL)), int(round((y0 - oy) / CELL)) + 1)):
                cells[(i, j)] = max(cells.get((i, j), 0.0), b["top"])
    return cells, ox, oy


def cluster_cells(cells, predicate):
    keep = {c for c, top in cells.items() if predicate(top)}
    seen, out = set(), []
    for c in sorted(keep):
        if c in seen:
            continue
        group, stack = [], [c]
        seen.add(c)
        while stack:
            i, j = stack.pop()
            group.append((i, j))
            for n in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if n in keep and n not in seen:
                    seen.add(n)
                    stack.append(n)
        out.append(group)
    return out


def describe(group, cells, ox, oy, floor):
    tops = np.array([cells[c] for c in group])
    ii = np.array([c[0] for c in group])
    jj = np.array([c[1] for c in group])
    rect = (ox + ii.min() * CELL, oy + jj.min() * CELL, ox + (ii.max() + 1) * CELL, oy + (jj.max() + 1) * CELL)
    boundary_gap = min(rect[0] - floor[0], rect[1] - floor[1], floor[2] - rect[2], floor[3] - rect[3])
    return {"cells": len(group), "rect": rect, "top": float(np.median(tops)), "maxTop": float(tops.max()),
            "area": len(group) * CELL * CELL, "boundaryGap": float(boundary_gap),
            "centre": ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)}


def landmark(ident, label, kind, rect, top, **extra):
    x0, y0, x1, y1 = rect
    out = {"id": ident, "label": label, "kind": kind,
           "position": [round(float(x0 + x1) / 2, 4), round(float(y0 + y1) / 2, 4), round(float(top) / 2, 4)],
           "size": [round(float(x1 - x0), 4), round(float(y1 - y0), 4), round(float(top), 4)]}
    out.update(extra)
    return out


def compute_landmarks(scene=None, use_mesh=True, objects=None):
    """Landmarks plus saved SimReady objects (default scene only, unless objects=True)."""
    info = _compute_landmarks(scene, use_mesh)
    if objects if objects is not None else scene is None:
        door = info["landmarks"].pop()  # keep the door last
        info["landmarks"] += object_landmarks() + [door]
    return info


def _compute_landmarks(scene=None, use_mesh=True):
    """Return {landmarks, door, floor, rects, notes, mesh}. Deterministic.

    Uses the companion collider mesh (pipeline.room_mesh) for the room interior,
    landmarks and spawn when the GLB is present; otherwise the box-only heuristic.
    """
    scene = load_scene() if scene is None else scene
    floor, floor_z, boxes = split_boxes(scene)
    rects = np.array([b["rect"] for b in boxes]) if boxes else np.zeros((0, 4))
    if use_mesh:
        from . import room_mesh
        grids = room_mesh.load_grids(scene)
        if grids is not None:
            return compute_mesh_landmarks(floor, floor_z, boxes, rects, grids)
    door = find_door(floor, rects)
    notes = []
    cells, ox, oy = column_grid(boxes)
    # Reachability: a surface counts only if some free cell (stretch-sized) lies near it.
    reach_free, rxs, rys = free_grid(floor, rects, 0.39)
    labels, sizes = components(reach_free)
    si, sj = int(round((door["spawn"][1] - rys[0]) / GRID)), int(round((door["spawn"][0] - rxs[0]) / GRID))
    main = labels[si, sj] or int(np.argmax(sizes))
    gx, gy = np.meshgrid(rxs, rys)
    reachable_pts = np.stack([gx[labels == main], gy[labels == main]], 1)

    def accessible(rect):
        return bool(len(reachable_pts)) and rect_distance(reachable_pts, [rect]).min() <= 0.39 + 0.25

    surfaces = [describe(g, cells, ox, oy, floor) for g in cluster_cells(cells, lambda t: SURFACE_BAND[0] <= t <= SURFACE_BAND[1])]
    surfaces = [s for s in surfaces if s["cells"] >= 2]
    for s in surfaces:
        s["accessible"] = accessible(s["rect"])
    talls = [describe(g, cells, ox, oy, floor) for g in cluster_cells(cells, lambda t: t > TALL_MIN)]
    centre = np.array([(floor[0] + floor[2]) / 2, (floor[1] + floor[3]) / 2])

    usable = [s for s in surfaces if s["accessible"]] or surfaces
    tables = [s for s in usable if s["boundaryGap"] > 0.3 and 0.65 <= s["top"] <= 1.0 and s["area"] >= 0.5]
    if tables:
        table = max(tables, key=lambda s: s["area"])
    else:
        big = [s for s in usable if s["cells"] >= 4] or usable
        interior = sorted(big, key=lambda s: (-s["boundaryGap"], -s["area"]))
        table = interior[0]
        notes.append("No interior cluster met the table rule (0.65-1.00 m top, >=0.5 m2, >0.3 m from walls); "
                     "the interior-most accessible counter-height cluster is labelled table.")
    counters = [s for s in usable if s is not table and s["boundaryGap"] <= 0.3 and 0.6 <= s["top"] <= 1.1]
    if not counters:
        counters = [s for s in usable if s is not table]
        notes.append("No wall-touching cluster met the 0.80-1.00 m counter rule; using remaining surfaces.")
    counters.sort(key=lambda s: -s["cells"])
    door_c = np.array(door["centre"])
    opposite = [s for s in counters if np.dot(np.array(s["centre"]) - centre, door_c - centre) < 0]
    pool = opposite or counters
    sink = min(pool, key=lambda s: float(np.linalg.norm(np.array(s["centre"]) - centre))) if pool else None
    if sink is not None and not opposite:
        notes.append("No counter lies on the side opposite the door; sink is the counter closest to the room centre.")

    out = [landmark("table", "Table", "table", table["rect"], table["top"])]
    if sink is not None:
        out.append(landmark("sink", "Sink counter (reviewed)" if sink.get("reviewed") else "Counter (sink assumed)", "sink", sink["rect"], sink["top"]))
    for n, s in enumerate([c for c in counters if c["rect"] != sink["rect"]] if sink is not None else counters, 1):
        out.append(landmark(f"counter_{n}", f"Counter {n}", "counter", s["rect"], s["top"]))
    for n, s in enumerate(sorted(talls, key=lambda s: -s["cells"])[:6], 1):
        out.append(landmark(f"tall_{n}", f"Tall unit {n}", "tall", s["rect"], s["maxTop"]))
    along, normal = np.array(door["along"]), np.array(door["normal"])
    half = along * door["width"] / 2
    a, b = door_c - half, door_c + half + normal * 0.1
    out.append(landmark("door", "Kitchen door", "door",
                        (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])), 2.0))
    out[-1]["size"] = [max(s, 0.1) for s in out[-1]["size"][:2]] + [2.0]
    return {"landmarks": out, "door": door, "floor": floor, "floorZ": floor_z, "rects": rects, "boxes": boxes, "notes": notes,
            "mesh": None}


# ---------------------------------------------------------------- mesh-based landmarks
MIN_SURFACE_AREA = 0.08     # m2; smaller horizontal patches are clutter
ACCESS_BAND = 0.75          # surface cells further than this from the interior are behind walls


def mesh_free(grids, rects, clearance):
    """Interior cells whose centre is >= clearance from every mesh obstacle cell and every box."""
    from .room_mesh import obstacle_distance
    xs, ys = grids["xs"], grids["ys"]
    gx, gy = np.meshgrid(xs, ys)
    box_ok = (rect_distance(np.stack([gx.ravel(), gy.ravel()], 1), rects) >= clearance).reshape(gx.shape)
    return grids["interior"] & (obstacle_distance(grids) >= clearance) & box_ok


def geodesic(free, sources):
    """8-connected Dijkstra distance in cells over `free` from source cells (list of (i, j))."""
    import heapq
    h, w = free.shape
    dist = np.full(free.shape, np.inf)
    heap = []
    for s in sources:
        dist[s] = 0.0
        heap.append((0.0, s))
    heapq.heapify(heap)
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5)]
    while heap:
        d, (i, j) = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for di, dj, c in moves:
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w and free[a, b] and d + c < dist[a, b]:
                dist[a, b] = d + c
                heapq.heappush(heap, (d + c, (a, b)))
    return dist


def mesh_clusters(grids, values, predicate, side, min_area=MIN_SURFACE_AREA):
    """4-connected clusters of cells passing predicate, split by which side of the room they face."""
    from scipy import ndimage
    step = GRID
    out = []
    keep = predicate(values) & grids["obstacle"]
    for s in np.unique(side[keep]):
        mask = keep & (side == s)
        labels, n = ndimage.label(mask)
        for k in range(1, n + 1):
            cells = labels == k
            if cells.sum() * step * step < min_area:
                continue
            out.append(cells)
    return out


def describe_mesh(cells, values, grids, void_dist):
    xs, ys = grids["xs"], grids["ys"]
    ii, jj = np.nonzero(cells)
    rect = (xs[jj].min() - GRID / 2, ys[ii].min() - GRID / 2, xs[jj].max() + GRID / 2, ys[ii].max() + GRID / 2)
    tops = values[cells]
    return {"cells": int(cells.sum()), "mask": cells, "rect": tuple(float(v) for v in rect), "top": float(np.median(tops)),
            "maxTop": float(tops.max()), "area": float(cells.sum() * GRID * GRID),
            "boundaryGap": float(void_dist[cells].min()), "centre": ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)}


BOX_INTERIOR_DROP = 0.10    # boxes whose footprint is >=10% observed open floor are ceiling-filled columns


def filter_boxes(boxes, grids):
    """Split boxes into (kept, dropped). The 0.22 m columns fill everything under the ceiling, so columns standing
    on floor the mesh shows as open interior contradict the mesh; the mesh obstacle grid covers what they meant."""
    xs, ys = grids["xs"], grids["ys"]
    kept, dropped = [], []
    for b in boxes:
        x0, y0, x1, y1 = b["rect"]
        sub = grids["interior"][np.ix_((ys > y0) & (ys < y1), (xs > x0) & (xs < x1))]
        (dropped if sub.size and sub.mean() >= BOX_INTERIOR_DROP else kept).append(b)
    return kept, dropped


REVIEW = SCENE.parent / "landmark-review.json"
SIMREADY = REPO / "output" / "simready" / "scene.json"


def object_landmarks(path=SIMREADY):
    """Saved SimReady replacements as kind "object" landmarks (position/size = the replacement's removal box).

    Extra fields: objectId (replacement id), assetId, category, halfHeight (scaled asset half height, m),
    meshPosition (the replacement's mesh origin, world Z-up). Empty when no saved scene exists.
    """
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return []
    out = []
    for n, item in enumerate(data.get("replacements") or [], 1):
        try:
            asset = item.get("asset") or {}
            category = str(asset.get("category") or item["assetId"].split("_")[1]).lower()
            sel = item["selection"]
            dims = asset.get("dimensions") or sel["size"]
            out.append({"id": f"object_{n}", "label": f"{category.capitalize()} (SimReady {item['assetId']})", "kind": "object",
                        "position": [round(float(v), 4) for v in sel["position"]],
                        "size": [round(float(v), 4) for v in sel["size"]],
                        "objectId": item["id"], "assetId": item["assetId"], "category": category,
                        "halfHeight": round(float(dims[2]) * float(item.get("scale", 1.0)) / 2, 4),
                        "meshPosition": [round(float(v), 4) for v in item["position"]]})
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out


def load_review():
    try:
        return json.loads(REVIEW.read_text()) if REVIEW.exists() else {}
    except (ValueError, OSError):
        return {}


def compute_mesh_landmarks(floor, floor_z, boxes, rects, grids):
    from scipy import ndimage
    from .room_mesh import obstacle_distance
    notes = []
    xs, ys = grids["xs"], grids["ys"]
    all_boxes = boxes
    boxes, dropped = filter_boxes(boxes, grids)
    rects = np.array([b["rect"] for b in boxes]) if boxes else np.zeros((0, 4))
    if dropped:
        notes.append(f"{len(dropped)} of {len(all_boxes)} collision columns stand on floor the mesh shows as open "
                     "(they were filled up to the ceiling) and are ignored for planning; the mesh obstacle grid is used instead.")
    interior, obstacle = grids["interior"], grids["obstacle"]
    # Distance to the outside of the room (cells that are neither interior floor nor scanned obstacle).
    void = ~(interior | obstacle | grids["floor"])
    void_dist = ndimage.distance_transform_edt(~void) * GRID
    # Which side of the interior each obstacle cell faces (offset to its nearest interior cell).
    idist, (ni, nj) = ndimage.distance_transform_edt(~interior, return_indices=True)
    di, dj = np.indices(interior.shape)[0] - ni, np.indices(interior.shape)[1] - nj
    side = np.where(np.abs(dj) >= np.abs(di), np.where(dj >= 0, 1, 3), np.where(di >= 0, 0, 2))  # 0 N,1 E,2 S,3 W
    near = idist * GRID <= ACCESS_BAND
    work = np.where(grids["surface"] > 0, grids["surface"], grids["top"])

    # Reachability: stretch-sized free interior, largest component.
    reach = mesh_free(grids, rects, 0.39)
    labels, n = ndimage.label(reach, structure=np.ones((3, 3)))
    main = int(np.argmax(ndimage.sum(reach, labels, range(1, n + 1)))) + 1 if n else 0
    gx, gy = np.meshgrid(xs, ys)
    reachable_pts = np.stack([gx[labels == main], gy[labels == main]], 1) if main else np.zeros((0, 2))

    def accessible(rect):
        return bool(len(reachable_pts)) and rect_distance(reachable_pts, [rect]).min() <= 0.39 + 0.25

    surfaces = [describe_mesh(c, work, grids, void_dist) for c in
                mesh_clusters(grids, np.where(near, work, 0), lambda t: (t >= SURFACE_BAND[0]) & (t <= SURFACE_BAND[1]), side)]
    for s in surfaces:
        s["accessible"] = accessible(s["rect"])
    talls = [describe_mesh(c, grids["top"], grids, void_dist) for c in
             mesh_clusters(grids, np.where(near, grids["top"], 0), lambda t: t > TALL_MIN, side)]

    usable = [s for s in surfaces if s["accessible"]] or surfaces
    tables = [s for s in usable if s["boundaryGap"] > 0.3 and 0.65 <= s["top"] <= 1.0 and s["area"] >= 0.5]
    if tables:
        table = max(tables, key=lambda s: s["area"])
    else:
        big = [s for s in usable if s["area"] >= 0.2] or usable
        table = sorted(big, key=lambda s: (-round(s["boundaryGap"], 1), -s["area"]))[0]
        notes.append("No interior cluster met the table rule (0.65-1.00 m top, >=0.5 m2, >0.3 m from walls); "
                     "the interior-most accessible counter-height cluster is labelled table.")
    counters = [s for s in usable if s is not table and s["boundaryGap"] <= 0.3 and 0.6 <= s["top"] <= 1.1]
    if not counters:
        counters = [s for s in usable if s is not table]
        notes.append("No wall-touching cluster met the 0.80-1.00 m counter rule; using remaining surfaces.")
    counters.sort(key=lambda s: -s["cells"])

    # Spawn: interior cell with Go2 clearance, geodesically farthest from the table/counters.
    spawn_free = mesh_free(grids, rects, DOOR_CLEARANCE)
    labels, n = ndimage.label(spawn_free, structure=np.ones((3, 3)))
    if not n:
        raise ValueError("no interior cell has Go2 clearance")
    big = labels == int(np.argmax(ndimage.sum(spawn_free, labels, range(1, n + 1)))) + 1
    fi, fj = np.nonzero(big)
    pts = np.stack([xs[fj], ys[fi]], 1)
    # Counters line both long walls of the galley, so the distance to the *nearest* surface peaks mid-aisle.
    # Use the mean geodesic distance to the table and every counter instead: it peaks at the far end.
    dists = []
    for s in [table] + counters:
        k = int(np.argmin(np.linalg.norm(pts - np.array(s["centre"]), axis=1)))
        dists.append(geodesic(big, [(int(fi[k]), int(fj[k]))]))
    dist = np.mean(dists, axis=0)
    dist[~big] = -1
    # Tie-break deterministically by clearance (prefer the centre of the aisle).
    clear = obstacle_distance(grids)
    score = np.where(big, dist + 1e-3 * clear, -1)
    si, sj = np.unravel_index(int(np.argmax(score)), score.shape)
    spawn = np.array([xs[sj], ys[si]])
    tc = np.array(table["centre"])
    yaw = math.atan2(-(tc - spawn)[0], (tc - spawn)[1])
    room_centre = np.array([xs[np.nonzero(interior)[1]].mean(), ys[np.nonzero(interior)[0]].mean()])
    door = {"side": "far end", "centre": spawn, "width": float(2 * clear[si, sj]), "along": (1.0, 0.0),
            "normal": (-math.sin(yaw), math.cos(yaw)), "spawn": spawn, "yaw": yaw,
            "geodesicFromSurfaces": float(dist[si, sj] * GRID), "clearance": float(clear[si, sj])}

    opposite = [s for s in counters if np.dot(np.array(s["centre"]) - room_centre, spawn - room_centre) < 0]
    pool = opposite or counters
    sink = min(pool, key=lambda s: float(np.linalg.norm(np.array(s["centre"]) - room_centre))) if pool else None
    if sink is not None and not opposite:
        notes.append("No counter lies on the side opposite the door; sink is the counter closest to the room centre.")
    # A person can pin the sink from the scan: output/kitchen/landmark-review.json {"sink": {"near": [x, y]}}.
    review = load_review()
    if counters and isinstance(review.get("sink", {}).get("near"), list) and len(review["sink"]["near"]) == 2:
        near = np.array(review["sink"]["near"], dtype=float)
        sink = min(counters, key=lambda s: float(np.linalg.norm(np.array(s["centre"]) - near)))
        notes.append("Sink location pinned by human review of the scan (output/kitchen/landmark-review.json).")
        sink = {**sink, "reviewed": True}

    out = [landmark("table", "Table", "table", table["rect"], table["top"])]
    if sink is not None:
        out.append(landmark("sink", "Sink counter (reviewed)" if sink.get("reviewed") else "Counter (sink assumed)", "sink", sink["rect"], sink["top"]))
    for n, s in enumerate([c for c in counters if c["rect"] != sink["rect"]] if sink is not None else counters, 1):
        out.append(landmark(f"counter_{n}", f"Counter {n}", "counter", s["rect"], s["top"]))
    for n, s in enumerate(sorted(talls, key=lambda s: -s["cells"])[:6], 1):
        out.append(landmark(f"tall_{n}", f"Tall unit {n}", "tall", s["rect"], s["maxTop"]))
    half = 0.3
    out.append(landmark("door", "Kitchen entrance (far end of the room)", "door",
                        (spawn[0] - half, spawn[1] - half, spawn[0] + half, spawn[1] + half), 2.0))
    notes.append("Room interior, landmarks and spawn come from the companion collider mesh "
                 "(public/samples/rustic-kitchen-collider.glb); heights are above the fitted floor plane.")
    return {"landmarks": out, "door": door, "floor": floor, "floorZ": floor_z, "rects": rects, "boxes": boxes,
            "notes": notes, "mesh": grids, "meshClearance": clear, "droppedBoxes": [b["id"] for b in dropped]}


def main():
    result = compute_landmarks()
    d = result["door"]
    print(f"floor {np.round(result['floor'], 3).tolist()}  obstacles {len(result['boxes'])}")
    print(f"door: {d['side']}, centre {np.round(d['centre'], 2).tolist()}, gap {d['width']:.2f} m, "
          f"spawn {np.round(d['spawn'], 2).tolist()}, yaw {d['yaw']:.3f}")
    for lm in result["landmarks"]:
        print(f"{lm['id']:<10} {lm['kind']:<8} {lm['label']:<24} pos {lm['position']} size {lm['size']}")
    for note in result["notes"]:
        print("note:", note)


if __name__ == "__main__":
    main()
