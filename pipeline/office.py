"""Office (study) environment for the task stage: colliders, landmarks and spawn.

The office is the sparkjs.dev "fireplace" sample splat (public/samples/spark-fireplace.spz).
output/office/scene.json holds splat-only collision columns proposed by the environment
builder (no companion collider mesh exists for this scan, so the planner uses the box-only
path). Landmarks are anchored on objects that gpt-6.1-terra detected and the pipeline lifted
to 3D (output/discovery/office/report.json, same world frame as the builder): the reading
armchair, the side table holding two books, the waste bin and the bookshelf.

Scale is assumed (1 source unit = 1 m) and unreviewed, exactly like the kitchen fixture.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .landmarks import find_door, landmark, rect_distance, split_boxes

REPO = Path(__file__).resolve().parents[1]
ENV_ID = "office"
SCENE = REPO / "output" / ENV_ID / "scene.json"
ROLLOUTS = REPO / "rollouts" / ENV_ID
REPLAYS = REPO / "output" / ENV_ID / "replays"
SPLAT = "spark-fireplace.spz"
HEADLINE = "put-the-book-in-the-bin"
TASKS = ("put the book in the bin", "bring the book to the armchair", "go to the fireplace")

# Anchors in the shared world frame (metres, Z-up), from Terra detections lifted to 3D.
# Each landmark rect is the union of collision columns within `radius` of the anchor.
ANCHORS = [
    # id, label, kind, anchor xy, radius, usable top (m; None = tallest column)
    # The reachable floor is the rug strip down the middle (x -0.95..0.75); low book piles (0.2-0.3 m) block the rest.
    ("armchair", "Reading armchair", "table", (-1.55, 0.40), 0.05, 0.55),
    ("table", "Side table", "table", (1.10, -2.20), 0.30, None),
    ("bin", "Waste bin", "counter", (-0.90, 1.16), 0.08, 0.55),
    ("fireplace", "Fireplace", "counter", (0.40, 0.75), 0.10, 0.45),
]
BOOK = {"position": (1.22, -2.17), "size": (0.24, 0.18, 0.06)}


def load_scene(path=SCENE):
    return json.loads(Path(path).read_text())


def _cluster(boxes, xy, radius):
    p = np.array(xy, float)
    near = [b for b in boxes if rect_distance([p], [b["rect"]])[0] <= radius]
    if not near:
        raise ValueError(f"no collision columns near anchor {xy}")
    rects = np.array([b["rect"] for b in near])
    return (float(rects[:, 0].min()), float(rects[:, 1].min()), float(rects[:, 2].max()), float(rects[:, 3].max())), near


def compute_office_landmarks(scene=None):
    """Same shape as landmarks.compute_landmarks(): {landmarks, door, floor, floorZ, rects, boxes, notes, mesh, meshClearance}."""
    scene = scene or load_scene()
    floor, floor_z, boxes = split_boxes(scene)
    rects = [b["rect"] for b in boxes]
    out = []
    for ident, label, kind, xy, radius, top in ANCHORS:
        rect, near = _cluster(boxes, xy, radius)
        height = top if top is not None else max(b["top"] for b in near)
        out.append(landmark(ident, label, kind, rect, height))
    table = next(lm for lm in out if lm["id"] == "table")
    bx, by = BOOK["position"]
    sx, sy, sz = BOOK["size"]
    top = table["size"][2]
    out.append({"id": "object_1", "label": "Book (Terra detection on the side table)", "kind": "object",
                "position": [bx, by, round(top + sz / 2, 4)], "size": [sx, sy, sz],
                "objectId": "office-book-1", "assetId": "magazines_stack_a01", "category": "book",
                "halfHeight": sz / 2, "meshPosition": [bx, by, round(top + sz / 2, 4)]})
    door = find_door(floor, rects)
    spawn = np.array(door["spawn"], float)
    half = 0.3
    out.append(landmark("door", "Study door", "door", (spawn[0] - half, spawn[1] - half, spawn[0] + half, spawn[1] + half), 2.0))
    notes = ["Office colliders are splat-only columns from the environment builder (no companion mesh for this scan).",
             "Armchair, side table, bin and fireplace are anchored on gpt-6.1-terra detections lifted to 3D.",
             "Scale is assumed (1 unit = 1 m) and unreviewed."]
    return {"landmarks": out, "door": door, "floor": floor, "floorZ": floor_z, "rects": rects, "boxes": boxes,
            "notes": notes, "mesh": None, "meshClearance": None, "droppedBoxes": [], "spawnLabel": "Spawned at the study door"}


def main():
    info = compute_office_landmarks()
    d = info["door"]
    print(f"floor {np.round(info['floor'], 2).tolist()} obstacles {len(info['boxes'])} door {d['side']} spawn {np.round(d['spawn'], 2).tolist()} yaw {d['yaw']:.2f}")
    for lm in info["landmarks"]:
        print(f"{lm['id']:<10} {lm['kind']:<8} {lm['label']:<44} pos {lm['position']} size {lm['size']}")




# Rollouts: python -m pipeline.task_planner "<task>" --env office --all --rules  (one per entry in TASKS).


if __name__ == "__main__":
    main()
