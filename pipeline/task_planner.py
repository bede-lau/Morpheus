"""Plan a natural-language kitchen task for each robot and emit replay envelopes.

Offline and deterministic by default (rules planner). When OPENAI_API_KEY is
configured the Responses API may choose the step sequence; its answer is
validated against the landmark IDs and falls back to rules on any error.
The output is a planned rollout reviewed against collision boxes only: it is not
a physics simulation and says nothing about real task success.

CLI: python -m pipeline.task_planner "wipe the table" --all | --robot stretch
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

import numpy as np

from .landmarks import GRID, compute_landmarks, free_grid, rect_distance
from .read_api import ROOT as OUTPUT_ROOT, EnvironmentStore, validate_replay

REPO = Path(__file__).resolve().parents[1]
ROLLOUTS = REPO / "rollouts" / "kitchen"
REPLAYS = OUTPUT_ROOT / "kitchen" / "replays"
HZ = 10
MARGIN = 0.05
APPROACH_GAP = 0.10
TURN_RATE = 1.6       # rad/s in-place turning
ARM_RATE = 0.6        # armExtension units per second
OPENAI_TIMEOUT = 25
OPENAI_MODELS = ("gpt-5", )   # extended at runtime by discover_models()

ROBOTS = {
    "stretch": {"name": "Stretch", "base": "wheeled", "radius": 0.34, "reach": 0.52, "armHeight": (0.15, 1.10), "eye": 1.30, "speed": 0.45},
    "go2": {"name": "Go2", "base": "quadruped", "radius": 0.42, "reach": 0.0, "armHeight": None, "eye": 0.45, "speed": 0.80},
    "soarm": {"name": "SO-ARM100", "base": "stationary", "radius": 0.18, "reach": 0.36, "armHeight": (0.05, 0.40), "eye": 0.35, "speed": 0.0},
}
ROBOT_IDS = ("stretch", "go2", "soarm")
ACTIONS = ("navigate", "wipe", "fill", "place")
MANIPULATION = {"wipe", "fill", "place"}


def slugify(task):
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:80].strip("-")
    return slug or "task"


def yaw_towards(dx, dy):
    """yaw 0 faces +Y; forward vector = (-sin yaw, cos yaw)."""
    return math.atan2(-dx, dy)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# ---------------------------------------------------------------- geometry
class World:
    def __init__(self, scene=None):
        info = compute_landmarks(scene)
        self.info = info
        self.landmarks = info["landmarks"]
        self.by_id = {lm["id"]: lm for lm in self.landmarks}
        self.rects = info["rects"]
        self.floor = info["floor"]
        self.door = info["door"]
        self.spawn = np.array(info["door"]["spawn"], float)
        self.spawn_yaw = float(info["door"]["yaw"])
        self.grids = {}

    def grid(self, robot):
        if robot not in self.grids:
            r = ROBOTS[robot]["radius"]
            free, xs, ys = free_grid(self.floor, self.rects, r + MARGIN, floor_margin=r)
            self.grids[robot] = (free, xs, ys)
        return self.grids[robot]

    def clearance(self, points):
        return rect_distance(points, self.rects)

    def segment_clear(self, a, b, need):
        n = max(2, int(np.linalg.norm(b - a) / 0.01) + 1)
        pts = a[None, :] + np.linspace(0, 1, n)[:, None] * (b - a)[None, :]
        x0, y0, x1, y1 = self.floor
        inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        return bool(inside.all() and self.clearance(pts).min() >= need)

    def landmark_rect(self, lm):
        (x, y, _), (sx, sy, _) = lm["position"], lm["size"]
        return (x - sx / 2, y - sy / 2, x + sx / 2, y + sy / 2)

    def astar(self, robot, start, goal):
        free, xs, ys = self.grid(robot)
        h, w = free.shape

        def cell(p):
            return (int(np.clip(round((p[1] - ys[0]) / GRID), 0, h - 1)), int(np.clip(round((p[0] - xs[0]) / GRID), 0, w - 1)))

        s, g = cell(start), cell(goal)
        if not free[s] or not free[g]:
            return None
        moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
        dist = {s: 0.0}
        parent = {s: None}
        heap = [(0.0, s)]
        while heap:
            _, c = heapq.heappop(heap)
            if c == g:
                break
            base = dist[c]
            for di, dj, cost in moves:
                n = (c[0] + di, c[1] + dj)
                if not (0 <= n[0] < h and 0 <= n[1] < w) or not free[n]:
                    continue
                if di and dj and not (free[c[0] + di, c[1]] and free[c[0], c[1] + dj]):
                    continue
                nd = base + cost
                if nd < dist.get(n, 1e18):
                    dist[n], parent[n] = nd, c
                    heapq.heappush(heap, (nd + math.hypot(n[0] - g[0], n[1] - g[1]), n))
        if g not in parent:
            return None
        path, c = [], g
        while c is not None:
            path.append(np.array([xs[c[1]], ys[c[0]]]))
            c = parent[c]
        path.reverse()
        path[0], path[-1] = np.array(start, float), np.array(goal, float)
        return path

    def smooth(self, robot, path):
        need = ROBOTS[robot]["radius"] + MARGIN * 0.5
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.segment_clear(path[i], path[j], need):
                j -= 1
            out.append(path[j])
            i = j
        return out

    def reachable_cells(self, robot):
        """Cells reachable from spawn (BFS on the inflated grid)."""
        free, xs, ys = self.grid(robot)
        h, w = free.shape
        si = (int(round((self.spawn[1] - ys[0]) / GRID)), int(round((self.spawn[0] - xs[0]) / GRID)))
        seen = np.zeros_like(free)
        if not free[si]:
            return seen
        stack = [si]
        seen[si] = True
        while stack:
            i, j = stack.pop()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1), (i + 1, j + 1), (i - 1, j - 1), (i + 1, j - 1), (i - 1, j + 1)):
                if 0 <= a < h and 0 <= b < w and free[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    stack.append((a, b))
        return seen

    def approach(self, robot, lm):
        """Free reachable pose with the footprint APPROACH_GAP from the landmark rect, facing its centre."""
        free, xs, ys = self.grid(robot)
        seen = self.reachable_cells(robot)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx[seen], gy[seen]], 1)
        if not len(pts):
            return None
        rect = self.landmark_rect(lm)
        d = rect_distance(pts, [rect])
        target = ROBOTS[robot]["radius"] + APPROACH_GAP
        centre = np.array(lm["position"][:2])
        for tol in (0.03, 0.08, 0.15, 0.3, 0.6):
            ok = np.abs(d - target) <= tol
            if ok.any():
                cand = pts[ok]
                # Prefer the middle of a face (closest to centre), then closest to spawn.
                score = np.linalg.norm(cand - centre, axis=1) + 0.05 * np.linalg.norm(cand - self.spawn, axis=1)
                p = cand[int(np.argmin(score))]
                return {"position": p, "yaw": yaw_towards(*(centre - p)), "distance": float(rect_distance([p], [rect])[0]),
                        "gapError": float(abs(rect_distance([p], [rect])[0] - target))}
        return None


# ---------------------------------------------------------------- task parsing
def rules_plan(task, landmarks):
    ids = {lm["id"] for lm in landmarks}
    text = task.lower()
    words = set(re.findall(r"[a-z]+", text))
    sink = "sink" if "sink" in ids else next((i for i in ids if i.startswith("counter")), "table")
    if words & {"wipe", "clean", "wash", "scrub"} and not words & {"glass", "water", "drink", "fill"}:
        target = "table"
        for lm in landmarks:
            if lm["kind"] in {"counter", "sink"} and lm["kind"] in words:
                target = lm["id"]
        return [{"action": "navigate", "target": target}, {"action": "wipe", "target": target}]
    if words & {"fill", "water", "glass", "drink"}:
        return [{"action": "navigate", "target": sink}, {"action": "fill", "target": sink},
                {"action": "navigate", "target": "table"}, {"action": "place", "target": "table"}]
    best, score = "table", 0
    for lm in landmarks:
        if lm["kind"] == "door":
            continue
        tokens = set(re.findall(r"[a-z]+", (lm["id"] + " " + lm["label"] + " " + lm["kind"]).lower()))
        overlap = len(tokens & words)
        if overlap > score:
            best, score = lm["id"], overlap
    return [{"action": "navigate", "target": best}]


def validate_plan(steps, landmarks):
    ids = {lm["id"] for lm in landmarks if lm["kind"] != "door"}
    if not isinstance(steps, list) or not 1 <= len(steps) <= 8:
        raise ValueError("plan needs 1-8 steps")
    at = None
    clean = []
    for step in steps:
        if not isinstance(step, dict) or step.get("action") not in ACTIONS or step.get("target") not in ids:
            raise ValueError("unknown action or landmark")
        if step["action"] == "navigate":
            at = step["target"]
        elif step["target"] != at:
            raise ValueError("manipulation before navigating to its target")
        clean.append({"action": step["action"], "target": step["target"]})
    return clean


def load_key():
    try:
        from .object_server import load_local_key
        return load_local_key()
    except Exception:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        return key


def _openai_request(url, key, payload=None, timeout=OPENAI_TIMEOUT):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


_MODEL_CACHE = None


def discover_models(key):
    """Ordered candidate Responses models: newest gpt-6 / gpt-5.x chat models first, then gpt-5."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    names = []
    try:
        listing = _openai_request("https://api.openai.com/v1/models", key, timeout=10)
        names = [m.get("id", "") for m in listing.get("data", []) if isinstance(m, dict)]
    except Exception:
        names = []
    skip = ("audio", "realtime", "tts", "transcribe", "image", "search", "embedding", "codex", "instruct", "nano", "luna", "chat-latest", "pro")
    suffix_rank = {"": 0, "-sol": 0, "-terra": 1, "-astra": 1, "-mini": 2}

    def version(name):
        m = re.match(r"^gpt-(\d+)(?:\.(\d+))?(.*)$", name)
        if not m or any(s in m.group(3) for s in skip) or m.group(3) not in suffix_rank:
            return None
        return (int(m.group(1)), int(m.group(2) or 0), -suffix_rank[m.group(3)])

    ranked = sorted((n for n in names if version(n) and version(n)[0] >= 5 and version(n)[2] >= -1), key=version, reverse=True)
    out = []
    for n in ranked[:3] + list(OPENAI_MODELS):
        if n not in out:
            out.append(n)
    _MODEL_CACHE = out[:4]
    return _MODEL_CACHE


PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["steps", "rationale"],
    "properties": {
        "rationale": {"type": "string"},
        "steps": {"type": "array", "minItems": 1, "maxItems": 8, "items": {
            "type": "object", "additionalProperties": False, "required": ["action", "target"],
            "properties": {"action": {"type": "string", "enum": list(ACTIONS)}, "target": {"type": "string"}}}},
    },
}


def openai_plan(task, landmarks):
    """Return (steps, model) or raise. Never logs the key."""
    key = load_key()
    if not key or os.environ.get("MORPHEUS_PLANNER") == "rules":
        raise ValueError("OpenAI planner disabled")
    usable = [{"id": lm["id"], "label": lm["label"], "kind": lm["kind"], "topHeightM": lm["size"][2]} for lm in landmarks if lm["kind"] != "door"]
    for lm in usable:
        lm["schemaNote"] = "sink is assumed from geometry" if lm["kind"] == "sink" else ""
    prompt = ("You plan household tasks for a mobile robot in a reconstructed kitchen. Choose the minimal ordered step "
              "sequence using only these actions: navigate (drive to a landmark), wipe (wipe its surface), fill (fill a "
              "glass with water; only at a sink), place (put a held object on a surface). Every manipulation must follow "
              "a navigate to the same target. Use only the landmark ids given. House conventions: a glass of water is "
              "filled at the sink and then served by navigating to the table and placing it there; wiping or cleaning "
              "without a named surface means the table; the robot starts at the door with nothing in hand.\n"
              f"Landmarks: {json.dumps(usable)}\nTask: {task}")
    last = None
    deadline = time.monotonic() + OPENAI_TIMEOUT
    for model in discover_models(key):
        remaining = deadline - time.monotonic()
        if remaining < 2:
            break
        payload = {"model": model, "input": prompt,
                   "text": {"format": {"type": "json_schema", "name": "task_plan", "schema": PLAN_SCHEMA, "strict": True}}}
        try:
            resp = _openai_request("https://api.openai.com/v1/responses", key, payload, timeout=remaining)
            text = resp.get("output_text")
            if not text:
                for item in resp.get("output", []):
                    for part in item.get("content", []) or []:
                        if part.get("type") == "output_text":
                            text = part.get("text")
            steps = validate_plan(json.loads(text)["steps"], landmarks)
            return steps, model
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code} from {model}"
        except Exception as exc:  # timeout, bad JSON, invalid plan
            last = f"{type(exc).__name__} from {model}"
    raise ValueError(last or "no model available")


def decisions_feasibility(robot, steps, world):
    """Optional Decisions predicates; returns {stepIndex: note}. Empty on any error."""
    key = load_key()
    if not key or os.environ.get("MORPHEUS_PLANNER") == "rules":
        return {}
    prof = ROBOTS[robot]
    questions = []
    for i, step in enumerate(steps):
        if step["action"] not in MANIPULATION:
            continue
        lm = world.by_id[step["target"]]
        questions.append({"name": f"step_{i}", "type": "predicate",
                          "instructions": (f"Robot {prof['name']} ({prof['base']} base, arm reach {prof['reach']} m, arm height "
                                           f"range {prof['armHeight']}) must {step['action']} at a surface whose top is "
                                           f"{lm['size'][2]:.2f} m high. Can this robot physically do that?")})
    if not questions:
        return {}
    try:
        result = _openai_request("https://api.openai.com/v1/decisions", key, {
            "model": "gpt-6-luna", "input": [{"role": "user", "content": [{"type": "input_text", "text": "Answer feasibility predicates for a robot task plan."}]}],
            "questions": questions}, timeout=OPENAI_TIMEOUT)
        notes = {}
        for ans in result.get("answers", []):
            name, prob = ans.get("name", ""), ans.get("probability")
            if ans.get("type") == "predicate" and name.startswith("step_") and isinstance(prob, (int, float)):
                notes[int(name[5:])] = f"Decisions (gpt-6-luna) feasibility estimate p={prob:.2f}."
        return notes
    except Exception:
        return {}


def plan_steps(task, landmarks, use_openai=True):
    if use_openai:
        try:
            steps, model = openai_plan(task, landmarks)
            return steps, "openai", model
        except Exception:
            pass
    return rules_plan(task, landmarks), "rules", None


# ---------------------------------------------------------------- frames
class Timeline:
    def __init__(self, pos, yaw):
        self.frames = []
        self.events = []
        self.pos = np.array(pos, float)
        self.yaw = float(yaw)
        self.arm = 0.0
        self.emit()

    @property
    def time(self):
        return (len(self.frames) - 1) / HZ

    def emit(self):
        self.frames.append({"time": round(len(self.frames) / HZ, 1), "position": [round(float(self.pos[0]), 4), round(float(self.pos[1]), 4), 0.0],
                            "yaw": round(wrap(self.yaw), 4), "armExtension": round(min(1.0, max(0.0, self.arm)), 4)})

    def event(self, label):
        self.events.append({"time": round(self.time, 1), "label": label})

    def hold(self, seconds):
        for _ in range(max(1, int(round(seconds * HZ)))):
            self.emit()

    def turn_to(self, yaw):
        delta = wrap(yaw - self.yaw)
        n = int(math.ceil(abs(delta) / (TURN_RATE / HZ)))
        start = self.yaw
        for k in range(1, n + 1):
            self.yaw = start + delta * k / n
            self.emit()
        self.yaw = start + delta

    def arm_to(self, value):
        value = min(1.0, max(0.0, value))
        n = int(math.ceil(abs(value - self.arm) / (ARM_RATE / HZ)))
        start = self.arm
        for k in range(1, n + 1):
            self.arm = start + (value - start) * k / n
            self.emit()
        self.arm = value

    def drive(self, path, speed):
        for a, b in zip(path, path[1:]):
            seg = b - a
            length = float(np.linalg.norm(seg))
            if length < 1e-6:
                continue
            self.turn_to(yaw_towards(*seg))
            n = max(1, int(math.ceil(length / (speed / HZ))))
            for k in range(1, n + 1):
                self.pos = a + seg * k / n
                self.emit()
            self.pos = np.array(b, float)

    def wipe(self, base_yaw, seconds=6.0):
        self.arm_to(0.8)
        n = int(seconds * HZ)
        for k in range(1, n + 1):
            phase = 2 * math.pi * k / (2.0 * HZ)  # 0.5 Hz
            self.arm = 0.7 + 0.15 * math.sin(phase)
            self.yaw = base_yaw + 0.15 * math.sin(phase / 2)
            self.emit()
        self.yaw = base_yaw
        self.arm_to(0.8)


ACTION_TEXT = {"wipe": "Wipe", "fill": "Fill glass at", "place": "Place glass on"}


def arm_target(robot, distance):
    reach = ROBOTS[robot]["reach"]
    return min(1.0, max(0.2, (distance - 0.15) / reach)) if reach else 0.0


def simulate(world, robot, task, steps):
    prof = ROBOTS[robot]
    tl = Timeline(world.spawn, world.spawn_yaw)
    tl.event("Spawned at the kitchen door")
    results, at, arrival = [], None, None
    aborted = False
    for i, step in enumerate(steps):
        lm = world.by_id[step["target"]]
        name = lm["label"].split(" (")[0].lower()
        title = f"Navigate to {name}" if step["action"] == "navigate" else f"{ACTION_TEXT[step['action']]} {name}"
        if aborted:
            results.append({"name": title, "status": "skipped", "note": "Skipped after an earlier step failed."})
            continue
        rect = world.landmark_rect(lm)
        if step["action"] == "navigate":
            if prof["speed"] <= 0:
                d = float(rect_distance([tl.pos], [rect])[0])
                if d <= prof["reach"]:
                    at, arrival = lm["id"], {"distance": d, "yaw": yaw_towards(*(np.array(lm["position"][:2]) - tl.pos))}
                    results.append({"name": title, "status": "done", "note": f"Already within reach ({d:.2f} m)."})
                    continue
                tl.event(f"{prof['name']} has a stationary base and cannot drive to the {name}")
                tl.turn_to(yaw_towards(*(np.array(lm["position"][:2]) - tl.pos)))
                tl.arm_to(1.0)
                tl.hold(1.0)
                tl.arm_to(0.0)
                results.append({"name": title, "status": "failed", "note": f"Stationary base; the {name} is {d:.2f} m away, beyond its {prof['reach']:.2f} m reach."})
                aborted = True
                continue
            pose = world.approach(robot, lm)
            path = world.astar(robot, tl.pos, pose["position"]) if pose is not None else None
            if path is None:
                tl.event(f"No collision-free path to the {name}")
                tl.hold(1.0)
                results.append({"name": title, "status": "failed", "note": "No collision-free path on the inflated occupancy grid."})
                aborted = True
                continue
            path = world.smooth(robot, path)
            tl.event(f"Navigating to the {name}")
            length = sum(float(np.linalg.norm(b - a)) for a, b in zip(path, path[1:]))
            tl.drive(path, prof["speed"])
            tl.turn_to(pose["yaw"])
            tl.event(f"Arrived at the {name}")
            at, arrival = lm["id"], pose
            note = f"{length:.1f} m path, stopped {pose['distance'] - prof['radius']:.2f} m from the surface."
            results.append({"name": title, "status": "done", "note": note})
            continue
        # manipulation
        if at != lm["id"]:
            results.append({"name": title, "status": "failed", "note": "Robot is not at this landmark."})
            aborted = True
            continue
        top = lm["size"][2]
        if not prof["reach"]:
            tl.event(f"{prof['name']} has no manipulator: cannot {step['action']} the {name}")
            tl.hold(2.0)
            results.append({"name": title, "status": "failed", "note": f"{prof['name']} has no arm."})
            aborted = True
            continue
        lo, hi = prof["armHeight"]
        horizontal = arrival["distance"] - 0.15
        if not lo <= top <= hi or horizontal > prof["reach"]:
            tl.event(f"The {name} is out of reach")
            tl.arm_to(1.0)
            tl.hold(1.0)
            tl.arm_to(0.0)
            why = f"surface top {top:.2f} m is outside the {lo:.2f}-{hi:.2f} m arm range" if not lo <= top <= hi else f"surface is {horizontal:.2f} m beyond the base, reach {prof['reach']:.2f} m"
            results.append({"name": title, "status": "failed", "note": why.capitalize() + "."})
            aborted = True
            continue
        tl.turn_to(yaw_towards(*(np.array(lm["position"][:2]) - tl.pos)))
        ext = arm_target(robot, arrival["distance"])
        if step["action"] == "wipe":
            tl.event(f"Wiping the {name}")
            tl.wipe(tl.yaw)
            tl.arm_to(0.0)
            tl.event(f"Finished wiping the {name}")
            results.append({"name": title, "status": "done", "note": "Arm oscillated 0.55-0.85 for 6 s over an approx. 0.6 m x 0.4 m patch."})
        elif step["action"] == "fill":
            tl.arm_to(ext)
            tl.event("Filling glass")
            tl.hold(4.0)
            tl.arm_to(0.0)
            results.append({"name": title, "status": "done", "note": "Held 4 s at the assumed sink (sink location is a geometric guess)."})
        else:
            tl.arm_to(ext)
            tl.hold(0.5)
            tl.arm_to(0.0)
            tl.event("Placed glass on table" if lm["id"] == "table" else f"Placed glass on the {name}")
            results.append({"name": title, "status": "done", "note": f"Arm extended to {ext:.2f}."})
    tl.hold(1.0)
    done = sum(r["status"] == "done" for r in results)
    status = "completed" if done == len(results) else ("partial" if done else "failed")
    return tl, results, status


def summarize(robot, task, status, results):
    name = ROBOTS[robot]["name"]
    failed = next((r for r in results if r["status"] == "failed"), None)
    if status == "completed":
        return f"{name} completed '{task}': " + "; ".join(r["name"] for r in results) + "."
    done = [r["name"] for r in results if r["status"] == "done"]
    return (f"{name} {'partially completed' if status == 'partial' else 'could not complete'} '{task}'"
            + (f" (done: {', '.join(done)})" if done else "") + f". Failed at '{failed['name']}': {failed['note']}")


def plan_task(task, robot, world=None, revision=None, use_openai=True, steps_info=None):
    world = world or World()
    revision = revision or EnvironmentStore().snapshot()["revision"]
    steps, planner, model = steps_info or plan_steps(task, world.landmarks, use_openai)
    tl, results, status = simulate(world, robot, task, steps)
    if planner == "openai":
        for i, note in decisions_feasibility(robot, steps, world).items():
            if i < len(results):
                results[i]["note"] = (results[i]["note"] + " " + note).strip()
    tl.event({"completed": "Task completed", "partial": "Task partially completed", "failed": "Task failed"}[status])
    duration = tl.frames[-1]["time"]
    verdict = {"status": status, "summary": summarize(robot, task, status, results), "steps": results, "planner": planner}
    if model:
        verdict["model"] = model
    envelope = {
        "environmentId": "kitchen", "environmentRevision": revision, "task": task, "taskSlug": slugify(task),
        "replay": {"version": 1, "source": "backend", "robot": robot, "duration": duration, "frames": tl.frames,
                   "task": task, "events": tl.events, "verdict": verdict, "landmarks": world.landmarks,
                   "plannerNotes": world.info["notes"]},
    }
    validate_replay(envelope, robot, revision)
    return envelope


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False))
    tmp.replace(path)


def plan_and_write(task, robots=ROBOT_IDS, use_openai=True, rollouts=ROLLOUTS, replays=REPLAYS):
    task = task.strip()
    world = World()
    revision = EnvironmentStore().snapshot()["revision"]
    steps_info = plan_steps(task, world.landmarks, use_openai)
    slug = slugify(task)
    results = {}
    for robot in robots:
        env = plan_task(task, robot, world, revision, use_openai, steps_info)
        write_json(Path(rollouts) / slug / f"{robot}.json", env)
        write_json(Path(replays) / f"{robot}.json", env)
        if slug == "wipe-the-table":
            write_json(Path(rollouts) / "default" / f"{robot}.json", env)
        results[robot] = env
    return slug, results


def main():
    parser = argparse.ArgumentParser(description="Plan a kitchen task for the demo robots")
    parser.add_argument("task")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true")
    group.add_argument("--robot", choices=ROBOT_IDS)
    parser.add_argument("--rules", action="store_true", help="skip the OpenAI planner")
    args = parser.parse_args()
    if not 1 <= len(args.task.strip()) <= 200:
        parser.error("task must be 1-200 characters")
    robots = (args.robot,) if args.robot else ROBOT_IDS
    slug, results = plan_and_write(args.task, robots, use_openai=not args.rules)
    for robot, env in results.items():
        v = env["replay"]["verdict"]
        print(f"{slug}/{robot}: {v['status']} planner={v['planner']}{' model=' + v['model'] if 'model' in v else ''} duration={env['replay']['duration']}s")
        print("   ", v["summary"])


if __name__ == "__main__":
    main()
