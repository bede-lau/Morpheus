"""HTTP logic for the task stage (see docs/TASKS.md). Routes:
GET /api/tasks, GET /api/tasks/<slug>/<robot>, POST /api/tasks/plan.
Paths are built only from a validated slug and the fixed robot tuple.
"""
from __future__ import annotations

import json
import re
import threading

from .read_api import EnvironmentStore, validate_replay
from .task_planner import ROBOT_IDS, ROLLOUTS, REPLAYS, plan_and_write, slugify, write_json

SLUG = re.compile(r"^[a-z0-9-]{1,80}$")
PLAN_LOCK = threading.Lock()


def _json(status, value):
    return status, json.dumps(value, allow_nan=False).encode(), "application/json"


class TaskStore:
    def __init__(self, rollouts=ROLLOUTS, replays=REPLAYS, environment=None, builtin_rollouts=None):
        self.rollouts = rollouts
        self.replays = replays
        self.environment = environment or EnvironmentStore()
        self.builtin_rollouts = builtin_rollouts

    def rollout_file(self, slug, robot):
        if self.builtin_rollouts is not None:
            shipped = self.builtin_rollouts / slug / f"{robot}.json"
            if shipped.is_file():
                return shipped
        return self.rollouts / slug / f"{robot}.json"

    def listing(self):
        roots = [self.rollouts]
        if self.builtin_rollouts is not None:
            roots.append(self.builtin_rollouts)
        slugs = {folder.name for root in roots if root.is_dir() for folder in root.iterdir()
                 if folder.is_dir() and folder.name != "default" and SLUG.fullmatch(folder.name)}
        tasks = []
        for slug in sorted(slugs):
            robots, text = {}, slug.replace("-", " ")
            for robot in ROBOT_IDS:
                path = self.rollout_file(slug, robot)
                if path.is_file():
                    robots[robot] = f"/api/tasks/{slug}/{robot}"
                    try:
                        text = json.loads(path.read_bytes()).get("task") or text
                    except (ValueError, OSError):
                        pass
            if robots:
                tasks.append({"slug": slug, "task": text, "robots": robots})
        return {"version": 1, "tasks": tasks}

    def read(self, path):
        """Return (status, bytes, content type) or None when the route is not ours."""
        if path != "/api/tasks" and not path.startswith("/api/tasks/"):
            return None
        if path == "/api/tasks":
            return _json(200, self.listing())
        parts = path.split("/")
        if len(parts) != 5 or not SLUG.match(parts[3]) or parts[4] not in ROBOT_IDS:
            return _json(404, {"error": "Unknown task route"})
        file = self.rollout_file(parts[3], parts[4])
        if not file.is_file():
            return _json(404, {"error": "No rollout for this task and robot"})
        try:
            revision = self.environment.snapshot()["revision"]
            return _json(200, validate_replay(json.loads(file.read_bytes()), parts[4], revision))
        except FileNotFoundError as exc:
            return _json(404, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            return _json(409, {"error": str(exc)})

    def plan(self, body):
        if not isinstance(body, dict):
            return _json(400, {"error": "Expected a JSON object"})
        task, robot = body.get("task"), body.get("robot")
        if not isinstance(task, str) or not 1 <= len(task.strip()) <= 200:
            return _json(400, {"error": "task must be 1-200 characters"})
        if robot is not None and robot not in ROBOT_IDS:
            return _json(400, {"error": "Unknown robot"})
        with PLAN_LOCK:
            robots = (robot,) if robot else ROBOT_IDS
            slug = slugify(task)
            recorded = {}
            if self.builtin_rollouts is not None and slug == "aisle-clearance":
                for robot_id in robots:
                    status, content, _ = self.read(f"/api/tasks/{slug}/{robot_id}")
                    if status == 200:
                        envelope = json.loads(content)
                        if envelope["replay"].get("physics"):
                            recorded[robot_id] = envelope
            if len(recorded) == len(robots):
                results = recorded
                for robot_id, envelope in results.items():
                    write_json(self.replays / f"{robot_id}.json", envelope)
            else:
                slug, results = plan_and_write(task, robots, rollouts=self.rollouts, replays=self.replays)
        return _json(200, {"version": 1, "slug": slug, "task": task.strip(), "results": results})
