"""HTTP logic for the task stage (see docs/TASKS.md). Routes:
GET /api/tasks, GET /api/tasks/<slug>/<robot>, POST /api/tasks/plan.
Paths are built only from a validated slug and the fixed robot tuple.
"""
from __future__ import annotations

import json
import re
import threading

from .read_api import EnvironmentStore, validate_replay
from .task_planner import ROBOT_IDS, ROLLOUTS, REPLAYS, plan_and_write

SLUG = re.compile(r"^[a-z0-9-]{1,80}$")
PLAN_LOCK = threading.Lock()


def _json(status, value):
    return status, json.dumps(value, allow_nan=False).encode(), "application/json"


class TaskStore:
    def __init__(self, rollouts=ROLLOUTS, replays=REPLAYS, environment=None):
        self.rollouts = rollouts
        self.replays = replays
        self.environment = environment or EnvironmentStore()

    def listing(self):
        tasks = []
        if self.rollouts.is_dir():
            for folder in sorted(self.rollouts.iterdir()):
                if not folder.is_dir() or folder.name == "default" or not SLUG.match(folder.name):
                    continue
                robots, text = {}, folder.name.replace("-", " ")
                for robot in ROBOT_IDS:
                    path = folder / f"{robot}.json"
                    if path.is_file():
                        robots[robot] = f"/api/tasks/{folder.name}/{robot}"
                        try:
                            text = json.loads(path.read_bytes()).get("task") or text
                        except (ValueError, OSError):
                            pass
                if robots:
                    tasks.append({"slug": folder.name, "task": text, "robots": robots})
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
        file = self.rollouts / parts[3] / f"{parts[4]}.json"
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
            slug, results = plan_and_write(task, (robot,) if robot else ROBOT_IDS, rollouts=self.rollouts, replays=self.replays)
        return _json(200, {"version": 1, "slug": slug, "task": task.strip(), "results": results})
