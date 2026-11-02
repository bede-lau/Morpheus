"""Serve the Morpheus UI and both APIs from one Railway service."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import os
from pathlib import Path
import re
import shutil
import threading
import urllib.request
from urllib.parse import urlparse

from . import landmarks, object_server, server
from .read_api import EnvironmentStore, ROBOTS, validate_replay
from .task_api import TaskStore

ROOT = Path(__file__).resolve().parents[1]
VISITOR_LOCK = threading.Lock()


class VisitorEnvironmentStore(EnvironmentStore):
    def __init__(self, visitor_root):
        super().__init__()
        self.visitor_root = visitor_root

    def replay(self, robot, revision):
        if robot not in ROBOTS:
            raise FileNotFoundError("Unknown robot")
        path = self.visitor_root / "replays" / f"{robot}.json"
        if not path.is_file():
            path = ROOT / "rollouts" / "kitchen" / "default" / f"{robot}.json"
        if not path.is_file():
            raise FileNotFoundError("No rollout supplied for this robot")
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Replay exceeds 32 MB")
        import json
        return validate_replay(json.loads(path.read_bytes()), robot, revision)


def seed_output():
    source = Path(os.environ.get("MORPHEUS_SEED_DIR", "/seed-output"))
    target = ROOT / "output"
    if not source.is_dir():
        return
    for item in source.rglob("*"):
        destination = target / item.relative_to(source)
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif item.is_file() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)


def prepare_collider():
    configured = os.environ.get("MORPHEUS_COLLIDER_PATH")
    if not configured:
        return
    destination = Path(configured)
    expected = "38b9f692edeb671b7106f3e36cfe7f4c55be20e98fe90c014a8714b09874f6fa"
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
        return
    url = "https://wlt-ai-cdn.art/example_exports/rustic_kitchen_with_natural_light/rustic_kitchen_with_natural_light_collider.glb"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(8_000_001)
    if len(data) > 8_000_000 or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Downloaded collider did not match the expected checksum")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)


class HostedHandler(server.Handler, object_server.Handler, SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "dist"), **kwargs)

    def _allowed_origin(self):
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlparse(origin)
        return parsed.scheme in {"http", "https"} and parsed.netloc == self.headers.get("Host")

    def allowed(self):
        return self._allowed_origin()

    def prepare_visitor(self):
        token = self.headers.get("X-Morpheus-Visitor", "")
        if not re.fullmatch(r"[0-9a-f]{64}", token):
            self._json(401, {"error": "A valid visitor session is required"})
            return False
        visitor = ROOT / "output" / "visitors" / hashlib.sha256(token.encode()).hexdigest()
        rollouts = visitor / "rollouts"
        with VISITOR_LOCK:
            if not (visitor / ".ready").is_file():
                visitor.mkdir(parents=True, exist_ok=True)
                shutil.copytree(ROOT / "rollouts" / "kitchen", rollouts, dirs_exist_ok=True)
                (visitor / ".ready").touch()
        self.scene_path = visitor / "scene.json"
        baseline = Path("/seed-output/simready/scene.json")
        self.scene_fallback = baseline if baseline.is_file() else ROOT / "output" / "simready" / "scene.json"
        self.environment_store = VisitorEnvironmentStore(visitor)
        self.task_store = TaskStore(rollouts=rollouts, replays=visitor / "replays", environment=self.environment_store)
        return True

    def do_GET(self):
        path = urlparse(self.path).path
        if (path == "/api/objects/scene" or path == "/api/environments" or path.startswith("/api/environments/") or path == "/api/tasks" or path.startswith("/api/tasks/")) and not self.prepare_visitor():
            return
        if path.startswith("/api/objects/"):
            return object_server.Handler.do_GET(self)
        if path.startswith("/api/"):
            return server.Handler.do_GET(self)
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        path = urlparse(self.path).path
        if (path == "/api/objects/scene" or path == "/api/tasks/plan") and not self.prepare_visitor():
            return
        if path == "/api/tasks/plan":
            marker = landmarks.visitor_scene_path.set(self.scene_path if self.scene_path.is_file() else self.scene_fallback)
            try:
                return server.Handler.do_POST(self)
            finally:
                landmarks.visitor_scene_path.reset(marker)
        if path.startswith("/api/objects/"):
            return object_server.Handler.do_POST(self)
        return server.Handler.do_POST(self)


def main():
    seed_output()
    prepare_collider()
    port = int(os.environ.get("PORT", "8080"))
    httpd = ThreadingHTTPServer(("0.0.0.0", port), HostedHandler)
    print(f"Morpheus listening on {port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
