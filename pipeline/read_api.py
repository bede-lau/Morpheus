"""Read fixed, locally prepared environments and revision-bound teammate replays."""
import hashlib
import io
import json
import math
from pathlib import Path
import threading
import zipfile

from .export import export_files, validate_manifest, validate_physics

ROOT = Path(__file__).resolve().parents[1] / "output"
ROBOTS = ("go2", "g1", "stretch")
# Committed pre-rendered rollouts (docs/TASKS.md); used when output/kitchen/replays/<robot>.json is absent.
DEFAULT_ROLLOUTS = Path(__file__).resolve().parents[1] / "rollouts" / "kitchen" / "default"
CDN = "https://wlt-ai-cdn.art/example_exports/rustic_kitchen_with_natural_light/rustic_kitchen_with_natural_light"
# SHA-256 of the full SPZ bytes, verified against the downloaded official sample.
# Pinned with the CDN asset (also checked by scripts/fetch-sample.mjs), not the
# scene manifest: exposing provenance must not invalidate existing rollouts.
SOURCE_SHA256 = "351f5f2ceb6ee9095a4b4100f864db9780a21cc6bdb8a70c6f88117cbb1d5fe0"
# Local copies fetched by `npm run fetch-sample` (served by Vite under /samples/). The CDN
# is the fallback: it is the same bytes but far slower, which stalls the demo load.
SAMPLES = Path(__file__).resolve().parents[1] / "public" / "samples"
LOCAL_SPLAT, LOCAL_MESH = "rustic-kitchen-500k.spz", "rustic-kitchen-collider.glb"


def visual_urls():
    splat = "/samples/" + LOCAL_SPLAT if (SAMPLES / LOCAL_SPLAT).is_file() else CDN + "_500k.spz"
    mesh = "/samples/" + LOCAL_MESH if (SAMPLES / LOCAL_MESH).is_file() else CDN + "_collider.glb"
    return splat, mesh


def finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def validate_replay(value, robot, revision):
    if not isinstance(value, dict) or value.get("environmentId") != "kitchen" or value.get("environmentRevision") != revision:
        raise ValueError("Replay does not match this environment revision")
    replay = value.get("replay")
    if not isinstance(replay, dict) or replay.get("version") != 1 or replay.get("source") != "backend" or replay.get("robot") != robot:
        raise ValueError("Expected a backend replay for this robot")
    duration, frames = replay.get("duration"), replay.get("frames")
    if not finite_number(duration) or duration <= 0 or not isinstance(frames, list) or not 2 <= len(frames) <= 100000:
        raise ValueError("Replay duration or frame count is invalid")
    previous = -1
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Invalid replay frame")
        time, pos, yaw, extension = (frame.get(k) for k in ("time", "position", "yaw", "armExtension"))
        if (not finite_number(time) or not previous < time <= duration or time < 0
                or not isinstance(pos, list) or len(pos) != 3 or not all(map(finite_number, pos))
                or not finite_number(yaw) or not finite_number(extension) or not 0 <= extension <= 1):
            raise ValueError("Invalid replay frame coordinates or timing")
        previous = time
    if frames[0]["time"] != 0 or frames[-1]["time"] != duration:
        raise ValueError("Replay must span time zero through duration")
    # Reject non-finite extras too. The source tag is producer metadata, not proof
    # of physical validity or task success.
    json.dumps(value, allow_nan=False)
    return value


class EnvironmentStore:
    def __init__(self, root=ROOT, fallback=None):
        self.root = Path(root)
        # Only the real output root falls back to the repository rollouts; temporary roots stay isolated.
        self.fallback = Path(fallback) if fallback else (DEFAULT_ROLLOUTS if self.root == ROOT else None)
        self.lock = threading.Lock()
        self.cached = None

    def snapshot(self):
        path = self.root / "kitchen" / "scene.json"
        if not path.is_file():
            raise FileNotFoundError("Kitchen fixture is missing; run npm run prepare-sample")
        raw = path.read_bytes()
        if len(raw) > 5 * 1024 * 1024:
            raise ValueError("Scene exceeds 5 MB")
        scene = validate_manifest(json.loads(raw))
        canonical = json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        revision = hashlib.sha256(canonical).hexdigest()
        with self.lock:
            if self.cached is None or self.cached["revision"] != revision:
                self.cached = {"revision": revision, "scene": scene, "validation": validate_physics(scene)}
            return self.cached

    def replay(self, robot, revision):
        # Caller must pass a known robot, never a caller-supplied filesystem path.
        if robot not in ROBOTS:
            raise FileNotFoundError("Unknown robot")
        path = self.root / "kitchen" / "replays" / f"{robot}.json"
        if not path.is_file() and self.fallback is not None:
            path = self.fallback / f"{robot}.json"
        if not path.is_file():
            raise FileNotFoundError("No teammate rollout supplied for this robot")
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Replay exceeds 32 MB")
        return validate_replay(json.loads(path.read_bytes()), robot, revision)

    def detail(self, snapshot):
        statuses = {}
        for robot in ROBOTS:
            try:
                self.replay(robot, snapshot["revision"])
                statuses[robot] = {"status": "available", "url": f"/api/environments/kitchen/replays/{robot}"}
            except FileNotFoundError as exc:
                statuses[robot] = {"status": "unavailable", "url": None, "reason": str(exc)}
            except (ValueError, TypeError) as exc:
                statuses[robot] = {"status": "invalid", "url": None, "reason": str(exc)}
        splat_url, mesh_url = visual_urls()
        return {"version": 1, "id": "kitchen", **snapshot,
                "visual": {"splatUrl": splat_url, "meshUrl": mesh_url, "sourceSha256": SOURCE_SHA256,
                           "sourceUpAxis": "Y-down", "transform": snapshot["scene"].get("source", {}).get("transform")},
                "replays": statuses}

    def read(self, path):
        """Return (status, bytes, content type); None means route not owned here."""
        base = "/api/environments"
        if path != base and not path.startswith(base + "/"):
            return None
        allowed = {base, base + "/kitchen"}
        files = {"scene.json", "scene.xml", "environment.xml", "validation.json", "bundle.zip"}
        allowed.update(base + "/kitchen/" + name for name in files)
        allowed.update(base + "/kitchen/replays/" + robot for robot in ROBOTS)
        if path not in allowed:
            return self.json(404, {"error": "Unknown environment route"})
        try:
            snap = self.snapshot()
            if path == base:
                return self.json(200, {"version": 1, "environments": [{"id": "kitchen", "name": snap["scene"]["name"], "revision": snap["revision"], "url": base + "/kitchen"}]})
            if path == base + "/kitchen":
                return self.json(200, self.detail(snap))
            if "/replays/" in path:
                return self.json(200, self.replay(path.rsplit("/", 1)[1], snap["revision"]))
            name = path.rsplit("/", 1)[1]
            if name == "scene.json":
                return self.json(200, snap["scene"])
            if name == "validation.json":
                return self.json(200, snap["validation"])
            exported = export_files(snap["scene"])
            if name == "bundle.zip":
                if snap["validation"]["status"] == "failed":
                    return self.json(422, {"error": "Environment failed native validation", "validation": snap["validation"]})
                exported["validation.json"] = json.dumps(snap["validation"], indent=2)
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
                    for filename, text in exported.items():
                        archive.writestr(filename, text)
                return 200, buf.getvalue(), "application/zip"
            return 200, exported[name].encode(), "application/xml"
        except FileNotFoundError as exc:
            return self.json(404, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            return self.json(409, {"error": str(exc)})

    @staticmethod
    def json(status, value):
        return status, json.dumps(value, allow_nan=False).encode(), "application/json"
