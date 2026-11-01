import json
from pathlib import Path
from http.server import ThreadingHTTPServer
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np

from pipeline import room_mesh
from pipeline.landmarks import rect_distance
from pipeline.read_api import EnvironmentStore, validate_replay
from pipeline.task_planner import ROBOT_IDS, ROBOTS, World, plan_task, rules_plan, validate_plan

TASKS = ("wipe the table", "fill a glass of water")


class TaskPlannerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = World()
        cls.revision = EnvironmentStore().snapshot()["revision"]
        cls.envelopes = {(t, r): plan_task(t, r, cls.world, cls.revision, use_openai=False) for t in TASKS for r in ROBOT_IDS}

    def test_landmarks_include_contract_kinds(self):
        kinds = {lm["kind"] for lm in self.world.landmarks}
        self.assertTrue({"table", "sink", "door"} <= kinds)
        ids = [lm["id"] for lm in self.world.landmarks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_paths_keep_footprint_clear_of_every_box(self):
        rects = self.world.rects
        for (task, robot), env in self.envelopes.items():
            frames = env["replay"]["frames"]
            pts = np.array([f["position"][:2] for f in frames])
            # also sample between consecutive frames
            mids = (pts[1:] + pts[:-1]) / 2
            clearance = rect_distance(np.vstack([pts, mids]), rects).min()
            self.assertGreaterEqual(clearance, ROBOTS[robot]["radius"], (task, robot, clearance))

    def test_frames_satisfy_typescript_rules(self):
        for key, env in self.envelopes.items():
            replay = env["replay"]
            frames = replay["frames"]
            self.assertEqual(frames[0]["time"], 0)
            self.assertEqual(frames[-1]["time"], replay["duration"])
            self.assertTrue(all(b["time"] > a["time"] for a, b in zip(frames, frames[1:])), key)
            self.assertTrue(all(abs(b["time"] - a["time"] - 0.1) < 1e-6 for a, b in zip(frames, frames[1:])), key)
            for f in frames:
                self.assertTrue(0 <= f["armExtension"] <= 1)
                self.assertEqual(len(f["position"]), 3)
                self.assertTrue(np.isfinite(f["position"]).all() and np.isfinite(f["yaw"]))
            self.assertIn(replay["robot"], ROBOT_IDS)
            self.assertTrue(all(e["time"] <= replay["duration"] for e in replay["events"]))

    def test_envelopes_validate(self):
        for (task, robot), env in self.envelopes.items():
            validate_replay(json.loads(json.dumps(env)), robot, self.revision)
            self.assertEqual(env["replay"]["verdict"]["planner"], "rules")

    def test_verdicts_per_robot(self):
        go2 = self.envelopes[("wipe the table", "go2")]["replay"]["verdict"]
        self.assertIn(go2["status"], {"failed", "partial"})
        self.assertTrue(any(s["status"] == "failed" and s["name"].startswith("Wipe") for s in go2["steps"]))
        self.assertEqual(self.envelopes[("wipe the table", "soarm")]["replay"]["verdict"]["status"], "failed")
        self.assertEqual(self.envelopes[("wipe the table", "stretch")]["replay"]["verdict"]["status"], "completed")
        self.assertEqual(self.envelopes[("fill a glass of water", "stretch")]["replay"]["verdict"]["status"], "completed")
        soarm = self.envelopes[("wipe the table", "soarm")]["replay"]
        self.assertEqual(len({tuple(f["position"]) for f in soarm["frames"]}), 1)

    def test_rules_and_plan_validation(self):
        lms = self.world.landmarks
        self.assertEqual([s["action"] for s in rules_plan("Fill a glass of water", lms)], ["navigate", "fill", "navigate", "place"])
        self.assertEqual(rules_plan("clean the table", lms)[1], {"action": "wipe", "target": "table"})
        with self.assertRaises(ValueError):
            validate_plan([{"action": "wipe", "target": "table"}], lms)
        with self.assertRaises(ValueError):
            validate_plan([{"action": "navigate", "target": "../etc"}], lms)


ROLLOUT_ROOT = Path(__file__).resolve().parents[1] / "rollouts" / "kitchen"


@unittest.skipUnless(room_mesh.GLB.exists(), "companion collider GLB missing")
class RolloutsInsideRoomTest(unittest.TestCase):
    def test_committed_rollouts_stay_inside_the_mesh_interior(self):
        grids = room_mesh.load_grids()
        clear = room_mesh.obstacle_distance(grids)
        files = sorted(ROLLOUT_ROOT.glob("**/*.json"))
        self.assertGreaterEqual(len(files), 9)
        for path in files:
            replay = json.loads(path.read_text())["replay"]
            radius = ROBOTS[replay["robot"]]["radius"]
            pts = np.array([f["position"][:2] for f in replay["frames"]])
            i, j, ok = room_mesh.cell_index(grids, pts)
            name = str(path.relative_to(ROLLOUT_ROOT))
            self.assertTrue(ok.all(), name)
            outside = ~grids["interior"][i, j]
            self.assertFalse(outside.any(), (name, pts[outside][:3].tolist()))
            self.assertGreaterEqual(float(clear[i, j].min()), radius, name)


class TaskApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pipeline.server import Handler
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.load(resp)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    def test_listing_includes_default_task(self):
        status, body = self.request("/api/tasks")
        self.assertEqual(status, 200)
        slugs = {t["slug"]: t for t in body["tasks"]}
        self.assertIn("wipe-the-table", slugs)
        self.assertEqual(set(slugs["wipe-the-table"]["robots"]), set(ROBOT_IDS))
        status, env = self.request("/api/tasks/wipe-the-table/stretch")
        self.assertEqual(status, 200)
        self.assertEqual(env["replay"]["robot"], "stretch")
        self.assertEqual(self.request("/api/tasks/../stretch")[0], 404)
        self.assertEqual(self.request("/api/tasks/wipe-the-table/robot")[0], 404)

    def test_plan_rejects_bad_task_text(self):
        self.assertEqual(self.request("/api/tasks/plan", {"task": ""})[0], 400)
        self.assertEqual(self.request("/api/tasks/plan", {"task": "   "})[0], 400)
        self.assertEqual(self.request("/api/tasks/plan", {"task": "x" * 201})[0], 400)
        self.assertEqual(self.request("/api/tasks/plan", {"task": "wipe", "robot": "r2d2"})[0], 400)


if __name__ == "__main__":
    unittest.main()
