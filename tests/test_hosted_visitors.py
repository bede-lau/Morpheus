import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from pipeline import hosted, landmarks


class HostedVisitorIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        original = hosted.ROOT
        shutil.copytree(original / 'rollouts' / 'kitchen', cls.root / 'rollouts' / 'kitchen')
        (cls.root / 'output' / 'simready').mkdir(parents=True)
        shutil.copy2(original / 'output' / 'simready' / 'scene.json', cls.root / 'output' / 'simready' / 'scene.json')
        cls.root_patch = patch.object(hosted, 'ROOT', cls.root)
        cls.root_patch.start()
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), hosted.HostedHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.root_patch.stop()
        cls.tmp.cleanup()

    def request(self, path, token=None, body=None):
        headers = {'X-Morpheus-Visitor': token} if token else {}
        if body is not None:
            headers['Content-Type'] = 'application/json'
        req = Request(self.base + path, headers=headers, data=json.dumps(body).encode() if body is not None else None)
        try:
            with urlopen(req, timeout=10) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    def test_existing_visitors_receive_the_current_builtin_catalog(self):
        token = 'c' * 64
        visitor = self.root / 'output' / 'visitors' / hashlib.sha256(token.encode()).hexdigest()
        shipped = self.root / 'rollouts' / 'kitchen'
        for slug in ('wipe-the-table', 'previous-custom-task'):
            folder = visitor / 'rollouts' / slug
            folder.mkdir(parents=True)
            replay = json.loads((shipped / 'wipe-the-table' / 'stretch.json').read_text())
            replay['task'] = 'Saved before deployment'
            (folder / 'stretch.json').write_text(json.dumps(replay))
        (visitor / '.ready').touch()

        status, body = self.request('/api/tasks', token)
        self.assertEqual(status, 200)
        slugs = {item['slug'] for item in body['tasks']}
        self.assertIn('drawer-opening', slugs)
        self.assertIn('previous-custom-task', slugs)
        status, current = self.request('/api/tasks/wipe-the-table/stretch', token)
        self.assertEqual(status, 200)
        self.assertEqual(current, json.loads((shipped / 'wipe-the-table' / 'stretch.json').read_text()))
        status, saved = self.request('/api/tasks/previous-custom-task/stretch', token)
        self.assertEqual(status, 200)
        self.assertEqual(saved['task'], 'Saved before deployment')

    def test_aisle_task_uses_recorded_physics_for_all_robots(self):
        token = "d" * 64
        status, body = self.request("/api/tasks/plan", token, {"task": "Aisle clearance"})
        self.assertEqual(status, 200)
        self.assertEqual(set(body["results"]), {"g1", "go2", "stretch"})
        for robot, result in body["results"].items():
            self.assertEqual(result["replay"]["physics"]["heldOutPassed"], 12)
            self.assertEqual(self.request(f"/api/tasks/aisle-clearance/{robot}", token), (200, result))
            self.assertEqual(self.request(f"/api/environments/kitchen/replays/{robot}", token), (200, result))

    def test_counter_task_serves_recorded_contact_for_all_robots(self):
        token = "e" * 64
        status, body = self.request("/api/tasks/plan", token, {"task": "Counter reach"})
        self.assertEqual(status, 200)
        self.assertEqual(set(body["results"]), {"g1", "go2", "stretch"})
        for robot, result in body["results"].items():
            replay = result["replay"]
            self.assertEqual(replay["physics"]["heldOutPassed"], 12)
            self.assertGreaterEqual(replay["physics"]["contactDwellSeconds"], .25)
            self.assertEqual(replay["frames"][-1]["contactForce"], 0)
            self.assertEqual(self.request(f"/api/tasks/counter-reach/{robot}", token), (200, result))

    def test_drawer_task_preserves_recorded_runs_for_all_robots(self):
        token = "f" * 64
        status, body = self.request("/api/tasks/plan", token, {"task": "Drawer opening"})
        self.assertEqual(status, 200)
        self.assertEqual(set(body["results"]), {"g1", "go2", "stretch"})
        for robot, result in body["results"].items():
            replay = result["replay"]
            self.assertEqual(replay["physics"]["heldOutPassed"], 12)
            self.assertGreaterEqual(replay["frames"][-1]["drawerDisplacement"], .12)
            self.assertEqual(replay["frames"][-1]["contactForce"], 0)
            self.assertEqual(self.request(f"/api/tasks/drawer-opening/{robot}", token), (200, result))

    def test_table_wipe_serves_recorded_cloth_and_scripted_previews(self):
        token = "9" * 64
        status, body = self.request("/api/tasks/plan", token, {"task": "Wipe the table"})
        self.assertEqual(status, 200)
        self.assertEqual(set(body["results"]), {"g1", "go2", "stretch"})
        result = body["results"]["stretch"]
        self.assertGreaterEqual(len(result["replay"]["cloth"]["frames"][-1]["wiped"]), 8)
        self.assertEqual(self.request("/api/tasks/wipe-the-table/stretch", token), (200, result))
        for robot in ("g1", "go2"):
            self.assertNotIn("physics", body["results"][robot]["replay"])

    def test_scene_and_plan_isolation_and_reload(self):
        a, b = 'a' * 64, 'b' * 64
        self.assertEqual(self.request('/api/objects/scene')[0], 401)
        self.assertEqual(self.request('/api/tasks', '../bad')[0], 401)
        status, original_a = self.request('/api/objects/scene', a)
        self.assertEqual(status, 200)
        self.assertEqual(original_a, self.request('/api/objects/scene', b)[1])
        changed = copy.deepcopy(original_a)
        changed['baseRevision'] = changed['revision']
        changed['replacements'] = []
        status, saved = self.request('/api/objects/scene', a, changed)
        self.assertEqual(status, 200)
        self.assertEqual(saved['replacements'], [])
        self.assertEqual(self.request('/api/objects/scene', a)[1], saved)
        self.assertEqual(self.request('/api/objects/scene', b)[1], original_a)

        seen = []
        def fake_plan(task, robots, rollouts, replays):
            seen.append(landmarks.visitor_scene_path.get())
            folder = Path(rollouts) / 'private-task'
            folder.mkdir(parents=True)
            baseline = json.loads((self.root / 'rollouts' / 'kitchen' / 'wipe-the-table' / 'stretch.json').read_text())
            (folder / 'stretch.json').write_text(json.dumps(baseline))
            Path(replays).mkdir(parents=True, exist_ok=True)
            (Path(replays) / 'stretch.json').write_text(json.dumps(baseline))
            return 'private-task', {'stretch': baseline}

        with patch('pipeline.task_api.plan_and_write', side_effect=fake_plan):
            self.assertEqual(self.request('/api/tasks/plan', a, {'task': 'private task', 'robot': 'stretch'})[0], 200)
        self.assertTrue(seen[0].is_file())
        self.assertIn('private-task', [t['slug'] for t in self.request('/api/tasks', a)[1]['tasks']])
        self.assertNotIn('private-task', [t['slug'] for t in self.request('/api/tasks', b)[1]['tasks']])
        self.assertEqual(self.request('/api/tasks/private-task/stretch', a)[0], 200)
        self.assertEqual(self.request('/api/tasks/private-task/stretch', b)[0], 404)
        self.assertEqual(self.request('/api/environments/kitchen/replays/stretch', a)[0], 200)


if __name__ == '__main__':
    unittest.main()
