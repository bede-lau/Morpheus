import copy
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
            baseline = json.loads((Path(rollouts) / 'wipe-the-table' / 'stretch.json').read_text())
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
