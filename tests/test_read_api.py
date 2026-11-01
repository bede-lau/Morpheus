import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from pipeline.read_api import EnvironmentStore


class ReadApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'kitchen' / 'replays').mkdir(parents=True)
        self.scene = {'version': 1, 'name': 'Kitchen', 'units': 'm', 'upAxis': 'Z',
                      'source': {'scaleConfirmed': False, 'transform': {'matrix': [1,0,0,0,0,1,0,0,0,0,1,0,2,3,4,1], 'matrixOrder': 'column-major'}},
                      'objects': [{'id': 'floor', 'type': 'box', 'role': 'floor', 'position': [0,0,-.1], 'size': [4,4,.2]}]}
        self.path = self.root / 'kitchen' / 'scene.json'
        self.path.write_text(json.dumps(self.scene))
        self.store = EnvironmentStore(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def detail(self):
        status, body, _ = self.store.read('/api/environments/kitchen')
        self.assertEqual(status, 200)
        return json.loads(body)

    def test_scene_transform_and_native_evidence_share_revision(self):
        snapshot_before = self.store.snapshot()
        detail = self.detail()
        self.assertEqual(detail['visual']['sourceSha256'], '351f5f2ceb6ee9095a4b4100f864db9780a21cc6bdb8a70c6f88117cbb1d5fe0')
        self.assertEqual(detail['revision'], snapshot_before['revision'])
        self.assertEqual(detail['scene'], snapshot_before['scene'])
        self.assertNotIn('sha256', detail['scene']['source'])
        self.assertEqual(detail['visual']['transform'], self.scene['source']['transform'])
        self.assertFalse(detail['scene']['source']['scaleConfirmed'])
        self.assertEqual(detail['validation']['status'], 'passed')
        self.assertTrue(all(r['status'] == 'unavailable' for r in detail['replays'].values()))
        status, body, content_type = self.store.read('/api/environments/kitchen/bundle.zip')
        self.assertEqual((status, content_type), (200, 'application/zip'))
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            self.assertEqual(json.loads(archive.read('scene.json')), detail['scene'])
            self.assertEqual(json.loads(archive.read('validation.json')), detail['validation'])

    def test_stale_and_fabricated_demo_replay_never_marked_available(self):
        detail = self.detail()
        replay = {'environmentId': 'kitchen', 'environmentRevision': detail['revision'],
                  'replay': {'version': 1, 'source': 'backend', 'robot': 'go2', 'duration': 1,
                             'frames': [{'time': t, 'position': [0,0,.3], 'yaw': 0, 'armExtension': 0} for t in [0,1]]}}
        path = self.root / 'kitchen' / 'replays' / 'go2.json'
        path.write_text(json.dumps(replay))
        self.assertEqual(self.detail()['replays']['go2']['status'], 'available')
        self.scene['objects'][0]['size'][0] = 5
        self.path.write_text(json.dumps(self.scene))
        changed = self.detail()
        self.assertNotEqual(changed['revision'], detail['revision'])
        self.assertEqual(changed['replays']['go2']['status'], 'invalid')
        self.assertEqual(self.store.read('/api/environments/kitchen/replays/go2')[0], 409)
        replay['environmentRevision'] = changed['revision']
        replay['replay']['source'] = 'demo'
        path.write_text(json.dumps(replay))
        self.assertEqual(self.detail()['replays']['go2']['status'], 'invalid')

    def test_unknown_routes_and_missing_assets_are_explicit(self):
        self.assertEqual(self.store.read('/api/environments/kitchen/replays/../../scene.json')[0], 404)
        self.assertEqual(self.store.read('/api/environments/kitchen/replays/stretch')[0], 404)
        self.path.unlink()
        self.assertEqual(self.store.read('/api/environments/kitchen')[0], 404)


if __name__ == '__main__':
    unittest.main()
