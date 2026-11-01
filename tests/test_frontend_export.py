"""Exercise the TypeScript browser exporter with the real native MuJoCo parser.

Requires npm ci and the Python requirements. No browser, HTTP server, generated
sample downloads, or additional Python dependencies are required.
"""
import base64
import io
import json
import math
from pathlib import Path
import subprocess
import unittest
import zipfile

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NODE_EXPORT = """
import { readFileSync } from 'node:fs';
import { buildEnvironmentFiles, createEnvironmentZip } from './src/export.ts';
const source = JSON.parse(readFileSync(0, 'utf8'));
const files = buildEnvironmentFiles(source);
const zipBase64 = Buffer.from(createEnvironmentZip(source)).toString('base64');
process.stdout.write(JSON.stringify({files, zipBase64}));
"""


def frontend_export(source):
    process = subprocess.run(
        ["node", "--import", "tsx", "--input-type=module", "-e", NODE_EXPORT],
        input=json.dumps(source), capture_output=True, text=True,
        cwd=PROJECT_ROOT, timeout=30, check=True,
    )
    return json.loads(process.stdout)


def representative_scene():
    return {
        "version": 1, "name": 'Cross-language <kitchen> & "room"',
        "units": "m", "upAxis": "Z",
        "source": {
            "scaleConfirmed": False, "reviewed": False,
            # This source transform must never be applied to collider poses.
            "transform": {"matrix": [1,0,0,0,0,1,0,0,0,0,1,0,20,30,40,1], "matrixOrder": "column-major"},
        },
        "objects": [
            {"id": "floor", "name": "Floor", "type": "box", "role": "floor", "position": [0,0,-.05], "size": [4,5,.1]},
            {"id": "rotated_wall", "name": "Rotated wall", "type": "box", "role": "wall", "position": [1.125,-.75,.9], "size": [.24,1.6,1.8], "quaternion": [3*math.cos(.35),0,0,3*math.sin(.35)]},
            {"id": "counter", "name": "Counter", "type": "box", "role": "surface", "position": [-1.25,.3456789,.45], "size": [.7,1.2,.9], "quaternion": [2,0,0,0], "friction": [.6,.01,.001]},
            {"id": "disabled", "name": "Removed proposal", "type": "box", "position": [0,0,.5], "size": [1,1,1], "enabled": False},
        ],
    }


class FrontendExportIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = representative_scene()
        cls.output = frontend_export(cls.source)

    def assert_native_geometry_matches(self, source, files):
        normalized = json.loads(files["scene.json"])
        model = mujoco.MjModel.from_xml_string(
            files["scene.xml"], {"environment.xml": files["environment.xml"].encode()},
        )
        active = [obj for obj in normalized["objects"] if obj.get("enabled") is not False]
        self.assertEqual(model.ngeom, len(active))
        self.assertEqual(normalized["source"], source["source"])
        for obj in active:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "env_"+obj["id"])
            self.assertGreaterEqual(index, 0)
            self.assertEqual(model.geom_type[index], mujoco.mjtGeom.mjGEOM_BOX)
            source_obj = next(original for original in source["objects"] if original["id"] == obj["id"])
            for axis in range(3):
                self.assertAlmostEqual(model.geom_pos[index][axis], source_obj["position"][axis], places=9)
                self.assertAlmostEqual(model.geom_size[index][axis] * 2, source_obj["size"][axis], places=9)
            quaternion = source_obj.get("quaternion", [1,0,0,0])
            length = math.hypot(*quaternion)
            expected = [component/length for component in quaternion]
            actual = model.geom_quat[index]
            # q and -q encode the same orientation.
            alignment = abs(sum(a*b for a,b in zip(actual, expected)))
            self.assertAlmostEqual(alignment, 1.0, places=10)
        for obj in normalized["objects"]:
            if obj.get("enabled") is False:
                self.assertEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "env_"+obj["id"]), -1)
        return model

    def test_browser_mjcf_compiles_and_preserves_geometry(self):
        self.assert_native_geometry_matches(self.source, self.output["files"])
        # Cross-language normalization must not mutate caller-owned source values.
        self.assertEqual(self.source["objects"][2]["quaternion"], [2,0,0,0])

    def test_browser_zip_remains_unverified_even_when_mjcf_is_valid(self):
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(self.output["zipBase64"]))) as archive:
            self.assertEqual(set(archive.namelist()), {"environment.xml", "scene.xml", "scene.json", "validation.json", "README.txt"})
            files = {name: archive.read(name).decode() for name in archive.namelist()}
        self.assert_native_geometry_matches(self.source, files)
        report = json.loads(files["validation.json"])
        self.assertTrue(report["schemaValid"])
        self.assertEqual(report["status"], "unverified")
        self.assertEqual(report["compile"], "not_run")
        self.assertEqual(report["dropProbe"]["status"], "not_run")
        self.assertEqual(report["activeObjectCount"], 3)
        # Our external integration test does not retroactively certify or alter
        # the browser's original export evidence.
        self.assertEqual(files["validation.json"], self.output["files"]["validation.json"])


if __name__ == "__main__":
    unittest.main()
