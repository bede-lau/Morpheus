import copy
import io
import json
from http.server import ThreadingHTTPServer
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

from pipeline import ManifestError, export_files, package_environment, validate_manifest, validate_physics
from pipeline.server import Handler


def scene():
    return {"version": 1, "name": 'Test kitchen & "room"', "units": "m", "upAxis": "Z", "objects": [
        {"id": "floor", "name": "Floor", "type": "box", "role": "floor", "position": [0, 0, -.1], "size": [4, 4, .2]},
        {"id": "counter", "name": "Counter", "type": "box", "role": "surface", "position": [1.5, 0, .5], "size": [.6, 2, 1]},
    ]}


class ExportTest(unittest.TestCase):
    def test_invalid_geometry_is_rejected(self):
        for field, value in [("size", [1, -1, 1]), ("position", [0, float("nan"), 0]), ("quaternion", [0,0,0,0]), ("id", 'bad"/>')]:
            with self.subTest(field=field):
                s = scene()
                s["objects"][0][field] = value
                with self.assertRaises(ManifestError):
                    validate_manifest(s)

    def test_duplicates_and_wrong_frame_rejected(self):
        s = scene()
        s["objects"].append(copy.deepcopy(s["objects"][0]))
        with self.assertRaises(ManifestError): validate_manifest(s)
        s = scene()
        s["upAxis"] = "Y"
        with self.assertRaises(ManifestError): validate_manifest(s)

    def test_mujoco_compile_and_contact(self):
        import mujoco
        original = scene()
        files = export_files(original)
        model = mujoco.MjModel.from_xml_string(files["scene.xml"], {"environment.xml": files["environment.xml"].encode()})
        self.assertEqual(model.ngeom, 2)
        self.assertAlmostEqual(model.geom_size[0][0], 2)
        self.assertNotIn("quaternion", original["objects"][0])
        result = validate_physics(original)
        self.assertEqual(result["status"], "passed", result)
        self.assertTrue(result["dropProbe"]["floorContact"])

    def test_no_floor_does_not_claim_drop_success(self):
        s = scene()
        s["objects"] = s["objects"][1:]
        result = validate_physics(s)
        self.assertEqual(result["compile"], "passed")
        self.assertEqual(result["dropProbe"]["status"], "skipped")

    def test_disabled_geometry_is_preserved_but_not_simulated(self):
        s = scene()
        s["objects"][1]["enabled"] = False
        files = export_files(s)
        self.assertNotIn("env_counter", files["environment.xml"])
        self.assertEqual(len(json.loads(files["scene.json"])["objects"]), 2)
        result = validate_physics(s)
        self.assertEqual(result["compiledGeomCount"], 1)
        self.assertEqual(result["activeObjectCount"], 1)

    def test_archive_contains_reusable_scene_and_evidence(self):
        data, report = package_environment(scene())
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            self.assertEqual(set(archive.namelist()), {"scene.json", "scene.xml", "environment.xml", "validation.json", "README.txt"})
            self.assertEqual(json.loads(archive.read("validation.json"))["status"], "passed")
        self.assertEqual(report["status"], "passed")


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class QuietHandler(Handler):
            def log_message(self, *args):
                pass
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, path, payload, origin=None):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        return urlopen(Request(self.base+path, data=json.dumps(payload).encode(), headers=headers), timeout=30)

    def test_validate_and_export_over_http(self):
        with self.post("/api/validate", scene(), "http://localhost:5173") as response:
            self.assertEqual(json.load(response)["status"], "passed")
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
        with self.post("/api/export", scene()) as response:
            self.assertEqual(response.headers["Content-Type"], "application/zip")
            with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                self.assertIn("scene.xml", archive.namelist())

    def test_invalid_scene_and_remote_origin_are_rejected(self):
        with self.assertRaises(HTTPError) as error:
            self.post("/api/validate", {})
        self.assertEqual(error.exception.code, 400)
        with self.assertRaises(HTTPError) as error:
            self.post("/api/validate", scene(), "https://untrusted.example")
        self.assertEqual(error.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
