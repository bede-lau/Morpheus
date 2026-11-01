"""Portable, static collision environments. All manifest dimensions are full sizes.

The scene manifest is the coordinate authority: metres, right-handed, Z-up.
source.transform is provenance describing how assets entered that frame; it is
not reapplied to the already-transformed collider positions during export.
"""
from __future__ import annotations

import copy
import io
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Any


class ManifestError(ValueError):
    """User-correctable scene errors, rather than a simulator crash."""


def _vector(value: Any, length: int, path: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ManifestError(f"{path} must contain {length} numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ManifestError(f"{path} must contain finite numbers")
    return [float(v) for v in value]


def validate_manifest(manifest: Any) -> dict:
    """Validate and return a normalized copy; never mutate the caller's scene."""
    if not isinstance(manifest, dict):
        raise ManifestError("Scene must be a JSON object")
    scene = copy.deepcopy(manifest)
    if scene.get("version") != 1 or isinstance(scene.get("version"), bool):
        raise ManifestError("version must be 1")
    if scene.get("units") != "m" or scene.get("upAxis") != "Z":
        raise ManifestError("Expected units='m' and upAxis='Z'; calibrate before export")
    if not isinstance(scene.get("name"), str) or not scene["name"].strip():
        raise ManifestError("name must be a non-empty string")
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ManifestError("Add at least one collision object before export")
    if len(objects) > 10000:
        raise ManifestError("At most 10000 primitive colliders are supported")
    ids = set()
    for index, obj in enumerate(objects):
        path = f"objects[{index}]"
        if not isinstance(obj, dict):
            raise ManifestError(f"{path} must be an object")
        name = obj.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,95}", name):
            raise ManifestError(f"{path}.id must start with a letter and use letters, digits, _ or - (max 96 characters)")
        if name in ids:
            raise ManifestError(f"Duplicate object id: {name}")
        ids.add(name)
        if obj.get("type") != "box":
            raise ManifestError(f"{path}.type must be 'box'")
        if "enabled" in obj and not isinstance(obj["enabled"], bool):
            raise ManifestError(f"{path}.enabled must be boolean")
        obj["position"] = _vector(obj.get("position"), 3, f"{path}.position")
        obj["size"] = _vector(obj.get("size"), 3, f"{path}.size")
        if any(v <= 0 for v in obj["size"]):
            raise ManifestError(f"{path}.size must be positive full dimensions in metres")
        quat = _vector(obj.get("quaternion", [1, 0, 0, 0]), 4, f"{path}.quaternion")
        norm = math.hypot(*quat)
        if not math.isfinite(norm) or norm < 1e-12:
            raise ManifestError(f"{path}.quaternion must have a finite nonzero norm")
        obj["quaternion"] = [v / norm for v in quat]
        if "friction" in obj:
            obj["friction"] = _vector(obj["friction"], 3, f"{path}.friction")
            if any(v < 0 for v in obj["friction"]):
                raise ManifestError(f"{path}.friction cannot be negative")
    try:
        json.dumps(scene, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"Scene metadata must be finite JSON: {exc}") from exc
    return scene


def _numbers(values):
    return " ".join(format(v, ".12g") for v in values)


def _xml(root):
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def _worldbody(scene, parent):
    world = ET.SubElement(parent, "worldbody")
    for obj in scene["objects"]:
        if obj.get("enabled") is False:
            continue
        role = obj.get("role", "obstacle")
        rgba = {"floor": "0.72 0.75 0.72 1", "wall": "0.85 0.85 0.81 1", "surface": "0.27 0.55 0.45 1"}.get(role, "0.52 0.60 0.62 1")
        ET.SubElement(world, "geom", {
            "name": f"env_{obj['id']}", "type": "box",
            "pos": _numbers(obj["position"]),
            "size": _numbers([v / 2 for v in obj["size"]]),
            "quat": _numbers(obj["quaternion"]), "rgba": rgba,
            "friction": _numbers(obj.get("friction", [0.8, 0.005, 0.0001])),
            "contype": "1", "conaffinity": "1", "group": "0",
        })
    return world


def export_files(manifest: Any) -> dict[str, str]:
    """Return standalone MJCF, reusable include, manifest, and integration notes."""
    scene = validate_manifest(manifest)
    include = ET.Element("mujocoinclude")
    _worldbody(scene, include)
    root = ET.Element("mujoco", {"model": scene["name"]})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    ET.SubElement(root, "include", {"file": "environment.xml"})
    return {
        "environment.xml": _xml(include),
        "scene.xml": _xml(root),
        "scene.json": json.dumps(scene, indent=2, allow_nan=False) + "\n",
        "README.txt": (
            "Morpheus collision environment\n\n"
            "Coordinates: right-handed, metres, Z up. Box sizes in scene.json are FULL dimensions; MJCF sizes are half-extents.\n"
            "Quaternions: w x y z. Collider coordinates are already in the calibrated world frame.\n"
            "source.transform records the visual asset alignment; do not apply it again to colliders.\n\n"
            "Standalone: mujoco.MjModel.from_xml_path('scene.xml')\n"
            "Integration: add <include file=\"environment.xml\"/> inside your robot's <mujoco> root.\n"
            "Environment geoms use the env_ prefix and standard collision masks (contype=1 conaffinity=1).\n"
            "Remove or reconcile your robot scene's existing floor; do not add two overlapping floors.\n"
            "No robot, policy, moving furniture, visual splat, or third-party asset is bundled.\n"
            "All environment geoms are static. Geometry and friction are approximations, not measurements unless independently calibrated.\n"
            "A passing drop probe checks basic collision support only, not reconstruction accuracy or robot task feasibility.\n"
        ),
    }


def validate_physics(manifest: Any) -> dict:
    """Load actual exported MJCF, then optionally drop a sphere above an exposed floor."""
    scene = validate_manifest(manifest)
    active = [o for o in scene["objects"] if o.get("enabled") is not False]
    report = {"schemaValid": True, "engine": "MuJoCo", "objectCount": len(scene["objects"]), "activeObjectCount": len(active), "compile": "not_run", "dropProbe": {"status": "not_run"}, "warnings": ["Checks model loading and collision support only; scan alignment, metric scale, and robot task success are not established."]}
    try:
        import mujoco
    except ImportError:
        report["warnings"].append("MuJoCo is not installed. Schema validation only; physics has not been tested.")
        report["status"] = "unverified"
        return report
    report["engineVersion"] = mujoco.__version__
    files = export_files(scene)
    try:
        model = mujoco.MjModel.from_xml_string(files["scene.xml"], {"environment.xml": files["environment.xml"].encode()})
        report["compile"] = "passed"
        report["compiledGeomCount"] = int(model.ngeom)
    except Exception as exc:
        report["compile"] = "failed"
        report["error"] = str(exc)
        report["status"] = "failed"
        return report

    # Pick an exposed point on an unrotated floor. Never claim a passing test
    # merely because a sphere landed on an unrelated table above the floor.
    floors = [o for o in active if o.get("role") == "floor" and abs(abs(o["quaternion"][0]) - 1) < 1e-8]
    radius = 0.04
    candidate = None
    for floor in floors:
        if min(floor["size"][:2]) < radius * 6:
            continue
        top = floor["position"][2] + floor["size"][2] / 2
        for fx, fy in [(0, 0), (-.3, -.3), (.3, -.3), (-.3, .3), (.3, .3), (0, .3), (.3, 0), (-.3, 0), (0, -.3)]:
            x = floor["position"][0] + fx * floor["size"][0]
            y = floor["position"][1] + fy * floor["size"][1]
            blocked = False
            for other in active:
                if other["id"] == floor["id"]:
                    continue
                # Conservative bounding sphere handles rotated boxes without
                # making a false claim of an unobstructed drop trajectory.
                q = other["quaternion"]
                if abs(abs(q[0]) - 1) < 1e-8:
                    half = [v / 2 for v in other["size"]]
                else:
                    half = [math.sqrt(sum(v * v for v in other["size"])) / 2] * 3
                p = other["position"]
                if abs(x-p[0]) < half[0]+radius and abs(y-p[1]) < half[1]+radius and p[2]+half[2] > top+.005 and p[2]-half[2] < top+1:
                    blocked = True
                    break
            if not blocked:
                candidate = (floor, x, y, top)
                break
        if candidate:
            break
    if not candidate:
        report["dropProbe"] = {"status": "skipped", "reason": "No exposed, unrotated floor found for a meaningful drop test."}
        report["status"] = "compiled"
        return report
    floor, x, y, top = candidate
    root = ET.Element("mujoco", {"model": "environment_collision_probe"})
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    world = _worldbody(scene, root)
    body = ET.SubElement(world, "body", {"name": "probe_body", "pos": _numbers([x, y, top+.5])})
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", {"name": "probe_geom", "type": "sphere", "size": str(radius), "mass": "0.1"})
    try:
        model = mujoco.MjModel.from_xml_string(_xml(root))
        data = mujoco.MjData(model)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        actual = float(data.qpos[2])
        expected = top + radius
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"env_{floor['id']}")
        probe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe_geom")
        contact = any({int(c.geom1), int(c.geom2)} == {floor_id, probe_id} for c in data.contact)
        passed = math.isfinite(actual) and abs(actual-expected) < .01 and contact and abs(float(data.qvel[2])) < .05
        report["dropProbe"] = {"status": "passed" if passed else "failed", "floorId": floor["id"], "startPosition": [x,y,top+.5], "expectedRestZ": expected, "actualRestZ": actual if math.isfinite(actual) else None, "floorContact": contact, "durationSeconds": 2, "radiusMeters": radius}
        report["status"] = "passed" if passed else "failed"
    except Exception as exc:
        report["dropProbe"] = {"status": "failed", "error": str(exc)}
        report["status"] = "failed"
    return report


def package_environment(manifest: Any, *, verify: bool = True) -> tuple[bytes, dict]:
    """Build an in-memory ZIP. Original scans are deliberately not redistributed."""
    files = export_files(manifest)
    report = validate_physics(manifest) if verify else {"status": "unverified", "schemaValid": True, "compile": "not_run", "dropProbe": {"status": "not_run"}}
    files["validation.json"] = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue(), report
