"""Read-only validation for upstream Scene JSON and MJCF bundles.

This module deliberately does not create collision geometry.  It verifies that
the upstream JSON description names and describes the geometry already present
in MJCF, preserving the original files for later composition and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Mapping
import xml.etree.ElementTree as ET
import zipfile


MAX_ARCHIVE_MEMBERS = 10_000
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_STRUCTURED_FILE_BYTES = 4 * 1024 * 1024
MAX_OBJECTS = 4_096
_FLOAT_TOLERANCE = 1e-6
_QUAT_TOLERANCE = 1e-5


class SceneValidationError(ValueError):
    """Raised when a supplied scene cannot be safely interpreted as v1 input."""


@dataclass(frozen=True)
class Box:
    id: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    role: str


@dataclass(frozen=True)
class Scene:
    name: str
    objects: tuple[Box, ...]
    source: Mapping[str, Any]
    root: Path
    mjcf_path: Path
    hashes: Mapping[str, str]
    caveats: tuple[str, ...]

    @property
    def floors(self) -> tuple[Box, ...]:
        return tuple(box for box in self.objects if box.role == "floor")

    @property
    def obstacles(self) -> tuple[Box, ...]:
        return tuple(box for box in self.objects if box.role != "floor")

    @property
    def bounds_xy(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds of the finite supplied floor support."""
        if len(self.floors) != 1:
            raise SceneValidationError("exactly one enabled floor is required")
        floor = self.floors[0]
        corners = _xy_corners(floor)
        return (
            min(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[0] for point in corners),
            max(point[1] for point in corners),
        )


def load_scene(path: str | Path) -> Scene:
    """Load a ZIP, scene directory, or scene.json without changing its contents."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise SceneValidationError(f"scene input does not exist: {source_path}")

    if source_path.is_file() and zipfile.is_zipfile(source_path):
        root, archive_hash = _extract_archive(source_path)
        hashes = {"bundle": archive_hash}
    elif source_path.is_dir():
        root = source_path
        hashes = {}
    elif source_path.is_file() and source_path.name == "scene.json":
        root = source_path.parent
        hashes = {}
    else:
        raise SceneValidationError("input must be a ZIP, a directory, or a file named scene.json")

    json_path = _single_scene_json(root)
    raw_json = _read_bounded(json_path)
    try:
        document = json.loads(raw_json)
    except json.JSONDecodeError as error:
        raise SceneValidationError(f"invalid scene.json: {error.msg}") from error
    if not isinstance(document, dict):
        raise SceneValidationError("scene.json root must be an object")

    objects, source = _parse_scene_json(document)
    mjcf_path, xml_paths, geoms = _load_mjcf(root)
    _validate_json_mjcf(objects, geoms)

    hashes.update(_source_hashes(root))
    caveats = _caveats(source)
    return Scene(
        name=_required_string(document, "name"),
        objects=tuple(objects),
        source=source,
        root=root,
        mjcf_path=mjcf_path,
        hashes=hashes,
        caveats=caveats,
    )


def verify_source_snapshot(scene: Scene) -> None:
    """Fail if a recorded input file changed after scene validation.

    This verifies the bounded source paths and hashes captured by ``load_scene``
    immediately before or after a consuming operation.  It narrows the mutable
    directory input window; a full immutable VFS snapshot is a separate
    deployment concern.
    """
    root = scene.root.resolve()
    for relative, expected_hash in scene.hashes.items():
        if relative == "bundle":
            continue
        _verify_snapshot_path(root, relative, expected_hash)


def _verify_snapshot_path(root: Path, relative: str, expected_hash: str) -> None:
    path_parts = PurePosixPath(relative)
    if path_parts.is_absolute() or not relative or any(part in {"", ".", ".."} for part in path_parts.parts):
        raise SceneValidationError(f"invalid recorded source path: {relative!r}")
    candidate = root
    for part in path_parts.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise SceneValidationError(f"source snapshot contains a symlink: {relative}")
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise SceneValidationError(f"recorded source path escapes scene root: {relative}") from error
    if not candidate.is_file():
        raise SceneValidationError(f"source snapshot file is missing: {relative}")
    if _sha256_file(candidate) != expected_hash:
        raise SceneValidationError(f"source snapshot changed: {relative}")


def _extract_archive(archive: Path) -> tuple[Path, str]:
    digest = _sha256_file(archive)
    cache_root = Path.cwd() / ".morpheus" / "inputs" / digest
    marker = cache_root / ".complete"
    if marker.exists():
        return cache_root, digest

    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise SceneValidationError("archive has too many members")
        expanded = 0
        for info in infos:
            _validate_zip_member(info)
            expanded += info.file_size
            if expanded > MAX_EXPANDED_BYTES:
                raise SceneValidationError("archive exceeds expanded-size limit")
            if info.filename.lower().endswith((".json", ".xml")) and info.file_size > MAX_STRUCTURED_FILE_BYTES:
                raise SceneValidationError(f"structured archive member is too large: {info.filename}")

        cache_root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f"{digest[:12]}-", dir=cache_root.parent))
        try:
            for info in infos:
                if info.is_dir():
                    continue
                relative = PurePosixPath(info.filename)
                target = temporary.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as input_file, target.open("wb") as output_file:
                    shutil.copyfileobj(input_file, output_file)
            (temporary / ".complete").touch()
            if cache_root.exists():
                shutil.rmtree(temporary)
            else:
                temporary.replace(cache_root)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    return cache_root, digest


def _validate_zip_member(info: zipfile.ZipInfo) -> None:
    member = PurePosixPath(info.filename)
    if member.is_absolute() or not info.filename or any(part in {"", ".", ".."} for part in member.parts):
        raise SceneValidationError(f"unsafe archive member path: {info.filename!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise SceneValidationError(f"archive symlink is not allowed: {info.filename}")


def _single_scene_json(root: Path) -> Path:
    matches = [path for path in root.rglob("scene.json") if path.is_file()]
    if len(matches) != 1:
        raise SceneValidationError("bundle must contain exactly one scene.json")
    return matches[0]


def _read_bounded(path: Path) -> bytes:
    if not path.is_file() or path.stat().st_size > MAX_STRUCTURED_FILE_BYTES:
        raise SceneValidationError(f"structured input is missing or too large: {path.name}")
    return path.read_bytes()


def _parse_scene_json(document: Mapping[str, Any]) -> tuple[list[Box], Mapping[str, Any]]:
    if document.get("version") != 1:
        raise SceneValidationError("only scene schema version 1 is supported")
    if document.get("units") != "m" or document.get("upAxis") != "Z":
        raise SceneValidationError("scene must declare metres and Z-up coordinates")
    source = document.get("source")
    if not isinstance(source, dict):
        raise SceneValidationError("scene source metadata must be an object")
    raw_objects = document.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise SceneValidationError("scene must contain objects")
    if len(raw_objects) > MAX_OBJECTS:
        raise SceneValidationError("scene has too many objects")

    seen: set[str] = set()
    objects: list[Box] = []
    for raw in raw_objects:
        if not isinstance(raw, dict):
            raise SceneValidationError("each scene object must be an object")
        if raw.get("enabled", True) is not True:
            continue
        identifier = _required_string(raw, "id")
        if identifier in seen:
            raise SceneValidationError(f"duplicate object id: {identifier}")
        seen.add(identifier)
        if raw.get("type") != "box":
            raise SceneValidationError(f"object {identifier} must be a box")
        role = _required_string(raw, "role")
        position = _finite_vector(raw.get("position"), 3, f"object {identifier} position")
        size = _finite_vector(raw.get("size"), 3, f"object {identifier} size")
        if any(value <= 0 for value in size):
            raise SceneValidationError(f"object {identifier} has a degenerate size")
        quaternion = _finite_vector(raw.get("quaternion"), 4, f"object {identifier} quaternion")
        if not math.isclose(math.sqrt(sum(value * value for value in quaternion)), 1.0, abs_tol=_QUAT_TOLERANCE):
            raise SceneValidationError(f"object {identifier} quaternion must be normalized WXYZ")
        objects.append(Box(identifier, position, size, quaternion, role))

    if not objects:
        raise SceneValidationError("scene has no enabled objects")
    if sum(box.role == "floor" for box in objects) != 1:
        raise SceneValidationError("scene must have exactly one enabled object with role 'floor'")
    return objects, source


def _load_mjcf(root: Path) -> tuple[Path, tuple[Path, ...], Mapping[str, Mapping[str, str]]]:
    candidates = [path for path in root.rglob("scene.xml") if path.is_file()]
    if len(candidates) != 1:
        raise SceneValidationError("bundle must contain exactly one scene.xml")
    scene_xml = candidates[0]
    xml_paths: list[Path] = []
    geoms: dict[str, Mapping[str, str]] = {}
    pending = [scene_xml]
    seen: set[Path] = set()
    while pending:
        xml_path = pending.pop()
        if xml_path in seen:
            continue
        seen.add(xml_path)
        raw = _read_bounded(xml_path)
        try:
            root_element = ET.fromstring(raw)
        except ET.ParseError as error:
            raise SceneValidationError(f"invalid XML in {xml_path.name}: {error}") from error
        if root_element.tag not in {"mujoco", "mujocoinclude"}:
            raise SceneValidationError(f"unexpected XML root in {xml_path.name}: {root_element.tag}")
        if any(element.tag == "plugin" for element in root_element.iter()):
            raise SceneValidationError("MJCF plugins are not accepted from supplied input")
        for element in root_element.iter():
            filename = element.attrib.get("file")
            if filename and element.tag != "include":
                _safe_asset_path(xml_path.parent, filename, root)
        xml_paths.append(xml_path)
        for geom in root_element.iter("geom"):
            name = geom.attrib.get("name")
            if name and name.startswith("env_"):
                if name in geoms:
                    raise SceneValidationError(f"duplicate environment geom: {name}")
                geoms[name] = dict(geom.attrib)
        for include in root_element.iter("include"):
            filename = include.attrib.get("file")
            if not filename:
                raise SceneValidationError("MJCF include is missing file")
            included = _safe_include(xml_path.parent, filename, root)
            pending.append(included)
    return scene_xml, tuple(xml_paths), geoms


def _safe_include(parent: Path, filename: str, bundle_root: Path) -> Path:
    if "://" in filename or filename.startswith(("/", "\\")):
        raise SceneValidationError(f"remote or absolute MJCF include is not allowed: {filename}")
    candidate = (parent / filename).resolve()
    try:
        candidate.relative_to(bundle_root.resolve())
    except ValueError as error:
        raise SceneValidationError(f"MJCF include escapes bundle: {filename}") from error
    if candidate.suffix.lower() != ".xml" or not candidate.is_file():
        raise SceneValidationError(f"MJCF include is missing or not XML: {filename}")
    return candidate


def _safe_asset_path(parent: Path, filename: str, bundle_root: Path) -> Path:
    if "://" in filename or filename.startswith(("/", "\\")):
        raise SceneValidationError(f"remote or absolute MJCF asset is not allowed: {filename}")
    candidate = (parent / filename).resolve()
    try:
        candidate.relative_to(bundle_root.resolve())
    except ValueError as error:
        raise SceneValidationError(f"MJCF asset escapes bundle: {filename}") from error
    if not candidate.is_file():
        raise SceneValidationError(f"MJCF asset is missing: {filename}")
    return candidate


def _validate_json_mjcf(objects: list[Box], geoms: Mapping[str, Mapping[str, str]]) -> None:
    expected = {f"env_{box.id}": box for box in objects}
    if set(geoms) != set(expected):
        missing = sorted(set(expected) - set(geoms))
        unexpected = sorted(set(geoms) - set(expected))
        details = []
        if missing:
            details.append(f"missing={','.join(missing[:3])}")
        if unexpected:
            details.append(f"unexpected={','.join(unexpected[:3])}")
        raise SceneValidationError(f"JSON/MJCF environment geom mismatch ({'; '.join(details)})")
    for name, box in expected.items():
        geom = geoms[name]
        if geom.get("type") != "box":
            raise SceneValidationError(f"{name} is not an MJCF box")
        _compare_vector(box.position, _finite_vector(_split_xml_vector(geom, "pos"), 3, f"{name} pos"), name, "position")
        half_size = _finite_vector(_split_xml_vector(geom, "size"), 3, f"{name} size")
        _compare_vector(tuple(value / 2 for value in box.size), half_size, name, "half-size")
        xml_quaternion = _finite_vector(_split_xml_vector(geom, "quat", default="1 0 0 0"), 4, f"{name} quat")
        _compare_quaternion(box.quaternion, xml_quaternion, name)
        for mask in ("contype", "conaffinity"):
            try:
                value = int(geom.get(mask, "1"))
            except ValueError as error:
                raise SceneValidationError(f"{name} has an invalid {mask}") from error
            if value < 0:
                raise SceneValidationError(f"{name} has a negative {mask}")


def _split_xml_vector(attributes: Mapping[str, str], key: str, default: str | None = None) -> list[float]:
    raw = attributes.get(key, default)
    if raw is None:
        raise SceneValidationError(f"MJCF geom is missing {key}")
    try:
        return [float(value) for value in raw.split()]
    except ValueError as error:
        raise SceneValidationError(f"invalid MJCF {key}") from error


def _compare_vector(expected: tuple[float, ...], actual: tuple[float, ...], name: str, label: str) -> None:
    if any(not math.isclose(left, right, abs_tol=_FLOAT_TOLERANCE) for left, right in zip(expected, actual)):
        raise SceneValidationError(f"{name} {label} does not agree with scene.json")


def _compare_quaternion(expected: tuple[float, ...], actual: tuple[float, ...], name: str) -> None:
    same = all(math.isclose(left, right, abs_tol=_QUAT_TOLERANCE) for left, right in zip(expected, actual))
    negated = all(math.isclose(left, -right, abs_tol=_QUAT_TOLERANCE) for left, right in zip(expected, actual))
    if not (same or negated):
        raise SceneValidationError(f"{name} quaternion does not agree with scene.json")


def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise SceneValidationError(f"{label} must contain {length} values")
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise SceneValidationError(f"{label} must be numeric") from error
    if not all(math.isfinite(item) for item in vector):
        raise SceneValidationError(f"{label} must be finite")
    return vector


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise SceneValidationError(f"{key} must be a non-empty string")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_name(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _source_hashes(root: Path) -> dict[str, str]:
    """Hash every source file while excluding the cache's lifecycle marker."""
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != ".complete":
            result[_relative_name(root, path)] = _sha256_file(path)
    return result


def _caveats(source: Mapping[str, Any]) -> tuple[str, ...]:
    caveats: list[str] = []
    if source.get("synthetic") is True:
        caveats.append("source geometry is synthetic")
    if source.get("scaleConfirmed") is not True:
        caveats.append("metric scale is not confirmed")
    if source.get("reviewed") is not True:
        caveats.append("source geometry is not reviewed")
    approximation = source.get("approximation")
    if isinstance(approximation, str) and approximation:
        caveats.append(f"geometry approximation: {approximation}")
    return tuple(caveats)


def _xy_corners(box: Box) -> tuple[tuple[float, float], ...]:
    yaw = _yaw(box.quaternion)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    half_x, half_y = box.size[0] / 2, box.size[1] / 2
    return tuple(
        (box.position[0] + cosine * x - sine * y, box.position[1] + sine * x + cosine * y)
        for x, y in ((-half_x, -half_y), (-half_x, half_y), (half_x, -half_y), (half_x, half_y))
    )


def _yaw(quaternion: tuple[float, float, float, float]) -> float:
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
