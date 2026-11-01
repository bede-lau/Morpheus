#!/usr/bin/env python3
"""Fetch only the pinned G1 model files needed by the verified policy."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from hashlib import sha256
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as element_tree


REPOSITORY = "unitreerobotics/unitree_rl_gym"
COMMIT = "276801e46c5d433564f24658bac64f254b7d2d4b"
API_TREE_URL = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}"
ROBOT_SOURCE_ROOT = "resources/robots/g1_description"
MODEL_SOURCE = f"{ROBOT_SOURCE_ROOT}/g1_12dof.xml"
SCENE_SOURCE = f"{ROBOT_SOURCE_ROOT}/scene.xml"
CHECKPOINT_SOURCE = "deploy/pre_train/g1/motion.pt"
LICENSE_SOURCE = "LICENSE"
CHECKPOINT_SHA256 = "cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d"
CHECKPOINT_BYTES = 145_745

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_EXTRACTED_BYTES = 512 * 1024 * 1024
MAX_WORKERS = 4
CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(url: str):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "morpheus-policy-fetcher/1"}), timeout=30
    )


def fetch_json(url: str) -> dict:
    with _request(url) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_JSON_BYTES:
            raise ValueError("GitHub tree response exceeds the configured size bound")
        content = response.read(MAX_JSON_BYTES + 1)
    if len(content) > MAX_JSON_BYTES:
        raise ValueError("GitHub tree response exceeds the configured size bound")
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("GitHub tree response is not an object")
    return value


def pinned_tree() -> dict[str, str]:
    response = fetch_json(API_TREE_URL)
    if response.get("truncated"):
        raise ValueError("GitHub tree response was truncated")
    records = response.get("tree")
    if not isinstance(records, list) or len(records) > 10_000:
        raise ValueError("GitHub tree response has an invalid number of members")
    tree: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        path, kind, blob = record.get("path"), record.get("type"), record.get("sha")
        parsed = PurePosixPath(path) if isinstance(path, str) else None
        if parsed is not None and not parsed.is_absolute() and ".." not in parsed.parts and kind == "blob" and isinstance(blob, str):
            tree[path] = blob
    return tree


def download_file(source_path: str, destination: Path) -> tuple[str, int]:
    parsed = PurePosixPath(source_path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"unsafe source path: {source_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest, written = sha256(), 0
    with _request(f"{RAW_BASE_URL}/{source_path}") as response, destination.open("wb") as file:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_MEMBER_BYTES:
            raise ValueError(f"upstream asset exceeds per-file size bound: {source_path}")
        while chunk := response.read(CHUNK_BYTES):
            written += len(chunk)
            if written > MAX_MEMBER_BYTES:
                raise ValueError(f"upstream asset exceeds per-file size bound: {source_path}")
            digest.update(chunk)
            file.write(chunk)
    return digest.hexdigest(), written


def referenced_meshes(model_xml: Path) -> list[str]:
    root = element_tree.parse(model_xml).getroot()
    compiler = root.find("compiler")
    mesh_directory = compiler.get("meshdir", "meshes") if compiler is not None else "meshes"
    mesh_dir = PurePosixPath(mesh_directory)
    if mesh_dir.is_absolute() or ".." in mesh_dir.parts:
        raise ValueError("G1 MJCF has an unsafe meshdir")
    paths: set[str] = set()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("file")
        if filename is None:
            continue
        relative = PurePosixPath(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("G1 MJCF has an unsafe mesh filename")
        paths.add((PurePosixPath(ROBOT_SOURCE_ROOT) / mesh_dir / relative).as_posix())
    if not paths:
        raise ValueError("G1 MJCF does not reference mesh files")
    return sorted(paths)


def _assert_tree_members(tree: dict[str, str], paths: list[str]) -> None:
    missing = [path for path in paths if path not in tree]
    if missing:
        raise ValueError(f"pinned tree is missing required files: {missing}")


def asset_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "provenance.json"
    }


def install(output: Path) -> dict[str, object]:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = pinned_tree()
    roots = [MODEL_SOURCE, SCENE_SOURCE, CHECKPOINT_SOURCE, LICENSE_SOURCE]
    _assert_tree_members(tree, roots)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        destinations = {
            MODEL_SOURCE: staging / "robot/g1_12dof.xml",
            SCENE_SOURCE: staging / "robot/scene.xml",
            CHECKPOINT_SOURCE: staging / "motion.pt",
            LICENSE_SOURCE: staging / "LICENSE",
        }
        for source, destination in destinations.items():
            download_file(source, destination)
        mesh_sources = referenced_meshes(destinations[MODEL_SOURCE])
        _assert_tree_members(tree, mesh_sources)
        total_size = sum(path.stat().st_size for path in destinations.values())
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(download_file, source, staging / "robot/meshes" / PurePosixPath(source).name): source
                for source in mesh_sources
            }
            for future in as_completed(futures):
                _, size = future.result()
                total_size += size
                if total_size > MAX_EXTRACTED_BYTES:
                    raise ValueError("selected G1 assets exceed the extraction size bound")
        checkpoint = staging / "motion.pt"
        if checkpoint.stat().st_size != CHECKPOINT_BYTES or sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("downloaded checkpoint does not match the pinned artifact")
        provenance: dict[str, object] = {
            "schema_version": "1.0",
            "repository": f"https://github.com/{REPOSITORY}",
            "commit": COMMIT,
            "tree_url": API_TREE_URL,
            "raw_base_url": RAW_BASE_URL,
            "source_git_blobs": {source: tree[source] for source in sorted([*roots, *mesh_sources])},
            "checkpoint": {"path": "motion.pt", "source_path": CHECKPOINT_SOURCE, "sha256": CHECKPOINT_SHA256, "bytes": CHECKPOINT_BYTES},
            "robot": {"model": "robot/g1_12dof.xml", "scene": "robot/scene.xml", "mesh_directory": "robot/meshes", "mesh_count": len(mesh_sources)},
            "license": {"path": "LICENSE", "source_path": "LICENSE", "spdx": "BSD-3-Clause"},
            "asset_file_sha256": asset_file_hashes(staging),
            "retrieved_at_unix_s": time.time(),
        }
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, output)
        return provenance
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".morpheus/assets/g1"))
    arguments = parser.parse_args(argv)
    try:
        provenance = install(arguments.output)
    except (OSError, ValueError, urllib.error.URLError, element_tree.ParseError) as error:
        print(f"G1 asset fetch failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
