#!/usr/bin/env python3
"""Render an authoritative local MuJoCo replay for a demo; never re-simulate it.

Use `uv run --with pillow --with imageio --with imageio-ffmpeg python
scripts/render_replay.py outputs/g1-verified-preview` to create a still and a
short MP4. The supplied MJZ/replay/report are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np


class ReplayRenderError(ValueError):
    """The replay cannot be truthfully rendered with its supplied model."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayRenderError(f"Could not read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ReplayRenderError(f"{path.name} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_artifacts(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], mujoco.MjModel]:
    """Load the exported composed model and prove its result artifacts agree."""
    if not directory.is_dir():
        raise ReplayRenderError(f"Artifact directory does not exist: {directory}")
    replay = _read_json(directory / "replay.json")
    report = _read_json(directory / "report.json")
    manifest = _read_json(directory / "manifest.json")
    model_path = directory / "model.mjz"
    if replay.get("schema_version") != "1.0" or replay.get("authority") != "python-mujoco":
        raise ReplayRenderError("Replay is not an authoritative schema 1.0 Python MuJoCo artifact")
    if report.get("schema_version") != "1.0":
        raise ReplayRenderError("Report schema is not supported")
    if manifest.get("replay_sha256") != _sha256(directory / "replay.json"):
        raise ReplayRenderError("manifest replay SHA-256 does not match replay.json")
    if manifest.get("report_sha256") != _sha256(directory / "report.json"):
        raise ReplayRenderError("manifest report SHA-256 does not match report.json")
    if manifest.get("readiness") != report.get("readiness"):
        raise ReplayRenderError("manifest/report readiness mismatch")
    if replay.get("robot", {}).get("nq") is None or not isinstance(replay.get("samples"), list):
        raise ReplayRenderError("Replay is missing robot nq or samples")
    try:
        model = mujoco.MjSpec.from_zip(str(model_path)).compile()
    except (OSError, ValueError, mujoco.FatalError) as error:
        raise ReplayRenderError(f"Could not compile exported model.mjz: {error}") from error
    nq = replay["robot"]["nq"]
    if not isinstance(nq, int) or nq != model.nq:
        raise ReplayRenderError("Replay robot nq does not match exported model")
    if report.get("hashes", {}).get("policy_sha256") != replay.get("hashes", {}).get("policy_sha256"):
        raise ReplayRenderError("Report/replay policy provenance mismatch")
    return replay, report, manifest, model


def _sample(replay: dict[str, Any], index: int, model: mujoco.MjModel) -> dict[str, Any]:
    samples = replay["samples"]
    if not samples:
        raise ReplayRenderError("Replay has no samples")
    if index < 0 or index >= len(samples):
        raise ReplayRenderError(f"Sample index outside 0..{len(samples) - 1}")
    sample = samples[index]
    qpos = np.asarray(sample.get("qpos"), dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ReplayRenderError(f"Sample {index} qpos is not a finite vector of length {model.nq}")
    if not isinstance(sample.get("time_s"), (int, float)) or not math.isfinite(sample["time_s"]):
        raise ReplayRenderError(f"Sample {index} has invalid time_s")
    return sample


def _camera(model: mujoco.MjModel, data: mujoco.MjData, replay: dict[str, Any]) -> mujoco.MjvCamera:
    """Frame the actual robot plus its local collision-proxy neighbourhood."""
    points = np.asarray(replay["route"]["points_m"], dtype=float)
    route_center = np.mean(points, axis=0)
    center = np.array([route_center[0], route_center[1], max(0.35, float(data.qpos[2]) * 0.55)])
    # Preserve nearby supplied proxy context while preventing a large scanned
    # envelope from reducing the robot to a few pixels.
    route_span = float(np.max(np.ptp(points[:, :2], axis=0), initial=0.0))
    span = max(3.5, route_span + 2.5)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center
    camera.distance = span * 1.55
    camera.azimuth = 65
    camera.elevation = -52
    return camera


def _append_marker(scene: mujoco.MjvScene, position: np.ndarray, *, radius: float, color: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise ReplayRenderError("Renderer overlay geometry budget is exhausted")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius]),
        position,
        np.eye(3).reshape(9),
        np.array(color, dtype=np.float32),
    )
    geom.emission = 1.0
    geom.shininess = 0.0
    scene.ngeom += 1


def _append_overlays(renderer: mujoco.Renderer, replay: dict[str, Any]) -> None:
    route = replay.get("route", {}).get("points_m", [])
    for point in route:
        if isinstance(point, list) and len(point) == 3 and np.isfinite(np.asarray(point, dtype=float)).all():
            _append_marker(renderer.scene, np.asarray(point, dtype=float) + [0, 0, 0.18], radius=0.11, color=(0.1, 0.95, 0.55, 1.0))
    if route:
        _append_marker(renderer.scene, np.asarray(route[-1], dtype=float) + [0, 0, 0.28], radius=0.18, color=(0.1, 0.75, 1.0, 1.0))


def render_sample(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData, replay: dict[str, Any], index: int) -> np.ndarray:
    """Set an exported exact state and render it; this intentionally never calls mj_step."""
    sample = _sample(replay, index, model)
    data.qpos[:] = sample["qpos"]
    data.time = float(sample["time_s"])
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=_camera(model, data, replay))
    _append_overlays(renderer, replay)
    return renderer.render()


def _label(image: np.ndarray, report: dict[str, Any], sample: dict[str, Any]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise ReplayRenderError("Install Pillow: add --with pillow to the uv command") from error
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()
    metrics = sample.get("metrics", {})
    draw.rectangle((12, 12, 590, 92), fill=(8, 14, 24, 210))
    lines = [
        "UNITREE G1 · PPO/LSTM 12-DoF · actuated MuJoCo replay",
        f"{report.get('readiness', 'UNKNOWN')} · hardware {report.get('hardware_readiness', 'UNKNOWN')}",
        "Assumed-scale synthetic collision proxies · simulation preview only",
        f"t={sample['time_s']:.2f}s  clearance={metrics.get('min_clearance_m', float('nan')):.2f}m",
    ]
    for row, line in enumerate(lines):
        draw.text((20, 20 + row * 17), line, fill=(255, 214, 100, 255) if row == 1 else (244, 248, 255, 255), font=font)
    return np.asarray(canvas)


def save_png(path: Path, image: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as error:
        raise ReplayRenderError("Install Pillow: add --with pillow to the uv command") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_video(path: Path, *, renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData,
               replay: dict[str, Any], report: dict[str, Any], seconds: float, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise ReplayRenderError("Install video dependencies: add --with imageio --with imageio-ffmpeg") from error
    if not 0 < seconds <= 15:
        raise ReplayRenderError("--seconds must be greater than 0 and at most 15")
    if not 1 <= fps <= 30:
        raise ReplayRenderError("--fps must be in 1..30")
    frames = max(1, round(seconds * fps))
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in range(frames):
            replay_index = min(len(replay["samples"]) - 1, round(frame * (len(replay["samples"]) - 1) / max(1, frames - 1)))
            image = render_sample(renderer, model, data, replay, replay_index)
            writer.append_data(_label(image, report, _sample(replay, replay_index, model)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, help="Directory containing model.mjz, replay.json, report.json, manifest.json")
    parser.add_argument("--png", type=Path, help="Output native-rendered preview PNG")
    parser.add_argument("--mp4", type=Path, help="Output MP4 of no more than 15 seconds")
    parser.add_argument("--sample", type=int, default=-1, help="Replay sample for the still; -1 means final sample")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.png is None and args.mp4 is None:
        args.png = args.artifacts / "preview.png"
        args.mp4 = args.artifacts / "demo.mp4"
    if not 64 <= args.width <= 1920 or not 64 <= args.height <= 1080:
        raise ReplayRenderError("Render dimensions must be within 64..1920 by 64..1080")
    replay, report, _, model = load_artifacts(args.artifacts)
    data = mujoco.MjData(model)
    sample_index = args.sample if args.sample >= 0 else len(replay["samples"]) - 1
    # The exported visual config fixes an offscreen framebuffer. Render no larger
    # than it declares rather than mutating this immutable evidence model.
    width = min(args.width, int(model.vis.global_.offwidth))
    height = min(args.height, int(model.vis.global_.offheight))
    with mujoco.Renderer(model, height=height, width=width, max_geom=model.ngeom + len(replay.get("route", {}).get("points_m", [])) + 4) as renderer:
        if args.png is not None:
            image = _label(render_sample(renderer, model, data, replay, sample_index), report, _sample(replay, sample_index, model))
            save_png(args.png, image)
            print(json.dumps({"png": str(args.png), "sample": sample_index, "width": width, "height": height, "readiness": report["readiness"], "hardware_readiness": report["hardware_readiness"]}))
        if args.mp4 is not None:
            save_video(args.mp4, renderer=renderer, model=model, data=data, replay=replay, report=report, seconds=args.seconds, fps=args.fps)
            print(json.dumps({"mp4": str(args.mp4), "seconds": args.seconds, "fps": args.fps, "readiness": report["readiness"], "hardware_readiness": report["hardware_readiness"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayRenderError as error:
        print(json.dumps({"execution": "unavailable", "readiness": "BLOCKED", "hardware_readiness": "UNVERIFIED", "error": str(error)}), file=sys.stderr)
        raise SystemExit(2)
