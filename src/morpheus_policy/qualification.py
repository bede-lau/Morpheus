"""Fail-closed qualification over held-out physics reports.

Qualification is a report validator, not a policy trainer or a source-confidence
repair mechanism. It preserves every submitted episode and its reasons.
"""

from collections.abc import Mapping
from dataclasses import replace
import math

import numpy as np


_REQUIRED_METRICS = ("duration_s", "goal_error_m", "forbidden_contacts", "max_tilt_rad")
_PROVENANCE_KEYS = ("scene_json_sha256", "mjcf_sha256", "policy_sha256")


def qualification_tasks(task, *, seed=20261101, episodes=20):
    if not 20 <= episodes <= 100:
        raise ValueError("Qualification requires 20 to 100 held-out episodes")
    rng = np.random.default_rng(seed)
    for index in range(episodes):
        offset = rng.uniform(-0.02, 0.02, 2)
        yield replace(task, start_xy=tuple(np.asarray(task.start_xy) + offset),
                      start_yaw=float(task.start_yaw + rng.uniform(-0.03, 0.03)),
                      id=f"{task.id}:holdout:{seed}:{index}")


def _episode_reasons(report, index, expected_scene_hash):
    prefix = f"episode[{index}]"
    reasons = []
    if not isinstance(report, Mapping):
        return [f"{prefix}: report is not an object"], None, None
    if report.get("execution") != "completed":
        reasons.append(f"{prefix}: execution is not completed")
    if report.get("readiness") not in {"PREVIEW_ONLY", "SIM_READY"}:
        reasons.append(f"{prefix}: readiness must be PREVIEW_ONLY or SIM_READY")

    failures = report.get("failures")
    if not isinstance(failures, list):
        reasons.append(f"{prefix}: failures must be a list")
    elif failures:
        reasons.extend(f"{prefix}: reported failure: {failure}" for failure in failures)

    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        reasons.append(f"{prefix}: metrics must be an object")
    else:
        for name in _REQUIRED_METRICS:
            value = metrics.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                reasons.append(f"{prefix}: metric {name} must be finite")
        if metrics.get("forbidden_contacts") != 0:
            reasons.append(f"{prefix}: forbidden_contacts must be zero")

    hashes = report.get("hashes")
    provenance = None
    if not isinstance(hashes, Mapping):
        reasons.append(f"{prefix}: hashes must be an object")
    else:
        provenance = {key: hashes.get(key) for key in _PROVENANCE_KEYS}
        for key, value in provenance.items():
            if not isinstance(value, str) or not value:
                reasons.append(f"{prefix}: missing {key}")
        if expected_scene_hash is not None and provenance["scene_json_sha256"] != expected_scene_hash:
            reasons.append(f"{prefix}: scene_json_sha256 does not match supplied scene")

    config = report.get("controller_config")
    if not isinstance(config, Mapping):
        reasons.append(f"{prefix}: controller_config must be an object")
        config = None
    else:
        config = dict(config)
    return reasons, provenance, config


def _source_disqualifiers(scene, task):
    source = scene.source
    reasons = []
    if source.get("scaleConfirmed") is not True:
        reasons.append("Source scaleConfirmed is not true")
    if source.get("reviewed") is not True:
        reasons.append("Source reviewed is not true")
    if source.get("synthetic") is not False:
        reasons.append("Source synthetic is not false")
    if task.synthetic:
        reasons.append("Task is synthetic")
    return reasons


def summarize_qualification(scene, task, reports, *, seed):
    """Return an auditable result; invalid/mismatched input remains blocked."""
    reports = list(reports)
    expected_scene_hash = scene.hashes.get("scene.json")
    episode_reasons, reference_provenance, reference_config = [], None, None
    valid_episodes = 0
    for index, report in enumerate(reports):
        reasons, provenance, config = _episode_reasons(report, index, expected_scene_hash)
        if provenance is not None and reference_provenance is not None and provenance != reference_provenance:
            reasons.append(f"episode[{index}]: scene/model/policy hashes differ from qualification run")
        elif provenance is not None:
            reference_provenance = provenance
        if config is not None and reference_config is not None and config != reference_config:
            reasons.append(f"episode[{index}]: controller_config differs from qualification run")
        elif config is not None:
            reference_config = config
        if reasons:
            episode_reasons.extend(reasons)
        else:
            valid_episodes += 1

    if len(reports) < 20:
        episode_reasons.append(f"Qualification needs at least 20 reports; received {len(reports)}")
    physics_complete = not episode_reasons
    source_disqualifiers = _source_disqualifiers(scene, task)
    caveats = list(scene.caveats)
    caveats.append("No friction/mass/perception/dynamic-obstacle robustness or physical trials established.")
    readiness = "BLOCKED" if not physics_complete else ("SIM_READY" if not source_disqualifiers else "PREVIEW_ONLY")
    return {
        "schema_version": "1.0", "kind": "held_out_start_pose_qualification",
        "execution": "completed" if physics_complete else "failed",
        "readiness": readiness, "hardware_readiness": "UNVERIFIED", "seed": seed,
        "coverage": {"episodes": len(reports), "successes": valid_episodes, "required_success_rate": 1.0,
                     "qualification_complete": physics_complete,
                     "tested_variation": "start XY +/-0.02 m, yaw +/-0.03 rad; same static geometry/task"},
        "provenance": {"hashes": reference_provenance, "controller_config": reference_config},
        "source_disqualifiers": source_disqualifiers,
        "disqualifiers": episode_reasons,
        "caveats": caveats,
        "episodes": reports,
    }
