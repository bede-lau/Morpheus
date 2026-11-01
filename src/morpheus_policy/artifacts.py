"""Versioned evidence artifacts, written locally and never used as hardware commands."""

import hashlib
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifacts(output, report, replay, *, spec=None, model=None, third_party_license_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model_digest = None
    if spec is not None:
        if model is None:
            raise ValueError("A model is required for portable model export")
        model_path = output / "model.mjz"
        spec.encode(str(model_path), model=model)
        if third_party_license_path is not None:
            license_bytes = Path(third_party_license_path).read_bytes()
            (output / "THIRD_PARTY_LICENSE.txt").write_bytes(license_bytes)
            with zipfile.ZipFile(model_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("THIRD_PARTY_LICENSE.txt", license_bytes)
        restored = mujoco.MjSpec.from_zip(str(model_path)).compile()
        for field in ("nq", "nv", "nu", "ngeom", "nbody"):
            if getattr(model, field) != getattr(restored, field):
                raise ValueError(f"Exported model roundtrip changed {field}")
        for field in ("body_mass", "body_inertia", "jnt_range", "jnt_actfrcrange", "geom_size", "geom_pos", "geom_quat", "geom_friction", "dof_damping", "dof_armature", "qpos0", "actuator_gainprm", "actuator_biasprm"):
            # MuJoCo's canonical XML writer rounds geometric decimals. Preserve a
            # declared 5-micrometre presentation tolerance; physics used source values.
            tolerance = 5e-6 if field in ("geom_size", "geom_pos") else 1e-10
            if not np.allclose(getattr(model, field), getattr(restored, field), rtol=1e-9, atol=tolerance):
                raise ValueError(f"Exported model roundtrip changed {field}")
        if bytes(model.names) != bytes(restored.names):
            raise ValueError("Exported model roundtrip changed name mappings")
        model_digest = sha256_file(model_path)
        with zipfile.ZipFile(model_path) as archive:
            xml_names = [name for name in archive.namelist() if name.endswith(".xml")]
            if len(xml_names) != 1:
                raise ValueError("Portable model must contain one canonical XML")
            xml_digest = hashlib.sha256(archive.read(xml_names[0])).hexdigest()
        for artifact in (report, replay):
            artifact["hashes"].update(model_mjz_sha256=model_digest, mjcf_sha256=xml_digest)
        report["portable_model_roundtrip"] = {
            "checked": True, "geometry_tolerance_m": 5e-6,
            "max_geometry_difference_m": max(float(np.max(np.abs(model.geom_size - restored.geom_size))),
                                              float(np.max(np.abs(model.geom_pos - restored.geom_pos)))),
            "note": "Canonical XML rounds geometry; recorded physics used the unchanged source model. Viewer is replay-only.",
        }
    for name, value in (("report.json", report), ("replay.json", replay)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0",
        "report_sha256": sha256_file(output / "report.json"),
        "replay_sha256": sha256_file(output / "replay.json"),
        "model_mjz_sha256": model_digest,
        "readiness": report["readiness"], "hardware_readiness": "UNVERIFIED",
    }, indent=2) + "\n")
    return output
