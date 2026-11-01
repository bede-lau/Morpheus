# Visualization handoff contract

The upstream application owns the supplied MJCF/Scene JSON, Gaussian splat, Three.js + Spark renderer, robot meshes, and interface. This policy package exports calculation evidence only.

```
provided environment -> Python MuJoCo policy/evaluator -> replay bundle + report
existing Three.js/Spark viewer -----------------------> exact replay + evidence overlay
```

## Authoritative data

The replay schema is `1.0`, `authority: "python-mujoco"`, and includes source/MJCF/exported-MJZ/policy digests, evaluator version, robot identity and `nq`, source frame, route, timed full `qpos`, and optional body poses/events/metrics. The bridge rejects a different robot ID, `nq`, scene/MJCF/MJZ/policy digest, non-finite values, non-monotonic samples, or a frame other than metres/Z-up/WXYZ.

The report is also authoritative. Its `execution`, `readiness`, and `hardware_readiness: "UNVERIFIED"` values pass through unchanged. Browser preview cannot upgrade `PREVIEW_ONLY`, `BLOCKED`, or `UNSUPPORTED`.

## Application rules

At each sampled time, use exact state replay: set the full qpos, invoke MuJoCo `mj_forward`, then render. Do not call `mj_step` or infer a result from browser physics. Exported body poses are provenance/overlay data; a matched WASM model should calculate its render transforms from qpos.

Use source positions and orientations unchanged for robot nodes below one source-to-Y-up root rotation `(x,y,z) -> (x,z,-y)`. Keep the scan's pre-existing transform separate. Standalone overlay primitives need explicit basis conversion: vectors use the same mapping, and WXYZ quaternions use a basis conjugation then convert to Three.js XYZW ordering.

The evaluator samples G1 control state at 50 Hz. The viewer may interpolate purely for visual smoothness, but must not interpolate across a failure event or hide the first failing verified state. Suggested overlays: route/target, minimum clearance, contact object IDs, first failure, hashes, and the evaluator’s status.

## Verification

The bridge test covers exact qpos-before-forward ordering, model/hash identity mismatch, qpos length, finite/monotonic sample checks, no hardware-readiness escalation, and the source-frame conversion invariant. The existing viewer should add one integration fixture that compares root pose and named body transforms from replay against its matched WASM model at time zero.
