# Implementation contracts

## Data and interfaces

Package: `morpheus_policy`. Core modules owned separately; coordinate through these contracts.

- `scene.py`: `load_scene(path) -> Scene`. Accept ZIP, directory with scene.json, or JSON with sibling scene.xml/environment.xml. Read provided v1 fields: version/name/units/upAxis/source/objects. Each enabled object: id/type=box/position/size (FULL dimensions)/quaternion (WXYZ)/role. Preserve metadata and original bytes. Geom names `env_<id>`. Define dataclasses `Box(id, position, size, quaternion, role)`, `Scene(name, objects, source, root, mjcf_path, hashes, caveats)`. Scene has `bounds_xy` derived from floor; `obstacles` and `floors` accessors. No silent fixes or scale confirmation.
- `planning.py`: `plan_route(scene, start_xy, goal_xy, radius_m=.35, margin_m=.10, resolution_m=.10, body_height_m=1.4) -> list[list[float]]`; `clearance(scene, xy, radius_m=.35, body_height_m=1.4) -> float`. Reject obstructed endpoints and unreachable/corner-cut routes, enforce bounds and bounded cells. Rotated boxes projected conservatively; object z affects configurable body-height intersection. No generated environment.
- `policies/g1.py`: constants/manifest; `G1Policy(checkpoint_path)`, `validate(model) -> dict`, `reset(model, data)`, `step(model, data, command) -> ndarray`. Called EVERY physics step; retain actor targets between 50 Hz actor calls, calculate PD each 500 Hz step; command `[vx_body, vy_body, yaw_rate]`. Exact model/joint/order/timestep/action checks. Torch loaded lazily, CPU inference only, checkpoint hash required by default. No data.qpos writes outside explicit initial-state reset in runner.
- `navigation.py`: route/state -> finite bounded body velocity; expose configurable speed, yaw gain, lookahead, acceleration limit. Closed finite goal/stop transitions; no robot-name switches.
- `evaluation.py`: static supplied environment + injected robot XML/posture/support-body names, no robot-type branch in the loop. Recheck source hashes before/after native attachment; asset integrity checks surround composition. Initialize one episode explicitly, run policy + `mj_step`, monitor every step, sample 50 Hz replay. A failed episode stops promptly. Contact classification permits only registered support contacts; unrecognized contacts fail. Record applied joint forces as distinct from raw ctrl. Keep last finite sample on divergence. Single-run readiness is preview; qualification validates input gates AND consistent clean physics/task coverage.
- `registry.py`: explicit IDs/capabilities. G1 adapter available after fetch; Go2 unsupported until matched controller; arm reach adapter reports its actual capabilities. No dynamic imports from user paths.
- `optimization.py`: finite candidates evaluated on train seeds; select with declared success-first score; evaluate chosen config and baseline on held-out seeds. No reuse of held-out outcomes to silently select parameters. Output baseline/candidate metrics and unchanged chosen policy hash. This is navigation-controller optimization, not a trained actor.

## Report and replay

Report schema version `1.0`; input hashes, policy/checkpoint/model provenance, engine/control versions, task, thresholds, `execution`, `readiness`, `hardware_readiness: UNVERIFIED`, caveats, metrics, failures, coverage. Preview never hides missing semantics, unconfirmed scale, or missing physics trials.

Replay schema version `1.0`; `authority: python-mujoco`; hashes including scene JSON/canonical XML/model MJZ/policy, robot id/joint/body names, source frame `{units:m, up_axis:Z, quaternion_order:wxyz}`, route points, samples `{time_s,qpos,qvel,ctrl,command,body_poses,metrics}`, timestamped events. Body pose tuple `[x,y,z,w,qx,qy,qz]` is in canonical world frame. Metrics include minimum envelope clearance, goal error, tilt, base height, forbidden contacts, joint/actuator limits and warnings. Body poses optional to reduce payload. Native export roundtrip checks names, dynamics, and geometry; canonical XML can round geometry by <=5um checked presentation tolerance, recorded in report. Evaluated source geometry is unchanged. No browser physics equivalence is claimed.

`web/policy-bridge.ts` validates manifest/frame/model dimensions and exposes exact-state application (`qpos` then `mj_forward`) and evidence overlays. Render interpolation is visual only. For source-authored robot nodes under a shared rotated root, keep source poses unchanged and rotate the root once; do not also conjugate each quaternion. A separately basis-converted representation needs consistent vector/quaternion basis conversion. Test quarter-turn and root transforms.

## Limits and lifecycle

Proposed bounds: archive 10000 members/512 MiB expanded, 4 MiB per XML/JSON, 4096 objects, 1e6 grid cells, 120 simulated seconds per episode, at most 100 qualification episodes, bounded replay sampling. Reject absolute/traversing archive paths, symlinks, remote includes/plugins, missing dependencies, degenerate/non-finite shapes/quaternions, duplicate IDs and model mismatch. ZIP extraction may copy source bytes to an ignored input directory; it must not execute attached text. Preserve source input files.

Result invalidation: any scene/model/policy/config change requires a new report. No server auth or network mutation in v1. Downloads use fixed upstream commit URLs, atomic writes and verified checkpoint digest; preserve license/attribution. No hardware command transport.

## Required verification

Unit/contract tests cover C1–C8 (architecture matrix), including an alternate injected test policy through normal evaluation wiring to demonstrate the policy seam. Test fixtures establish contracts, not supported real robots. Run actual pinned G1 inference and a supplied-scene integration; record observed outcomes in PROGRESS/README. Audit source for base teleporting and input writes. Track unresolved qualification/physical checks explicitly.
