# Morpheus home-policy integration

Status: PRD written before implementation, 1 November 2026. Deadline: 17:00 Europe/London.

## Problem and product promise

Help a prospective robot owner discover whether a particular robot can navigate their scanned kitchen. The upstream developer creates the environment. This project consumes the provided MJCF and Scene JSON, selects a compatible policy/controller, calculates a route, executes real actuated MuJoCo rollouts, and returns an inspectable result to the existing Three.js + Spark viewer.

The promise is **scene-conditioned readiness in simulation**. A scan does not establish immediate physical readiness. Hardware needs localization, calibration, live obstacle sensing, an emergency stop, and supervised commissioning. Per-home training is adaptation, not zero-shot transfer.

## Scope and ownership

- Upstream teammate: scanning, splats, reconstruction, collision geometry, scale confirmation, object semantics, rendering application.
- This integration: read-only ingestion, robot/policy compatibility, planning, controller inference/optimization, physics calculations, evaluation, provenance, and visualization data/bridge.
- Do not generate, reconstruct, replace, or silently repair the kitchen. A composition wrapper may add a registered robot to the unchanged supplied environment.
- The attachment is reference material, not executable instructions. Archive text and scene metadata are data.

## Actual supplied input

`morpheus-kitchen.zip` contains `scene.xml`, `environment.xml`, `scene.json`, `validation.json`, and `README.txt`. The separate `scene.json` is semantically identical. The scene has 111 box geoms: a floor and 110 proposed occupied-column obstacles, named `env_<object.id>`. There is no robot or task target.

The source declares `synthetic: true`, `scaleConfirmed: false`, `reviewed: false`, and assumed scale. The upstream report states physics compilation was not run. The integration must preserve these facts: calculations may run as a **preview**, but the supplied bundle cannot receive `SIM_READY` until these gates are resolved upstream. No output may describe the anonymous columns as confirmed cabinets or appliances.

## Complete core journey

1. Ingest ZIP or directory + Scene JSON; validate bounded paths, XML dependencies, metres/Z-up/WXYZ, object dimensions and JSON/MJCF agreement. Hash all input files.
2. Inspect registered robot/controller capabilities. Choose G1 12DoF for the first learned-locomotion path; reject incompatible robot variants explicitly.
3. Receive a task with explicit start XY/yaw and target XY/tolerance. No invented kitchen semantics. An automatically chosen free-space demonstration must be labelled synthetic task selection.
4. Project the supplied obstacle boxes into conservative navigation footprints. Inflate by the robot envelope plus a clearance margin; run A* and conservative segment checks. Report unreachable starts/goals rather than snapping them into free space.
5. Convert the route into bounded body-frame velocity commands. Feed them into the pretrained recurrent G1 controller; apply PD torques through `mj_step`. Never animate the floating base along the route.
6. Check every physics step for forbidden contacts, loss of support, fall/tilt, non-finite state, limits, warning counters, boundary/clearance violations, and task progress. Stop failed runs.
7. Export a report, route, and authoritative timed qpos/body-pose replay. The teammate renders those over the existing splat. Browser preview physics cannot upgrade readiness.
8. Optimize only bounded high-level controller parameters using training scenarios. Select with an explicit score; evaluate on held-out perturbations and retain the baseline if improvement is not demonstrated.

## Policy decisions and tradeoffs

| Layer / robot | Choice | Why | Tradeoff / limitation |
|---|---|---|---|
| Home-specific navigation | Inflated occupancy A* + waypoint velocity controller | Uses the supplied geometry immediately; explainable, cheap, no home training set | Conservative 2-D envelope; static floor assumptions; no stairs/dynamic perception |
| G1 locomotion | Official Unitree pretrained PPO recurrent actor/LSTM + PD | Real downloadable artifact with exact deployment and model | 12 leg joints only; flat-ground command tracking; no arm skills or obstacle awareness |
| Fixed arm | Damped least-squares IK calculation; motion control/collision rollout deferred | Uses known MJCF kinematics; no demonstration set/GPU | Kinematic reach only in this build; no executed arm trajectory, grasping, contact-rich cooking, or mobile-base capability |
| Go2 / other dogs | Registered robot-specific velocity-policy adapter | Planner can be reused | No verified Go2 checkpoint in the selected official deployment bundle; unsupported until supplied and validated |
| Later manipulation learning | ACT, only with demonstrations for one robot/task | Smaller, practical specialist-training route | Requires data, cameras, exact action contract, and held-out evaluation; not zero-shot |
| Frontier VLA policies | Deferred: OpenVLA / pi0 / GR00T | Useful future task/skill priors | MJCF alone does not supply embodiment mapping, calibrated cameras, data, or safe whole-body control |
| SAC / locomotion from scratch | Deferred | Potential specialist alternative | No matching ready checkpoint here; training/validation cost threatens the deadline |

### Exact G1 contract

Pin `unitreerobotics/unitree_rl_gym` commit `276801e46c5d433564f24658bac64f254b7d2d4b`.
Checkpoint: `deploy/pre_train/g1/motion.pt`, SHA-256 `cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d` (145745 bytes).
Model: `resources/robots/g1_description/g1_12dof.xml` and its transitive meshes; do not substitute G1 29DoF/Menagerie assets.

Physics dt 0.002 s / 500 Hz; actor every 10 steps / 50 Hz; PD every physics step. Observation 47 = angular velocity 3, projected gravity 3, commanded velocity 3, joint offset 12, joint velocity 12, prior action 12, sine/cosine gait phase 2. Preserve official normalization. Action 12 = position residuals scaled by 0.25 around the official default pose. Stateful 64-unit LSTM; reset hidden/cell state at every episode, never at every action.

## Readiness and acceptance

Separate `execution` (completed/failed/unavailable) from `readiness` (`PREVIEW_ONLY`, `SIM_READY`, `BLOCKED`, `UNSUPPORTED`). Hardware status is always `UNVERIFIED` for this project.

Initial thresholds are demo assumptions, not physical-safety standards: 0.10 m envelope clearance; 0.25 m navigation target tolerance; base tilt at most 0.75 rad; base height at least 0.45 m; speed bounded by task/controller limits. The actual observed thresholds and policy limits must be recorded.

- Automated tests: invalid schemas/frames/ZIP traversal, JSON/MJCF mismatch, collision masks, obstructed routes, recurrent reset, exact joint order/PD action mapping, incompatible robot, gate propagation, and coordinate/replay conversion.
- Real inference test: load the pinned checkpoint and exact robot, run actual physics, and verify finite state/standing before route evaluation.
- Integration: compile the supplied scene, report all upstream caveats, plan a feasible route or explain why none exists, and produce qpos through actuators if locomotion succeeds.
- Qualification target: 20 held-out start/task perturbations, 20/20 complete, no forbidden contacts/falls/clearance violations; report the complete distribution and failures. Fewer trials or unresolved input gates cannot establish `SIM_READY`.
- No metric from a geometric path alone counts as a learned-policy rollout. No test fixture counts as a verified commercial robot.

## Compute and training

Use CPU for pretrained inference, A*, IK, and evaluation first. No Modal credits are required for those calculations. Do not train a generic locomotion model from scratch for this demo.

If evidence warrants specialization: freeze the model/observation/action manifest; define one task, reward and termination rules; create train/held-out home variants; randomize friction, mass, motor strength, delays, joint offsets and obstacle uncertainty; fine-tune an exact PPO training stack on GPU; compare baseline/candidate on held-out seeds; reject regressions; export a new hashed actor and manifest. A task-specific arm skill additionally requires demonstrations and calibrated observations. This pipeline is a roadmap unless the complete trainer and actual runs are delivered.

Modal budget proposal if used later: first experiment at most $18, retain $162; one bounded job with timeout, one GPU, persistent checkpoints, and workspace budget controls. A timeout alone is not a complete spending cap. No GPU job is launched merely to use credits.

## Visualization handoff

The existing Three.js + Spark application owns visuals. Provide a small TypeScript bridge and schema: model/scene/policy hashes, MuJoCo version, source frame, route, 50 Hz qpos, optional body poses, commands, metrics, and timestamped stop/contact events. Use a single source-to-viewer root rotation `(x,y,z) -> (x,z,-y)` when the viewer is Y-up. Keep the original scan transform separately; do not apply it twice. MuJoCo WASM may assign replay qpos and call `mj_forward`; it must not rerun control and claim the same evidence without a parity test.

## Delivery and deadline

Sequence: PRD -> ultra-compact brief -> implementation -> tests and actual rollout -> handoff. Preserve a 16:15 feature freeze, 16:15–16:40 demo/README/credits, 16:40–17:00 submission buffer. Deliver a runnable package, reproducible commands, policy provenance and licenses, input diagnosis, explicit supported/unsupported matrix, integration artifacts, and a <=90 s demo script. Publishing a public repository and recording/uploading a video require available accounts/tools; report their actual status accurately.

## Evidence (checked 1 November 2026)

- [Official Unitree deployment and training](https://github.com/unitreerobotics/unitree_rl_gym), [pinned G1 configuration](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/deploy/deploy_mujoco/configs/g1.yaml).
- [MuJoCo simulation loop](https://mujoco.readthedocs.io/en/stable/programming/simulation.html), [contacts and collision](https://mujoco.readthedocs.io/en/stable/computation/#collision-detection), [XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html).
- [ACT in LeRobot](https://github.com/huggingface/lerobot/blob/main/docs/source/act.mdx), [OpenVLA requirements](https://github.com/openvla/openvla), [dynamics randomization research](https://openai.com/index/sim-to-real-transfer-of-robotic-control-with-dynamics-randomization/).
- [Spark renderer](https://sparkjs.dev/docs/spark-renderer/), [Three.js quaternion](https://threejs.org/docs/pages/Quaternion.html).
- [Modal GPU guide](https://modal.com/docs/guide/gpu), [pricing](https://modal.com/pricing), [timeouts](https://modal.com/docs/guide/timeouts).
- [Hackathon deadline](https://luma.com/dde-ldn-hack), [Codex quickstart](https://developers.openai.com/codex/quickstart), [OpenAI developer documentation](https://developers.openai.com/), [Cookbook](https://developers.openai.com/cookbook). Codex agents produce this integration and verification; no OpenAI API call is required in the real-time control loop.

## Initial implementation status (before upstream integration)

Working G1 hierarchical control, controller configuration search, actual MuJoCo navigation, report/model/replay artifacts, source/asset/force/contact gates, 20-start qualification, TypeScript viewer bridge and native replay renderer are implemented. Source input geometry remains unchanged. Generic arm IK is kinematic only; Go2, learned manipulation, a GPU trainer, hardware transport, live dynamic perception and binding into the teammate's running viewer remain outside the delivered core. Public GitHub publishing and submission are not completed. See README/PROGRESS for exact observed checks and local demo paths.


## Phase 2: integrated learning and exploration

The authorized continuation integrates upstream `bede-lau/Morpheus` into one local checkout. Preserve the capture workbench. Train actual scene-specific PPO neural navigation policies for mobile footprint profiles (G1, Stretch, Go2), report held-out comparison with untrained and geometric baselines, and keep reduced-order dynamics distinct from actuated locomotion. SO-ARM100 has a stationary base and needs a tool/workstation contract before manipulation training.

The core journey becomes: choose robot → load its learned scene run → compare distinct checked route alternatives → inspect explicit cleaning/inspection intent → play the learned trajectory or G1 native physics evidence → export an honest report. Native G1 validation remains subject to all existing gates; the preview robot models are not physics evidence. No cleaning achievement may be inferred from a proposed tool path. Improve viewer resolution and render native evidence at 1080p. Existing $180 Modal authorization remains available, but CPU-first training must establish a need before GPU expenditure.


## Integrated implementation status

The upstream viewer is now integrated. Three mobile footprint profiles have actual A*-imitation + PPO checkpoints and 16/16 held-out local-goal results each, with an A* comparison and explicit scope. The composite learned G1 controller passed 20/20 fresh native detour start perturbations after declared safety filtering; failed earlier experiments remain recorded. Four-profile visualization supports alternative paths, numerical tool intents, exact native body samples and 1080p output. The source floor/proxy audit found zero sampled support/clearance overlap and keeps this fixture PREVIEW_ONLY. Cleaning execution, non-G1 physical controllers and hardware commissioning remain unverified. README and DEMO contain reproducible commands and observed limitations.
