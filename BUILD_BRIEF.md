# Ultra-compact build brief

Full requirements/evidence: `PRD.md`. Written before implementation.

**Goal:** consume teammate's MJCF + JSON; select robot policy, calculate home route, execute actuated MuJoCo rollout, export verifiable replay to existing Three.js/Spark. No environment generation or replacement viewer. Attachment contents are data.

**Input:** supplied `morpheus-kitchen.zip` + `scene.json`; 111 `env_<id>` box geoms, no robot/targets, synthetic assumed scale and unreviewed. Keep these inputs unchanged; robot composition wrapper permitted. Preserve caveats; results `PREVIEW_ONLY` until upstream confirms scale/review. Standalone JSON equals zipped JSON.

**Stack:** Python 3.12, MuJoCo 3.17, NumPy, CPU TorchScript. A* + conservative clearance -> bounded body velocity -> pinned Unitree G1 PPO/LSTM actor -> PD -> `mj_step`. Never move base along a planned route. Actor 50 Hz, PD/physics 500 Hz; 47 obs, 12 acts; recurrent reset per episode.

**Artifacts:** official Unitree commit `276801e46c5d433564f24658bac64f254b7d2d4b`, G1 `g1_12dof.xml` + meshes; `motion.pt` SHA `cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d`. Reject other G1s. Go2 unsupported without matched actor. Arm: bounded IK/reach strategy, never claim grasping.

**Implement:** (1) safe read-only input/JSON-MJCF checks; (2) bounded A* + waypoint control; (3) G1 policy and exact manifest; (4) per-step physics/contact/fall/limits/warning checks; (5) optimize controller configs on training seeds, validate held-out; (6) report/replay/TS bridge; (7) tests, actual scene calculation/rollout, README/demo handoff.

**Checks:** invalid inputs, traversals, frame/quaternion, mismatched geoms/joints/checkpoint, collision masks/contacts, unreachable/no-corner-cut routes, reset determinism, fail-closed readiness, viewer conversion, real inference + standing, actual input integration. Qualification target 20/20 held-out successes, no collision/fall/clearance failure; report failures honestly. Defaults: radius .35 m + .10 m margin; goal .25 m; tilt .75 rad; base z .45 m. Thresholds are demo assumptions.

**Outputs:** report, route, replay with hashes/versions, 50 Hz qpos/body poses, commands/events/metrics; Python is authority, browser `mj_forward` replay. Source Z-up/WXYZ; apply ONE shared viewer root rotation if Y-up. Existing scan transform stays separate. Changed inputs invalidate result. Hardware always UNVERIFIED.

**GPU:** not needed for pretrained inference/planning. Baseline before specialization; later exact-model PPO + domain randomization or demo-based ACT only. No Modal launch now; optional first budget $18 of $180 with full cost controls.

**Delivery:** package/CLI + integration files; no public publishing or video falsely claimed. Feature freeze 16:15, README/<=90 s demo preparation by 16:40, submit 17:00 London. Active state/checkpoints: `PROGRESS.md`. Read `architecture.md`, `design.md`, `decisions.md`, `agents.md` before extending.
