# Architecture

Implemented as one local Python package plus a dependency-free TypeScript handoff. Evidence below covers the delivered simulation core; it does not establish physical or production readiness.

```
provided ZIP/MJCF + JSON -> scene adapter -> occupancy/planner/task plan -> velocity commands
                                        |                         -> frozen gait Policy strategy -> MuJoCo evaluator
                                        |                         -> local PPO strategy -> learned artifact/export adapter
existing Three.js/Spark/MuJoCo WASM <-------------------------- TypeScript bridge + native-hash gates
```

All house geometry is upstream-owned. Input adapters may unpack bounded archives and compose the registered robot with unchanged environment XML. They never reconstruct a home. Stable orchestration depends on contracts, not robot-specific switches. Registration is centralized. External Torch/MuJoCo/download/file APIs stay behind adapters.

| ID | Responsibility / contract | Expected variation / extension | Stable paths | Evidence / verdict |
|---|---|---|---|---|
| C1 scene | Ingest Scene JSON + bounded bundle, preserve caveats and hash provenance | Separate source adapter; current contract deliberately restricted to supplied v1 static boxes | `scene.py` | ZIP/schema/XML agreement, mutation/symlink tests pass; actual111-geoms compile; OCP reviewed for v1 contract |
| C2 planning | Conservative static footprint route and clearance | Envelope/height/grid config; injected planner would require a deliberate contract change | `planning.py` | Blocked/corridor/rotated/overhead tests; actual 17-point detour executed in MuJoCo, 20/20 perturbed starts passed; no robot-name branching; OCP reviewed |
| C3 control | Convert route/state to bounded velocity command | NavigationConfig varies speed/yaw/lookahead/limits without loop edits | `navigation.py` | Settling/bounds/stop tests; .2/.3/.4m/s configs evaluated through normal path; OCP verified for configuration variation |
| C4 policy | `validate/reset/step`; output controls for actual model | Robot-specific injected Policy protocol implementations | `policies/` | Real G1 and alternate fixture through evaluator; recurrent reset, joint/force/prefix/integrity tests; OCP verified for velocity-policy seam |
| C5 evaluation | Physics loop, mandatory gates and metrics | Inject policy/posture/support names/thresholds; mandatory failure semantics fixed | `evaluation.py` | Actual G1 and alternate test policy, contact/NaN/timeout/mask tests; no robot-name selection in loop; OCP reviewed |
| C6 optimization | Bounded config candidates, separate train/held-out evaluation | Inject evaluator callback; finite configuration candidates | `optimization.py` | Success/regression/split/budget/failure tests; actual8-run audit; OCP verified for evaluator/config seam |
| C7 replay | Versioned evidence results and coordinate bridge | Fixed v1 schema; viewer callbacks supply binding without changing exporter | `artifacts.py`, `web/` | Native asset/model/body roundtrip+hash/license check;4TS tests incl every failure kind; binding to real teammate WASM unrun; OCP reviewed |
| C8 composition | CLI, exact trusted asset fetching, robot registration | Explicit registry wiring; static-environment/model attachment | `cli.py`, `registry.py`, `scripts/` | Pinned scoped fetch, actual CLI workflows, original source hashes unchanged; unsupported robots rejected; OCP reviewed |
| C9 arm | Scalar-joint/site reach calculation | Supplied joint mapping, target/config; direct DLS solver | `arm.py` | Reachable/unreachable generic fixtures; no executed arm policy claim; direct design reviewed |
| C10 qualification | Validate consistent clean held-out reports and input gates | Fixed evidence schema, bounded task perturbation config | `qualification.py` |20clean/forged contact/hash/config/NaN/source tests plus actual20starts; direct validation design reviewed |
| C11 profiles | Declare mobile footprint/speed/height contracts without robot-name logic in learner | Add a `NavigationProfile`; fixed arms remain non-mobile | `profiles.py` | G1/Stretch/Go2 proxy profiles and SO-ARM non-mobile contract; OCP reviewed |
| C12 task plans | Keep task start/goal/route intent separate from scene and policy | Add bounded task plan/route inputs; no environment reconstruction | `task_plans.py` | Static local-goal tasks; obstacle/dynamic task claims require new evidence; OCP reviewed |
| C13 learner | Scene-specific reduced-order local navigation | Inject scene/profile/config; 23-observation, 3-action 64x64 MLP artifact | `learning.py` | A*-BC then PPO, seed/profile/scene/checkpoint metadata, 5 learner tests; not gait or sim-to-real |
| C14 learned strategy | Load a hash-bound learned artifact and emit bounded velocity commands | Swap only compatible scene/profile artifact; native embodiment stays separate | `learned_navigation.py` | Reject mismatched scene/profile/checkpoint contracts; local-goal learning plus separate 20/20 native detour start checks with declared safety filters; filter/actor identity included in provenance; OCP reviewed |
| C15 export/UI gates | Bind viewer replay/learned playback to exact native evidence | Extend schema/adapter rather than trust display state | `artifacts.py`, `web/` | Native replay requires matching learned/native hashes; source/native/preview labels stay distinct; viewer is not physics authority |

Local CLI deployment; no database, service queue, multi-user auth, or external AI inference required. One scene/run per process; bounded archive/input size, grid cells, episode duration, candidate count, and replay length. Measured example: 5.266 simulated seconds in roughly 1.2 CPU wall seconds; not a load/concurrency/service benchmark. Persist runs in user-selected output directories with hashes and status. Failures retain report/reason where possible. Retrying uses a new output directory; no hidden training job retries or costs.

MJCF/JSON are read-only. Runtime outputs and downloaded third-party artifacts are ignored by Git. No secrets are required. The local PPO experiment ran on CPU (about 20 seconds/profile, Modal spend $0) and emits immutable scene/profile/checkpoint contracts. Its 16/16 direct-goal proxy episodes for G1/Stretch/Go2 do not establish obstacle detours (0/16 held-out detours), visual-floor support, gait learning, or hardware deployment. Hardware deployment remains outside this integration.
