# Morpheus: home-policy integration

Consume a teammate's MJCF + Scene JSON, calculate a robot-compatible route, execute a real learned G1 walking controller in MuJoCo, and export the resulting state and evidence to an existing Three.js + Spark viewer.

**This repository does not capture or generate environments.** The supplied geometry is read-only. The only composition step attaches a registered robot to that geometry. `PREVIEW_ONLY` is a successful calculation with unresolved input/qualification gates; it is not physical readiness.

## Policy and supported scope

| Component | Implementation | Why / tradeoff |
|---|---|---|
| Home navigation | Conservative footprint A*, bounded waypoint velocity control | Immediate use of supplied geometry; static, flat-floor envelope model |
| Humanoid locomotion | Official Unitree G1 12DoF PPO/LSTM checkpoint, 47 observations / 12 actions | Actual learned walking with exact model pairing; legs only, no kitchen manipulation |
| Low-level actuation | 50 Hz recurrent actor, 500 Hz PD and MuJoCo physics | Preserves original deployment contract; CPU inference |
| Optimization | Bounded controller-config search, disjoint train/held-out seeds | Actor stays frozen; results are controller tuning, not new policy training |
| Fixed arm | Generic damped least-squares IK on supplied joint/site mapping | Kinematic reach calculation only; no commercial arm, grasp, collision-rollout or manipulation certification |
| Go2 / other dogs | Capability registry, explicitly unavailable | Requires a separate matched policy/model contract; G1 weights cannot transfer by changing MJCF |

The actor is pinned to [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym) commit `276801e46c5d433564f24658bac64f254b7d2d4b`; checkpoint SHA-256 is `cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d`. Use its `g1_12dof.xml`, not a generic G1/29DoF model. The BSD-3-Clause license is fetched and carried in exported model bundles.

## Run the core journey

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). No API key, GPU, Modal account, browser server, or hardware connection is required.

```sh
uv sync --python 3.12 --extra dev
uv run python scripts/fetch_g1.py
uv run morpheus-policy inspect /path/to/morpheus-kitchen.zip
uv run morpheus-policy robots
uv run morpheus-policy run /path/to/morpheus-kitchen.zip \
  --demo-task --output outputs/demo
```

The fetcher installs only the pinned robot XML, referenced meshes, checkpoint, license, and hash inventory into `.morpheus/assets/g1`. It refuses an existing output. Reuse an intact installation; use a new `--output` path when fetching another copy.

The provided bundle has no semantic task target. `--demo-task` explicitly chooses a numerical free-space route; it does not label a target as a sink, appliance, or real household task. For a specified numerical route:

```sh
uv run morpheus-policy plan /path/to/morpheus-kitchen.zip \
  --start=-2.5,-1.5 --goal=0,2.5
uv run morpheus-policy run /path/to/morpheus-kitchen.zip \
  --start=-2.5,-1.5 --goal=0,2.5 --duration 45 --output outputs/crossing
```

Positions are metres in the existing scene's Z-up frame. `--yaw` is radians; when omitted it points toward the goal. Initial posture/base pose are assigned only at episode reset. Every subsequent state comes from `policy → PD controls → mj_step`; the floating base is never moved along a planned path.

Outputs:

- `report.json`: task, thresholds, source caveats, execution/readiness, provenance and per-step aggregate metrics.
- `replay.json`: 50 Hz full qpos/qvel, controls, velocity commands, named body poses, route, failure events.
- `model.mjz`: portable composed robot + supplied geometry, external meshes and Unitree license.
- `manifest.json`: report/replay/model digests; the model digest is also in report/replay.
- `THIRD_PARTY_LICENSE.txt`: redistributed robot-asset license.

Output directories must be new. A blocked/failed run exits with code 2. Native XML export can round geometric decimals by a few micrometres; the report declares the checked 5 µm presentation tolerance. Physics uses original input values; body-pose replay is checked. Browser physics parity is not claimed.

## Verification and optimization

```sh
uv run morpheus-policy qualify /path/to/morpheus-kitchen.zip \
  --demo-task --episodes 20 --output outputs/qualification
uv run morpheus-policy optimize /path/to/morpheus-kitchen.zip \
  --demo-task --output outputs/controller-search
uv run pytest -q
MORPHEUS_SCENE_BUNDLE=/path/to/morpheus-kitchen.zip uv run pytest -q
node --experimental-strip-types --test web/policy-bridge.test.ts
```

The actual-input integration test requires that environment variable and fetched assets. Generic robot fixtures verify contracts only. Qualification changes starting XY by ±2 cm and heading by ±0.03 rad for the same task/map; it does not establish robustness to a different home, friction, perception, or dynamic obstacles.

Every physics step checks finite state, forbidden contacts, support bodies, body tilt/height, envelope clearance, joint position/speed, applied joint-force limits, and simulator warnings. Failed episodes stop. Qualification also rejects mismatched source/model/policy/configuration and fabricated completion with contacts or failures.

The supplied fixture declares synthetic geometry, assumed metric scale and no review. These flags are preserved, so its result remains `PREVIEW_ONLY` even with successful trials. Physical hardware is always `UNVERIFIED`.

## Observed results on the supplied fixture

Measured locally on 1 November 2026 with MuJoCo 3.17.0 / Torch 2.15.0:

- Learned G1 walking: 5.266 simulated seconds, approximately 1.2 CPU wall seconds, final target error 0.2205 m within the declared 0.25 m tolerance, zero forbidden contacts/falls/force-limit overruns.
- 20/20 perturbed starts completed; minimum envelope clearance approximately 0.869 m and maximum tilt approximately 0.0882 rad.
- A separate 4.72 m straight crossing completed in 18.11 simulated seconds with zero forbidden contacts and 0.339 m minimum envelope clearance.
- A genuine 17-point obstacle detour (1.60 m planned route vs 1.265 m straight separation) completed in 7.146 simulated seconds, with 0.2036 m final error, zero forbidden contacts and 0.1779 m minimum envelope clearance.
- The detour also passed 20/20 held-out starts: worst final error 0.2162 m, minimum clearance 0.1758 m, zero forbidden contacts and applied force-limit overruns.
- Exploratory search on two train and two held-out starts selected 0.4 m/s over the 0.3 m/s baseline: mean held-out duration 5.135 s vs 5.271 s, zero contacts in both. Two held-out cases are limited evidence; the baseline remains the default and has the 20-trial qualification.

These are simulation calculations on proposed collision proxies. No actor was retrained, no hardware was tested, and no Modal credits were spent.

Final validation: 44 Python tests passed with the actual-input integration enabled; 4 TypeScript tests passed; both generated replay bundles passed the actual bridge identity/state-application check. Torch emits a deprecation warning for the official TorchScript artifact; no test fails from it.

## Existing viewer integration

Use [web/README.md](web/README.md) and [visualization-contract.md](visualization-contract.md). Import `web/policy-bridge.ts` into the teammate's viewer. Verify the model artifact digest, set complete qpos on the exact compiled model, then call `mj_forward` to obtain transforms. Keep readiness from `report.json`; Three.js collision tests cannot upgrade it.

Spark/environment rendering remains upstream-owned. Apply the declared source-to-viewer root transform once; keep the original scan transform separate. Actual integration into the teammate's running browser/WASM wrapper is still an external handoff, not an executed browser end-to-end test.

Render a native replay clip without building a new viewer:

```sh
uv run --with pillow --with imageio --with imageio-ffmpeg \
  python scripts/render_replay.py outputs/demo \
  --png outputs/demo/preview.png --mp4 outputs/demo/demo.mp4 --seconds 12 --fps 20
```

This renders recorded states only, including route overlays and preview labels. The 12-second clip stretches the short simulation for inspection; its on-screen timestamp is simulation time. It is a local demo artifact, not an uploaded submission video.

## Training and GPU decision

Pretrained inference, planning, IK and evaluation fit CPU execution. Modal would be useful for a defined specialist-training experiment, not this inference loop. [POLICY_TRAINING.md](POLICY_TRAINING.md) defines the conditional next step, data/reward/compatibility requirements, and $180 budget boundary. No GPU trainer is implemented or launched here.

## Handoff, attribution and release

[PRD.md](PRD.md) was written first; [BUILD_BRIEF.md](BUILD_BRIEF.md) is its ultra-compact pre-build handoff. Engineering contracts are in [architecture.md](architecture.md), [design.md](design.md), [decisions.md](decisions.md), and [agents.md](agents.md).

Codex coordinated parallel policy research, scene/planning implementation, visualization, verification and independent review. Concrete review fixes include recurrent-memory reset tests, model-name mapping, portable assets, force/contact checks, failure-replay events, and hash-linked evidence. See [DEMO.md](DEMO.md) for a <=90-second presentation script.

Original integration code: MIT. Existing code/assets: Unitree RL Gym/G1 and checkpoint (BSD-3-Clause), MuJoCo (Apache-2.0), Torch (BSD-style), upstream fixture attributed to World Labs in its source metadata. The capture pipeline, Spark, Three.js, and the viewer are supplied by the other developer and are not bundled here. Dependencies retain their own licenses. See [THIRD_PARTY.md](THIRD_PARTY.md).

Inputs, robot downloads, output geometry/replays/media, secrets and local environment files are excluded from Git. Review outputs before sharing home scans. A local Git repository is initialized; there is no configured public remote, uploaded video, or submitted hackathon entry. Reserve time before the [17:00 London deadline](https://luma.com/dde-ldn-hack) to publish and submit.
