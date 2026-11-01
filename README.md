# Morpheus

Choose a robot, train a navigation policy against a supplied room collision scene, compare routes, and inspect its simulation evidence in the existing Three.js + Spark workspace. The capture workbench remains available at `/?view=builder`.

The policy package **consumes MJCF + Scene JSON**. Environment capture and collision authoring belong to the upstream workbench. [PRD](PRD.md) → [compact build brief](BUILD_BRIEF.md) → [architecture](architecture.md) → [policy methods](POLICY_TRAINING.md).

## What is implemented

| Layer | Method and reason |
| --- | --- |
| Global route calculation | Conservative A* with robot footprint inflation; returns up to three distinct checked routes. Predictable and cheap on static collision boxes. |
| Learned local navigation | PyTorch actor/critic MLP, 23 observations and 3 bounded velocity actions. A* route imitation warm start, then clipped PPO with GAE. Separate scene/profile-bound checkpoints for G1, Stretch and Go2. |
| Native G1 locomotion | Frozen official Unitree G1 12-DoF PPO/LSTM; 47 observations, 12 joint actions, 50 Hz actor, 500 Hz PD/physics. Exact model/checkpoint contract is verified before use. |
| Robustness wrapper | Global waypoints feed the local actor; heading alignment, goal slowing and predictive footprint braking suppress unsafe translation. Per-step MuJoCo contact, clearance, fall, force and joint gates still decide success. |
| Arm calculations | Generic bounded damped least-squares IK with explicit joint/site mapping. SO-ARM100 keeps a stationary preview base; no manipulation training or cleaning execution is claimed. |
| Visualization | Four profiles, alternative paths, proposed wipe/sweep traces, trained trajectory playback, hash-checked native G1 body poses, high-DPI rendering and 1920×1080 capture/video. |

Stretch, Go2 and SO-ARM100 in the browser are procedural previews. Only G1 currently has a matched physical model and locomotion checkpoint. A shared navigation envelope does not make robot joint controllers interchangeable. Cleaning traces are numerical **tool intentions**: the supplied input has no verified surface, mounted tool, force controller or cleaning-effect model.

## Measured results

All results below are from the supplied synthetic collision proxy, whose scale and review remain unconfirmed.

| Mobile profile | Untrained | After route imitation | After PPO | A* reference |
| --- | ---: | ---: | ---: | ---: |
| G1 | 0/16 | 5/16 | 16/16 | 16/16 |
| Stretch | 0/16 | 3/16 | 16/16 | 16/16 |
| Go2 | 0/16 | 3/16 | 16/16 | 16/16 |

The frozen final configuration uses 160 imitation epochs, then 32 PPO updates (16,384 transitions), about 20 CPU seconds per profile. Training seed 7; fresh held-out seed 10007. These are **local 0.75–2 m goals**, with initial heading noise σ=0.35 rad in the same scene; every held-out geometric route was straight. A* remains a strong reference. The first short G1 training run reached only 2/16.

The learned G1 controller also ran real actuated MuJoCo tests. A straight-task start-pose qualification passed 20/20. The first unfiltered detour failed clearance; an initial safety wrapper passed 19/20. With the final braking buffer and a fresh native seed 20007, the detour passed **20/20**, zero forbidden contacts, minimum clearance **0.10535 m** against the unchanged 0.10 m gate. That tests one static task with starts perturbed ±0.02 m and yaw ±0.03 rad. It establishes neither arbitrary-room generalization nor hardware readiness. Raw failed runs are retained locally alongside successful evidence.

**Input problem:** the mesh-support audit found 973 downward-ray-supported samples and zero overlap with positive 0.45 m-envelope clearance, over 4,556 grid samples. The occupied-column export blocks the visible aisle while proxy-free regions lack sampled mesh support. Runs therefore default to the supplied collision view, with this limitation visible. Corrected/reviewed upstream geometry is needed for a credible walk through the visible kitchen. This project does not silently repair it.

Modal spending: **$0 of the available $180**. This small MLP trains quickly on CPU. GPU-backed robot gait or contact-rich manipulation training requires a separate, matched model/task contract; [the training guide](POLICY_TRAINING.md) explains that path.

## Install and run

Tested with Node.js 24, Python 3.12, MuJoCo 3.17.0, Torch 2.15.0 and a WebGL2 desktop browser. No OpenAI API key is required.

```sh
npm ci
uv sync --extra dev
npm run fetch-sample
uv run python scripts/fetch_g1.py
```

The explicit fetch commands download pinned/hashed third-party assets. They remain ignored by Git. Read [sample provenance](docs/ASSETS.md) and [third-party credits](THIRD_PARTY.md) before redistributing assets.

Set the path to the teammate's existing export, then train and export the mobile profiles. The fixed arm skips base-navigation training.

```sh
export SCENE_BUNDLE="/path/to/morpheus-kitchen.zip"
uv run morpheus-policy inspect "$SCENE_BUNDLE"
uv run morpheus-learn train "$SCENE_BUNDLE" --output outputs/home --demo-cleaning
uv run morpheus-learn validate-g1 "$SCENE_BUNDLE" --output outputs/home --episodes 20
uv run morpheus-learn export "$SCENE_BUNDLE" --output outputs/home --demo-cleaning --native-artifacts outputs/home/g1-native
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://127.0.0.1:5174`; this session uses `http://127.0.0.1:5175`). Choose a robot, compare route alternatives, preview a cleaning intention for Stretch/SO-ARM100, play the learned trajectory, and use **Replay native G1 dynamics** for separately verified physical evidence. Native playback uses exact recorded body samples. The optional full-model video link appears after a video is rendered and the run is exported again.

The browser reads `public/policies/{g1,stretch,go2,soarm}.json`. Import/export also works through the UI. Generate artifacts before `npm run build`; the build copies policy evidence into `dist/policies`, while excluding downloaded scan binaries. Generated evidence can contain room data, so it stays ignored by Git. A static host does not run training or native validation.

To reproduce the input audit without changing the scene:

```sh
node scripts/audit_floor_support.mjs --scene /path/to/scene.json --mesh public/samples/rustic-kitchen-collider.glb --output outputs/input-floor-audit.json
uv run morpheus-learn export "$SCENE_BUNDLE" --output outputs/home --demo-cleaning --floor-audit outputs/input-floor-audit.json --native-artifacts outputs/home/g1-native
```

To render native evidence in 1080p:

```sh
uv run --with pillow --with imageio --with imageio-ffmpeg python scripts/render_replay.py outputs/home/g1-native --width 1920 --height 1080
```

Existing outputs are protected: use a new training/native directory for another experiment. CLI options expose seeds, imitation/PPO settings, explicit start/goal and native episode counts. [The initial policy guide](POLICY_README.md) documents lower-level inspect/plan/run/qualify/optimize/reach commands; [UI handoff](UI_HANDOFF.md) describes the viewer's original preview contract; [environment contract](docs/CONTRACT.md) documents upstream MJCF/JSON.

## Checks and demo

```sh
uv run pytest
npm test
node --experimental-strip-types --test web/policy-bridge.test.ts
npm run build
uv run morpheus-policy --help
uv run morpheus-learn --help
```

The actual-bundle Python integration check is enabled by `MORPHEUS_SCENE_BUNDLE`. The legacy mesh-clearance test additionally needs the provided manifest at `output/kitchen/scene.json`; copying it there does not alter its geometry. See [DEMO.md](DEMO.md) for the ≤90-second core journey and exact recorded outputs.

Codex coordinated research, implementation and independent review; built scene/profile/checkpoint contracts, trained and compared real PPO policies, found the floor/proxy mismatch, and caught a native high-level checkpoint provenance gap. The code, tests, failed/successful reports and audit are inspectable evidence. Existing libraries, model weights and sample assets retain their upstream terms; [THIRD_PARTY.md](THIRD_PARTY.md) records attribution.
