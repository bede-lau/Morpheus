# Morpheus

Explore a captured room, prepare its collision scene, and evaluate robot navigation policies in the same workspace. The integrated policy layer consumes the existing MJCF + Scene JSON without generating the environment.

- **Viewer and capture setup:** follow this page.
- **Policy installation, real G1 simulation, evidence and limitations:** [Policy guide](POLICY_README.md).
- **Architecture and compact build contract:** [architecture](architecture.md), [build brief](BUILD_BRIEF.md).

Turn a captured room into an editable collision environment for robot simulation.

Morpheus keeps a Gaussian splat as the visual reference, proposes simple collision geometry, lets a person correct it, and exports a MuJoCo scene with its provenance and validation status. The core journey is **load a room → establish scale → generate and review collision shapes → validate → export**. Robot models, controllers, policy training, and task evaluation are integrated through the policy package; see the policy guide above.

Built during the DevDay Community Hackathon in London on 1 November 2026. See the [90-second demo outline](docs/DEMO.md), [environment handoff contract](docs/CONTRACT.md), and [sample asset provenance](docs/ASSETS.md).

## What works

- Open the synthetic reference kitchen, or import a Gaussian `.spz`, `.ply`, or `.splat`, optionally alongside an aligned `.glb` companion mesh.
- Inspect the visual scan, reference mesh, and collision layer; select the source up-axis and calibrate scale from a known distance.
- Generate editable obstacle columns from sampled geometry. A companion mesh is preferred; splat-center proposals are a rougher fallback.
- Select, move, resize, rotate, rename, add, or remove box colliders. Save and reopen the collision manifest.
- Export `environment.xml`, `scene.xml`, `scene.json`, `validation.json`, and integration notes in a ZIP.
- With the local Python service running, compile the exported scene in native MuJoCo and run a sphere-drop floor-contact probe.

## Local setup

Tested development tools: Node.js 24, Python 3.12, and MuJoCo 3.17.0. Use a desktop browser with WebGL2; Chrome is the primary demo target. No OpenAI API key is required to run this version.

From the repository root:

```sh
npm ci
npm run fetch-sample
uv venv pipeline/.venv
uv pip install --python pipeline/.venv/bin/python -r requirements.txt
```

The explicit sample-fetch command downloads the publisher's test assets into `public/samples/`, checks their sizes and SHA-256 hashes, and can resume interrupted transfers. Read [the asset notes](docs/ASSETS.md) before redistributing any sample files. They are excluded from Git.

Start the frontend in one terminal:

```sh
npm run dev
```

Open [http://127.0.0.1:5174](http://127.0.0.1:5174). In a second terminal, start native validation:

```sh
pipeline/.venv/bin/python -m pipeline.server
```

The service listens on `127.0.0.1:8000`; Vite proxies `/api` to it. It accepts collision manifests, not scan binaries. Check [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health) for engine availability. The server preloads MuJoCo before announcing readiness because its first native import can take several seconds.

## Use the app

1. Click **Open sample kitchen**, or **Import your scan**. Select an aligned GLB in the same file selection if one is available.
2. Inspect **Scale & alignment**. The sample's metre scale is assumed, not independently measured. For a real capture, select two known surface points and enter their measured separation. Do not invent a measurement for the synthetic sample.
3. Click **Generate collision shapes**. Compare the **Scan**, **Reference mesh**, and **Collisions** views. Correct false obstacles and inspect passageways; review is part of the workflow.
4. Click **Validate in MuJoCo** when the local service is running. Read compilation and floor-probe results separately.
5. Click **Export environment** and hand the ZIP to the robot-training developer. A browser-only export is schema checked and explicitly marked `unverified` for native physics.

Local imports are processed in the browser. Native validation sends only the collision manifest to the local Python service. Loading the sample in a production build fetches its files directly from the publisher's host.

## Training handoff

The scene uses metres, a right-handed Z-up frame, full box dimensions in JSON, and WXYZ quaternions. MuJoCo receives half-extents as required by MJCF. Static geometry names are prefixed `env_`.

Unzip a bundle and load its standalone scene:

```python
import mujoco

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
```

To combine it with a robot scene, place this inside the robot's `<mujoco>` root:

```xml
<include file="environment.xml"/>
```

Reconcile the robot scene's existing floor to avoid duplicate overlapping geometry. Preserve the exported world frame; do not reapply `source.transform` to the colliders. See [CONTRACT.md](docs/CONTRACT.md) for the exact schema and validation meanings.

## Reproduce the reference export

After fetching the sample:

```sh
npm run prepare-sample
pipeline/.venv/bin/python -m pipeline validate output/kitchen/scene.json
pipeline/.venv/bin/python -m pipeline package output/kitchen/scene.json -o output/kitchen/morpheus-kitchen.zip
```

The preparation script samples the companion mesh, applies a shared frame conversion, and writes an **assumed-scale, unreviewed** fixture. It does not infer semantic household objects. The output directory is ignored by Git.

Recorded native check on 1 November 2026: the reference export contained 111 static colliders, compiled in MuJoCo 3.17.0, and passed the two-second floor-contact probe. A sphere of radius 0.04 m dropped from `[0, 0, 0.5]` rested at Z ≈ 0.03963 m against the floor. That evidence establishes basic physical support in the generated scene; it does not establish metric accuracy, reconstruction quality, or successful robot tasks.

## Checks and builds

```sh
npm test
pipeline/.venv/bin/python -m unittest discover -s tests -p 'test_pipeline.py' -v
npm run build
npm run preview
```

The recorded implementation checks comprise 11 TypeScript tests and 8 Python tests. They cover geometry proposal behavior, scale and transform math, manifest validation, MJCF export, disabled colliders, actual native floor contact, ZIP contents, and local API validation/export. Browser interaction checks and production deployment must be assessed separately; these test counts do not certify every UI interaction.

`npm run build` produces a static `dist/`. It deliberately does **not** copy downloaded sample binaries. Production sample loading uses official remote URLs and requires network access. A static host alone does not run the Python service, so its browser export records native validation as not run. The local Vite proxy is a development feature, not a deployed backend.

## Scope and limitations

- This is a reviewed collision-geometry preparation tool. It is not arbitrary splat-to-mesh reconstruction, semantic segmentation, exact furniture retrieval, or a robot policy trainer.
- The sample kitchen is AI-generated by World Labs and uses its companion mesh. It is not a measured scan of a real home.
- Proposals use occupied XY columns extending from the proposed floor to observed obstacle heights. This fills space beneath elevated surfaces and can block table openings, overhangs, and other real free space. Sparse or missing observations can also omit real obstacles. Unknown space is not certified free.
- Floor estimation, cropping, scale, orientation, and collision proposals require review. Source provenance records whether scale is assumed or confirmed; validation does not change those claims.
- Boxes are static and have assumed contact parameters. Doors, drawers, deformable objects, material identification, and articulated scene reconstruction are outside this version.
- A passing floor probe proves only model compilation and contact support at the sampled point. It does not prove reachability, graspability, navigation safety, or task success.

## Credits and Codex contribution

The hackathon implementation includes the room-preparation interface, geometry proposal and editing flow, scene contract, dual browser/Python exporters, local validation service, and tests. It builds on existing libraries and a publisher-provided sample:

| Component | Source |
| --- | --- |
| Gaussian splat rendering | [Spark 2.4.0](https://github.com/sparkjsdev/spark) |
| 3D scene and interaction | [Three.js 0.180.0](https://github.com/mrdoob/three.js) |
| UI | [React 19.2.0](https://github.com/facebook/react), [Lucide icons](https://github.com/lucide-icons/lucide) |
| Development/build tools | [Vite](https://github.com/vitejs/vite), [TypeScript](https://github.com/microsoft/TypeScript) |
| Physics validation | [MuJoCo 3.17.0](https://github.com/google-deepmind/mujoco) |
| Browser ZIP export | [fflate](https://github.com/101arrowz/fflate) |
| Synthetic sample and companion collider | [World Labs — Rustic kitchen with natural light](https://docs.worldlabs.ai/marble/export/specs) |

Upstream code and assets retain their respective terms. No open redistribution license was established for the World Labs sample; downloaded binaries are not committed or bundled in the production build. Full file hashes and provenance are in [ASSETS.md](docs/ASSETS.md).

Codex coordinated bounded research and implementation tasks, checked sample integrity and relative splat/mesh alignment, authored the collision preparation and export code, and ran the native tests. A concrete contribution was identifying the full-room convex-hull collision problem, exporting separate box colliders instead, and validating the actual generated 111-collider scene with a physical contact probe. The code, tests, and `validation.json` are inspectable evidence of that work.
