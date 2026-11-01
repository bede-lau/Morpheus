<p align="center"><img src="src/assets/morpheus-gradient-logo-transparent.png" width="480" alt="Morpheus — gradient ribbon logo" /></p>

<p align="center">
  🏆 <b>1st Place @ OpenAI DevDay Exchange Community Hackathon, London</b><br/>
  <sub>1 November 2026 · Built in one day · An event with 1,000+ applicants</sub>
</p>

<p align="center">
  <b>Scan your room. Morph it into a simulation. Watch real robots do your chores in it.</b><br/>
  <sub>Gaussian splat → collision world → SimReady objects → planned robot rollouts → an honest verdict on which robot fits your home.</sub>
</p>

<p align="center">
  <img alt="Three.js" src="https://img.shields.io/badge/three.js-0.180-000?style=flat-square&logo=threedotjs"/>
  <img alt="Spark" src="https://img.shields.io/badge/gaussian%20splats-Spark%202.4-19c37d?style=flat-square"/>
  <img alt="MuJoCo" src="https://img.shields.io/badge/MuJoCo-3.17-0b3d31?style=flat-square"/>
  <img alt="OpenAI Decisions" src="https://img.shields.io/badge/OpenAI-Decisions%20%C2%B7%20gpt--6--luna-111?style=flat-square&logo=openai"/>
  <img alt="Terra" src="https://img.shields.io/badge/vision-gpt--6.1--terra-111?style=flat-square&logo=openai"/>
  <img alt="Codex" src="https://img.shields.io/badge/built%20with-Codex-111?style=flat-square&logo=openai"/>
  <img alt="SimReady" src="https://img.shields.io/badge/assets-NVIDIA%20SimReady-76b900?style=flat-square&logo=nvidia&logoColor=white"/>
  <img alt="Tests" src="https://img.shields.io/badge/tests-43%20TS%20%C2%B7%2034%20Py-19c37d?style=flat-square"/>
</p>

<p align="center">
  <img src="docs/media/03-luna.jpg" alt="Luna Decisions classifying a scanned bowl against the NVIDIA SimReady catalogue" width="100%"/>
</p>

---

## ▍The problem

Robots are about to move into homes. The question stops being *"can a robot do chores?"* and becomes **"which robot can do *my* chores, in *my* kitchen?"** Spec sheets can't answer that. Your counter height, your aisle width and where your sink actually is can.

Morpheus takes a phone-grade capture of a room and turns it into something a robot can be dropped into. Then it lets several robots attempt the same household task inside it, from the robot's own camera, and records which steps each one could and couldn't do.

## ▍The pipeline at a glance

```mermaid
flowchart LR
    classDef input fill:#0b0f0e,stroke:#19c37d,color:#e8f3ec,stroke-width:1.5px
    classDef stage fill:#ffffff,stroke:#111,color:#111,stroke-width:1px
    classDef model fill:#111,stroke:#111,color:#fff,stroke-width:1px
    classDef output fill:#19c37d,stroke:#0b3d31,color:#04120d,stroke-width:1.5px

    SPZ[("Gaussian splat<br/>500k gaussians<br/>.spz / .ply")]:::input
    GLB[("Companion mesh<br/>collider .glb")]:::input

    subgraph S1["01 · SCAN → ENVIRONMENT"]
      direction TB
      A1["Floor-plane fit<br/>+ surface sampling"]:::stage
      A2["Box collider proposal<br/>111 static geoms"]:::stage
      A3["MuJoCo compile<br/>+ drop probe"]:::stage
      A1 --> A2 --> A3
    end

    subgraph S2["02 · OBJECTS → SIMREADY"]
      direction TB
      B1["Render 16 orbit views"]:::stage
      B2["Detect tableware<br/>per view"]:::model
      B3["Raycast into splat<br/>dedupe in 3D"]:::stage
      B4["Classify + rank<br/>catalogue candidates"]:::model
      B5["Mask splats<br/>drop in USD→GLB mesh"]:::stage
      B1 --> B2 --> B3 --> B4 --> B5
    end

    subgraph S3["03 · TASK → ROBOTS"]
      direction TB
      C1["Room interior<br/>from mesh grid"]:::stage
      C2["Landmarks<br/>table · sink · entrance · objects"]:::stage
      C3["A* on inflated<br/>occupancy grid"]:::stage
      C4["Scripted manipulation<br/>wipe · pick · carry · place"]:::stage
      C5["Per-step feasibility"]:::model
      C1 --> C2 --> C3 --> C4 --> C5
    end

    SPZ --> S1
    GLB --> S1
    S1 --> S2 --> S3
    S3 --> V{{"Verdict per robot<br/>Stretch · Go2 · Unitree G1"}}:::output
```

| Stage | Model | What it decides |
| --- | --- | --- |
| **Terra vision** | `gpt-6.1-terra` via the Responses API, or `gpt-6-astra` subagents via `codex exec` | Which pixels in each rendered view are a plate, bowl, mug, cup, glass or spoon |
| **Luna decisions** | `gpt-6-luna` via `POST /v1/decisions` | The object's category (with probabilities), handle/shallow predicates, and a 0–3 match score for every catalogue candidate |
| **Feasibility** | `gpt-6-luna` predicates | Whether each manipulation step is plausible for the chosen robot body |

---

## ▍The demo

The app opens straight into the show. Every number on screen is read from data recorded in this repository: the scan, the terra detections, the luna traces, the SimReady meshes, the MuJoCo report and the planned rollouts.

<table>
<tr>
<td width="50%"><img src="docs/media/00-bloom.jpg" alt="Splat bloom-in"/><br/><sub><b>00 · BLOOM.</b> Click the kitchen. Half a million gaussians reveal through an expanding sphere mask while the camera dollies in.</sub></td>
<td width="50%"><img src="docs/media/01-scan.jpg" alt="Scan sweep"/><br/><sub><b>01 · SCAN.</b> Floor plane fitted on a tilted capture, 90k surface samples, room crop and cutaway.</sub></td>
</tr>
<tr>
<td><img src="docs/media/02-terra.jpg" alt="Terra contact sheet"/><br/><sub><b>02 · TERRA VISION.</b> Sixteen orbit renders, one vision request each. Brackets are the model's own pixel detections.</sub></td>
<td><img src="docs/media/03-luna.jpg" alt="Luna decisions card"/><br/><sub><b>03 · LUNA DECISIONS.</b> Category probabilities, then every NVIDIA candidate scored. Only matches ≥ 0.5 / 3 are accepted.</sub></td>
</tr>
<tr>
<td><img src="docs/media/05-carry.jpg" alt="Stretch carrying the SimReady bowl"/><br/><sub><b>06 · ROBOTS.</b> Stretch carries the swapped-in SimReady bowl to the sink. The inset is the robot's own camera.</sub></td>
<td><img src="docs/media/06-robots.jpg" alt="Verdict"/><br/><sub><b>VERDICT.</b> Six of six steps completed. Run Go2 and G1 on the same chore and the verdicts stack up side by side.</sub></td>
</tr>
</table>

```mermaid
sequenceDiagram
    autonumber
    actor U as Presenter
    participant D as Demo page (/)
    participant V as RoomViewer<br/>Three.js + Spark
    participant E as Env API :8000
    participant O as Object API :8001
    participant R as RobotScene + POV

    U->>D: click "Rustic kitchen"
    D->>E: GET /api/environments/kitchen
    E-->>D: scene · validation · splat/mesh URLs · world transform
    D->>V: load splat + mesh, applyWorldTransform, crop, cutaway
    V-->>U: sphere-mask bloom-in
    U->>D: MORPH TO SIM
    D->>V: 01 scan sweep (floor → cutaway)
    D->>D: 02 replay terra detections (/discovery/*.json)
    D->>O: GET /api/objects/scene
    O-->>D: replacements + luna Decisions traces
    D->>D: 03 animate category bars + candidate scores
    D->>V: 04 SavedObjectLayer: mask splats, scale-in SimReady GLBs
    D->>V: 05 colliders + MuJoCo stamp
    Note over D,U: waits — nothing moves until a robot is chosen
    U->>D: pick robot + task → RUN TASK
    D->>E: GET /api/tasks/{slug}/{robot}
    E-->>D: replay · events · verdict · carried[]
    D->>R: play 10 Hz frames, carry bowl on gripper, render POV inset
    R-->>U: captions, build log, verdict card
```

**Controls:** `Space` pause / start · `→` next phase · `R` restart. In the workspace, `F` toggles a creative-mode fly camera (WASD, Space/Shift, mouse look, wheel for speed).

---

## ▍Architecture

![Morpheus architecture: room capture and task inputs flow through the React client to environment and object services, geometry validation, task planning and the object pipeline.](docs/media/architecture.svg)

The environment service serves scenes and planned replays and validates MJCF with MuJoCo. The object service manages catalogue matching and saved replacements. Object discovery can also run offline. Robot routes and manipulation are planned or scripted; they are not learned locomotion. The sample scene's scale is assumed.

<sub>Visual direction inspired by [Supabase's architecture diagrams](https://supabase.com/docs/guides/getting-started/architecture).</sub>

**Committed state the client reads:** `output/kitchen/scene.json` (111 colliders) · `rollouts/kitchen/*` (3 tasks × 3 robots) · `output/simready/` (prepared meshes + saved scene with Decisions traces) · `public/discovery/` (16 views + detections) · `src/robots/*.glb` (MuJoCo Menagerie meshes).

### Object discovery → SimReady swap

```mermaid
flowchart LR
    classDef step fill:#fff,stroke:#111,color:#111
    classDef ai fill:#111,stroke:#111,color:#fff
    classDef ok fill:#19c37d,stroke:#0b3d31,color:#04120d
    classDef no fill:#f0a429,stroke:#7a4d06,color:#1d1200

    classDef q fill:#0b0f0e,stroke:#19c37d,color:#e8f3ec
    R["render<br/>16 orbit views<br/>850×874"]:::step --> D{"detect"}:::q
    D -->|"OPENAI_TERRA_API_KEY set<br/>≤ 4 requests in flight"| T["gpt-6.1-terra<br/>Responses API<br/>json_schema output"]:::ai
    D -->|"no key"| A["codex exec ×16<br/>gpt-6-astra<br/>ChatGPT login"]:::ai
    T --> J[("detections.json<br/>view · label · x · y · conf")]:::q
    A --> J
    J --> RC["raycast pixel → splat<br/>reject z ∉ 0.35–1.6 m<br/>dedupe < 0.18 m"]:::step
    RC --> L["luna Decisions<br/>1 · category choice<br/>2 · score every candidate 0–3"]:::ai
    L -->|"score ≥ 0.5"| S["mask splats (SplatEdit BOX)<br/>place SimReady GLB<br/>merge into saved scene"]:::ok
    L -->|"weak / unknown"| K["leave as scan<br/>log reason"]:::no
```

The API-key policy is deliberate: the main key only calls **Decisions**. Detection uses either a separate terra-only key (the code refuses any non-terra model on that path) or the Codex CLI login. Nothing above terra is ever called with an API key.

### Task planning

```mermaid
flowchart TB
    classDef s fill:#fff,stroke:#111,color:#111
    classDef ai fill:#111,stroke:#111,color:#fff
    classDef v fill:#19c37d,stroke:#0b3d31,color:#04120d

    M["collider .glb → world frame<br/>400k surface samples"]:::s --> G["5 cm grids<br/>floor · obstacle · interior<br/>(tilted floor plane fitted)"]:::s
    G --> LM["landmarks<br/>table · sink (reviewed) · counters<br/>entrance · SimReady objects"]:::s
    classDef q fill:#0b0f0e,stroke:#19c37d,color:#e8f3ec
    TXT["'wipe the table and<br/>move the bowl to the sink'"]:::q --> P{"plan"}:::q
    P -->|"default"| RU["rule planner"]:::s
    P -->|"--steps plan.json"| CX["plan written by a<br/>Codex CLI agent"]:::ai
    RU & CX --> STEPS["navigate → wipe → navigate →<br/>pick → navigate → place"]:::s
    LM --> STEPS
    STEPS --> AS["A* (8-connected) on grid inflated<br/>by robot radius + 5 cm, string-pulled"]:::s
    AS --> SIM["10 Hz frames · yaw · arm extension<br/>events · carried[] pick/place times"]:::s
    SIM --> FE["luna feasibility predicates<br/>per manipulation step"]:::ai
    FE --> V["verdict<br/>completed · partial · failed<br/>+ the step that broke"]:::v
```

| Robot | Body | Reach | Bowl chore |
| --- | --- | --- | --- |
| **Stretch 3** | wheeled mobile manipulator, telescoping arm | 0.52 m, 0.15–1.10 m height | ✅ completed · 6 / 6 steps |
| **Unitree G1** | humanoid | 0.50 m, 0.30–1.45 m height | ✅ completed |
| **Unitree Go2** | quadruped, no arm | — | ⚠️ partial · reaches the table, cannot wipe |

Every frame of every committed rollout is tested to stay inside the room interior with full footprint clearance from the mesh.

---

## ▍Run it

Requirements: Node 24, Python 3.12, a WebGL2 browser (Chrome).

```sh
git clone https://github.com/bede-lau/Morpheus && cd Morpheus
npm ci
npm run fetch-sample                                    # World Labs kitchen splat + collider (checksummed)
uv venv pipeline/.venv
uv pip install --python pipeline/.venv/bin/python -r requirements.txt
```

```sh
npm run dev                                             # http://127.0.0.1:5174
pipeline/.venv/bin/python -m pipeline.server            # :8000  environments · tasks · MuJoCo
pipeline/.venv/bin/python -m pipeline.object_server     # :8001  SimReady objects · Decisions
```

| URL | What |
| --- | --- |
| `/` | The demo: blank canvas → bloom → morph to sim → robots |
| `/?view=workspace` | Task runner, robot POV, verdicts, compare, fly camera |
| `/?view=objects` | Manual select → identify with Decisions → replace |
| `/?view=builder` | Collider editing, scale calibration, MuJoCo validate/export |

**No key needed to run the demo.** Rollouts, detections, Decisions traces, prepared meshes and the saved object scene are all committed. Keys only matter for producing new data:

```sh
# .env
OPENAI_API_KEY=sk-...           # Decisions only (gpt-6-luna): matching + feasibility notes
OPENAI_TERRA_API_KEY=sk-...     # optional, terra-class vision only
```

### Regenerate the data

```sh
# object discovery: render → detect (terra API or Codex subagents) → apply
python3 scripts/discover-objects.py render --views 16
python3 scripts/discover-objects.py detect             # --backend api|codex
python3 scripts/discover-objects.py apply              # Decisions match, replace, merge-save

# task rollouts for all robots
pipeline/.venv/bin/python -m pipeline.task_planner "wipe the table and move the bowl to the sink" --all
pipeline/.venv/bin/python -m pipeline.task_planner "fill a glass of water" --all --steps plan.json

# inspect what the planner sees
pipeline/.venv/bin/python -m pipeline.landmarks
```

### Tests

```sh
npm test                                                                   # 43 TypeScript
pipeline/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'     # 34 Python
npm run build
```

Coverage includes collider proposal and export, native MuJoCo floor contact, browser-vs-native MJCF parity, revision-bound replays, landmark and spawn placement, rollout-inside-room clearance, pick/carry/place timing, POV eye pose, fly-camera math and SimReady scene persistence.

---

## ▍Repository map

```
Morpheus/
├── src/
│   ├── demo.tsx · demo-fx.ts          morph-to-sim show (route /)
│   ├── red-pill.tsx                   opening screen
│   ├── workspace.tsx                  task runner UI
│   ├── viewer.ts · cutaway.ts         Three.js + Spark room viewer, display-only crop/cutaway
│   ├── robot-scene.ts · robot-model.ts  articulated Menagerie robots, gait, carry
│   ├── pov.ts · fly.ts                robot camera inset, creative-mode fly cam
│   ├── saved-objects.ts · object-scene.ts  SimReady meshes + splat masks
│   └── robots/                        Stretch 3 · Go2 · G1 GLBs + joint sidecars
├── pipeline/
│   ├── server.py · read_api.py · task_api.py   :8000
│   ├── object_server.py · simready.py          :8001, Decisions, USD → GLB
│   ├── room_mesh.py · landmarks.py · task_planner.py
│   └── export.py                      MJCF + validation
├── scripts/discover-objects.py        render / detect / apply
├── rollouts/kitchen/                  committed rollouts (3 tasks × 3 robots)
├── output/                            scene.json, SimReady prepared meshes, discovery record
└── docs/                              contracts, API, demo script
```

| Doc | |
| --- | --- |
| [docs/DEMO.md](docs/DEMO.md) | 90-second script with talking points |
| [docs/TASKS.md](docs/TASKS.md) | Task-stage contract: robots, envelope, landmarks, carry, API policy |
| [docs/CONTRACT.md](docs/CONTRACT.md) | Environment handoff: MJCF, manifest schema, frames, validation meaning |
| [docs/READ_API.md](docs/READ_API.md) | HTTP routes for environments, replays and tasks |
| [docs/ASSETS.md](docs/ASSETS.md) | Provenance and licensing for every asset |
| [docs/UI_HANDOFF.md](docs/UI_HANDOFF.md) | Workspace UI internals |

### Drop the room into your own MuJoCo scene

```python
import mujoco
model = mujoco.MjModel.from_xml_path("scene.xml")   # from the exported bundle
```

```xml
<mujoco> <include file="environment.xml"/> <!-- your robot here --> </mujoco>
```

Metres, right-handed, Z-up, WXYZ quaternions, static geoms prefixed `env_`. Details in [CONTRACT.md](docs/CONTRACT.md).

---

## ▍What this is, and isn't

We'd rather be precise than impressive.

- The kitchen is a **synthetic World Labs capture** with a companion mesh, not a measured scan of a real home. Scale is assumed.
- Rollouts are **planned motions checked against the room geometry**, not physics simulations or trained policies. A verdict says which steps are geometrically and kinematically plausible for a body, not that a robot will succeed.
- Landmarks are inferred from geometry. The sink was pinned by a person after looking at the scan, and the UI says so.
- SimReady swaps are **visual + static collision** replacements. Mass, articulation and grasp quality are not validated.
- The table bowl sits slightly past Stretch's nominal reach; the pick uses a declared 0.20 m gripper allowance, shown in the step note.

## ▍Credits

| | |
| --- | --- |
| Gaussian splat rendering | [Spark](https://github.com/sparkjsdev/spark) |
| 3D | [Three.js](https://github.com/mrdoob/three.js) |
| Physics | [MuJoCo](https://github.com/google-deepmind/mujoco) |
| Robot models | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie): Hello Robot Stretch 3 (Clear BSD), Unitree Go2 and G1 (BSD-3-Clause) |
| Objects | [NVIDIA SimReady Assets](https://huggingface.co/datasets/nvidia/simready-assets) (CC-BY-4.0) |
| Sample room | [World Labs — Rustic kitchen with natural light](https://docs.worldlabs.ai/marble/export/specs) (not redistributed; fetched by script) |
| Models | OpenAI Decisions API (`gpt-6-luna`), Responses API (`gpt-6.1-terra`), Codex CLI (`gpt-6-astra`) |

<p align="center"><sub>Built in one day at the OpenAI DevDay Community Hackathon, London, 1 November 2026, with Codex.</sub></p>

