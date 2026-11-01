# Task stage contract — "tell the robot what to do"

This is the final stage of the Morpheus demo. The environment is already
reconstructed (collision boxes in `output/kitchen/scene.json`, optional
SimReady object replacements in `output/simready/scene.json`). A robot is
spawned at the kitchen door, given a natural-language task such as
**"wipe the table"** or **"fill a glass of water"**, and attempts it inside
the reconstructed room. Three robots (Stretch, Go2, SO-ARM100) attempt the
same task so the user can decide which robot suits their home.

Everything in this stage must work **offline and deterministically** (no API
key) and merely get *better* when `OPENAI_API_KEY` is set. Pre-rendered
rollouts are committed to the repository under `rollouts/kitchen/`.

## Files and ownership

| Owner | Files |
| --- | --- |
| Planner (Python) | `pipeline/task_planner.py`, `pipeline/landmarks.py`, task routes in `pipeline/server.py`, `pipeline/read_api.py` fallback to `rollouts/`, `tests/test_tasks.py`, `rollouts/kitchen/**` |
| Frontend (TS) | `src/replay.ts` (type extension only, keep existing exports), `src/robot-scene.ts`, `src/workspace.tsx`, `src/workspace.css`, `src/pov.ts`, `src/tasks.ts`, `src/*.test.ts` |

Do not edit the other owner's files. Shared contract is this document.

## Robot profiles (shared constants)

| id | name | base | footprint radius | arm reach (m) | arm height range (m) | eye height (m) | speed (m/s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `stretch` | Stretch | wheeled | 0.34 | 0.52 | 0.15–1.10 | 1.30 | 0.45 |
| `go2` | Go2 | quadruped | 0.42 | 0 (no arm) | – | 0.45 | 0.80 |
| `soarm` | SO-ARM100 | stationary | 0.18 | 0.36 | 0.05–0.40 | 0.35 | 0 |

The SO-ARM100 cannot navigate: it is placed at the door and must report that
it cannot reach anything that is not within 0.36 m of its base. Go2 can
navigate anywhere but has no manipulator; it can reach the target and then
must report the manipulation step as failed. Stretch can do both (counter
height 0.9 m is inside its arm range).

## Replay envelope (extends READ_API.md)

```json
{
  "environmentId": "kitchen",
  "environmentRevision": "<sha256 from GET /api/environments/kitchen>",
  "task": "wipe the table",
  "taskSlug": "wipe-the-table",
  "replay": {
    "version": 1,
    "source": "backend",
    "robot": "stretch",
    "duration": 31.5,
    "frames": [ {"time": 0, "position": [x,y,z], "yaw": 0, "armExtension": 0}, ... ],
    "task": "wipe the table",
    "events": [ {"time": 0, "label": "Spawned at the kitchen door"}, {"time": 4.2, "label": "Navigating to the table"} ],
    "verdict": {
      "status": "completed" | "partial" | "failed",
      "summary": "Stretch reached the table and wiped a 0.6 m × 0.4 m patch.",
      "steps": [ {"name": "Navigate to table", "status": "done" | "failed" | "skipped", "note": "..."} ],
      "planner": "rules" | "openai"
    },
    "landmarks": [ {"id": "table", "label": "Table", "position": [x,y,z], "size": [dx,dy,dz], "kind": "table"|"counter"|"sink"|"tall"|"door"} ]
  }
}
```

- `frames` keep the existing semantics (seconds, metres Z-up, yaw radians with
  0 = +Y, `armExtension` 0..1). Frames must strictly increase from 0 to
  `duration`, 10 Hz sampling. Positions are the robot body origin; use the
  same contact-height convention as `src/demo-placement.ts` (floor z ≈ 0, the
  frontend grounds against the mesh when available).
- `events`, `verdict`, `landmarks`, `task` are **additive**; the existing
  Python `validate_replay` and TypeScript `parseReplay` must keep accepting
  envelopes that lack them.
- Yaw must face the direction of travel while moving and face the target
  landmark while manipulating. Wiping = arm extends to ~0.8 then oscillates
  0.55–0.85 for ~6 s with small yaw sweeps (±0.15 rad). Filling a glass =
  navigate to the counter/sink, extend arm, hold 4 s ("Filling glass"),
  retract, navigate to the table, extend, retract ("Placed glass on table").
- Spawn point = the kitchen door. The door is the largest obstacle-free gap on
  the floor-box perimeter in `scene.json` (prefer the gap that opens onto free
  interior grid cells). Robots start 0.45 m inside the door facing the room.
- Paths are planned on a 0.05 m occupancy grid built from every enabled
  non-floor box in `scene.json`, inflated by the robot's footprint radius +
  0.05 m margin, with A* and then corner-cut smoothing. Never interpolate
  through obstacles.

## Landmarks

Boxes in `scene.json` are unlabeled columns. `pipeline/landmarks.py` clusters
adjacent obstacle columns (4-neighbour adjacency at the 0.22 m cell size) and
labels clusters by geometry:

- `table`: interior cluster (not touching the floor boundary within 0.3 m),
  top height 0.65–1.00 m, footprint ≥ 0.5 m²
- `counter`: boundary-touching cluster, top height 0.80–1.00 m
- `sink`: the counter cluster segment closest to the geometric room centre on
  the counter side opposite the door (heuristic; label it "Counter (sink
  assumed)")
- `tall`: top height > 1.6 m (cabinets / fridge)
- `door`: the spawn gap

**API usage policy.** The `OPENAI_API_KEY` in `.env` is reserved for the
Decisions API (`POST /v1/decisions`, `gpt-6-luna`): object matching in stage 2
and the per-step feasibility predicates ("can this robot reach a 0.9 m
surface?") recorded in `verdict.steps[].note`. Language-model reasoning
(choosing the step sequence, spotting objects in rendered views) is done by
the Codex CLI agents, not by API calls:

- `python -m pipeline.task_planner "<task>" --all --steps plan.json` takes a
  plan written by the CLI agent (`{"steps":[{"action":"navigate","target":"table"},...]}`,
  actions `navigate|wipe|fill|place`, targets from `python -m pipeline.landmarks`)
  and records `verdict.planner: "codex-cli"`. Without `--steps` the keyword
  rule planner is used (`"rules"`).
- `scripts/discover-objects.py render` writes camera views for the CLI agents
  to inspect; they write `output/discovery/detections.json`; `apply` raycasts,
  matches with Decisions, replaces and saves.
- A Responses-API planner exists behind `MORPHEUS_PLANNER=openai` for
  experiments only; it is off by default.

## HTTP routes (port 8000, proxied by Vite under `/api`)

- `GET /api/tasks` → `{version:1, tasks:[{slug, task, robots:{stretch:"/api/tasks/<slug>/stretch", go2:..., soarm:...}}]}` listing committed rollouts under `rollouts/kitchen/<slug>/<robot>.json` plus anything planned in this session.
- `GET /api/tasks/<slug>/<robot>` → the envelope above (404 when absent, 409 when its revision does not match the current environment).
- `POST /api/tasks/plan` body `{task:string, robot?:string}` → plans for one robot, or for all three when `robot` is omitted, returns `{version:1, slug, task, results:{stretch:envelope,...}}`, and writes `rollouts/kitchen/<slug>/<robot>.json` **and** `output/kitchen/replays/<robot>.json` (so the existing `GET /api/environments/kitchen` reports the latest plan as `available`). Task text max 200 chars, slugified `[a-z0-9-]`.
- `pipeline/read_api.py` serves `output/kitchen/replays/<robot>.json` first and falls back to `rollouts/kitchen/default/<robot>.json` so a fresh clone shows rollouts immediately.

CLI: `pipeline/.venv/bin/python -m pipeline.task_planner "wipe the table" --all` writes the same files. Pre-render both demo tasks for all robots before committing.

## Frontend behaviour

- The workspace (`/?view=workspace`; the demo page is the default route) gains a **Task** section: text input with preset chips
  ("Wipe the table", "Fill a glass of water"), **Run on this robot** and
  **Run on all robots** (POST `/api/tasks/plan`), and a list of prerendered
  tasks (GET `/api/tasks`). Selecting a task + robot loads its envelope via
  `parseBackendReplay`.
- **Robot POV**: a picture-in-picture panel (bottom-right of the viewport,
  ~34% width, 16:10) rendered every frame from a camera at the robot's eye
  height, 0.12 m forward of its origin, looking along yaw (and pitched down
  ~12°). Use `renderer.setScissorTest` + `setViewport` on the same canvas
  after the main render; keep the Spark sort stable by rendering the POV at
  reduced size and accepting the main camera's sort order. A button swaps the
  POV and the main view (POV fullscreen, orbit in the inset).
- **Fly camera**: `F` or the gamepad button toggles a creative-mode free
  camera (`src/fly.ts`): WASD along the look heading, Space/Shift up and
  down, drag or pointer-lock mouse look, wheel for speed, Esc exits. While it
  is on, the POV render loop drives it instead of OrbitControls.
- **Event captions**: show the most recent `events` label (time ≤ current
  time) as a subtitle over the viewport; show the verdict in the right panel
  with the step list; the compare modal shows all three robots' verdicts for
  the current task side by side.
- Nothing in the UI may call a rollout "validated". Labels: "Planned rollout
  · reviewed against collision geometry".
