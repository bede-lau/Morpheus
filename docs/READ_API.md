# Local environment and replay read API

Run `pipeline/.venv/bin/python -m pipeline.server` on port 8000. Vite proxies `/api` there; the independent object service uses `/api/objects` on port 8001.

The prepared environment is `output/kitchen/scene.json`. Generate it with `npm run fetch-sample` and `npm run prepare-sample`. Unsaved builder edits are not this stored fixture. Replace the file atomically to publish a new local revision. No client-supplied filesystem paths are accepted.

- `GET /api/environments`: `{version:1,environments:[{id,name,revision,url}]}`.
- `GET /api/environments/kitchen`: `{version:1,id,revision,scene,validation,visual,replays}`.
- `GET /api/environments/kitchen/{scene.json,scene.xml,environment.xml,validation.json,bundle.zip}`: download the current revision's artifact.
- `GET /api/environments/kitchen/replays/{go2,g1,stretch}`: a revision-bound replay envelope, or 404 if absent / 409 if invalid or stale.

`revision` is a SHA-256 of the normalized manifest serialized by Python with sorted keys and compact separators. Consumers should use the returned value rather than recomputing it. Native validation is computed from exactly that manifest and cached by revision; historical validation.json files on disk are not trusted as current evidence.

`visual` contains `splatUrl`, `meshUrl`, `sourceUpAxis:'Y-down'`, and `transform` from `scene.source.transform`. Assets are the official World Labs CDN URLs, not redistributed binaries. `transform.matrix` is a column-major raw-source-to-world matrix. Load raw assets, then call `RoomViewer.applyWorldTransform(matrix)` after loading and before setting world-space colliders. This replaces the viewer's provisional alignment; never apply it twice. Units are metres, Z up; scale remains assumed until measured.

`replays` is keyed by robot. Each entry has `status:'available'|'unavailable'|'invalid'`, `url:string|null`, and a `reason` when unavailable/invalid. Show an explicit empty state; do not substitute demonstration motion for a missing backend rollout.

`visual.sourceSha256` is the pinned SHA-256 of the full SPZ file served by `visual.splatUrl`, verified against the official downloaded sample and the fetch script checksum. It is separate from the scene manifest and does not change `revision`. For saved object replacements, require both matching source hash and matching source-to-world transform; matching the file alone does not establish alignment. The current sample hash is `351f5f2ceb6ee9095a4b4100f864db9780a21cc6bdb8a70c6f88117cbb1d5fe0`.

To supply a teammate rollout, atomically write `output/kitchen/replays/<robot>.json`:

```json
{
  "environmentId": "kitchen",
  "environmentRevision": "COPY_FROM_GET_ENVIRONMENT",
  "replay": {
    "version": 1,
    "source": "backend",
    "robot": "go2",
    "duration": 1,
    "frames": [
      {"time": 0, "position": [0,0,0.3], "yaw": 0, "armExtension": 0},
      {"time": 1, "position": [0.1,0,0.3], "yaw": 0, "armExtension": 0}
    ]
  }
}
```

The snippet documents the format only; it is not installed as a real rollout. Time is seconds, yaw radians, position is the model-specific body origin in the exported world, and armExtension is a normalized visual control in [0,1]. Frames strictly increase from zero to duration. The existing simplified viewer cannot reproduce arbitrary joint trajectories or a full physics state. No robot meshes, joint layout, policy weights, or trained rollout have been supplied. `source:'backend'` is producer metadata, not proof of simulation, successful tasks, or physical validity.

A revision change invalidates existing rollouts until rerun for that scene. Compilation/contact evidence says nothing about task completion, scan accuracy, or robot suitability. Health and POST validate/export routes remain available.

## Task stage

The "tell the robot what to do" stage is specified in [docs/TASKS.md](TASKS.md). `pipeline/task_planner.py` (CLI: `pipeline/.venv/bin/python -m pipeline.task_planner "wipe the table" --all`) labels landmarks from the collision boxes (`pipeline/landmarks.py`), plans an A* path on a 0.05 m grid inflated by each robot's footprint + 0.05 m, and writes envelopes with additive `task`, `events`, `verdict` and `landmarks` fields to `rollouts/kitchen/<slug>/<robot>.json` and `output/kitchen/replays/<robot>.json`. Routes: `GET /api/tasks`, `GET /api/tasks/<slug>/<robot>`, `POST /api/tasks/plan` `{task, robot?}`. When `output/kitchen/replays/<robot>.json` is absent, the replay route falls back to the committed `rollouts/kitchen/default/<robot>.json` ("wipe the table"). The planner runs offline with rules; with `OPENAI_API_KEY` set it asks the Responses API for the step sequence (validated, `verdict.planner:"openai"`, rules fallback on any error, 25 s budget; set `MORPHEUS_PLANNER=rules` or pass `--rules` to force rules). Rollouts are planned against collision boxes only; they are not physics simulations or evidence of task success.
