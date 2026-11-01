# Local environment and replay read API

Run `pipeline/.venv/bin/python -m pipeline.server` on port 8000. Vite proxies `/api` there; the independent object service uses `/api/objects` on port 8001.

The prepared environment is `output/kitchen/scene.json`. Generate it with `npm run fetch-sample` and `npm run prepare-sample`. Unsaved builder edits are not this stored fixture. Replace the file atomically to publish a new local revision. No client-supplied filesystem paths are accepted.

- `GET /api/environments`: `{version:1,environments:[{id,name,revision,url}]}`.
- `GET /api/environments/kitchen`: `{version:1,id,revision,scene,validation,visual,replays}`.
- `GET /api/environments/kitchen/{scene.json,scene.xml,environment.xml,validation.json,bundle.zip}`: download the current revision's artifact.
- `GET /api/environments/kitchen/replays/{go2,soarm,stretch}`: a revision-bound replay envelope, or 404 if absent / 409 if invalid or stale.

`revision` is a SHA-256 of the normalized manifest serialized by Python with sorted keys and compact separators. Consumers should use the returned value rather than recomputing it. Native validation is computed from exactly that manifest and cached by revision; historical validation.json files on disk are not trusted as current evidence.

`visual` contains `splatUrl`, `meshUrl`, `sourceUpAxis:'Y-down'`, and `transform` from `scene.source.transform`. Assets are the official World Labs CDN URLs, not redistributed binaries. `transform.matrix` is a column-major raw-source-to-world matrix. Load raw assets, then call `RoomViewer.applyWorldTransform(matrix)` after loading and before setting world-space colliders. This replaces the viewer's provisional alignment; never apply it twice. Units are metres, Z up; scale remains assumed until measured.

`replays` is keyed by robot. Each entry has `status:'available'|'unavailable'|'invalid'`, `url:string|null`, and a `reason` when unavailable/invalid. Show an explicit empty state; do not substitute demonstration motion for a missing backend rollout.

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
