# Robot workspace UI

The default `/` entry connects to the saved environment backend. The preparation app remains at `/?view=builder`; the SimReady editor is at `/?view=objects`.

## Stack and launch

React 19, TypeScript, Vite 7, Three.js 0.180, Spark 2.4, Lucide. No new dependencies. Reuse Vite on http://127.0.0.1:5174, environment service on8000, and object service on8001. See `docs/READ_API.md` for backend setup and contracts. Production continues fetching official source asset URLs; source binaries are not distributed.

## Connected environment

`backend.ts` validates API responses; `use-backend.ts` polls every15 seconds and supports manual refresh and environment selection. The workspace loads the saved scene, applies `visual.transform.matrix` exactly once to source visuals, and displays the already-world-aligned enabled colliders. The Collisions tab shows actual saved collider geometry. MuJoCo compile/contact results come from the backend; robot task compatibility stays Not evaluated. Missing visuals fall back explicitly to saved collision geometry. Backend failures are shown and pause playback. Export environment downloads the native backend bundle, including validation evidence.

## Robot replay handoff

The user confirmed their teammate has not produced a rollout yet. No robot is shown and play/scrub/save are disabled until a matching rollout exists. The separate **Open illustrative demo** action enables the frontend demonstration.

Publish `output/kitchen/replays/stretch.json`, `go2.json`, or `soarm.json` with this envelope (replace the revision with the current GET response):

```json
{
  "environmentId": "kitchen",
  "environmentRevision": "<exact current revision>",
  "replay": {
    "version": 1,
    "source": "backend",
    "robot": "stretch",
    "duration": 1,
    "frames": [
      { "time": 0, "position": [0, 0, 0], "yaw": 0, "armExtension": 0 },
      { "time": 1, "position": [0, 0, 0], "yaw": 0, "armExtension": 0 }
    ]
  }
}
```

This is a schema example, not an approved pose or generated rollout. The API serves valid files and reports availability; the UI picks up newly available replays automatically. Manual **Import replay** accepts the same envelope. Wrong environment/revision/robot and demo provenance are rejected. Frames span0..duration with strictly increasing times. Positions are already-aligned world metres, Z-up; yaw is radians about +Z with zero pointing along +Y. `armExtension` is normalized0..1 preview control. Do not transform or re-ground backend trajectories. Robot meshes remain procedural previews, not the teammate's calibrated model; no leg gait or autonomous arm animation is invented for backend frames. Arbitrary robot joint-state/model support is a future contract extension.

## SimReady saved objects

The workspace polls GET `/api/objects/scene` and reads the research editor's **Save to simulation workspace** result. `saved-objects.ts` requires matching source SHA256 and world transform, validates poses/masks, loads prepared GLBs, and hides the original selected splat boxes while preserving cutaway edits. World poses are applied directly. Collision mode hides these visual meshes because they are not part of the base environment's validated collision bundle or replay revision. The UI explicitly labels interactions unvalidated. The initial service contains no saved replacements. Exporting replacement physics remains the research editor's separate object-scene export.

## Demonstration and clipping correction

Demo motion uses `demo-placement.ts` floor probing and the corrected x1.95, y−1.2..1.2 aisle route. `robot-clearance.test.ts` reproduces the old collision and checks all robot parts at481 poses per model against the real source GLB with5mm clearance. It runs when local fixture assets exist, otherwise explicitly skips. This is sampled geometry regression, not continuous collision or task-success validation. Demo results remain illustrative and scale remains assumed.

## Verification

27 frontend tests pass, including stale-revision/robot rejection, backend HTTP failures, no invented backend gait, exact transforms, saved-object source/alignment/mask checks, and actual source-mesh clipping regression. TypeScript/Vite production build passes (existing large Spark/Three chunk warning). Browser verified connected room,111 saved collider shapes, missing-replay controls, native ZIP download with passed validation, demo playback, and clearing demo motion when switching back to backend. No real robot rollout is available for an end-to-end motion check yet.
