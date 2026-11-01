# Robot workspace UI

The default `/` entry is the mockup-based robot workspace. The existing environment preparation app stays available at `/?view=builder` and the environment owner's `/builder.html`.

## Stack and source ownership

React 19 + TypeScript + Vite 7, Three.js 0.180, Spark 2.4, Lucide icons. No new dependencies were added. `src/workspace.tsx` and `src/workspace.css` own the interface; `src/robot-scene.ts` adds procedural robot previews and a trajectory overlay to the existing `RoomViewer`. `src/replay.ts` validates and samples replay data. The source scan and mesh share the viewer's transform unchanged.

Run `npm run dev` from this directory and open the Vite URL (normally http://127.0.0.1:5174). The currently running development server can be reused. Run `npm run fetch-sample` only if the local sample assets are missing. Production requests the official World Labs asset URLs rather than bundling those binaries.

## Implemented interactions

- Automatic real 3D kitchen scan, orbit, pan, zoom, top view, camera reset, robot follow, full-screen request, and reversible roof cutaway.
- Stretch, Go2, and SO-ARM100 procedural preview models; Go2 leg motion; Stretch arm extension; stationary SO-ARM100 arm movement.
- Play/pause, scrubbing, 0.5×/1×/2× speed, route visibility, and Scan / Collisions / Reach views. Collisions currently shows the reference mesh, not evaluated robot contacts.
- Room renaming, task selection, robot comparison, preview report JSON export, replay JSON export/import, settings and keyboard controls.
- Responsive desktop / tablet / mobile layout, labeled controls, native modal focus handling, reduced-motion CSS, loading and error states.

## Replay handoff

Use the Import replay control, or pass a parsed replay to `RobotScene.setReplay`. The UI currently has no polling or WebSocket connection; a future transport should validate with `parseReplay` before calling `setReplay`.

```json
{
  "version": 1,
  "source": "backend",
  "robot": "stretch",
  "duration": 10,
  "frames": [
    { "time": 0, "position": [1.5, -1.5, 0.06], "yaw": 0, "armExtension": 0.1 },
    { "time": 10, "position": [1.5, 0.8, 0.06], "yaw": 0, "armExtension": 0.6 }
  ]
}
```

- `robot`: `stretch`, `go2`, or `soarm`.
- `time` and `duration`: seconds. Strictly increasing frames must span exactly 0 through duration.
- `position`: finite `[x,y,z]` in the already-aligned scene's Z-up frame, assumed metres for this sample. Do not apply `source.transform` twice.
- `yaw`: radians about +Z; zero points the preview robot along +Y. Interpolation takes the shortest angular path.
- `armExtension`: normalized 0..1 preview control, not a calibrated joint value.
- `source: backend` switches the compatibility panel to Not evaluated; importing motion never asserts task success.

## Current boundaries

The scan is the existing synthetic World Labs kitchen, not the exact room pictured in the mockup. The robot meshes and 24-second trajectories are frontend demos, not URDF models, physics simulation, or robot policies. The corrected demo paths are regression-checked against the actual reference mesh at 0.05-second samples using each model part’s bounding volume and 5 mm clearance; this is not a continuous physics or task-success validation. Reach shading and demo compatibility values are illustrative and identified as such in the UI and report. SO-ARM100 base motion remains stationary. Scene scale remains unverified. Reports contain `evaluationStatus: not-evaluated` and `scaleVerified: false`.

The room and selections are in-memory UI state; reload resets them. Save replay downloads trajectories, not a video. Native environment preparation and validation remain in the existing builder. Source asset licensing/provenance stays governed by `docs/ASSETS.md`.

## Verification

- Production TypeScript/Vite build succeeds (existing large Spark/Three chunk warning remains).
- 15 TypeScript tests pass, including replay interpolation, stationary base, malformed replay rejection, and angle wrapping.
- Browser checked at 1440×940 and 390×844. Mobile document width equals viewport width.
- Browser verified scene load, top view, visible robot replay, pointer scrub, model/profile switching, task exclusion, reference mesh, reach envelope, speed selection, modal comparison, and actual report download.

## Clipping correction

The sample route now follows x=1.95, y=-1.2..1.2, retaining full base and arm clearance. `demo-placement.ts` probes nine points under the model footprint against the aligned reference mesh, adjusts for the model contact origin, and retains a small allowance for scan roughness. Imported backend trajectories remain unchanged. The original x=1.5 route overlapped the counter and had an incorrect fixed ground height. `robot-clearance.test.ts` reproduces that collision and checks all three corrected models at 481 poses each against the source mesh. This source-backed regression runs when the local sample GLB and scene manifest are available; otherwise it is explicitly skipped.
