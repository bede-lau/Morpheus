# 90-second demonstration

Before presenting, run `node scripts/fetch-sample.mjs` for the local sample, start the frontend and native validation service, and check the complete flow once. The sample must not be silently bundled into a public release; see [ASSETS.md](ASSETS.md). Pause at actual loading or validation completion rather than narrating a result early.

| Time | Action | What to say |
| --- | --- | --- |
| 0–12 s | Open **sample kitchen**; orbit the room. | “A scan looks like a room, but a robot simulator also needs solid geometry. This is a synthetic World Labs kitchen with a companion mesh, used to test our pipeline.” |
| 12–25 s | Inspect **Scale & alignment**, toggle **Reference mesh**. | “We keep the visual scan and geometry aligned. Scale is still assumed here. For a real capture, we select two points and enter a measured distance.” Do not invent a measurement for this synthetic fixture. |
| 25–40 s | Click **Generate collision shapes**, then **Collisions**. | “Morpheus proposes editable collision shapes from the companion mesh. The proposal is approximate: columns can fill open space beneath furniture.” |
| 40–58 s | Select one obstructing shape, move or resize it; show the scan again. | “We review the physical layer against the room and correct false obstacles. These are geometry proposals, not automatic object recognition.” |
| 58–75 s | Mark the shapes reviewed, click **Validate in MuJoCo**. | If the report passes: “MuJoCo compiled this environment and its floor-contact probe passed.” If unavailable: “Native validation is offline; the export will record that it was not run.” |
| 75–90 s | Click **Export environment**, inspect the downloaded bundle. | “We hand off MJCF collision geometry, a scene manifest, provenance, and validation status. Our teammate’s robot training pipeline consumes this environment. Robot behavior and task success have not been evaluated here.” |

## Evidence of Codex contribution

Codex created the Three.js/Spark interface, authored the editable collision-geometry and export workflow, researched the official sample assets, and verified binary integrity and splat/mesh coordinate alignment. The recorded asset check compares 20,000 splat centers with mesh vertices; it is documented in [ASSETS.md](ASSETS.md). The frontend and Python validation source provide inspectable implementation evidence. Run the repository's checks before claiming a test pass; a recorded code change is not evidence that every interaction works.

## Demo boundaries

The kitchen walkthrough is mesh-assisted and synthetic. It does not demonstrate arbitrary splat-to-mesh reconstruction, exact furniture retrieval, semantic segmentation, verified real-world scale, or robot policy training. Export status must distinguish a schema-checked bundle from a scene actually compiled and probed by MuJoCo.
