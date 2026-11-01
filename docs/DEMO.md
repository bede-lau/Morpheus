# 90-second demonstration

Before presenting: `npm run fetch-sample`, start `npm run dev`, `python -m pipeline.server` (8000) and, if stage 2 is shown live, `python -m pipeline.object_server` (8001). Rollouts for both demo tasks are already committed under `rollouts/kitchen/`; run the planner again only if the scene changes. Pause at actual loading or verdicts rather than narrating a result early.

**Problem statement.** Robots are coming into homes. The question becomes *which* robot is right for *my* home. Morpheus captures your room, reconstructs it for simulation, and lets several robots attempt your chores inside it so you can choose.

| Time | Action | What to say |
| --- | --- | --- |
| 0–10 s | Open `/?view=builder`, orbit the kitchen scan, toggle **Collisions**. | "We start from a Gaussian splat of a kitchen. Morpheus proposes editable collision geometry, validates it in MuJoCo, and exports a scene any robot stack can load." |
| 10–25 s | Open `/?view=objects`, show a replaced mug/bowl (pre-replaced if the key is offline). | "Objects a robot can interact with are swapped for real NVIDIA SimReady meshes. With the OpenAI Decisions API we rank which catalogue asset matches the selected splats; the original scan stays intact underneath." |
| 25–45 s | Open `/`, pick **Wipe the table**, robot **Stretch**, press play. Point at the POV inset and the event captions. | "Now the part we care about. A robot spawns at the door and is told to wipe the table. The planner labels the table, counters and door from the reconstructed geometry, plans a collision-free route, and the inset is the robot's own camera." |
| 45–60 s | Switch to **Go2**, play. Then **SO-ARM100**. | "Same task, same kitchen, different robot. The quadruped gets there but has nothing to wipe with. The desk arm cannot leave the door. The verdict panel says exactly which step failed." |
| 60–75 s | Pick **Fill a glass of water**, Stretch, play; swap POV to fullscreen for the pour. | "A two-stop task: counter first, then the table. Still collision-free, still from the robot's eyes." |
| 75–90 s | Click **Compare robots**. | "Three robots, one home, one honest answer per task. That is how you choose. Everything here is reproducible from the repository: scan in, verdict out." |

## Evidence of Codex contribution

Codex built the Three.js/Spark workbench, the collision-geometry proposal and MuJoCo export/validation, the SimReady conversion and Decisions matching service, the landmark labelling and task planner, the POV renderer, and the tests that check every rollout against the reference mesh. Run the repository's checks before claiming a test pass.

## Demo boundaries

The kitchen is a synthetic World Labs sample with a companion mesh, not a measured scan of a real home, and its scale is assumed. Landmarks are inferred from collision boxes, not recognised semantically. Rollouts are planned motions reviewed against collision geometry, not physics simulations, trained policies, or proof of task success. Robot bodies are preview models, not URDFs. The Decisions and Responses API calls are optional and the UI records whether a plan came from rules or from the model.
