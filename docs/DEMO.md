# 90-second demonstration

Before presenting: `npm run fetch-sample`, `npm run dev`, `python -m pipeline.server` (8000) and `python -m pipeline.object_server` (8001). Open **http://127.0.0.1:5174/?view=demo**. Everything the page shows is real recorded data from this repository: the scan, the terra detections, the luna Decisions traces, the SimReady meshes, the MuJoCo validation and the planned rollouts.

**Problem statement.** Robots are coming into homes. The question becomes *which* robot is right for *my* home. Morpheus captures your room, reconstructs it for simulation, swaps the things a robot can touch for real simulation assets, and lets several robots attempt your chores inside it so you can choose.

| Time | Action | What to say |
| --- | --- | --- |
| 0–8 s | Empty white canvas. Click **Rustic kitchen**. The Gaussian splat blooms into the room. | "Start from a phone scan of your kitchen. Half a million gaussians." |
| 8–14 s | Click **MORPH TO SIM**. Phase 01 scan sweep. | "Morpheus fits the floor, samples the surfaces and finds the room interior." |
| 14–22 s | Phase 02: sixteen views, terra brackets the objects. | "Terra looks at the room from sixteen angles and spots everything a robot could pick up." |
| 22–32 s | Phase 03: luna Decisions cards. | "For each object the Decisions API classifies it and scores every candidate in NVIDIA's SimReady catalogue. Only confident matches are accepted." |
| 32–40 s | Phase 04 swap, phase 05 physics. | "The matched meshes drop in, the scan is masked underneath, and the collision scene compiles in MuJoCo." |
| 40–75 s | Phase 06: Stretch wipes the table, picks up the swapped bowl and carries it to the sink. POV inset. | "Now the part that matters: a robot in your kitchen, doing your chore, touching a real object. This is the robot's own camera." |
| 75–90 s | **Compare robots**. | "Same kitchen, same chore, three robots, one honest verdict each. That is how you choose." |

## Evidence of Codex contribution

Codex built the Three.js/Spark workbench, the collision-geometry proposal and MuJoCo export/validation, the SimReady conversion and Decisions matching service, the object discovery fan-out (Astra subagents through the Codex CLI, or terra through the Responses API), the mesh-based landmark labelling and task planner, the POV renderer, the morph-to-sim demo page, and the tests that check every rollout against the room mesh.

## Demo boundaries

The kitchen is a synthetic World Labs sample with a companion mesh, not a measured scan of a real home, and its scale is assumed. Landmarks are inferred from geometry; the sink position was pinned by a person after looking at the scan. Rollouts are planned motions reviewed against collision geometry, not physics simulations, trained policies or proof of task success. Robot bodies are preview models. The API key is used only for the Decisions API; terra detection uses a separate terra-only key, and Astra runs through the Codex CLI login.
