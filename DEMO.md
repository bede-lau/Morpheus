# Demo: policy learning and evidence in under 90 seconds

Use the existing workspace at the Vite URL. Generated artifacts are local and ignored by Git. The room is the teammate's supplied synthetic collision export; scale/review remain unconfirmed.

1. **0–15 s — choose a robot.** Select G1, Stretch or Go2. Show the 16 held-out local-goal results: untrained 0%, route imitation 31%/19%, then PPO 100%. State that these are learned static-footprint trajectories; only G1 has native locomotion evidence.
2. **15–30 s — compare paths.** Use Top view. Select the primary route and two alternatives; point out their measured lengths and clearance. The learned rollout is separate. The source audit warning explains why collision view is the default.
3. **30–45 s — preview a task.** Select Stretch and “Wipe demonstration patch,” then “Sweep demonstration patch.” These show numerical tool intentions; no tool contact or cleaning effect has been verified. SO-ARM100's base remains fixed. Go2/G1 have no cleaning controller.
4. **45–70 s — inspect real physics.** Select G1 and “Replay native G1 dynamics.” Play/scrub exact recorded body samples; open the report or the full-model 1080p video. The final composite learned/local + A*/safety controller completed 20/20 fresh detour starts, zero forbidden contacts, minimum clearance 0.10535 m against a 0.10 m gate.
5. **70–85 s — show the limit and contribution.** The supplied occupied columns block visible floor-supported space. Show the audit and preserve PREVIEW_ONLY/hardware UNVERIFIED. Codex found that input problem, trained/compared actual PPO weights, caught native checkpoint provenance gaps, and implemented the stop gates.

## Recorded local evidence

- `outputs/learning-robust/{g1,stretch,go2}/reduced_order_ppo.{pt,json}` — final profile-bound local policies and all three training stages.
- `outputs/learning-robust/g1-native-detour-v2/{report,replay,manifest,qualification}.json` — final filtered learned-controller native episode plus 20-start result, fresh native seed 20007.
- Same directory: `preview-1080p.png`, `demo-1080p.mp4` — 1920×1080 still and 12-second H264/24fps replay; the simulated episode is compressed into the clip duration.
- `outputs/learning-robust/g1-native-final/qualification.json` — 20/20 straight-task starts with the prior 0.03 m braking buffer; final detour wrapper uses 0.06 m.
- `outputs/input-floor-audit.json` — source/mesh hashes, 4,556 samples, 973 mesh-ray-supported, zero overlap at 0.45 m envelope.
- `outputs/learning/g1*`, `outputs/learning-robust/g1-native-detour*` — retain initial weak learning, failed unfiltered detour and 19/20 intermediate safety result. Do not present these as passing qualifications.
- `public/policies/` — final viewer adapters, source manifest, audit, and copied native evidence/video.

## Reproduce the final detour

After installation and training in README, run the following for this supplied fixture only. Use a fresh output directory; source geometry is unchanged.

```sh
uv run morpheus-learn validate-g1 "$SCENE_BUNDLE" --output outputs/home --native-output outputs/home/g1-detour --episodes 20 --heldout-seed 20007 --start=-2.529208821881345,-1.660230821881345 --goal=-1.3292088218813451,-1.2602308218813452
uv run --with pillow --with imageio --with imageio-ffmpeg python scripts/render_replay.py outputs/home/g1-detour
uv run morpheus-learn export "$SCENE_BUNDLE" --output outputs/home --native-artifacts outputs/home/g1-detour --floor-audit outputs/input-floor-audit.json --demo-cleaning
```

The source repository already exists on GitHub; local integration is on `integration/home-policies`. A hosted deployment, narrated screen recording, public video upload and hackathon submission have not been performed. The native 12-second clip is ready to include in the screen recording. Preserve credits from THIRD_PARTY.md and docs/ASSETS.md; exclude private scans, credentials and generated room data from the public repository.
