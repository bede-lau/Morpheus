# 90-second demonstration script

0–12 s: “Our teammate captures a home and supplies MJCF plus Scene JSON. We handle the robot policy and verification. We do not rebuild the environment.” Show the actual source caveats: synthetic, assumed scale, unreviewed.

12–28 s: Run `inspect` and show the compatibility matrix. “This G1 uses an official pretrained PPO/LSTM walking policy with its exact 12-joint model. Dogs need their own controller; arm reach is a separate calculation.”

28–43 s: Show `outputs/kitchen-detour/demo.mp4`. “The planner turns the supplied geometry into velocity commands and a 17-point detour. The learned controller drives actuators in MuJoCo. This is recorded physics, not a robot marker moved along a path.”

43–58 s: Show detour report: 0 forbidden contacts, target error ~0.204 m within the declared 0.25 m tolerance, and runtime. Show `outputs/detour-qualification/qualification.json`: 20/20 start-perturbation trials, including its limited coverage and PREVIEW_ONLY status.

58–72 s: Show the controller-search audit with separate train/holdout seeds. “We tune the route controller while freezing the actor. CPU execution suffices, so no Modal credits were used. Specialist training is gated on a measured need and a valid training setup.”

72–84 s: Show the bridge/model digests and tests. “Parallel Codex agents researched the real policy artifact, built adapters and tests, and caught an export bug plus failure-event mismatches before the demo.”

84–90 s: “The handoff plugs into our existing Three.js/Spark viewer. Physical home readiness still needs confirmed scale, sensing, calibration and commissioning.”

Before public submission: publish only the source/README/credits after a local secrets/data review; do not include a private home scan, environment bundle, credentials or a locally generated physics log. Upload the <=90-second recording and test its public link without authentication. Neither publishing nor uploading is completed by the local integration.
