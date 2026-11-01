# Policy replay bridge

`policy-bridge.ts` is the boundary between the authoritative Python MuJoCo calculation and the existing Three.js + Spark viewer. It does not construct the Gaussian splat, scene, robot asset, or user interface.

Call `createReplayBridge(bundle, expectedIdentity)`, then bind its `applyExactSample` target to the existing MuJoCo WASM wrapper:

```ts
const bridge = createReplayBridge(replay, expected);
bridge.applyExactSample(index, {
  setQpos: (qpos) => mujoco.setQpos(qpos),
  mjForward: () => mujoco.mjForward(),
  setExportedBodyPoses: (poses) => overlay.setVerifiedBodyPoses(poses),
});
```

This writes the evaluated qpos and calls forward kinematics only. It must not call `mj_step`, rerun the controller, or turn a viewer result into readiness. The replay identity includes the exported `model_mjz_sha256`, linking the exact rendered asset to the evidence. Display `validateViewerEvidence(report)` exactly as supplied by the evaluator.

Place source-frame robot and physics overlays below one Y-up root with `sourceFrameRootRotationWxyz()`. The supplied scan transform is separate and must not be applied again. Use `sourceVectorToYUp` and `sourceQuaternionToYUpXyzw` only for standalone viewer primitives that cannot live under the shared root.

Run the dependency-free contract tests with:

```sh
node --experimental-strip-types --test web/policy-bridge.test.ts
```
