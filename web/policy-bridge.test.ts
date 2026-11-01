import assert from "node:assert/strict";
import test from "node:test";
import { createReplayBridge, ReplayContractError, sourceQuaternionToYUpXyzw, sourceVectorToYUp, validateReplayBundle, validateViewerEvidence } from "./policy-bridge.ts";

const hash = "a".repeat(64);
const bundle = {
  schema_version: "1.0",
  authority: "python-mujoco",
  hashes: { scene_json_sha256: hash, mjcf_sha256: hash, model_mjz_sha256: hash, policy_sha256: hash, evaluator_version: "mujoco-3.17.0" },
  robot: { id: "unitree_g1_12dof", joint_names: ["left_hip_pitch_joint"], body_names: ["pelvis"], nq: 3 },
  frame: { units: "m", up_axis: "Z", quaternion_order: "wxyz" },
  route: { points_m: [[0, 0, 0], [1, 0, 0]], target_id: "explicit_target" },
  samples: [{ time_s: 0, qpos: [0, 0, 0], body_poses: { pelvis: [0, 0, 0, 1, 0, 0, 0] } }, { time_s: 0.02, qpos: [1, 2, 3] }],
  events: [],
};

test("applies exact qpos then forward, without a physics step", () => {
  const calls: string[] = [];
  const bridge = createReplayBridge(bundle, { scene_json_sha256: hash, mjcf_sha256: hash, model_mjz_sha256: hash, policy_sha256: hash, robot_id: "unitree_g1_12dof", nq: 3 });
  bridge.applyExactSample(0, { setQpos: (qpos) => calls.push(`qpos:${qpos.join(",")}`), mjForward: () => calls.push("forward"), setExportedBodyPoses: () => calls.push("poses") });
  assert.deepEqual(calls, ["qpos:0,0,0", "forward", "poses"]);
});

test("rejects mismatched identity, qpos dimension, and non-monotonic time", () => {
  assert.throws(() => createReplayBridge(bundle, { scene_json_sha256: hash, mjcf_sha256: hash, model_mjz_sha256: hash, policy_sha256: hash, robot_id: "wrong", nq: 3 }), ReplayContractError);
  assert.throws(() => createReplayBridge(bundle, { scene_json_sha256: hash, mjcf_sha256: hash, model_mjz_sha256: "b".repeat(64), policy_sha256: hash, robot_id: "unitree_g1_12dof", nq: 3 }), /replay identity/);
  assert.throws(() => validateReplayBundle({ ...bundle, samples: [{ time_s: 0, qpos: [0, 0] }] }), /qpos must have 3 values/);
  assert.throws(() => validateReplayBundle({ ...bundle, samples: [{ time_s: 0.02, qpos: [0, 0, 0] }, { time_s: 0.01, qpos: [0, 0, 0] }] }), /strictly increasing/);
});

test("keeps report gate authority and has a single source-to-Y-up conversion", () => {
  assert.deepEqual(validateViewerEvidence({ execution: "completed", readiness: "PREVIEW_ONLY", hardware_readiness: "UNVERIFIED" }), { execution: "completed", readiness: "PREVIEW_ONLY", hardware_readiness: "UNVERIFIED" });
  assert.throws(() => validateViewerEvidence({ execution: "completed", readiness: "SIM_READY", hardware_readiness: "READY" }), ReplayContractError);
  assert.deepEqual(sourceVectorToYUp([2, 3, 4]), [2, 4, -3]);
  assert.deepEqual(sourceQuaternionToYUpXyzw([1, 0, 0, 0]).map((value) => Number(value.toFixed(12))), [0, 0, 0, 1]);
});

test("accepts every evaluator failure event kind", () => {
  const kinds = ["contact", "low_clearance", "fall", "contract_failure", "non_finite", "simulation_warning", "joint_limit", "joint_speed", "timeout", "actuator_limit"];
  const parsed = validateReplayBundle({ ...bundle, events: kinds.map((kind) => ({ start_s: 0, end_s: 0, kind, severity: "fail", object_ids: [], detail: kind })) });
  assert.deepEqual(parsed.events.map((event) => event.kind), kinds);
});
