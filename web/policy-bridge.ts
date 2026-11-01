/**
 * A dependency-free handoff from the authoritative Python MuJoCo evaluator to
 * an existing viewer. This module creates no scene, geometry, or renderer.
 */

export type Vec3 = readonly [number, number, number];
export type QuaternionWxyz = readonly [number, number, number, number];
export type PoseWxyz = readonly [number, number, number, number, number, number, number];

export type ReplaySample = {
  time_s: number;
  qpos: number[];
  qvel?: number[];
  ctrl?: number[];
  command?: Vec3;
  body_poses?: Record<string, PoseWxyz>;
  metrics?: Record<string, number>;
};

export type ReplayEvent = {
  start_s: number;
  end_s: number;
  kind: "contact" | "low_clearance" | "fall" | "contract_failure" | "non_finite" | "simulation_warning" | "joint_limit" | "joint_speed" | "timeout" | "actuator_limit";
  severity: "warning" | "fail";
  object_ids: string[];
  detail: string;
};

export type PolicyReplayBundle = {
  schema_version: "1.0";
  authority: "python-mujoco";
  hashes: {
    scene_json_sha256: string;
    mjcf_sha256: string;
    model_mjz_sha256: string;
    policy_sha256: string;
    evaluator_version: string;
  };
  robot: { id: string; joint_names: string[]; body_names: string[]; nq: number };
  frame: { units: "m"; up_axis: "Z"; quaternion_order: "wxyz" };
  route: { points_m: Vec3[]; target_id: string };
  samples: ReplaySample[];
  events: ReplayEvent[];
};

export type ViewerEvidence = {
  execution: "completed" | "failed" | "unavailable";
  readiness: "PREVIEW_ONLY" | "SIM_READY" | "BLOCKED" | "UNSUPPORTED";
  hardware_readiness: "UNVERIFIED";
};

export type ExpectedReplayIdentity = {
  scene_json_sha256: string;
  mjcf_sha256: string;
  model_mjz_sha256: string;
  policy_sha256: string;
  robot_id: string;
  nq: number;
};

export type MuJoCoReplayTarget = {
  /** Set the complete, source-frame qpos vector without modifying it. */
  setQpos: (qpos: readonly number[]) => void;
  /** Recompute kinematics after setQpos; never advance physics in replay. */
  mjForward: () => void;
  /** Optional consumer for evaluator-exported source-frame body poses. */
  setExportedBodyPoses?: (poses: Readonly<Record<string, PoseWxyz>>) => void;
};

export class ReplayContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ReplayContractError";
  }
}

const SHA256 = /^[a-f0-9]{64}$/;
const FRAME_ROOT_ROTATION_WXYZ: QuaternionWxyz = [Math.SQRT1_2, -Math.SQRT1_2, 0, 0];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (!isRecord(value)) throw new ReplayContractError(`${path} must be an object`);
  return value;
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new ReplayContractError(`${path} must be a non-empty string`);
  }
  return value;
}

function requireFinite(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ReplayContractError(`${path} must be finite`);
  }
  return value;
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new ReplayContractError(`${path} must be an array`);
  return value;
}

function finiteVector(value: unknown, size: number, path: string): number[] {
  const vector = requireArray(value, path);
  if (vector.length !== size) throw new ReplayContractError(`${path} must have ${size} values`);
  return vector.map((entry, index) => requireFinite(entry, `${path}[${index}]`));
}

function finiteNumbers(value: unknown, path: string): number[] {
  return requireArray(value, path).map((entry, index) => requireFinite(entry, `${path}[${index}]`));
}

function stringArray(value: unknown, path: string): string[] {
  return requireArray(value, path).map((entry, index) => requireString(entry, `${path}[${index}]`));
}

function sha256(value: unknown, path: string): string {
  const digest = requireString(value, path);
  if (!SHA256.test(digest)) throw new ReplayContractError(`${path} must be a lowercase SHA-256 digest`);
  return digest;
}

function parseMetrics(value: unknown, path: string): Record<string, number> {
  const metrics = requireRecord(value, path);
  return Object.fromEntries(Object.entries(metrics).map(([key, entry]) => [key, requireFinite(entry, `${path}.${key}`)]));
}

function parseBodyPoses(value: unknown, path: string): Record<string, PoseWxyz> {
  const bodyPoses = requireRecord(value, path);
  return Object.fromEntries(Object.entries(bodyPoses).map(([name, pose]) => {
    const values = finiteVector(pose, 7, `${path}.${name}`);
    return [name, values as PoseWxyz];
  }));
}

/** Validate and return an immutable-shape replay bundle before it reaches a viewer. */
export function validateReplayBundle(value: unknown): PolicyReplayBundle {
  const root = requireRecord(value, "bundle");
  if (root.schema_version !== "1.0") throw new ReplayContractError("bundle.schema_version must be 1.0");
  if (root.authority !== "python-mujoco") throw new ReplayContractError("bundle.authority must be python-mujoco");

  const hashes = requireRecord(root.hashes, "bundle.hashes");
  const robot = requireRecord(root.robot, "bundle.robot");
  const frame = requireRecord(root.frame, "bundle.frame");
  const route = requireRecord(root.route, "bundle.route");

  if (frame.units !== "m" || frame.up_axis !== "Z" || frame.quaternion_order !== "wxyz") {
    throw new ReplayContractError("bundle.frame must declare m, Z, wxyz");
  }

  const nq = requireFinite(robot.nq, "bundle.robot.nq");
  if (!Number.isInteger(nq) || nq <= 0) throw new ReplayContractError("bundle.robot.nq must be a positive integer");
  const result: PolicyReplayBundle = {
    schema_version: "1.0",
    authority: "python-mujoco",
    hashes: {
      scene_json_sha256: sha256(hashes.scene_json_sha256, "bundle.hashes.scene_json_sha256"),
      mjcf_sha256: sha256(hashes.mjcf_sha256, "bundle.hashes.mjcf_sha256"),
      model_mjz_sha256: sha256(hashes.model_mjz_sha256, "bundle.hashes.model_mjz_sha256"),
      policy_sha256: sha256(hashes.policy_sha256, "bundle.hashes.policy_sha256"),
      evaluator_version: requireString(hashes.evaluator_version, "bundle.hashes.evaluator_version"),
    },
    robot: {
      id: requireString(robot.id, "bundle.robot.id"),
      joint_names: stringArray(robot.joint_names, "bundle.robot.joint_names"),
      body_names: stringArray(robot.body_names, "bundle.robot.body_names"),
      nq,
    },
    frame: { units: "m", up_axis: "Z", quaternion_order: "wxyz" },
    route: {
      points_m: requireArray(route.points_m, "bundle.route.points_m").map((point, index) => finiteVector(point, 3, `bundle.route.points_m[${index}]`) as Vec3),
      target_id: requireString(route.target_id, "bundle.route.target_id"),
    },
    samples: [],
    events: [],
  };

  let previousTime = -Infinity;
  result.samples = requireArray(root.samples, "bundle.samples").map((sample, index) => {
    const record = requireRecord(sample, `bundle.samples[${index}]`);
    const time_s = requireFinite(record.time_s, `bundle.samples[${index}].time_s`);
    if (time_s <= previousTime) throw new ReplayContractError("bundle.samples must have strictly increasing time_s");
    previousTime = time_s;
    const parsed: ReplaySample = { time_s, qpos: finiteVector(record.qpos, nq, `bundle.samples[${index}].qpos`) };
    if (record.qvel !== undefined) parsed.qvel = finiteNumbers(record.qvel, `bundle.samples[${index}].qvel`);
    if (record.ctrl !== undefined) parsed.ctrl = finiteNumbers(record.ctrl, `bundle.samples[${index}].ctrl`);
    if (record.command !== undefined) parsed.command = finiteVector(record.command, 3, `bundle.samples[${index}].command`) as Vec3;
    if (record.body_poses !== undefined) parsed.body_poses = parseBodyPoses(record.body_poses, `bundle.samples[${index}].body_poses`);
    if (record.metrics !== undefined) parsed.metrics = parseMetrics(record.metrics, `bundle.samples[${index}].metrics`);
    return parsed;
  });
  if (result.samples.length === 0) throw new ReplayContractError("bundle.samples must not be empty");

  result.events = requireArray(root.events, "bundle.events").map((event, index) => {
    const record = requireRecord(event, `bundle.events[${index}]`);
    const start_s = requireFinite(record.start_s, `bundle.events[${index}].start_s`);
    const end_s = requireFinite(record.end_s, `bundle.events[${index}].end_s`);
    if (end_s < start_s) throw new ReplayContractError(`bundle.events[${index}] ends before it starts`);
    if (!(["contact", "low_clearance", "fall", "contract_failure", "non_finite", "simulation_warning", "joint_limit", "joint_speed", "timeout", "actuator_limit"] as string[]).includes(String(record.kind))) {
      throw new ReplayContractError(`bundle.events[${index}].kind is unsupported`);
    }
    if (!(["warning", "fail"] as string[]).includes(String(record.severity))) {
      throw new ReplayContractError(`bundle.events[${index}].severity is unsupported`);
    }
    return {
      start_s, end_s,
      kind: record.kind as ReplayEvent["kind"], severity: record.severity as ReplayEvent["severity"],
      object_ids: stringArray(record.object_ids, `bundle.events[${index}].object_ids`),
      detail: requireString(record.detail, `bundle.events[${index}].detail`),
    };
  });
  return result;
}

/** Never infer or upgrade a result from visual replay. Pass evaluator evidence through unchanged. */
export function validateViewerEvidence(value: unknown): ViewerEvidence {
  const evidence = requireRecord(value, "report");
  const execution = evidence.execution;
  const readiness = evidence.readiness;
  if (!(["completed", "failed", "unavailable"] as unknown[]).includes(execution)) throw new ReplayContractError("report.execution is invalid");
  if (!(["PREVIEW_ONLY", "SIM_READY", "BLOCKED", "UNSUPPORTED"] as unknown[]).includes(readiness)) throw new ReplayContractError("report.readiness is invalid");
  if (evidence.hardware_readiness !== "UNVERIFIED") throw new ReplayContractError("report.hardware_readiness must be UNVERIFIED");
  return { execution: execution as ViewerEvidence["execution"], readiness: readiness as ViewerEvidence["readiness"], hardware_readiness: "UNVERIFIED" };
}

function identitiesMatch(bundle: PolicyReplayBundle, expected: ExpectedReplayIdentity): boolean {
  return bundle.hashes.scene_json_sha256 === expected.scene_json_sha256
    && bundle.hashes.mjcf_sha256 === expected.mjcf_sha256
    && bundle.hashes.model_mjz_sha256 === expected.model_mjz_sha256
    && bundle.hashes.policy_sha256 === expected.policy_sha256
    && bundle.robot.id === expected.robot_id
    && bundle.robot.nq === expected.nq;
}

export function createReplayBridge(bundleValue: unknown, expected: ExpectedReplayIdentity) {
  const bundle = validateReplayBundle(bundleValue);
  if (!identitiesMatch(bundle, expected)) throw new ReplayContractError("replay identity does not match the loaded scene, model, policy, or nq");
  return {
    bundle,
    applyExactSample(index: number, target: MuJoCoReplayTarget): ReplaySample {
      if (!Number.isInteger(index) || index < 0 || index >= bundle.samples.length) {
        throw new ReplayContractError("replay sample index is outside the bundle");
      }
      const sample = bundle.samples[index];
      target.setQpos(sample.qpos);
      target.mjForward();
      if (sample.body_poses && target.setExportedBodyPoses) target.setExportedBodyPoses(sample.body_poses);
      return sample;
    },
  };
}

/** Attach source-authored robot nodes beneath this one root when a viewer is Y-up. */
export const sourceFrameRootRotationWxyz = (): QuaternionWxyz => FRAME_ROOT_ROTATION_WXYZ;

/** For standalone overlay data only; robot nodes under the shared root retain source coordinates. */
export function sourceVectorToYUp([x, y, z]: Vec3): Vec3 {
  return [x, z, -y];
}

/** Basis-change a standalone source-frame quaternion; returns Three ordering x,y,z,w. */
export function sourceQuaternionToYUpXyzw(source: QuaternionWxyz): readonly [number, number, number, number] {
  const [w, x, y, z] = normalize(source);
  const rotated = multiply(multiply(FRAME_ROOT_ROTATION_WXYZ, [w, x, y, z]), [Math.SQRT1_2, Math.SQRT1_2, 0, 0]);
  return [rotated[1], rotated[2], rotated[3], rotated[0]];
}

function normalize([w, x, y, z]: QuaternionWxyz): QuaternionWxyz {
  const norm = Math.hypot(w, x, y, z);
  if (!Number.isFinite(norm) || norm === 0) throw new ReplayContractError("quaternion must be finite and non-zero");
  return [w / norm, x / norm, y / norm, z / norm];
}

function multiply(a: QuaternionWxyz, b: QuaternionWxyz): QuaternionWxyz {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw];
}
