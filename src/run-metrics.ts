import { sampleReplay, type Landmark, type Replay, type RobotId, type Verdict } from './replay';
import type { Vec3 } from './types';

/**
 * Post-run telemetry for the run report: kinematics, clearance against the reviewed landmark boxes,
 * phase timeline, per-step outcomes and a capability profile. Pure functions over the replay envelope
 * so the dashboard can be unit-tested without a renderer.
 */
export type Phase = { kind: 'navigate' | 'manipulate' | 'blocked' | 'idle'; start: number; end: number; label: string };
export type Sample = { t: number; x: number; y: number; speed: number; yawRate: number; accel: number; arm: number; clearance: number };
export type StepRow = { name: string; status: 'done' | 'failed' | 'skipped'; note?: string; start?: number; end?: number };
export type Capability = { axis: string; score: number; detail: string };
export type Feedback = { kind: 'fix' | 'keep' | 'switch'; title: string; detail: string };
export type RunMetrics = {
  robot: RobotId; task: string; status: Verdict['status'] | 'unknown'; summary: string; planner: string; duration: number;
  pathLength: number; straightLine: number; pathEfficiency: number; meanSpeed: number; peakSpeed: number; movingFraction: number; stops: number;
  totalTurn: number; jerkRms: number; accelRms: number; minClearance: number | null; meanClearance: number | null; nearMisses: number; nearMissTime: number;
  armDuty: number; armPeak: number; manipulationTime: number; navigationTime: number; blockedTime: number;
  samples: Sample[]; phases: Phase[]; steps: StepRow[]; capabilities: Capability[]; score: number; feedback: Feedback[];
  landmarks: Landmark[]; firstFailure: { time: number; label: string; position: [number, number] } | null;
};
/** Shared robot profile (docs/TASKS.md): footprint radius, arm reach and height window, nominal speed. */
export const ROBOT_PROFILE: Record<RobotId, { radius: number; reach: number; armHeight: [number, number] | null; speed: number; name: string }> = {
  stretch: { radius: .34, reach: .52, armHeight: [.15, 1.10], speed: .45, name: 'Stretch' },
  go2: { radius: .42, reach: 0, armHeight: null, speed: .80, name: 'Go2' },
  g1: { radius: .30, reach: .50, armHeight: [.30, 1.45], speed: .60, name: 'Unitree G1' },
};
/** Clearance under this while moving at walking pace counts as a near miss (approach stops end just outside it by design). */
export const NEAR_MISS_M = .10;
/** Physical body radius used for clearance (the profile radius above is the planner's inflation radius): Stretch 3 base, Go2 body, G1 stance. */
export const BODY_RADIUS: Record<RobotId, number> = { stretch: .20, go2: .25, g1: .15 };
const HZ = 10;
const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
/** 2D distance from a point to an axis-aligned landmark box (0 inside). */
export function boxDistance(x: number, y: number, l: Landmark) {
  const dx = Math.max(l.position[0] - l.size[0] / 2 - x, 0, x - (l.position[0] + l.size[0] / 2));
  const dy = Math.max(l.position[1] - l.size[1] / 2 - y, 0, y - (l.position[1] + l.size[1] / 2));
  return Math.hypot(dx, dy);
}
/** Base-level obstacles: tall clusters are mostly walls, the door is the spawn gap, objects sit on surfaces. */
const OBSTACLES = (l: Landmark) => l.kind !== 'tall' && l.kind !== 'door' && l.kind !== 'object';
/** Phase per event label; the planner's event vocabulary (pipeline/task_planner.py). */
export function classifyEvent(label: string): Phase['kind'] {
  const s = label.toLowerCase();
  if (/cannot|out of reach|no collision-free|stationary|beyond/.test(s)) return 'blocked';
  if (/navigating/.test(s)) return 'navigate';
  if (/wiping|filling|picked|placed|holding|carrying|wipe|fill/.test(s)) return 'manipulate';
  return 'idle';
}
export function buildPhases(replay: Replay): Phase[] {
  const events = [...(replay.events ?? [])].sort((a, b) => a.time - b.time);
  const phases: Phase[] = [];
  for (let i = 0; i < events.length; i++) {
    const e = events[i], end = i + 1 < events.length ? events[i + 1].time : replay.duration;
    if (end <= e.time) continue;
    phases.push({ kind: classifyEvent(e.label), start: e.time, end, label: e.label });
  }
  if (!phases.length) phases.push({ kind: replay.frames.some((f, i) => i && Math.hypot(f.position[0] - replay.frames[0].position[0], f.position[1] - replay.frames[0].position[1]) > .05) ? 'navigate' : 'idle', start: 0, end: replay.duration, label: 'Replay' });
  return phases;
}
function rms(values: number[]) { return values.length ? Math.sqrt(values.reduce((s, v) => s + v * v, 0) / values.length) : 0; }
export function computeRunMetrics(replay: Replay): RunMetrics {
  const profile = ROBOT_PROFILE[replay.robot], landmarks = replay.landmarks ?? [], spawn = replay.frames[0].position, body = BODY_RADIUS[replay.robot];
  // Boxes that already contain the spawn point are the door gap or an over-wide counter cluster, not something the robot can clear.
  const obstacles = landmarks.filter(OBSTACLES).filter(l => boxDistance(spawn[0], spawn[1], l) > 0);
  const n = Math.max(2, Math.round(replay.duration * HZ) + 1), dt = replay.duration / (n - 1);
  const raw = Array.from({ length: n }, (_, i) => sampleReplay(replay, i * dt));
  const samples: Sample[] = raw.map((f, i) => {
    const prev = raw[Math.max(0, i - 1)], next = raw[Math.min(n - 1, i + 1)], span = (Math.min(n - 1, i + 1) - Math.max(0, i - 1)) * dt || dt;
    const speed = Math.hypot(next.position[0] - prev.position[0], next.position[1] - prev.position[1]) / span;
    const yawRate = wrap(next.yaw - prev.yaw) / span;
    const clearance = obstacles.length ? Math.max(0, Math.min(...obstacles.map(l => boxDistance(f.position[0], f.position[1], l))) - body) : NaN;
    return { t: i * dt, x: f.position[0], y: f.position[1], speed, yawRate, accel: 0, arm: f.armExtension, clearance };
  });
  for (let i = 0; i < n; i++) { const a = samples[Math.max(0, i - 1)], b = samples[Math.min(n - 1, i + 1)]; samples[i].accel = (b.speed - a.speed) / (((Math.min(n - 1, i + 1) - Math.max(0, i - 1)) * dt) || dt); }
  const jerk = samples.map((s, i) => { const a = samples[Math.max(0, i - 1)], b = samples[Math.min(n - 1, i + 1)]; return (b.accel - a.accel) / (((Math.min(n - 1, i + 1) - Math.max(0, i - 1)) * dt) || dt); });
  let pathLength = 0; for (let i = 1; i < n; i++) pathLength += Math.hypot(samples[i].x - samples[i - 1].x, samples[i].y - samples[i - 1].y);
  const straightLine = Math.hypot(samples[n - 1].x - samples[0].x, samples[n - 1].y - samples[0].y);
  const phases = buildPhases(replay);
  // Path efficiency: straight-line over travelled distance inside each navigation phase (1 = no detour).
  const nav = phases.filter(p => p.kind === 'navigate').map(p => {
    const inside = samples.filter(s => s.t >= p.start && s.t <= p.end); if (inside.length < 2) return null;
    let len = 0; for (let i = 1; i < inside.length; i++) len += Math.hypot(inside[i].x - inside[i - 1].x, inside[i].y - inside[i - 1].y);
    const direct = Math.hypot(inside.at(-1)!.x - inside[0].x, inside.at(-1)!.y - inside[0].y);
    return len > .05 ? Math.min(1, direct / len) : null;
  }).filter((v): v is number => v !== null);
  const pathEfficiency = nav.length ? nav.reduce((a, b) => a + b, 0) / nav.length : (pathLength > .05 ? Math.min(1, straightLine / pathLength) : 1);
  const moving = samples.filter(s => s.speed > .02);
  let stops = 0, wasMoving = false; for (const s of samples) { if (s.speed > .02) wasMoving = true; else if (wasMoving) { stops++; wasMoving = false; } }
  const totalTurn = samples.reduce((a, s) => a + Math.abs(s.yawRate) * dt, 0);
  const clear = samples.map(s => s.clearance).filter(Number.isFinite);
  let nearMisses = 0, nearMissTime = 0, inMiss = false;
  for (const s of samples) { const near = Number.isFinite(s.clearance) && s.clearance < NEAR_MISS_M && s.speed > .08; if (near) { nearMissTime += dt; if (!inMiss) nearMisses++; } inMiss = near; }
  const sumKind = (k: Phase['kind']) => phases.filter(p => p.kind === k).reduce((a, p) => a + (p.end - p.start), 0);
  const steps: StepRow[] = (replay.verdict?.steps ?? []).map(s => ({ name: s.name, status: s.status, note: s.note }));
  // Attach phase windows to steps in order: each navigate step takes the next navigate phase, each manipulation step the next manipulate/blocked phase.
  let ni = 0, mi = 0; const navPhases = phases.filter(p => p.kind === 'navigate'), manPhases = phases.filter(p => p.kind === 'manipulate' || p.kind === 'blocked');
  for (const s of steps) { const p = /^navigate/i.test(s.name) ? navPhases[ni++] : manPhases[mi++]; if (p) { s.start = p.start; s.end = p.end; } }
  const blocked = phases.find(p => p.kind === 'blocked');
  const firstFailure = blocked ? { time: blocked.start, label: blocked.label, position: (() => { const f = sampleReplay(replay, blocked.start); return [f.position[0], f.position[1]] as [number, number]; })() } : null;
  const done = steps.filter(s => s.status === 'done').length, stepScore = steps.length ? done / steps.length : (replay.verdict?.status === 'completed' ? 1 : .5);
  const minClearance = clear.length ? Math.min(...clear) : null, meanClearance = clear.length ? clear.reduce((a, b) => a + b, 0) / clear.length : null;
  const clearanceScore = minClearance === null ? .6 : Math.max(0, Math.min(1, (minClearance + .05) / .35));
  const jerkRms = rms(jerk), accelRms = rms(samples.map(s => s.accel));
  const smoothness = Math.max(0, Math.min(1, 1 - jerkRms / 8));
  const score = Math.round(100 * (stepScore * .55 + clearanceScore * .2 + pathEfficiency * .1 + smoothness * .15));
  const surfaces = landmarks.filter(l => l.kind === 'table' || l.kind === 'counter' || l.kind === 'sink').map(l => l.size[2]);
  const inRange = profile.armHeight ? surfaces.filter(h => h >= profile.armHeight![0] && h <= profile.armHeight![1]).length : 0;
  const manipulationSteps = steps.filter(s => !/^navigate/i.test(s.name)), navSteps = steps.filter(s => /^navigate/i.test(s.name));
  const capabilities: Capability[] = [
    { axis: 'Navigation', score: navSteps.length ? navSteps.filter(s => s.status === 'done').length / navSteps.length : (profile.speed > 0 ? 1 : 0), detail: `${navSteps.filter(s => s.status === 'done').length}/${navSteps.length} navigation steps` },
    { axis: 'Reach', score: Math.min(1, profile.reach / .5), detail: `${profile.reach.toFixed(2)} m arm reach vs 0.50 m counter approach` },
    { axis: 'Height window', score: surfaces.length && profile.armHeight ? inRange / surfaces.length : (profile.armHeight ? 1 : 0), detail: profile.armHeight ? `${inRange}/${surfaces.length || 0} surfaces inside ${profile.armHeight[0].toFixed(2)}–${profile.armHeight[1].toFixed(2)} m` : 'No manipulator' },
    { axis: 'Clearance', score: clearanceScore, detail: minClearance === null ? 'No reviewed obstacles' : `${(minClearance * 100).toFixed(0)} cm minimum margin` },
    { axis: 'Tempo', score: Math.min(1, profile.speed / .8), detail: `${profile.speed.toFixed(2)} m/s nominal, ${(moving.length ? moving.reduce((a, s) => a + s.speed, 0) / moving.length : 0).toFixed(2)} m/s achieved` },
    { axis: 'Manipulation', score: manipulationSteps.length ? manipulationSteps.filter(s => s.status === 'done').length / manipulationSteps.length : (profile.reach ? 1 : 0), detail: `${manipulationSteps.filter(s => s.status === 'done').length}/${manipulationSteps.length} manipulation steps` },
  ];
  const feedback: Feedback[] = [];
  for (const s of steps.filter(s => s.status === 'failed')) {
    const note = s.note ?? ''; const alt = (Object.keys(ROBOT_PROFILE) as RobotId[]).filter(id => id !== replay.robot);
    if (/no arm|no manipulator/i.test(note)) feedback.push({ kind: 'switch', title: `“${s.name}” needs a manipulator`, detail: `${profile.name} has no arm. Re-run with ${alt.filter(id => ROBOT_PROFILE[id].reach > 0).map(id => ROBOT_PROFILE[id].name).join(' or ')}; the planner keeps the same route graph.` });
    else if (/reach|beyond|outside/i.test(note)) feedback.push({ kind: 'fix', title: `Approach closer for “${s.name}”`, detail: `${note} Tighten the approach gap or pick a robot with longer reach.` });
    else if (/collision-free|path/i.test(note)) feedback.push({ kind: 'fix', title: `Clear a corridor for “${s.name}”`, detail: `${note} Lower the planner inflation radius (${profile.radius.toFixed(2)} m) or move the flagged clutter in the scan.` });
    else feedback.push({ kind: 'fix', title: `Revisit “${s.name}”`, detail: note || 'Step failed without a recorded reason.' });
  }
  if (nearMisses) feedback.push({ kind: 'fix', title: `${nearMisses} near-miss${nearMisses > 1 ? 'es' : ''} under ${(NEAR_MISS_M * 100).toFixed(0)} cm`, detail: `${nearMissTime.toFixed(1)} s spent closer than ${(NEAR_MISS_M * 100).toFixed(0)} cm to reviewed geometry. Raise the planner clearance margin before a hardware trial.` });
  if (pathEfficiency < .8 && nav.length) feedback.push({ kind: 'fix', title: 'Route detours', detail: `Path efficiency ${(pathEfficiency * 100).toFixed(0)}%. Smooth the A* path or re-review the obstacle boxes that force the detour.` });
  if (!feedback.length) feedback.push({ kind: 'keep', title: 'Rollout clean', detail: `All ${steps.length} steps done with ${minClearance === null ? 'no obstacle contact' : `${(minClearance * 100).toFixed(0)} cm minimum clearance`}. Promote this plan to the comparison set.` });
  return {
    robot: replay.robot, task: replay.task ?? '', status: replay.verdict?.status ?? 'unknown', summary: replay.verdict?.summary ?? '', planner: replay.verdict?.planner ?? (replay.source === 'demo' ? 'illustrative' : 'backend'), duration: replay.duration,
    pathLength, straightLine, pathEfficiency, meanSpeed: moving.length ? moving.reduce((a, s) => a + s.speed, 0) / moving.length : 0, peakSpeed: Math.max(0, ...samples.map(s => s.speed)), movingFraction: moving.length / n, stops,
    totalTurn, jerkRms, accelRms, minClearance, meanClearance, nearMisses, nearMissTime,
    armDuty: samples.filter(s => s.arm > .05).length / n, armPeak: Math.max(0, ...samples.map(s => s.arm)), manipulationTime: sumKind('manipulate'), navigationTime: sumKind('navigate'), blockedTime: sumKind('blocked'),
    samples, phases, steps, capabilities, score, feedback, landmarks, firstFailure,
  };
}
/** Bounds of route plus landmarks for the top-down map. */
export function mapBounds(m: Pick<RunMetrics, 'samples' | 'landmarks'>) {
  const xs: number[] = [], ys: number[] = [];
  for (const s of m.samples) { xs.push(s.x); ys.push(s.y); }
  for (const l of m.landmarks.filter(OBSTACLES)) { xs.push(l.position[0] - l.size[0] / 2, l.position[0] + l.size[0] / 2); ys.push(l.position[1] - l.size[1] / 2, l.position[1] + l.size[1] / 2); }
  if (!xs.length) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  const pad = .3; return { minX: Math.min(...xs) - pad, maxX: Math.max(...xs) + pad, minY: Math.min(...ys) - pad, maxY: Math.max(...ys) + pad };
}
export const position = (f: { position: Vec3 }) => f.position;
