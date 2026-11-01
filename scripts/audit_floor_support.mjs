#!/usr/bin/env node
/**
 * Compare mesh ray support with the supplied Scene JSON's conservative
 * footprint clearance.  This is sampled evidence, not a continuous proof of
 * traversability, collision freedom, or metric calibration.
 *
 * Usage:
 * node scripts/audit_floor_support.mjs --scene scene.json --mesh room.glb \
 *   --output outputs/input-floor-audit.json [--grid-step .1] [--envelope .45]
 */

import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const DEFAULTS = { gridStep: 0.1, envelope: 0.45, bodyHeight: 1.4, rayOriginZ: 0.25, rayFar: 1, examples: 24 };

function usage() {
  return 'Usage: node scripts/audit_floor_support.mjs --scene scene.json --mesh room.glb --output audit.json [--grid-step .1] [--envelope .45] [--body-height 1.4] [--ray-origin-z .25] [--ray-far 1]';
}

function argumentsFrom(argv) {
  if (argv.length === 1 && (argv[0] === '--help' || argv[0] === '-h')) return null;
  const values = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    if (!key?.startsWith('--') || argv[index + 1] === undefined) throw new Error(usage());
    values[key.slice(2)] = argv[index + 1];
  }
  for (const required of ['scene', 'mesh', 'output']) if (!values[required]) throw new Error(usage());
  const number = (key, fallback) => {
    const value = values[key] === undefined ? fallback : Number(values[key]);
    if (!Number.isFinite(value) || value <= 0) throw new Error(`--${key} must be finite and positive`);
    return value;
  };
  return {
    scene: values.scene, mesh: values.mesh, output: values.output,
    gridStep: number('grid-step', DEFAULTS.gridStep), envelope: number('envelope', DEFAULTS.envelope),
    bodyHeight: number('body-height', DEFAULTS.bodyHeight), rayOriginZ: number('ray-origin-z', DEFAULTS.rayOriginZ),
    rayFar: number('ray-far', DEFAULTS.rayFar), examples: DEFAULTS.examples,
  };
}

function sha256(bytes) { return createHash('sha256').update(bytes).digest('hex'); }
function finiteVector(value, count, label) {
  if (!Array.isArray(value) || value.length !== count || !value.every(Number.isFinite)) throw new Error(`${label} must be ${count} finite numbers`);
  return value.map(Number);
}
function normalizedQuaternion(value, label) {
  const quaternion = finiteVector(value ?? [1, 0, 0, 0], 4, label);
  const magnitude = Math.hypot(...quaternion);
  if (magnitude === 0) throw new Error(`${label} must be non-zero`);
  return quaternion.map(item => item / magnitude);
}
function parseScene(bytes) {
  const document = JSON.parse(bytes.toString('utf8'));
  if (document?.version !== 1 || document.units !== 'm' || document.upAxis !== 'Z') throw new Error('scene must be v1, metres, Z-up');
  const transform = document.source?.transform;
  if (transform?.matrixOrder !== 'column-major') throw new Error('scene source transform must declare column-major order');
  const matrix = finiteVector(transform?.matrix, 16, 'source.transform.matrix');
  const objects = document.objects?.filter(object => object?.enabled !== false).map(object => ({
    id: String(object.id), role: String(object.role ?? 'obstacle'), position: finiteVector(object.position, 3, `object ${object.id}.position`),
    size: finiteVector(object.size, 3, `object ${object.id}.size`), quaternion: normalizedQuaternion(object.quaternion, `object ${object.id}.quaternion`),
  }));
  if (!Array.isArray(objects)) throw new Error('scene must contain objects');
  const floors = objects.filter(object => object.role === 'floor');
  if (floors.length !== 1) throw new Error('scene must have exactly one enabled floor');
  if (objects.some(object => object.size.some(value => value <= 0))) throw new Error('scene dimensions must be positive');
  return { matrix, floor: floors[0], obstacles: objects.filter(object => object.role !== 'floor') };
}

function rotationMatrix([w, x, y, z]) {
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}
function xyCorners(box) {
  const matrix = rotationMatrix(box.quaternion), half = box.size.map(value => value / 2), result = [];
  for (const x of [-half[0], half[0]]) for (const y of [-half[1], half[1]]) for (const z of [-half[2], half[2]]) {
    result.push([box.position[0] + matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
      box.position[1] + matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z]);
  }
  return result;
}
function signedDistanceAxisBox(point, minimumX, minimumY, maximumX, maximumY) {
  const dx = Math.max(minimumX - point[0], 0, point[0] - maximumX);
  const dy = Math.max(minimumY - point[1], 0, point[1] - maximumY);
  if (dx || dy) return Math.hypot(dx, dy);
  return -Math.min(point[0] - minimumX, maximumX - point[0], point[1] - minimumY, maximumY - point[1]);
}
function signedDistanceInsideFloor(point, floor) {
  const [w, x, y, z] = floor.quaternion;
  if (Math.abs(x) > 1e-6 || Math.abs(y) > 1e-6) throw new Error('tilted floors are unsupported by the matching planner contract');
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const cosine = Math.cos(yaw), sine = Math.sin(yaw), dx = point[0] - floor.position[0], dy = point[1] - floor.position[1];
  const localX = cosine * dx + sine * dy, localY = -sine * dx + cosine * dy;
  const halfX = floor.size[0] / 2, halfY = floor.size[1] / 2;
  const outsideX = Math.max(Math.abs(localX) - halfX, 0), outsideY = Math.max(Math.abs(localY) - halfY, 0);
  return outsideX || outsideY ? -Math.hypot(outsideX, outsideY) : Math.min(halfX - Math.abs(localX), halfY - Math.abs(localY));
}
function blocksNavigation(box, floor, bodyHeight) {
  return box.position[2] - box.size[2] / 2 < floor.position[2] + floor.size[2] / 2 + bodyHeight;
}
function clearance(point, floor, obstacles, envelope, bodyHeight) {
  let available = signedDistanceInsideFloor(point, floor) - envelope;
  for (const obstacle of obstacles) {
    if (!blocksNavigation(obstacle, floor, bodyHeight)) continue;
    const corners = xyCorners(obstacle), xs = corners.map(corner => corner[0]), ys = corners.map(corner => corner[1]);
    available = Math.min(available, signedDistanceAxisBox(point, Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)) - envelope);
  }
  return available;
}
function bounds(floor) {
  const corners = xyCorners(floor), xs = corners.map(corner => corner[0]), ys = corners.map(corner => corner[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}
function sample(record, limit) { return record.length < limit ? record : record.slice(0, limit); }

async function main() {
  const options = argumentsFrom(process.argv.slice(2));
  if (options === null) {
    process.stdout.write(`${usage()}\n`);
    return;
  }
  const sceneBytes = fs.readFileSync(options.scene), meshBytes = fs.readFileSync(options.mesh);
  const source = parseScene(sceneBytes);
  const { scene: room } = await new GLTFLoader().parseAsync(meshBytes.buffer.slice(meshBytes.byteOffset, meshBytes.byteOffset + meshBytes.byteLength), '');
  room.applyMatrix4(new THREE.Matrix4().fromArray(source.matrix));
  room.updateMatrixWorld(true);
  const [minimumX, minimumY, maximumX, maximumY] = bounds(source.floor);
  const raycaster = new THREE.Raycaster();
  const counts = { sampled: 0, ray_supported: 0, envelope_clear: 0, both: 0, ray_only: 0, clear_only: 0, neither: 0 };
  const examples = { both: [], ray_only: [], clear_only: [] };
  for (let x = minimumX; x <= maximumX + 1e-9; x += options.gridStep) for (let y = minimumY; y <= maximumY + 1e-9; y += options.gridStep) {
    const point = [Number(x.toFixed(9)), Number(y.toFixed(9))];
    raycaster.set(new THREE.Vector3(point[0], point[1], options.rayOriginZ), new THREE.Vector3(0, 0, -1));
    raycaster.far = options.rayFar;
    const rayHit = raycaster.intersectObject(room, true)[0];
    const clearanceM = clearance(point, source.floor, source.obstacles, options.envelope, options.bodyHeight);
    const supported = Boolean(rayHit), clear = clearanceM > 0;
    counts.sampled += 1; if (supported) counts.ray_supported += 1; if (clear) counts.envelope_clear += 1;
    const evidence = { xy_m: point, ...(supported ? { ray_height_m: Number(rayHit.point.z.toFixed(6)) } : {}), clearance_m: Number(clearanceM.toFixed(6)) };
    if (supported && clear) { counts.both += 1; if (examples.both.length < options.examples) examples.both.push(evidence); }
    else if (supported) { counts.ray_only += 1; if (examples.ray_only.length < options.examples) examples.ray_only.push(evidence); }
    else if (clear) { counts.clear_only += 1; if (examples.clear_only.length < options.examples) examples.clear_only.push(evidence); }
    else counts.neither += 1;
  }
  const classification = counts.both === 0 ? 'no_sampled_mesh_support_with_positive_envelope_clearance' : 'sampled_overlap_found';
  const report = {
    schema_version: '1.0', authority: 'input-floor-support-audit',
    inputs: { scene_sha256: sha256(sceneBytes), mesh_sha256: sha256(meshBytes) },
    frame: { units: 'm', up_axis: 'Z', source_transform_matrix_order: 'column-major' },
    sampling: { bounds_xy_m: [minimumX, minimumY, maximumX, maximumY], grid_step_m: options.gridStep, ray_origin_z_m: options.rayOriginZ, ray_far_m: options.rayFar, envelope_radius_m: options.envelope, body_height_m: options.bodyHeight, method: 'downward GLB ray plus planner-equivalent conservative box clearance' },
    counts, classification,
    examples,
    limitation: 'This finite grid samples downward mesh-ray support and static envelope clearance only. It is not continuous route proof, mesh collision validation, floor semantic classification, metric calibration, locomotion validation, or hardware readiness evidence.',
  };
  fs.mkdirSync(path.dirname(options.output), { recursive: true });
  fs.writeFileSync(options.output, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ classification, counts, scene_sha256: report.inputs.scene_sha256, mesh_sha256: report.inputs.mesh_sha256 })}\n`);
}

main().catch(error => { process.stderr.write(`floor-support audit failed: ${error.message}\n`); process.exitCode = 1; });
