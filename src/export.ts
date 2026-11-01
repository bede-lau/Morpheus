import { strToU8, zipSync } from 'fflate';
import type { SceneManifest } from './types';

type JsonObject = Record<string, unknown>;
type NormalizedCollider = JsonObject & {
  id: string; type: 'box'; position: number[]; size: number[]; quaternion: number[];
  friction?: number[]; enabled?: boolean; role?: string;
};
type NormalizedScene = JsonObject & { name: string; objects: NormalizedCollider[] };

const isObject = (value: unknown): value is JsonObject => value !== null && typeof value === 'object' && !Array.isArray(value);
const isVector = (value: unknown, length: number): value is number[] => Array.isArray(value)
  && value.length === length && value.every(v => typeof v === 'number' && Number.isFinite(v));

function finiteJson(value: unknown, ancestors = new Set<object>()): boolean {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (typeof value !== 'object' || ancestors.has(value)) return false;
  ancestors.add(value);
  const valid = Object.values(value).every(child => finiteJson(child, ancestors));
  ancestors.delete(value);
  return valid;
}

/** Schema checks only. This function does not compile MJCF or run a simulation. */
export function validateManifest(manifest: unknown): string[] {
  const errors: string[] = [];
  if (!isObject(manifest)) return ['Scene must be a JSON object.'];
  if (manifest.version !== 1) errors.push('version must be 1.');
  if (manifest.units !== 'm' || manifest.upAxis !== 'Z') errors.push('Expected units="m" and upAxis="Z"; calibrate before export.');
  if (typeof manifest.name !== 'string' || !manifest.name.trim()) errors.push('name must be a non-empty string.');
  if (!Array.isArray(manifest.objects) || !manifest.objects.length) errors.push('Add at least one collision object before export.');
  else {
    if (manifest.objects.length > 10000) errors.push('At most 10000 primitive colliders are supported.');
    const ids = new Set<string>();
    manifest.objects.forEach((object: unknown, index: number) => {
      const path = `objects[${index}]`;
      if (!isObject(object)) { errors.push(`${path} must be an object.`); return; }
      if (typeof object.id !== 'string' || !/^[A-Za-z][A-Za-z0-9_-]{0,95}$/.test(object.id)) {
        errors.push(`${path}.id must start with a letter and use letters, digits, _ or - (max 96 characters).`);
      } else {
        if (ids.has(object.id)) errors.push(`Duplicate object id: ${object.id}.`);
        ids.add(object.id);
      }
      if (object.type !== 'box') errors.push(`${path}.type must be "box".`);
      if ('enabled' in object && typeof object.enabled !== 'boolean') errors.push(`${path}.enabled must be boolean.`);
      if (!isVector(object.position, 3)) errors.push(`${path}.position must contain 3 finite numbers.`);
      if (!isVector(object.size, 3) || object.size.some(v => v <= 0)) errors.push(`${path}.size must contain 3 positive full dimensions in metres.`);
      const quaternion = object.quaternion ?? [1, 0, 0, 0];
      if (!isVector(quaternion, 4) || Math.hypot(...quaternion) < 1e-12 || !Number.isFinite(Math.hypot(...quaternion))) {
        errors.push(`${path}.quaternion must contain 4 finite numbers with a nonzero finite norm (w x y z).`);
      }
      if ('friction' in object && (!isVector(object.friction, 3) || object.friction.some(v => v < 0))) {
        errors.push(`${path}.friction must contain 3 nonnegative finite numbers.`);
      }
    });
  }
  if (!finiteJson(manifest)) errors.push('Scene metadata must be finite JSON, without cycles or undefined values.');
  return errors;
}

function normalizedScene(manifest: unknown): NormalizedScene {
  const errors = validateManifest(manifest);
  if (errors.length) throw new Error(errors.join('\n'));
  const scene = JSON.parse(JSON.stringify(manifest)) as NormalizedScene;
  scene.objects.forEach(object => {
    const q = object.quaternion ?? [1, 0, 0, 0];
    const length = Math.hypot(...q);
    object.quaternion = q.map(v => v / length);
  });
  return scene;
}

const escapeXml = (value: string) => value.replaceAll('&', '&amp;').replaceAll('"', '&quot;')
  .replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll("'", '&apos;');
const numbers = (values: number[]) => values.map(v => Number(v.toPrecision(12)).toString()).join(' ');

/** Portable collision-only package. Original splats and third-party meshes are not redistributed. */
export function buildEnvironmentFiles(manifest: SceneManifest | unknown): Record<string, string> {
  const scene = normalizedScene(manifest);
  const colors: Record<string, string> = {
    floor: '0.72 0.75 0.72 1', wall: '0.85 0.85 0.81 1', surface: '0.27 0.55 0.45 1',
  };
  const active = scene.objects.filter(object => object.enabled !== false);
  const geoms = active.map(object => {
    const attrs: Record<string, string> = {
      name: `env_${object.id}`, type: 'box', pos: numbers(object.position),
      size: numbers(object.size.map(v => v / 2)), quat: numbers(object.quaternion),
      rgba: colors[object.role ?? 'obstacle'] ?? '0.52 0.60 0.62 1',
      friction: numbers(object.friction ?? [0.8, 0.005, 0.0001]),
      contype: '1', conaffinity: '1', group: '0',
    };
    return `    <geom ${Object.entries(attrs).map(([key, value]) => `${key}="${escapeXml(value)}"`).join(' ')} />`;
  }).join('\n');
  const source = isObject(scene.source) ? scene.source : {};
  const warnings = [
    'Browser export checks the schema only. Native MuJoCo compilation and physics have NOT been tested.',
    source.scaleConfirmed === true
      ? 'Scale marked confirmed by the user; independently check reference measurements before training.'
      : 'Metric scale has NOT been confirmed. Verify a known reference distance before training.',
    'Automatic collider proposals require visual review. Obstacle columns fill space below observed surfaces; under-table openings may be blocked.',
  ];
  const validation = {
    status: 'unverified', schemaValid: true, compile: 'not_run',
    dropProbe: { status: 'not_run' }, objectCount: scene.objects.length,
    activeObjectCount: active.length, warnings,
  };
  return {
    'environment.xml': `<mujocoinclude>\n  <worldbody>\n${geoms}\n  </worldbody>\n</mujocoinclude>\n`,
    'scene.xml': `<mujoco model="${escapeXml(scene.name)}">\n  <compiler angle="radian" />\n  <option timestep="0.002" gravity="0 0 -9.81" />\n  <include file="environment.xml" />\n</mujoco>\n`,
    'scene.json': JSON.stringify(scene, null, 2) + '\n',
    'validation.json': JSON.stringify(validation, null, 2) + '\n',
    'README.txt': [
      'Morpheus collision environment', '',
      'Coordinates: right-handed, metres, Z up. scene.json box sizes are FULL dimensions; MJCF sizes are half-extents.',
      'Quaternions: w x y z. Collider coordinates are already in the calibrated world frame.',
      'source.transform records visual asset alignment; do not apply it again to colliders.', '',
      "Standalone: mujoco.MjModel.from_xml_path('scene.xml')",
      'Integration: add <include file="environment.xml"/> inside your robot\'s <mujoco> root.',
      'Environment geoms use the env_ prefix and collision masks contype=1 conaffinity=1.',
      'Remove or reconcile the robot scene\'s existing floor; do not add overlapping floors.',
      'All environment geoms are static. No robot, policy, moving furniture, visual splat, or third-party asset is bundled.',
      'Geometry and friction are approximations unless independently calibrated.', '',
      'VALIDATION AND ASSUMPTIONS', ...warnings.map(warning => `- ${warning}`), '',
      'Use the native Python validation command described in the project README to compile this environment and run a drop probe.',
      'A passing drop probe checks basic floor support, not reconstruction accuracy or robot task feasibility.', '',
      'SOURCE PROVENANCE', JSON.stringify(source, null, 2), '',
    ].join('\n'),
  };
}

/** No network calls. Only the manifest and generated collision files enter the ZIP. */
export function createEnvironmentZip(manifest: SceneManifest | unknown): Uint8Array {
  const files = buildEnvironmentFiles(manifest);
  return zipSync(Object.fromEntries(Object.entries(files).map(([name, text]) => [name, strToU8(text)])), { level: 6 });
}
