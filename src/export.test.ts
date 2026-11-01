import test from 'node:test';
import assert from 'node:assert/strict';
import { strFromU8, unzipSync } from 'fflate';
import { buildEnvironmentFiles, createEnvironmentZip, validateManifest } from './export.ts';

function manifest() {
  return {
    version: 1, name: 'Kitchen & <sample>', units: 'm', upAxis: 'Z',
    source: { scaleConfirmed: false, asset: 'user-scan.spz' },
    objects: [{ id: 'floor', type: 'box', role: 'floor', position: [0, 0, -0.05], size: [4, 6, 0.1], quaternion: [2, 0, 0, 0] }],
  };
}

test('export normalizes quaternion, converts full sizes and escapes XML', () => {
  const input = manifest();
  const files = buildEnvironmentFiles(input);
  assert.match(files['environment.xml'], /name="env_floor"/);
  assert.match(files['environment.xml'], /size="2 3 0.05"/);
  assert.match(files['environment.xml'], /quat="1 0 0 0"/);
  assert.match(files['scene.xml'], /Kitchen &amp; &lt;sample&gt;/);
  assert.deepEqual(input.objects[0].quaternion, [2, 0, 0, 0]);
  assert.deepEqual(JSON.parse(files['scene.json']).objects[0].quaternion, [1, 0, 0, 0]);
});

test('schema checks reject malformed geometry and metadata', () => {
  const input = manifest();
  input.objects[0].size[0] = -1;
  input.objects[0].position[0] = Infinity;
  input.objects[0].quaternion = [0, 0, 0, 0];
  const errors = validateManifest(input);
  assert.ok(errors.some(error => error.includes('positive')));
  assert.ok(errors.some(error => error.includes('position')));
  assert.ok(errors.some(error => error.includes('quaternion')));
  assert.throws(() => createEnvironmentZip(input), /positive/);
  assert.ok(validateManifest({ ...manifest(), units: 'cm' }).some(error => error.includes('units')));
  assert.ok(validateManifest({ ...manifest(), source: { scale: NaN } }).some(error => error.includes('metadata')));
});

test('invalid and duplicate IDs cannot create conflicting XML names', () => {
  const input = manifest();
  input.objects.push({ ...input.objects[0] });
  assert.ok(validateManifest(input).some(error => error.includes('Duplicate')));
  input.objects[1].id = 'bad"/>';
  assert.ok(validateManifest(input).some(error => error.includes('.id')));
});

test('disabled objects remain in manifest but leave the physics scene', () => {
  const input = manifest();
  const files = buildEnvironmentFiles({ ...input, objects: [{ ...input.objects[0], enabled: false }] });
  assert.doesNotMatch(files['environment.xml'], /<geom /);
  assert.equal(JSON.parse(files['scene.json']).objects[0].enabled, false);
  assert.equal(JSON.parse(files['validation.json']).activeObjectCount, 0);
});

test('ZIP contains exactly portable text artifacts and explicitly unverified physics', () => {
  const files = unzipSync(createEnvironmentZip(manifest()));
  assert.deepEqual(Object.keys(files).sort(), ['README.txt', 'environment.xml', 'scene.json', 'scene.xml', 'validation.json'].sort());
  const validation = JSON.parse(strFromU8(files['validation.json']));
  assert.equal(validation.schemaValid, true);
  assert.equal(validation.status, 'unverified');
  assert.equal(validation.compile, 'not_run');
  assert.equal(validation.dropProbe.status, 'not_run');
  assert.match(strFromU8(files['README.txt']), /Metric scale has NOT been confirmed/);
});
