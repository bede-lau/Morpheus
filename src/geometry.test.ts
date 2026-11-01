import test from 'node:test';
import assert from 'node:assert/strict';
import { computeBounds, estimateFloorZ, proposeColliders, scaleFromMeasurement, transformPoints } from './geometry.ts';

function roomPoints(): Float32Array {
  const points: number[] = [];
  for (let x = 0; x <= 4; x += 0.1) for (let y = 0; y <= 4; y += 0.1) {
    points.push(x, y, 0);
    // Counter along one wall; open central aisle must remain open.
    if (x < 0.5) points.push(x, y, 0.9);
  }
  return new Float32Array(points);
}

test('bounds ignore invalid points and suppress distant floaters', () => {
  const points = new Float32Array([...roomPoints(), 1e6, 1e6, 1e6, NaN, 0, 0]);
  const bounds = computeBounds(points);
  assert.ok(bounds.max[0] < 5);
  assert.ok(bounds.max[2] < 1);
  assert.equal(estimateFloorZ(points), 0);
});

test('floor and obstacle proposals preserve an open central aisle', () => {
  const colliders = proposeColliders(roomPoints(), { floorZ: 0, cellSize: 0.25 });
  assert.equal(colliders[0].role, 'floor');
  assert.equal(colliders[0].position[2] + colliders[0].size[2] / 2, 0);
  assert.ok(colliders.some(c => c.role === 'obstacle'));
  for (const box of colliders.filter(c => c.role === 'obstacle')) {
    assert.ok(box.position[0] + box.size[0] / 2 < 1);
    assert.ok(box.position[2] + box.size[2] / 2 >= 0.9);
    assert.ok(box.size.every(n => n > 0 && Number.isFinite(n)));
  }
});

test('box budget coarsens without returning more boxes than requested', () => {
  const points: number[] = [...roomPoints()];
  for (let x = 0; x < 4; x += 0.2) for (let y = 0; y < 4; y += 0.2) {
    points.push(x, y, 0.4 + ((Math.round(x * 5) + Math.round(y * 5)) % 3) * 0.2);
  }
  const colliders = proposeColliders(new Float32Array(points), { maxBoxes: 8, cellSize: 0.1, minPoints: 1 });
  assert.ok(colliders.length <= 8);
  assert.ok(colliders.length > 1);
});

test('known ceiling samples are excluded', () => {
  const points = new Float32Array([...roomPoints(), 2, 2, 2.5, 2.01, 2, 2.5, 2, 2.01, 2.5]);
  const colliders = proposeColliders(points, { ceilingZ: 2.5, trimFraction: 0, minPoints: 1 });
  assert.ok(colliders.filter(c => c.role === 'obstacle').every(c => c.position[2] + c.size[2] / 2 < 2));
});

test('measurement and transform agree on metric scale and quaternion rotation', () => {
  assert.equal(scaleFromMeasurement([0, 0, 0], [2, 0, 0], 1), 0.5);
  const points = transformPoints(new Float32Array([2, 0, 0]), 0.5, [0, 0, 2], [Math.SQRT1_2, 0, 0, Math.SQRT1_2]);
  assert.ok(Math.abs(points[0]) < 1e-6);
  assert.ok(Math.abs(points[1] - 1) < 1e-6);
  assert.equal(points[2], 2);
});

test('invalid input and calibration fail clearly', () => {
  assert.throws(() => computeBounds(new Float32Array()), /No finite/);
  assert.throws(() => computeBounds(new Float32Array([0, 0])), /triples/);
  assert.throws(() => scaleFromMeasurement([0, 0, 0], [0, 0, 0], 1), /distinct/);
  assert.throws(() => proposeColliders(roomPoints(), { cellSize: 0 }), /Invalid/);
  assert.throws(() => proposeColliders(roomPoints(), { maxBoxes: 1 }), /Invalid/);
  assert.throws(() => proposeColliders(roomPoints(), { floorZ: 100 }), /above/);
});
