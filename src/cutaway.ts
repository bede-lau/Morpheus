import * as THREE from 'three';
import { SplatEdit, SplatEditRgbaBlendMode, SplatEditSdf, SplatEditSdfType } from '@sparkjsdev/spark';
import type { SplatMesh } from '@sparkjsdev/spark';

/**
 * Display-only world-Z cutaway. It never changes source splats, samples,
 * collision proposals, or exported geometry.
 *
 * Spark 2.4 applies SplatEdits after the source object's world transform.
 * A PLANE SDF has distance local z; invert=true selects the upper half-space.
 * This edit stays detached from the scene graph, so its coordinates are world
 * coordinates and it does not become a global edit on unrelated splat meshes.
 * https://sparkjs.dev/docs/splat-editing/
 *
 * Reference mesh rendering can use `material.clippingPlanes = cutaway.planes`
 * with `renderer.localClippingEnabled = true`. Reset to [] when disabling.
 */
export function createSplatCutaway(splat: SplatMesh, initialHeight: number) {
  if (!Number.isFinite(initialHeight)) throw new Error('Cutaway height must be finite.');
  const edit = new SplatEdit({
    name: 'Morpheus display cutaway',
    rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY,
    softEdge: 0.025,
  });
  const sdf = new SplatEditSdf({
    type: SplatEditSdfType.PLANE,
    opacity: 0,
    invert: true,
  });
  // Attach only to the detached edit so updateMatrixWorld propagates normally.
  edit.add(sdf);
  const clippingPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), initialHeight);
  const originalImplicit = splat.edits === null;
  let enabled = false;
  let disposed = false;

  function setHeight(height: number) {
    if (!Number.isFinite(height)) throw new Error('Cutaway height must be finite.');
    if (disposed) return;
    sdf.position.set(0, 0, height);
    edit.updateMatrixWorld(true);
    clippingPlane.constant = height;
    splat.updateVersion();
  }

  function setEnabled(value: boolean) {
    if (disposed || enabled === value) return;
    enabled = value;
    if (value) {
      // Preserve any explicitly assigned edits. The viewer creates no implicit
      // child edits, so changing null to an explicit list is safe here.
      splat.edits = [...(splat.edits ?? []), edit];
    } else {
      const remaining = (splat.edits ?? []).filter(candidate => candidate !== edit);
      splat.edits = originalImplicit && remaining.length === 0 ? null : remaining;
    }
    splat.updateVersion();
  }

  setHeight(initialHeight);
  setEnabled(true);
  return {
    clippingPlane,
    planes: [clippingPlane],
    setHeight,
    setEnabled,
    get enabled() { return enabled; },
    get height() { return clippingPlane.constant; },
    dispose() {
      if (disposed) return;
      setEnabled(false);
      edit.remove(sdf);
      disposed = true;
    },
  };
}

/** Keep lower cabinets/counters while removing the ceiling and upper walls. */
export function suggestedCutawayHeight(floor: number, ceiling: number): number {
  if (!Number.isFinite(floor) || !Number.isFinite(ceiling) || ceiling <= floor) {
    throw new Error('Cutaway requires a ceiling above the floor.');
  }
  return floor + (ceiling - floor) * 0.65;
}

type Vec3Tuple = [number, number, number];

/**
 * Display-only room crop. Hides every splat OUTSIDE an axis-aligned world box
 * (garden through the window, floaters, stray exterior splats). Like the
 * cutaway it never changes source splats, samples, or exported geometry.
 *
 * Spark 2.4 BOX SDF: `abs(p) - sdf.scale.xyz` (scale.xyz are HALF extents,
 * radius rounds the corners). invert=true flips the sign, so opacity 0 applies
 * outside the box. The edit is detached, so coordinates are world coordinates.
 */
export function createSplatCrop(splat: SplatMesh, min: Vec3Tuple, max: Vec3Tuple) {
  const edit = new SplatEdit({
    name: 'Morpheus display room crop',
    rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY,
    softEdge: 0.05,
  });
  const sdf = new SplatEditSdf({
    type: SplatEditSdfType.BOX,
    opacity: 0,
    invert: true,
  });
  edit.add(sdf);
  const originalImplicit = splat.edits === null;
  let enabled = false;
  let disposed = false;
  let current = { min: [...min] as Vec3Tuple, max: [...max] as Vec3Tuple };

  function setBounds(nextMin: Vec3Tuple, nextMax: Vec3Tuple) {
    if (![...nextMin, ...nextMax].every(Number.isFinite)) throw new Error('Crop bounds must be finite.');
    if (nextMin.some((value, axis) => value >= nextMax[axis])) throw new Error('Crop min must be below max.');
    if (disposed) return;
    current = { min: [...nextMin] as Vec3Tuple, max: [...nextMax] as Vec3Tuple };
    sdf.position.set(
      (nextMin[0] + nextMax[0]) / 2,
      (nextMin[1] + nextMax[1]) / 2,
      (nextMin[2] + nextMax[2]) / 2,
    );
    sdf.scale.set(
      (nextMax[0] - nextMin[0]) / 2,
      (nextMax[1] - nextMin[1]) / 2,
      (nextMax[2] - nextMin[2]) / 2,
    );
    edit.updateMatrixWorld(true);
    splat.updateVersion();
  }

  function setEnabled(value: boolean) {
    if (disposed || enabled === value) return;
    enabled = value;
    if (value) {
      splat.edits = [...(splat.edits ?? []), edit];
    } else {
      const remaining = (splat.edits ?? []).filter(candidate => candidate !== edit);
      splat.edits = originalImplicit && remaining.length === 0 ? null : remaining;
    }
    splat.updateVersion();
  }

  setBounds(min, max);
  return {
    setBounds,
    setEnabled,
    get enabled() { return enabled; },
    get bounds() { return current; },
    dispose() {
      if (disposed) return;
      setEnabled(false);
      edit.remove(sdf);
      disposed = true;
    },
  };
}

/** Room crop box from aligned bounds: +margin in XY, floor-0.1 up to the top. */
export function roomCropBounds(
  bounds: { min: Vec3Tuple; max: Vec3Tuple },
  floor: number,
  margin = 0.12,
): { min: Vec3Tuple; max: Vec3Tuple } {
  const bottom = Math.min(floor, bounds.min[2]) - 0.1;
  const top = Math.max(bounds.max[2], bottom + 0.2) + 0.05;
  return {
    min: [bounds.min[0] - margin, bounds.min[1] - margin, bottom],
    max: [bounds.max[0] + margin, bounds.max[1] + margin, top],
  };
}
