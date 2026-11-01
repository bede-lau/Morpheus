/**
 * Reviewable collision proposals, not semantic reconstruction.
 * Coordinates are metres, right-handed Z-up. Sizes are full extents.
 * Dense surface samples are preferred; bare triangle vertices can miss surfaces.
 * The column approximation fills space below observed obstacles and cannot
 * represent openings under tables, overhangs, stairs, or moving parts.
 */
export type Vec3 = [number, number, number];
export type Quaternion = [number, number, number, number]; // w, x, y, z

export interface Collider {
  id: string;
  name: string;
  type: 'box';
  position: Vec3;
  size: Vec3;
  quaternion: Quaternion;
  source: 'geometry-proposal';
  role: 'floor' | 'obstacle';
  enabled: boolean;
}

export interface Bounds {
  min: Vec3;
  max: Vec3;
  center: Vec3;
  size: Vec3;
  pointCount: number;
}

export interface ProposalOptions {
  cellSize?: number;
  floorZ?: number;
  maxBoxes?: number;
  /** A known ceiling plane; exclude samples in its top 10 cm. */
  ceilingZ?: number;
  /** Minimum samples per occupied XY cell. Set to 1 for sparse geometry. */
  minPoints?: number;
  trimFraction?: number;
  floorThickness?: number;
  /** Ignore samples this close to the floor. Defaults to 8 cm. */
  floorClearance?: number;
}

const finite = (n: number) => Number.isFinite(n);
const round = (n: number) => Math.round(n * 1e6) / 1e6;
const quantile = (sorted: number[], fraction: number) => {
  const index = (sorted.length - 1) * fraction;
  const lo = Math.floor(index);
  return sorted[lo] + (sorted[Math.ceil(index)] - sorted[lo]) * (index - lo);
};

function validPoints(points: Float32Array): Vec3[] {
  if (points.length % 3 !== 0) throw new Error('Point data must contain complete XYZ triples.');
  const result: Vec3[] = [];
  for (let i = 0; i < points.length; i += 3) {
    const p: Vec3 = [points[i], points[i + 1], points[i + 2]];
    if (p.every(finite)) result.push(p);
  }
  if (!result.length) throw new Error('No finite 3D points were found.');
  return result;
}

function boundsOf(points: Vec3[], trimFraction: number): Bounds {
  if (!finite(trimFraction) || trimFraction < 0 || trimFraction >= 0.25) {
    throw new Error('trimFraction must be between 0 and 0.25 (exclusive).');
  }
  const axes = [0, 1, 2].map(axis => points.map(p => p[axis]).sort((a, b) => a - b));
  const min = axes.map(axis => quantile(axis, trimFraction)) as Vec3;
  const max = axes.map(axis => quantile(axis, 1 - trimFraction)) as Vec3;
  return {
    min, max,
    center: min.map((value, axis) => (value + max[axis]) / 2) as Vec3,
    size: min.map((value, axis) => max[axis] - value) as Vec3,
    pointCount: points.length,
  };
}

/** Percentile bounds suppress distant floaters; use trimFraction=0 for exact bounds. */
export function computeBounds(points: Float32Array, trimFraction = 0.01): Bounds {
  return boundsOf(validPoints(points), trimFraction);
}

/** Lower-height histogram mode. A proposal only: user must verify the floor plane. */
export function estimateFloorZ(points: Float32Array): number {
  const zs = validPoints(points).map(p => p[2]).sort((a, b) => a - b);
  const low = quantile(zs, 0.01);
  const high = quantile(zs, 0.35);
  const binWidth = Math.max(0.02, (high - low) / 32);
  const bins = new Map<number, number[]>();
  for (const z of zs) {
    if (z < low || z > high) continue;
    const index = Math.floor((z - low) / binWidth);
    const values = bins.get(index) ?? [];
    values.push(z);
    bins.set(index, values);
  }
  let winner: number[] = [low];
  for (const values of bins.values()) if (values.length > winner.length) winner = values;
  return round(quantile(winner, 0.5));
}

/** Determine uniform scale from a measured reference; never infer metric scale from appearance. */
export function scaleFromMeasurement(a: Vec3, b: Vec3, distanceMeters: number): number {
  if (![...a, ...b, distanceMeters].every(finite) || distanceMeters <= 0) {
    throw new Error('Reference coordinates must be finite and measured distance must be positive.');
  }
  const measured = Math.hypot(...a.map((v, i) => v - b[i]));
  if (measured < 1e-8) throw new Error('Choose two distinct reference points.');
  return distanceMeters / measured;
}

/** Apply scale, rotation (wxyz), then translation. Invalid input points stay invalid. */
export function transformPoints(
  points: Float32Array,
  scale = 1,
  translation: Vec3 = [0, 0, 0],
  quaternion: Quaternion = [1, 0, 0, 0],
): Float32Array {
  if (points.length % 3) throw new Error('Point data must contain complete XYZ triples.');
  if (![scale, ...translation, ...quaternion].every(finite) || scale <= 0) {
    throw new Error('Transform must be finite and scale must be positive.');
  }
  const length = Math.hypot(...quaternion);
  if (length < 1e-8) throw new Error('Rotation quaternion must be nonzero.');
  const [w, x, y, z] = quaternion.map(v => v / length);
  const result = new Float32Array(points.length);
  for (let i = 0; i < points.length; i += 3) {
    const px = points[i] * scale, py = points[i + 1] * scale, pz = points[i + 2] * scale;
    // v' = v + 2w(q_xyz cross v) + 2(q_xyz cross (q_xyz cross v)).
    const tx = 2 * (y * pz - z * py), ty = 2 * (z * px - x * pz), tz = 2 * (x * py - y * px);
    result[i] = px + w * tx + y * tz - z * ty + translation[0];
    result[i + 1] = py + w * ty + z * tx - x * tz + translation[1];
    result[i + 2] = pz + w * tz + x * ty - y * tx + translation[2];
  }
  return result;
}

type Cell = { x: number; y: number; count: number; top: number };
type Rect = { x: number; y: number; width: number; depth: number; height: number };

function occupiedRectangles(points: Vec3[], bounds: Bounds, floor: number, cellSize: number,
  clearance: number, ceiling: number, minPoints: number): Rect[] {
  const cells = new Map<string, Cell>();
  const nx = Math.max(1, Math.ceil(bounds.size[0] / cellSize));
  const ny = Math.max(1, Math.ceil(bounds.size[1] / cellSize));
  for (const [x, y, z] of points) {
    if (x < bounds.min[0] || x > bounds.max[0] || y < bounds.min[1] || y > bounds.max[1]
      || z <= floor + clearance || z > bounds.max[2] || z >= ceiling) continue;
    const gx = Math.min(nx - 1, Math.floor((x - bounds.min[0]) / cellSize));
    const gy = Math.min(ny - 1, Math.floor((y - bounds.min[1]) / cellSize));
    const key = `${gx},${gy}`;
    const cell = cells.get(key) ?? { x: gx, y: gy, count: 0, top: floor };
    cell.count++;
    cell.top = Math.max(cell.top, z);
    cells.set(key, cell);
  }
  // Quantize upwards so merging never lowers observed obstacle height.
  const heights = new Map<string, number>();
  for (const [key, cell] of cells) {
    if (cell.count >= minPoints) heights.set(key, Math.ceil((cell.top - floor) / 0.1) * 0.1);
  }
  const consumed = new Set<string>();
  const rectangles: Rect[] = [];
  const sorted = [...cells.values()].sort((a, b) => a.y - b.y || a.x - b.x);
  for (const cell of sorted) {
    const key = `${cell.x},${cell.y}`;
    const height = heights.get(key);
    if (height === undefined || consumed.has(key)) continue;
    const matches = (x: number, y: number) => !consumed.has(`${x},${y}`) && heights.get(`${x},${y}`) === height;
    let width = 1, depth = 1;
    while (cell.x + width < nx && matches(cell.x + width, cell.y)) width++;
    outer: while (cell.y + depth < ny) {
      for (let dx = 0; dx < width; dx++) if (!matches(cell.x + dx, cell.y + depth)) break outer;
      depth++;
    }
    for (let dy = 0; dy < depth; dy++) for (let dx = 0; dx < width; dx++) consumed.add(`${cell.x + dx},${cell.y + dy}`);
    rectangles.push({ x: cell.x, y: cell.y, width, depth, height });
  }
  return rectangles;
}

/**
 * Produces one floor box and obstacle columns. No inferred perimeter walls:
 * inventing a closed perimeter would seal real doorways. If the box budget is
 * exceeded, coarsen the grid rather than silently dropping occupied cells.
 */
export function proposeColliders(points: Float32Array, options: ProposalOptions = {}): Collider[] {
  const samples = validPoints(points);
  const bounds = boundsOf(samples, options.trimFraction ?? 0.01);
  const floor = options.floorZ ?? estimateFloorZ(points);
  let cellSize = options.cellSize ?? 0.25;
  const budget = options.maxBoxes ?? 200;
  const minPoints = options.minPoints ?? 3;
  const thickness = options.floorThickness ?? 0.1;
  const clearance = options.floorClearance ?? 0.08;
  if (![floor, cellSize, budget, minPoints, thickness, clearance].every(finite)
    || cellSize <= 0 || !Number.isInteger(budget) || budget < 2
    || !Number.isInteger(minPoints) || minPoints < 1 || thickness <= 0 || clearance < 0) {
    throw new Error('Invalid proposal options: positive cell size/thickness, maxBoxes >= 2, minPoints >= 1 required.');
  }
  if (bounds.size[0] < 0.05 || bounds.size[1] < 0.05) throw new Error('The scan needs a nonzero floor footprint.');
  if (Math.max(...bounds.size) > 200) throw new Error('Scene exceeds 200 metres. Verify its scale and crop distant points.');
  if (floor > bounds.max[2] + 0.1) throw new Error('Floor is above all scan points. Verify floor height.');
  if (options.ceilingZ !== undefined && (!finite(options.ceilingZ) || options.ceilingZ <= floor + clearance)) {
    throw new Error('Ceiling must be finite and above the floor.');
  }
  // No automatic ceiling exclusion: a high shelf is not necessarily a ceiling.
  const ceiling = options.ceilingZ === undefined ? Infinity : options.ceilingZ - 0.1;
  let rectangles = occupiedRectangles(samples, bounds, floor, cellSize, clearance, ceiling, minPoints);
  for (let attempts = 0; rectangles.length + 1 > budget && attempts < 24; attempts++) {
    cellSize *= 1.35;
    rectangles = occupiedRectangles(samples, bounds, floor, cellSize, clearance, ceiling, minPoints);
  }
  if (rectangles.length + 1 > budget) throw new Error('Too many collision boxes. Crop the scan or increase maxBoxes.');
  const make = (id: string, name: string, role: 'floor' | 'obstacle', position: Vec3, size: Vec3): Collider => ({
    id, name, type: 'box', position: position.map(round) as Vec3, size: size.map(round) as Vec3,
    quaternion: [1, 0, 0, 0], source: 'geometry-proposal', role, enabled: true,
  });
  const result = [make('floor', 'Proposed floor', 'floor',
    [bounds.center[0], bounds.center[1], floor - thickness / 2],
    [bounds.size[0] + 0.1, bounds.size[1] + 0.1, thickness])];
  rectangles.forEach((rect, index) => {
    const x = bounds.min[0] + rect.x * cellSize;
    const y = bounds.min[1] + rect.y * cellSize;
    const width = Math.min(rect.width * cellSize, bounds.max[0] - x);
    const depth = Math.min(rect.depth * cellSize, bounds.max[1] - y);
    if (width <= 0 || depth <= 0) return;
    result.push(make(`obstacle_${index + 1}`, `Proposed obstacle ${index + 1}`, 'obstacle',
      [x + width / 2, y + depth / 2, floor + rect.height / 2], [width, depth, rect.height]));
  });
  return result;
}
