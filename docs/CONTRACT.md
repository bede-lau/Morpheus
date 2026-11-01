# Environment handoff contract

This contract separates Morpheus room preparation from robot training. Morpheus exports static collision geometry and evidence about its preparation. The consumer supplies robot models, spawn states, controllers, tasks, and success criteria.

## Bundle files

| File | Contract |
| --- | --- |
| `scene.json` | Source of truth for collider geometry, provenance, review state, and scale assumptions. |
| `environment.xml` | Reusable MJCF include containing a `<worldbody>` with static box geoms. Root is `<mujocoinclude>`. |
| `scene.xml` | Standalone `<mujoco>` scene that includes `environment.xml`, uses a 0.002 s timestep, and gravity `[0,0,-9.81]`. |
| `validation.json` | Explicit schema/native compilation/drop-probe status. Browser-only export never claims native validation. |
| `README.txt` | Short integration and approximation notes. |

Original splats, companion meshes, robot assets, policies, and credentials are not included in the ZIP. A source URL in the manifest does not imply redistribution rights.

## Coordinate conventions

- `units: "m"`, `upAxis: "Z"`: right-handed world frame, metres.
- Object `position` is its box center in the exported world frame.
- Object `size` is **full** `[X,Y,Z]` dimensions in metres. Exporters divide by two for MJCF box `size` half-extents.
- Object `quaternion` uses `[w,x,y,z]`, unlike Three.js's commonly used XYZW ordering. Exporters normalize nonzero finite quaternions. Omitted quaternion means identity.
- Objects are already in the output frame. `source.transform` describes how original visual/mesh source coordinates entered that frame. **Do not apply that transform again to collider positions.**
- The recorded transform's `matrix`, when present, is column-major. Its associated `matrixOrder` must be preserved. The reference source transform maps raw `[x,y,z]` to `[x,z,-y]` before centering and floor alignment.
- `source.scaleConfirmed: false` means the numbers are expressed in an assumed metre frame, not measured physical dimensions. Do not treat an assumed-scale scene as a validated robot-clearance measurement.

## Manifest version 1

```json
{
  "version": 1,
  "name": "Example room",
  "units": "m",
  "upAxis": "Z",
  "source": {
    "fileName": "room.spz",
    "kind": "upload",
    "scaleConfirmed": false,
    "scaleStatus": "assumed",
    "reviewed": false,
    "geometrySource": "companion-mesh",
    "approximation": "Occupied columns; space below elevated surfaces is filled."
  },
  "objects": [
    {
      "id": "floor",
      "name": "Proposed floor",
      "type": "box",
      "role": "floor",
      "position": [0, 0, -0.05],
      "size": [4, 5, 0.1],
      "quaternion": [1, 0, 0, 0],
      "enabled": true,
      "source": "geometry-proposal"
    }
  ]
}
```

Required top-level fields are `version`, `name`, `units`, `upAxis`, and a non-empty `objects` array. The application also writes `source` metadata. IDs must be unique, start with an ASCII letter, contain only ASCII letters, digits, `_`, or `-`, and be at most 96 characters. All position/size/quaternion numbers must be finite; dimensions must be positive. Version 1 supports boxes only.

Optional `enabled: false` preserves an object in the manifest but excludes it from MJCF and physical checks. `name`, `role`, and `source` provide display/provenance context; object names are not inferred semantic identities. Optional `friction` is a three-number nonnegative MuJoCo friction tuple; the default is `[0.8,0.005,0.0001]`. These values are assumptions, not estimated material properties.

The reference fixture explicitly records `synthetic: true`, `reviewed: false`, and `scaleConfirmed: false`. A successful physics probe does not rewrite these flags.

## MJCF integration

Place this directly inside the consumer scene's `<mujoco>` root:

```xml
<include file="environment.xml"/>
```

The include contributes static geoms named `env_<manifest-id>`, such as `env_floor`. It does not create robot joints, actuators, sensors, or controls. Standard collision masks are `contype="1"` and `conaffinity="1"`; geoms are in group 0. Use an appropriate consumer collision configuration if the robot overrides these masks.

Reconcile any floor already present in the robot scene. Duplicate floors can create redundant contacts. Preserve the exported world frame when placing the robot and task objects. The standalone `scene.xml` is useful for loading the environment alone; the include avoids imposing its timestep/gravity options on the consumer scene.

A room cannot safely be treated as one ordinary convex collision mesh: the convex hull can fill its navigable interior. This version deliberately exports separate box shapes. It does not reproduce the full source mesh or its holes.

## Validation evidence

`POST /api/validate` accepts the manifest directly and returns a report. `POST /api/export` accepts the same body and returns a ZIP; a known native validation failure returns HTTP 422 instead. `GET /api/health` reports engine availability. The local service listens on `127.0.0.1:8000`, accepts localhost browser origins, and limits request bodies to 5 MB.

| Report field/status | Meaning |
| --- | --- |
| `schemaValid: true` | Supported frame, dimensions, identifiers, and finite JSON passed structural checks. |
| `compile: "passed"` | Native MuJoCo successfully loaded the actual exported MJCF. |
| `dropProbe.status: "passed"` | A 0.04 m radius sphere dropped above an exposed floor, settled near its expected height, and had actual contact with that floor after two simulated seconds. |
| `dropProbe.status: "skipped"` | No suitable exposed, unrotated floor was found; do not interpret this as a passed contact test. |
| `status: "passed"` | Both native compilation and the defined contact probe passed. |
| `status: "compiled"` | Native compilation passed, but the contact probe was skipped. |
| `status: "unverified"` | Native physics was not run, including browser-only exports and missing MuJoCo installations. |
| `status: "failed"` | A native check failed; inspect its error/probe details. |

The report also records the engine version, object counts, probe coordinates/heights, contact result, and limitations. Changing geometry invalidates previous evidence; validate the modified manifest again.

The actual reference fixture checked on 1 November 2026 compiled 111 colliders in MuJoCo 3.17.0 and passed its floor probe. The checked point was `[0,0]` on the proposed floor; that result does not test every obstacle, every floor location, or a robot trajectory.

## What the consumer must still establish

Before relying on the scene for training or evaluation, verify scale, gravity alignment, floor placement, openings, missing obstacles, free space beneath furniture, and collision approximations relevant to the task. Occupied-column proposals can fill open space beneath observed surfaces; unobserved space may still contain obstacles. No semantic object recognition, articulated drawers/doors, robot access, grasp quality, or policy performance is established by this export.
