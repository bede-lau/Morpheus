# Robot meshes

`stretch.glb`, `go2.glb`, `g1.glb` are exported from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(commit `b0655057a9df2abe39ea7fc989914ba6f248301c`): `hello_robot_stretch_3` (Hello Robot, Clear BSD), `unitree_go2` and
`unitree_g1` (Unitree, BSD-3-Clause). Metres, Z-up, local +Y forward, lowest point at z=0, decimated to ~90k faces, one
node per MJCF body; the `.json` sidecars list body positions and joints so `src/robot-model.ts` can pose them.

Regenerate: `python3 scripts/fetch-robot-models.py <menagerie dir> <xml> <out>` then
`uv run --with mujoco --with trimesh --with fast_simplification python scripts/export-robot-models.py <id> <xml> src/robots [joint=value...]`
(Stretch used `joint_lift=0.6 joint_arm_l0..l3=0.02`).
