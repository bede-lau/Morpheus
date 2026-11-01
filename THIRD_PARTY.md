# Existing software and assets

| Source | Use | License / handling |
|---|---|---|
| [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym) | Existing G1 12DoF model/meshes and pretrained PPO/LSTM actor | BSD-3-Clause, immutable commit pinned by fetcher; downloads excluded from Git; license carried with redistributed model |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Existing physics, model parsing/attachment and native rendering | Apache-2.0; dependency, no source copied |
| [PyTorch](https://github.com/pytorch/pytorch) | Existing CPU TorchScript inference | BSD-style; dependency; upstream TorchScript artifact retained despite current deprecation warning |
| [NumPy](https://numpy.org/) | Calculations | BSD-3-Clause dependency |
| Pillow / ImageIO / ImageIO-FFmpeg | Optional native-rendered clip | Optional dependencies with upstream licenses; not required for control |
| Supplied kitchen fixture | Existing upstream collision geometry + JSON | Source metadata attributes World Labs; no redistribution license inferred; input excluded from Git |
| Three.js / Spark / capture pipeline | Other developer's existing rendering and scan stack | Not bundled here; preserve their attribution and licenses in the combined application |

Original code in this repository is the new policy-integration/control/evaluation/bridge work authored with Codex. No pre-existing project template was imported. Do not describe the pretrained actor or robot assets as newly trained or created at the hackathon.
