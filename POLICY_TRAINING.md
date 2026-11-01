# Policy creation, optimization and possible specialization

## What runs now

The new home-conditioned navigation policy is a hierarchy: supplied scene -> conservative route -> bounded body-frame velocity command -> pretrained Unitree PPO/LSTM actor -> PD torque -> MuJoCo. The new integration creates this hierarchy and verifies it; it does not claim to have trained the underlying walking actor.

The actor consumes its original 47 normalized observations and outputs 12 leg-position residuals. It never sees a Gaussian splat or arbitrary scene JSON. Home geometry affects its velocity command through the planner. This separation allows a new map without retraining the low-level policy, subject to the supported flat-floor/model assumptions.

PPO is selected because there is a small, official, compatible checkpoint and deployment recipe. SAC would require a separate compatible trained artifact; choosing an algorithm alone does not make a robot move. IK is the current fixed-arm calculation because known kinematics are sufficient for reach feasibility. ACT and foundation action models need robot-specific data/observation/action contracts and do not turn a raw MJCF into manipulation competence.

## Calculation and optimization sequence

1. Validate the supplied metric geometry, frame, source confidence and existing collision masks.
2. Check exact robot asset/checkpoint/joint/force-limit/timestep contract.
3. Calculate inflated routes with a declared robot envelope; reject impossible endpoints.
4. Measure baseline actuated episodes; retain contacts, falls and all failures.
5. Search a bounded set of high-level controller configurations on training seeds.
6. Compare only the selected candidate against the frozen baseline on separate held-out starts. Retain the baseline without strict improvement. This is controller optimization.
7. Qualify the selected configuration on a fresh 20+ episode set before claiming its tested simulation coverage. Never reuse the small search holdout as a broad robustness claim.
8. Export policy/model/config/source identities and the precise coverage. Changed inputs require another evaluation.

## When a GPU experiment becomes justified

The observed baseline runs faster than real time on CPU. A GPU is not needed to infer the actor, compute routes, calculate IK, or validate this small scene.

Use Modal only after an identified failure cannot be corrected by input quality, model compatibility, route constraints or controller configuration. Examples: command tracking repeatedly fails on a supported robot after dynamics variation; or one specified arm skill needs learning from demonstrations. A scene with assumed scale is an upstream geometry problem, not a reward-learning problem.

For a locomotion specialist:

1. Freeze one robot model and its 47-observation/12-action recurrent contract.
2. Reproduce the official baseline before changing rewards or dynamics. The selected official training source uses its own training stack; an MJX/PPO port would be additional engineering and validation, not an existing feature of this repo.
3. Define a bounded velocity-tracking task with rewards for linear/yaw tracking and penalties for falls, forbidden contacts, limit violations, torque/energy and abrupt actions. Reward coefficients and termination thresholds are experimental inputs, not validated constants supplied by this project.
4. Split maps/tasks/seeds before tuning. Keep physical randomization variants out of the final held-out set.
5. Randomize bounded friction, mass/inertia, motor strength/latency, joint offsets, sensor noise and proxy uncertainty. Record the ranges and actual sampled seeds; do not modify the upstream environment source.
6. Fine-tune PPO with the actor/recurrent state structure preserved, save recoverable checkpoints, and export a new immutable actor plus manifest.
7. Compare candidate vs baseline on held-out tracking, route completion, falls, contacts, energy/forces and latency. Reject any required regression; report the complete distribution.
8. Repeat exact-model MuJoCo integration and keep hardware unverified until commissioning.

For a manipulation specialist, use one robot/gripper/camera/task and representative demonstrations before considering [ACT](https://github.com/huggingface/lerobot/blob/main/docs/source/act.mdx). Reachable TCP coordinates do not establish grasp success. A G1 leg actor cannot execute arm skills; whole-body control remains separate work.

## Modal budget

Available credits reported by the user: $180. Spending to date: $0. Proposed first specialist experiment ceiling: $18, leaving $162. Before launch, configure a workspace budget and one bounded function/GPU, enforce wall-time/checkpoint/retry limits, include CPU/memory/storage costs, and verify current [Modal pricing](https://modal.com/pricing) and [GPU configuration](https://modal.com/docs/guide/gpu). A timeout alone is not a total spending cap. No Modal job or trainer is silently staged or launched by this integration.

## Transfer to the delivered robot

Simulation qualification is conditional on its model and tested variations. At delivery: register the robot's sensors to the stored map, calibrate joint zeros/dynamics, verify e-stop/support, sense moved objects and people/pets, and run slow supervised trials. Dynamic-obstacle stopping/replanning and a hardware transport are not implemented here. “Zero-shot” may describe unchanged weights in a new simulation map; it does not certify physical home safety.
