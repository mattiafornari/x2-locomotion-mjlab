# Design notes, assumptions, limitations

## Scope
Legs-only (12 DoF) velocity tracking for the X2 Ultra; Phase 1 flat, Phase 2 gentle terrain, blind.
Not a sim-to-real project: there is no hardware, and the official X2 deployment I/O layout is not
documented publicly.

## Robot model
* Source: official MJCF `X2-Ultra.xml` (submodule, pinned commit). 31 joints, 44.8 kg. The 19 arm,
  waist and head joints are deleted at load time, so their bodies stay as fixed mass in the torso.
* Collisions: only the 12 sphere geoms per foot collide (named `{left,right}_foot{i}_collision`).
  Avoids relying on mesh/cylinder collision support in MuJoCo Warp.
* Pose: pelvis z 0.657 m with hip pitch −0.25, knee 0.5, ankle pitch −0.25 (flat-foot rule:
  hip_pitch + knee + ankle_pitch = 0).
* Hip-roll limits are asymmetric per side, hence `soft_joint_pos_limit_factor = 1.0`.

## Actuation (assumption, not identified)
Position actuators with `K = I·ω²`, `D = 2ζ·I·ω`, ω = 2π·10 Hz, ζ = 2, armature 0.03 (the official
joint default), giving K = 118.4 N·m/rad and D = 7.54 N·m·s/rad for all joints; torque limits 120 /
60 / 36 N·m (hip+knee / ankle pitch / ankle roll). Action scale `0.25·effort/K`. These are
reasonable starting values from the G1 recipe, **not** measured X2 values.
Consequence: zero action is not a standing pose (ankle gain is below `m·g·h`), so the smoke test only
checks finiteness and the initial height, not that the robot stands.

## Observations / commands
Actor 48-dim: base linear velocity, angular velocity, projected gravity, command, joint position
and velocity, last action. Critic 60-dim (privileged). **No terrain scan in either phase**, so
checkpoints transfer between phases without touching the network shape.
Limitation: the actor observes `base_lin_vel`, which on hardware would need a state estimator.
Commands: vx ∈ [−0.5, 1.0], vy ∈ [−0.5, 0.5], wz ∈ [−1, 1].

## Domain randomization (light)
Foot friction, torso COM, torso mass (±5 % pseudo-inertia), encoder bias, random pushes (±0.3 m/s
every 8–12 s), observation noise. No latency or actuator-model randomization.

## Phase 2
* Terrain: curriculum grid 10 levels × 5 sub-terrain types (flat 10 %, stairs up/down 30 % each,
  slope up/down 15 % each), difficulty = row / 9. Stair step height 0 → 5 cm (measured: 5.0 cm);
  slope gradient 0 → tan 10° with 1 mm height quantization. Measured on the generated heightfields:
  **≤ 9° averaged over 0.5 m, ≤ 10.1° per cell** (cell-level excess comes from the 8/79 m MuJoCo
  cell width vs the 0.1 m nominal).
* Fine-tuning loads actor and critic weights only (`X2FinetuneRunner`): optimizer state, iteration
  counter and curriculum step counter are reset. Learning rate starts at 5e-4 (half of phase 1).
  rsl-rl uses an adaptive-KL schedule, so the configured rate is a starting value, not a constant.
* Comparison arms: base zero-shot (phase-1 checkpoint), fine-tuned, optional from scratch.
  One change per run; seeds and terrain geometry fixed in `scripts/evaluate.py`.

## Verified vs unverified
Verified on CPU: asset compiles (nq 19, nv 18, nu 12, mass 44.801 kg), three tasks register, both
envs build and step with finite observations, terrain limits measured, evaluate/onnx-parity run
end-to-end (ONNX vs torch max diff ≈ 1e-7 with a random-init checkpoint), lint, 12 tests.
**Not verified:** training convergence, GPU throughput, VRAM at 2048 envs on 8 GB, mjlab/MuJoCo
Warp on WSL2, contact-buffer sizes (`njmax`/`nconmax` are estimates), whether the chosen gains and
rewards yield a stable gait.
