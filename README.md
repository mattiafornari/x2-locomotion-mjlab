# x2-locomotion-mjlab

Velocity-tracking locomotion for the **AgiBot X2 Ultra** humanoid (legs only, 12 DoF) in
**MuJoCo** via [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp), trained with PPO
([rsl-rl](https://github.com/leggedrobotics/rsl_rl)) in two phases:

1. **Phase 1 – flat ground.** Track `(vx, vy, wz)` commands, light domain randomization, goal:
   *walks 20 s without falling*.
2. **Phase 2 – gentle terrain, blind.** Fine-tune the **same** checkpoint (learning rate halved) on
   steps ≤ 5 cm and slopes ≤ 10° with a curriculum. Observations are unchanged, so no height scan.
   The deliverable is a table of *phase-1 policy zero-shot* vs *fine-tuned* (optionally vs
   *from scratch*) on fixed seeds and fixed terrain levels.

One repo, one submodule (the official X2 MJCF), one framework dependency (`mjlab==1.6.1`).

> **Status: code complete and CPU-verified, not yet trained.** The environments build and step on
> CPU, the evaluation and ONNX-parity scripts run end-to-end with a randomly initialised policy, and
> 12 tests pass. No training run exists yet, so there are **no results and no video** in this
> repository. The `Results` section below is filled in from `results/eval.csv` after the runs.
> mjlab on WSL2 and the 8 GB VRAM budget are untested (see `docs/DESIGN.md`).

## Quick start (WSL2 Ubuntu / Linux + NVIDIA GPU)

```bash
git clone --recurse-submodules <this repo> && cd x2-locomotion-mjlab
uv sync --extra cu128 --extra dev --extra deploy        # project venv; nothing system-wide is touched

# Phase 1 (flat)
uv run train X2-Velocity-Flat --env.scene.num-envs 2048 --agent.max-iterations 3000
# Phase 2: fine-tune the phase-1 checkpoint on gentle terrain (weights only, lr 5e-4)
uv run train X2-Velocity-GentleBlind-Finetune --env.scene.num-envs 2048 \
    --agent.resume True --agent.load-run '<phase-1 run dir>' --agent.load-checkpoint 'model_3000.pt'
# Optional: same terrain from scratch
uv run train X2-Velocity-GentleBlind-Scratch --env.scene.num-envs 2048
```

Full WSL2 setup, GPU checks and run hygiene: `docs/RUNBOOK.md` (Italian).

## Evaluate and export

```bash
uv run python scripts/evaluate.py --terrain gentle --label base_zero_shot --checkpoint <phase-1 model.pt>
uv run python scripts/evaluate.py --terrain gentle --label finetuned      --checkpoint <phase-2 model.pt>
uv run python scripts/make_table.py results/eval.csv > results/table.md
uv run python scripts/onnx_parity.py --terrain gentle --checkpoint <model.pt>   # CPU MuJoCo vs onnxruntime
uv run play X2-Velocity-GentleBlind-Finetune --agent trained --checkpoint-file <model.pt> --video True
```

Protocol (identical for every policy): nominal physics, forward command `vx = 0.8 m/s`, 20 s
episodes, seeds `0 1 2`, terrain geometry seed fixed, every env pinned to a terrain level 0–9
(level 9 = 5 cm steps / 10° slope). Reported: survival, velocity error, distance, and the hardest
level with survival ≥ 0.90 on every seed.

## Design in one paragraph

The official X2 MJCF is loaded with `MjSpec` and edited in memory: non-leg joints removed, actuators
replaced by position actuators (G1-style gains `K = I·ω²`, `D = 2ζIω`, 10 Hz, ζ = 2), collisions
reduced to the foot spheres, IMU sensors added. mjlab's `make_velocity_env_cfg()` supplies the task;
this repo changes the robot, the blind observation set, the terrain, the domain randomization and the
fine-tuning runner. Tasks register through the `mjlab.tasks` entry point, so the stock
`uv run train/play` CLIs work with no wrapper. Details, assumptions and **limitations** (no sim-to-real,
unidentified PD gains, the actor observes base linear velocity) are in `docs/DESIGN.md`.

## Layout

```
src/x2_locomotion/assets/x2.py            MJCF edit, actuators, collisions, sensors
src/x2_locomotion/tasks/velocity/         env cfg, terrains, PPO cfg, fine-tune runner, task registration
scripts/evaluate.py make_table.py onnx_parity.py
tests/                                     asset, registration, terrain limits, CPU env build
docs/DESIGN.md  docs/RUNBOOK.md
third_party/agibot_x2_urdf                 submodule (official model, pinned)
```

## License

BSD-3-Clause for this code. The X2 model is MulanPSL-2.0 (submodule, not redistributed); see `NOTICE.md`.

> `uv.lock` is not committed (it could not be generated offline from the PyTorch index); run `uv sync` once and commit it.
