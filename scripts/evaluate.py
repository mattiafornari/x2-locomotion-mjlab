#!/usr/bin/env python3
"""Evaluate a checkpoint with a fixed protocol and append the results to a CSV.

Protocol (identical for every checkpoint, so base zero-shot / fine-tuned / scratch are comparable):
  * nominal physics (foot friction 0.8, no noise, no pushes, no randomised mass), fixed terrain geometry
  * forward command vx = 0.8 m/s, vy = wz = 0, one 20 s episode per env
  * for each seed (default 0 1 2) and each terrain level: all envs are pinned to that level and rolled out
  * survival = fraction of envs that did not fall (``fell_over`` termination) within the 20 s
  * vel_err_lin / vel_err_yaw: mean |command - measured| in the body frame, over the steps before a fall
  * distance: planar distance from the spawn point at the end of the episode (or at the fall)

Terrain levels are the rows of the curriculum grid (0 = easiest, 9 = hardest = 5 cm steps / 10 deg slope).

Example:
  uv run python scripts/evaluate.py --terrain gentle --checkpoint logs/rsl_rl/x2_velocity/<run>/model_3000.pt \
      --label base_zero_shot
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401  (loads the entry-point task packages, incl. x2_locomotion)
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

from x2_locomotion.tasks.velocity.env_cfgs import EVAL_LIN_VEL_X, make_eval_env_cfg
from x2_locomotion.tasks.velocity.terrains import NUM_LEVELS, X2_GENTLE_TERRAINS_CFG

FIELDS = [
    "label",
    "terrain",
    "checkpoint",
    "git_commit",
    "seed",
    "level",
    "subterrain",
    "num_envs",
    "survival",
    "vel_err_lin",
    "vel_err_yaw",
    "distance",
]
RL_TASK = {"flat": "X2-Velocity-Flat", "gentle": "X2-Velocity-GentleBlind-Finetune"}


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout
        return out + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"


def pin_level(env: ManagerBasedRlEnv, level: int) -> None:
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        return
    terrain.terrain_levels[:] = level
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]


@torch.no_grad()
def rollout(env: ManagerBasedRlEnv, wrapped: RslRlVecEnvWrapper, policy) -> dict[str, np.ndarray]:
    """One full-length episode for every env; returns per-env metrics."""
    n = env.num_envs
    robot = env.scene["robot"]
    steps = env.max_episode_length
    dev = env.device

    obs, _ = wrapped.reset()
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    lin_err = torch.zeros(n, device=dev)
    yaw_err = torch.zeros(n, device=dev)
    count = torch.zeros(n, device=dev)
    dist = torch.zeros(n, device=dev)

    for _ in range(steps):
        cmd = env.command_manager.get_command("twist")
        lin = torch.linalg.norm(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=1)
        yaw = (cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2]).abs()
        planar = torch.linalg.norm(
            robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1
        )
        lin_err += torch.where(alive, lin, 0.0)
        yaw_err += torch.where(alive, yaw, 0.0)
        count += alive.float()
        dist = torch.where(alive, planar, dist)

        obs, _, dones, extras = wrapped.step(policy(obs))
        terminated = ~extras["time_outs"].bool() & dones.bool()
        newly_fell = alive & terminated
        fell |= newly_fell
        alive &= ~dones.bool()

    count = count.clamp(min=1)
    return {
        "survived": (~fell).float().cpu().numpy(),
        "vel_err_lin": (lin_err / count).cpu().numpy(),
        "vel_err_yaw": (yaw_err / count).cpu().numpy(),
        "distance": dist.cpu().numpy(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--terrain", choices=["flat", "gentle"], required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--label", required=True, help="row label, e.g. base_zero_shot / finetuned / scratch"
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument(
        "--levels", type=int, nargs="+", default=None, help="default: all rows (gentle) / 0 (flat)"
    )
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/eval.csv"))
    args = ap.parse_args()

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    levels = (
        args.levels
        if args.levels is not None
        else (list(range(NUM_LEVELS)) if args.terrain == "gentle" else [0])
    )

    cfg = make_eval_env_cfg(args.terrain, args.num_envs)
    torch.manual_seed(0)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env)

    rl_task = RL_TASK[args.terrain]
    agent_cfg = load_rl_cfg(rl_task)
    runner_cls = load_runner_cls(rl_task) or OnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    terrain = env.scene.terrain
    sub_names = list(X2_GENTLE_TERRAINS_CFG.sub_terrains) if args.terrain == "gentle" else ["flat"]
    types = (
        terrain.terrain_types.cpu().numpy()
        if args.terrain == "gentle"
        else np.zeros(env.num_envs, dtype=int)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    commit = git_commit()
    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for seed in args.seeds:
            for level in levels:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if args.terrain == "gentle":
                    pin_level(env, level)
                m = rollout(env, wrapped, policy)
                groups = {"all": np.ones(env.num_envs, dtype=bool)}
                groups |= {
                    name: types == i for i, name in enumerate(sub_names) if (types == i).any()
                }
                for sub, mask in groups.items():
                    row = {
                        "label": args.label,
                        "terrain": args.terrain,
                        "checkpoint": str(args.checkpoint),
                        "git_commit": commit,
                        "seed": seed,
                        "level": level,
                        "subterrain": sub,
                        "num_envs": int(mask.sum()),
                        "survival": float(m["survived"][mask].mean()),
                        "vel_err_lin": float(m["vel_err_lin"][mask].mean()),
                        "vel_err_yaw": float(m["vel_err_yaw"][mask].mean()),
                        "distance": float(m["distance"][mask].mean()),
                    }
                    writer.writerow(row)
                    if sub == "all":
                        print(
                            f"[{args.label}] seed={seed} level={level}: survival={row['survival']:.2f} "
                            f"vel_err={row['vel_err_lin']:.3f} m/s distance={row['distance']:.1f} m "
                            f"(cmd vx={EVAL_LIN_VEL_X} m/s)"
                        )
    print(f"appended to {args.out}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
