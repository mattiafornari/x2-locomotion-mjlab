#!/usr/bin/env python3
"""ONNX parity check on CPU MuJoCo.

Rolls the PyTorch policy in the CPU environment, records the actor observations it actually sees,
then replays the same observations through the exported ONNX model (onnxruntime, CPU) and compares
the actions. Exits non-zero if the max abs difference exceeds ``--tol``.

Training already writes ``<checkpoint>.onnx`` next to every ``model_*.pt``; use ``--export`` to
(re)generate it from the checkpoint instead.

Example:
  uv run python scripts/onnx_parity.py --terrain flat --checkpoint logs/rsl_rl/x2_velocity/<run>/model_3000.pt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401
import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner

from x2_locomotion.tasks.velocity.env_cfgs import make_eval_env_cfg

RL_TASK = {"flat": "X2-Velocity-Flat", "gentle": "X2-Velocity-GentleBlind-Finetune"}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--terrain", choices=["flat", "gentle"], required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--onnx", type=Path, default=None, help="default: <checkpoint dir>/<checkpoint stem>.onnx"
    )
    ap.add_argument(
        "--export", action="store_true", help="export the ONNX file from the checkpoint first"
    )
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    device = "cpu"  # the parity check is explicitly CPU MuJoCo + CPU onnxruntime
    env = ManagerBasedRlEnv(cfg=make_eval_env_cfg(args.terrain, args.num_envs), device=device)
    wrapped = RslRlVecEnvWrapper(env)
    task = RL_TASK[args.terrain]
    runner_cls = load_runner_cls(task) or OnPolicyRunner
    runner = runner_cls(wrapped, asdict(load_rl_cfg(task)), device=device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    onnx_path = args.onnx or args.checkpoint.with_suffix(".onnx")
    if args.export:
        runner.export_policy_to_onnx(str(onnx_path.parent), onnx_path.name)
    if not onnx_path.exists():
        print(f"missing {onnx_path}; pass --export", file=sys.stderr)
        return 2
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    worst = 0.0
    obs, _ = wrapped.reset()
    with torch.no_grad():
        for _ in range(args.steps):
            a_torch = policy(obs)
            x = obs["actor"] if hasattr(obs, "keys") and "actor" in obs else obs
            xs = x.detach().cpu().numpy().astype(np.float32)
            # the exported graph is traced with batch size 1 (as on the robot): feed env by env
            a_onnx = np.concatenate([sess.run(None, {in_name: row[None]})[0] for row in xs], axis=0)
            worst = max(worst, float(np.abs(a_torch.cpu().numpy() - a_onnx).max()))
            obs, *_ = wrapped.step(a_torch)

    print(
        f"max |torch - onnx| over {args.steps} steps x {args.num_envs} envs: {worst:.3e} (tol {args.tol:.0e})"
    )
    env.close()
    return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
