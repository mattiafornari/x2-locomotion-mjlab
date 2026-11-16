"""Slow: a phase-1 checkpoint loads into the phase-2 runner as weights only."""

from dataclasses import asdict

import pytest
import torch

pytestmark = pytest.mark.slow

FLAT, FT = "X2-Velocity-Flat", "X2-Velocity-GentleBlind-Finetune"


def _runner(task, terrain_envs=2):
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    cfg = load_env_cfg(task)
    cfg.scene.num_envs = terrain_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    runner = load_runner_cls(task)(RslRlVecEnvWrapper(env), asdict(load_rl_cfg(task)), device="cpu")
    return env, runner


def test_finetune_loads_weights_only(tmp_path):
    _, flat = _runner(FLAT)
    ckpt = tmp_path / "model_0.pt"
    flat.save(str(ckpt))
    src = torch.load(ckpt, weights_only=False)

    env, ft = _runner(FT)
    env.common_step_counter = 999
    ft.load(str(ckpt))

    key = next(k for k in src["actor_state_dict"] if k.endswith("mlp.0.weight"))
    assert torch.equal(ft.alg.actor.state_dict()[key], src["actor_state_dict"][key])
    assert ft.current_learning_iteration == 0  # iteration counter not restored
    assert env.common_step_counter == 0  # terrain curriculum clock restarts
    assert ft.alg.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
