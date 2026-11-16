"""Slow: build the environments on CPU and take a few steps."""

import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("terrain", ["flat", "gentle"])
def test_env_builds_and_steps(terrain):
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from x2_locomotion.tasks.velocity.env_cfgs import make_x2_env_cfg

    cfg = make_x2_env_cfg(terrain)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    obs, _ = env.reset()
    assert obs["actor"].shape == (2, 48)  # same dims in both phases -> checkpoints transfer
    assert obs["critic"].shape == (2, 60)
    for _ in range(5):
        obs, *_ = env.step(torch.zeros(2, 12))
    assert torch.isfinite(obs["actor"]).all()
    env.close()
