"""RSL-RL PPO configurations.

Network and PPO hyper-parameters are the mjlab G1 velocity ones. The three runs differ only in the
learning rate / iterations (the fine-tuning run halves the learning rate, as planned) and in the runner.
Both phases write to the same experiment folder so that ``--agent.load-run`` can find a phase-1 run
from a phase-2 launch (see ``X2FinetuneRunner``).
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

EXPERIMENT = "x2_velocity"
BASE_LR = 1.0e-3
FINETUNE_LR = BASE_LR / 2


def x2_ppo_cfg(
    *, run_name: str, learning_rate: float, max_iterations: int
) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=learning_rate,
            schedule="adaptive",  # the configured lr is only the starting value
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=EXPERIMENT,
        run_name=run_name,
        logger="tensorboard",
        upload_model=False,
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=max_iterations,
    )


def flat_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return x2_ppo_cfg(run_name="flat", learning_rate=BASE_LR, max_iterations=3000)


def finetune_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return x2_ppo_cfg(run_name="gentle_ft", learning_rate=FINETUNE_LR, max_iterations=2000)


def scratch_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return x2_ppo_cfg(run_name="gentle_scratch", learning_rate=BASE_LR, max_iterations=2000)
