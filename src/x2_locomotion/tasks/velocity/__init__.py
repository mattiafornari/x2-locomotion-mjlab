"""Registers the X2 velocity tasks with mjlab.

* ``X2-Velocity-Flat``                  phase 1: flat ground
* ``X2-Velocity-GentleBlind-Finetune``  phase 2: gentle terrain (steps <= 5 cm, slopes <= 10 deg),
                                        initialised from a phase-1 checkpoint, lr halved
* ``X2-Velocity-GentleBlind-Scratch``   phase 2 baseline trained from scratch on the same terrain
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import make_x2_env_cfg
from .rl_cfg import finetune_ppo_cfg, flat_ppo_cfg, scratch_ppo_cfg
from .runner import X2FinetuneRunner

register_mjlab_task(
    task_id="X2-Velocity-Flat",
    env_cfg=make_x2_env_cfg("flat"),
    play_env_cfg=make_x2_env_cfg("flat", play=True),
    rl_cfg=flat_ppo_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="X2-Velocity-GentleBlind-Finetune",
    env_cfg=make_x2_env_cfg("gentle"),
    play_env_cfg=make_x2_env_cfg("gentle", play=True),
    rl_cfg=finetune_ppo_cfg(),
    runner_cls=X2FinetuneRunner,
)

register_mjlab_task(
    task_id="X2-Velocity-GentleBlind-Scratch",
    env_cfg=make_x2_env_cfg("gentle"),
    play_env_cfg=make_x2_env_cfg("gentle", play=True),
    rl_cfg=scratch_ppo_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
