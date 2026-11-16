"""Runner for phase 2: initialise from a phase-1 checkpoint (weights only) and start a fresh run."""

from __future__ import annotations

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

WEIGHTS_ONLY = {"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False}


class X2FinetuneRunner(VelocityOnPolicyRunner):
    """``--agent.resume True --agent.load-run <phase-1 run>`` loads actor, critic and observation
    normalisers but *not* the optimizer state or the iteration counter, so that

    * the learning rate starts from the (halved) configured value instead of the phase-1 adaptive one,
    * the iteration counter and the env step counter (which drives the terrain curriculum) restart at 0.

    Explicit ``load_cfg`` arguments (e.g. ``{"actor": True}`` from ``play``) are respected.
    """

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(
            path,
            load_cfg=WEIGHTS_ONLY if load_cfg is None else load_cfg,
            strict=strict,
            map_location=map_location,
        )
        self.env.unwrapped.common_step_counter = 0
        return infos
