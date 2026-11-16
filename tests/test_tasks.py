"""Registration and terrain-limit checks (no simulation)."""

import math

import pytest

from x2_locomotion.tasks.velocity.terrains import (
    MAX_SLOPE_GRADIENT,
    MAX_STEP_HEIGHT_M,
    NUM_LEVELS,
    X2_GENTLE_TERRAINS_CFG,
)

TASKS = (
    "X2-Velocity-Flat",
    "X2-Velocity-GentleBlind-Finetune",
    "X2-Velocity-GentleBlind-Scratch",
)


def test_tasks_registered_through_entry_point():
    import mjlab.tasks  # noqa: F401  (triggers entry-point discovery)
    from mjlab.tasks.registry import list_tasks

    assert set(TASKS) <= set(list_tasks())


def test_terrain_limits_are_the_documented_ones():
    subs = X2_GENTLE_TERRAINS_CFG.sub_terrains
    assert NUM_LEVELS == 10
    assert subs["stairs_up"].step_height_range[1] == pytest.approx(MAX_STEP_HEIGHT_M)
    assert subs["stairs_down"].step_height_range[1] == pytest.approx(MAX_STEP_HEIGHT_M)
    assert subs["slope_up"].slope_range[1] == pytest.approx(math.tan(math.radians(10.0)))
    assert subs["slope_down"].slope_range[1] == pytest.approx(MAX_SLOPE_GRADIENT)
    assert MAX_STEP_HEIGHT_M <= 0.05


@pytest.mark.slow
def test_generated_stairs_and_slopes_within_limits():
    """Build the real terrain and measure it (steps <= 5 cm; slope averaged over 0.5 m <= 10 deg)."""
    import mujoco
    import numpy as np
    from mjlab.terrains.terrain_generator import TerrainGenerator

    spec = mujoco.MjSpec()
    gen = TerrainGenerator(X2_GENTLE_TERRAINS_CFG, device="cpu")
    gen.compile(spec)
    model = spec.compile()

    for h in range(model.nhfield):
        adr, nr, nc = model.hfield_adr[h], model.hfield_nrow[h], model.hfield_ncol[h]
        z = model.hfield_data[adr : adr + nr * nc].reshape(nr, nc) * model.hfield_size[h][2]
        dx = 2 * model.hfield_size[h][0] / (nc - 1)
        w = max(1, round(0.5 / dx))
        grad = np.abs(z[:, w:] - z[:, :-w]) / (w * dx)
        assert grad.max() <= MAX_SLOPE_GRADIENT + 1e-3

    # box stairs: heights of the pyramid steps, relative to the neighbouring box
    tops = sorted(
        {
            round(float(model.geom_pos[g][2] + model.geom_size[g][2]), 4)
            for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        }
    )
    steps = np.diff(tops)
    assert steps.size == 0 or steps[steps > 1e-3].max() <= MAX_STEP_HEIGHT_M + 1e-3
