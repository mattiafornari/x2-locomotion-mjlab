"""Gentle terrain for the blind fine-tuning phase: steps <= 5 cm, slopes <= 10 deg.

mjlab curriculum mode gives every sub-terrain type its own column and difficulty ``row / (num_rows - 1)``
(row 0 = easiest, row 9 = hardest), interpolated over each sub-terrain's range. The ranges below are
therefore the *hard limits* of the hardest row.
"""

from __future__ import annotations

import math

from mjlab.terrains.config import (
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

MAX_STEP_HEIGHT_M = 0.05
MAX_SLOPE_DEG = 10.0
# mjlab slope ranges are gradients (rise / run), not angles.
MAX_SLOPE_GRADIENT = math.tan(math.radians(MAX_SLOPE_DEG))

NUM_LEVELS = 10

X2_GENTLE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=NUM_LEVELS,
    num_cols=5,  # ignored in curriculum mode (one column per sub-terrain)
    curriculum=True,
    sub_terrains={
        "flat": flat(proportion=0.1),
        "stairs_up": pyramid_stairs(
            proportion=0.3, step_height_range=(0.0, MAX_STEP_HEIGHT_M), step_width=0.3
        ),
        "stairs_down": pyramid_stairs_inv(
            proportion=0.3, step_height_range=(0.0, MAX_STEP_HEIGHT_M), step_width=0.3
        ),
        # vertical_scale 1 mm (default 5 mm): finer height quantisation keeps the cell-to-cell
        # gradient close to the nominal slope instead of alternating between 8.4 and 11.2 deg.
        "slope_up": hf_pyramid_slope(
            proportion=0.15, slope_range=(0.0, MAX_SLOPE_GRADIENT), vertical_scale=0.001
        ),
        "slope_down": hf_pyramid_slope_inv(
            proportion=0.15, slope_range=(0.0, MAX_SLOPE_GRADIENT), vertical_scale=0.001
        ),
    },
    add_lights=True,
)
