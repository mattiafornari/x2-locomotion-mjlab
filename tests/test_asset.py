"""Asset checks (CPU, fast): the edited X2 MJCF compiles with the expected structure."""

import mujoco
import pytest

from x2_locomotion.assets.x2 import (
    DAMPING,
    STIFFNESS,
    X2_ACTION_SCALE,
    X2_XML,
    get_x2_legs_cfg,
)

pytestmark = pytest.mark.skipif(not X2_XML.exists(), reason="submodule not checked out")


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    # Entity adds the actuators / collision settings on top of the edited MJCF
    return get_x2_legs_cfg().build().spec.compile()


def test_legs_only_dimensions(model):
    assert model.nq == 19  # freejoint (7) + 12 leg joints
    assert model.nv == 18
    assert model.nu == 12


def test_actuators_follow_joint_order(model):
    jnt_names = [model.joint(i).name for i in range(1, model.njnt)]
    act_joints = [model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)]
    assert act_joints == jnt_names


def test_gains_and_limits(model):
    assert STIFFNESS == pytest.approx(118.4, abs=0.1)
    assert DAMPING == pytest.approx(7.54, abs=0.02)
    assert model.actuator_gainprm[0, 0] == pytest.approx(STIFFNESS)
    assert model.actuator_forcerange[:, 1].max() == pytest.approx(120.0)
    assert model.actuator_forcerange[:, 1].min() == pytest.approx(36.0)


def test_total_mass(model):
    assert model.body_mass.sum() == pytest.approx(44.801, abs=1e-2)


def test_only_feet_collide(model):
    colliding = [
        model.geom(i).name
        for i in range(model.ngeom)
        if model.geom_contype[i] or model.geom_conaffinity[i]
    ]
    assert colliding, "feet must collide"
    assert all("foot" in n for n in colliding if n), colliding


def test_action_scale_matches_effort_over_gain():
    assert max(X2_ACTION_SCALE.values()) == pytest.approx(0.25 * 120.0 / STIFFNESS)
    assert min(X2_ACTION_SCALE.values()) == pytest.approx(0.25 * 36.0 / STIFFNESS)
