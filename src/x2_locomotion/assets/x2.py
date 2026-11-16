"""AgiBot X2 Ultra, legs only, as an mjlab entity.

The official MJCF (``third_party/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra.xml``) is loaded as is and
edited through ``MjSpec``:

* its own actuators and sensors are dropped (mjlab creates PD actuators from :data:`X2_ARTICULATION`
  and the sensors below);
* the 19 non-leg joints (waist, head, arms) are deleted, so those bodies become rigid and stay at
  q = 0; the policy controls the 12 leg joints only;
* the 12 foot spheres per foot get names (``{side}_foot{i}_collision``) so a ``CollisionCfg`` can
  address them, and a ``{side}_foot`` site is added at the sole;
* ``imu_ang_vel`` / ``imu_lin_vel`` / ``root_angmom`` sensors are added (names expected by the
  mjlab velocity task).

Everything numeric that is *not* read from the official model (PD gains, armature, action scale) is a
starting value, not an identified one; see docs/DESIGN.md.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

REPO_ROOT = Path(os.environ.get("X2_REPO_ROOT", Path(__file__).resolve().parents[3]))
X2_XML = Path(
    os.environ.get(
        "X2_MJCF_PATH",
        REPO_ROOT / "third_party" / "agibot_x2_urdf" / "X2_URDF-v1.4.0" / "X2-Ultra.xml",
    )
)

LEG_JOINT_RE = re.compile(r"^(left|right)_(hip_(pitch|roll|yaw)|knee|ankle_(pitch|roll))_joint$")
NUM_LEG_JOINTS = 12
SIDES = ("left", "right")
FOOT_BODY = "{side}_ankle_roll_link"
FOOT_GEOM = "{side}_foot{i}_collision"
IMU_SITE = "imu_0"  # pelvis site already present in the official MJCF
# Sole height in the ankle_roll_link frame: lowest foot sphere centre z (-0.068) minus its radius (0.005).
FOOT_SITE_POS = (0.037, 0.0, -0.073)


def get_spec() -> mujoco.MjSpec:
    """Official MJCF with only the 12 leg joints articulated (plus the floating base)."""
    if not X2_XML.is_file():
        raise FileNotFoundError(
            f"{X2_XML} not found - run `git submodule update --init` (or set X2_MJCF_PATH)."
        )
    spec = mujoco.MjSpec.from_file(str(X2_XML))

    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for sensor in list(spec.sensors):
        spec.delete(sensor)

    for joint in list(spec.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE or LEG_JOINT_RE.match(joint.name):
            continue
        spec.delete(joint)

    for side in SIDES:
        body = spec.body(FOOT_BODY.format(side=side))
        spheres = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_SPHERE]
        for i, geom in enumerate(spheres, start=1):
            geom.name = FOOT_GEOM.format(side=side, i=i)
        body.add_site(name=f"{side}_foot", pos=list(FOOT_SITE_POS), size=[0.01, 0.0, 0.0])

    spec.add_sensor(
        name="imu_ang_vel",
        type=mujoco.mjtSensor.mjSENS_GYRO,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=IMU_SITE,
    )
    spec.add_sensor(
        name="imu_lin_vel",
        type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=IMU_SITE,
    )
    spec.add_sensor(
        name="root_angmom",
        type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname="pelvis",
    )
    return spec


##
# Actuators. Effort limits come from the official model (README: ankle pitch 60 N m, ankle roll 36 N m,
# everything else on the legs 120 N m). Armature is the official MJCF default (0.03). Stiffness and
# damping follow the mjlab G1 recipe: K = I * w^2, D = 2 * zeta * I * w with w = 2 pi * 10 Hz, zeta = 2.
##

ARMATURE = 0.03
NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0
STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2.0 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

HIP_KNEE_EFFORT = 120.0
ANKLE_PITCH_EFFORT = 60.0
ANKLE_ROLL_EFFORT = 36.0

_ACTUATOR_GROUPS: tuple[tuple[tuple[str, ...], float], ...] = (
    (
        (r".*_hip_pitch_joint", r".*_hip_roll_joint", r".*_hip_yaw_joint", r".*_knee_joint"),
        HIP_KNEE_EFFORT,
    ),
    ((r".*_ankle_pitch_joint",), ANKLE_PITCH_EFFORT),
    ((r".*_ankle_roll_joint",), ANKLE_ROLL_EFFORT),
)

X2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=names,
            stiffness=STIFFNESS,
            damping=DAMPING,
            effort_limit=effort,
            armature=ARMATURE,
        )
        for names, effort in _ACTUATOR_GROUPS
    ),
    # 1.0, not mjlab's 0.9: the hip-roll ranges of this model are asymmetric (left -0.24..2.91 rad,
    # right -2.91..0.24), so shrinking them around their centre would push q = 0 to the edge.
    soft_joint_pos_limit_factor=1.0,
)

# Same rule as the mjlab G1: action = 0.25 * effort_limit / stiffness (rad per unit action).
X2_ACTION_SCALE: dict[str, float] = {
    expr: 0.25 * effort / STIFFNESS for names, effort in _ACTUATOR_GROUPS for expr in names
}

##
# Initial state: knees bent, flat foot (hip_pitch + knee + ankle_pitch = 0, checked in MuJoCo).
# Pelvis height 0.657 m puts the lowest foot sphere on the ground for this posture.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.657),
    joint_pos={
        r".*_hip_pitch_joint": -0.25,
        r".*_knee_joint": 0.5,
        r".*_ankle_pitch_joint": -0.25,
    },
    joint_vel={r".*": 0.0},
)

# Feet only: every other geom is disabled, so the policy sees terrain contacts through the 24 foot
# spheres (sphere-vs-plane/box/heightfield only, cheap and supported by MuJoCo Warp).
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_foot\d+_collision$",),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)


def get_x2_legs_cfg() -> EntityCfg:
    """Fresh entity config each call (configs get mutated by the task builders)."""
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=X2_ARTICULATION,
        # ctrl / action order = joint definition order (left leg, right leg), independent of the
        # order of the actuator groups above. This is the order the exported policy uses.
        sort_actuators=True,
    )
