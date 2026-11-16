"""X2 velocity-tracking environments (legs only, blind).

Both phases share *exactly* the same observation, action and reward structure; only the terrain (and the
terrain curriculum) differs. This is what lets the phase-1 checkpoint be loaded unchanged in phase 2.

Built on top of ``mjlab``'s ``make_velocity_env_cfg()`` and the Unitree G1 task as reference.
"""

from __future__ import annotations

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from x2_locomotion.assets import X2_ACTION_SCALE, get_x2_legs_cfg

from .terrains import NUM_LEVELS, X2_GENTLE_TERRAINS_CFG

TerrainKind = Literal["flat", "gentle"]

FOOT_SITES = ("left_foot", "right_foot")
FOOT_GEOMS = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 13))
TORSO = "torso_link"  # rigidly attached to the pelvis (waist locked), carries most of the mass
BASE = "pelvis"

# Command ranges (m/s, rad/s). Fixed for the whole run: no command curriculum.
CMD_LIN_VEL_X = (-0.5, 1.0)
CMD_LIN_VEL_Y = (-0.5, 0.5)
CMD_ANG_VEL_Z = (-1.0, 1.0)
EVAL_LIN_VEL_X = 0.8  # fixed forward command used by scripts/evaluate.py


def _pose_std() -> tuple[dict, dict, dict]:
    """Per-joint tolerance of the posture reward, adapted from the mjlab G1 (legs only)."""
    standing = {r".*": 0.05}
    walking = {
        r".*hip_pitch.*": 0.3,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,
    }
    running = {
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.2,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.6,
        r".*ankle_pitch.*": 0.35,
        r".*ankle_roll.*": 0.15,
    }
    return standing, walking, running


def make_x2_env_cfg(terrain: TerrainKind, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": get_x2_legs_cfg()}

    ##
    # Sensors: blind policy, so no terrain height scan; per-foot height scan + contact for rewards.
    ##
    sensors = []
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            continue
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES)
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)
        sensors.append(sensor)
    sensors.append(
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(
                mode="subtree",
                pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        )
    )
    cfg.scene.sensors = tuple(sensors)

    # Observation dimensions must be identical in both phases (and blind): drop the height scan.
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    ##
    # Terrain.
    ##
    assert cfg.scene.terrain is not None
    if terrain == "flat":
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum.pop("terrain_levels", None)
        cfg.sim.njmax = 300
        cfg.sim.nconmax = None
        cfg.sim.contact_sensor_maxmatch = 64
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = _copy_terrain_cfg()
        cfg.scene.terrain.max_init_terrain_level = 3
        cfg.sim.njmax = 500
        cfg.sim.nconmax = 40
        cfg.sim.contact_sensor_maxmatch = 128
    cfg.curriculum.pop("command_vel", None)

    ##
    # Actions / commands.
    ##
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = dict(X2_ACTION_SCALE)

    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = CMD_LIN_VEL_X
    twist.ranges.lin_vel_y = CMD_LIN_VEL_Y
    twist.ranges.ang_vel_z = CMD_ANG_VEL_Z
    twist.viz.z_offset = 1.0

    cfg.viewer.body_name = TORSO

    ##
    # Light domain randomisation: foot friction, torso mass (+ COM), encoder bias, small pushes.
    ##
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
    cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO,)
    cfg.events["torso_mass"] = EventTermCfg(
        mode="startup",
        func=dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(TORSO,)),
            "alpha_range": (-0.05, 0.05),  # mass and inertia scale by exp(2 alpha): about +-10 %
        },
    )
    cfg.events["push_robot"].interval_range_s = (8.0, 12.0)
    cfg.events["push_robot"].params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}

    ##
    # Rewards.
    ##
    standing, walking, running = _pose_std()
    pose = cfg.rewards["pose"].params
    pose["std_standing"], pose["std_walking"], pose["std_running"] = standing, walking, running
    cfg.rewards["upright"].params["asset_cfg"].body_names = (TORSO,)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (TORSO,)
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = FOOT_SITES
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["air_time"].weight = 0.0

    if play:
        _apply_play_overrides(cfg, terrain)
    return cfg


def _copy_terrain_cfg():
    from dataclasses import replace

    return replace(X2_GENTLE_TERRAINS_CFG)


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg, terrain: TerrainKind) -> None:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    if terrain == "gentle":
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain, mode="reset", params={}
        )
        tg = cfg.scene.terrain.terrain_generator
        assert tg is not None
        tg.curriculum = False
        tg.num_cols = 5
        tg.num_rows = 5
        tg.border_width = 10.0
    else:
        twist = cfg.commands["twist"]
        assert isinstance(twist, UniformVelocityCommandCfg)
        twist.ranges.lin_vel_x = (-0.5, 1.2)


def make_eval_env_cfg(terrain: TerrainKind, num_envs: int) -> ManagerBasedRlEnvCfg:
    """Deterministic evaluation setup: nominal physics, no noise, no pushes, fixed forward command.

    The terrain curriculum is removed; ``scripts/evaluate.py`` pins every env to one terrain row
    (``level``) itself. Episode = 20 s, the 'walks 20 s without falling' criterion.
    """
    cfg = make_x2_env_cfg(terrain, play=False)
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 20.0
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.terminations.pop("out_of_terrain_bounds", None)
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.seed = 0  # same geometry for every evaluated checkpoint

    for name in ("push_robot", "encoder_bias", "base_com", "torso_mass"):
        cfg.events.pop(name, None)
    cfg.events["foot_friction"].params["ranges"] = (0.8, 0.8)

    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (EVAL_LIN_VEL_X, EVAL_LIN_VEL_X)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.heading_command = False
    twist.ranges.heading = None
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.init_velocity_prob = 0.0
    twist.resampling_time_range = (1e9, 1e9)
    twist.debug_vis = False
    return cfg


__all__ = [
    "BASE",
    "CMD_ANG_VEL_Z",
    "CMD_LIN_VEL_X",
    "CMD_LIN_VEL_Y",
    "EVAL_LIN_VEL_X",
    "FOOT_GEOMS",
    "FOOT_SITES",
    "NUM_LEVELS",
    "TORSO",
    "make_eval_env_cfg",
    "make_x2_env_cfg",
]
