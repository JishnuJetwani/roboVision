"""Coarse reset-only bootstrap for a centered stationary grasp specialist.

The first lesson begins near contact without initializing penetration or a grasp.
Later lessons open the jaws and broaden relative pose/velocity support. Every
action remains all five original learned forces. No control is applied at reset
or during a step, and the previous grasp experiment remains unchanged.
"""

from dataclasses import asdict, dataclass, replace
from numbers import Real
import math
import mujoco
import numpy as np
from .hierarchical_skills import GraspSkillEnv
from .joint_env import JointGraspEnv

VERSION = "centered-grasp-closure-bootstrap-v1"


@dataclass(frozen=True)
class BootstrapReset:
    opening: float
    relative_z_low: float
    relative_z_high: float


BOOTSTRAP_RESETS = (
    BootstrapReset(0.029, 0.005, 0.01),
    BootstrapReset(0.033, 0.0025, 0.0125),
    BootstrapReset(0.045, 0.0, 0.025),
    BootstrapReset(0.045, 0.0, 0.025),
)


def _optional_number(value, name, low, high):
    if value is None:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or (not math.isfinite(value))
        or (not low <= value <= high)
    ):
        raise ValueError(f"{name} must be finite and within [{low},{high}]")
    return float(value)


class CenteredGraspBootstrapEnv(GraspSkillEnv):
    """Learn near-contact stabilization before progressively reopening the hand.

    ``fixed_height`` or explicit ``height_bands`` retains GeneralizationGraspEnv's
    world-plane convention (hand Z=.310+height) for frozen comparison scenes.
    Without those overrides, reset Z is sampled relative to the actual cup center
    from BOOTSTRAP_RESETS. ``fixed_relative_z`` explicitly overrides that choice.
    ``fixed_opening=.045`` tests a fully open start at any curriculum stage.
    """

    version = VERSION
    target_relative_z = 0.0075
    lessons = tuple(
        (
            replace(lesson)
            for lesson, reset in zip(GraspSkillEnv.lessons, BOOTSTRAP_RESETS)
        )
    )

    def __init__(self, *, fixed_opening=None, fixed_relative_z=None, **kwargs):
        self.fixed_opening = _optional_number(
            fixed_opening, "fixed_opening", 0.028, 0.045
        )
        self.fixed_relative_z = _optional_number(
            fixed_relative_z, "fixed_relative_z", 0.0, 0.06
        )
        self._world_height_override = (
            kwargs.get("fixed_height") is not None or "height_bands" in kwargs
        )
        if self.fixed_relative_z is not None and self._world_height_override:
            raise ValueError(
                "Use either relative-Z or world-plane height overrides, not both"
            )
        super().__init__(**kwargs)

    def specification(self):
        spec = super().specification()
        spec.update(
            version=VERSION,
            bootstrap_resets=[asdict(lesson) for lesson in BOOTSTRAP_RESETS],
            fixed_opening=self.fixed_opening,
            fixed_relative_z=self.fixed_relative_z,
            world_height_override=self._world_height_override,
            default_reset_height_reference="Cup-relative Z; inherited height_bands are used only with explicit override",
            reset_height_info="Actual hand world Z minus .310 m; may be negative for centered bootstrap starts",
            initial_grasp=False,
            reset_control_force=0.0,
            reset_only_curriculum=True,
            previous_grasp_environment_unchanged=True,
        )
        return spec

    def reset(self, *, seed=None, options=None):
        render_images = self.render_images
        self.render_images = False
        try:
            super().reset(seed=seed, options=options)
        finally:
            self.render_images = render_images
        reset_lesson = BOOTSTRAP_RESETS[self.episode_skill_stage]
        if self.fixed_relative_z is not None:
            relative_z = self.fixed_relative_z
        elif self._world_height_override:
            relative_z = float(self.grasp_position[2] - self.cup_position[2])
        else:
            relative_z = float(
                self.rng.uniform(
                    reset_lesson.relative_z_low, reset_lesson.relative_z_high
                )
            )
        opening = (
            reset_lesson.opening if self.fixed_opening is None else self.fixed_opening
        )
        if not self._world_height_override:
            self.data.qpos[:4] = self.inverse_kinematics(
                [0.32, 0.0, self.cup_position[2] + relative_z]
            )
        self.data.qpos[4:6] = opening
        self.data.qvel[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._install_arrival_velocity()
        mujoco.mj_forward(self.model, self.data)
        self._check_simulation()
        fingers = set().union(*self._finger_geoms)
        for contact in self.data.contact:
            first, second = (int(contact.geom1), int(contact.geom2))
            robot_cup = (
                first in fingers
                and second in self._cup_geom_set
                or (second in fingers and first in self._cup_geom_set)
            )
            if robot_cup and contact.dist < -1e-07:
                raise ValueError("Reset opening causes finger/cup penetration")
        if self._table_collision():
            raise ValueError("Bootstrap reset causes a robot/table collision")
        self.episode_reset_height = float(self.grasp_position[2] - 0.31)
        self._previous_frame = None
        info = self._generalization_info(self._info())
        info.update(
            reset_relative_z=relative_z,
            reset_opening=opening,
            reset_height_reference="world Z minus .310 m",
            is_success=False,
            full_pickup_success=False,
        )
        metrics = self._phase_metrics(info)
        return (self._observation(), self._skill_info(info, metrics))


MOVING_VERSION = "moving-centered-grasp-bootstrap-v1"
FULL_PICKUP_VERSION = "full-pickup-bootstrap-evaluation-v1"


def centered_contact_quality(metrics):
    """Smooth distance outside the benchmark center band, with a flat valid band."""
    xy_outside = max(0.0, metrics["xy_error_m"] - 0.015)
    z_outside = max(0.0, -metrics["relative_z_m"], metrics["relative_z_m"] - 0.015)
    return math.exp(-math.hypot(xy_outside / 0.015, z_outside / 0.015))


class _CenteredFullPickupMetric:
    """Diagnostic counter matching the existing centered full-pickup benchmark."""

    def _reset_centered_full_pickup(self):
        self._centered_full_pickup_steps = 0
        self._centered_full_pickup_ever = False

    def _update_centered_full_pickup(self, info, metrics):
        geometry = (
            metrics["xy_error_m"] <= 0.015 + 1e-12
            and -1e-12 <= metrics["relative_z_m"] <= 0.015 + 1e-12
            and (metrics["downward"] >= math.cos(math.radians(15.0)) - 1e-12)
        )
        safe = info.get("reason") not in (
            "table_collision",
            "out_of_bounds",
            "grip_lost",
        )
        stable = (
            geometry
            and all(info["contacts"])
            and (info["clearance"] >= 0.06)
            and (info["upright"] >= math.cos(math.radians(20.0)))
            and (info["cup_speed"] < 0.1)
            and safe
        )
        self._centered_full_pickup_steps = (
            self._centered_full_pickup_steps + 1 if stable else 0
        )
        self._centered_full_pickup_ever |= self._centered_full_pickup_steps >= 25

    def _centered_full_pickup_info(self, info):
        return {
            **info,
            "centered_full_pickup_success": self._centered_full_pickup_steps >= 25,
            "centered_full_pickup_success_ever": self._centered_full_pickup_ever,
            "centered_full_pickup_stable_steps": self._centered_full_pickup_steps,
        }

    @staticmethod
    def _centered_full_pickup_specification():
        return dict(
            centered_full_pickup_metric="Existing centered benchmark: XY<=15 mm, relative Z0–15 mm, downward tilt<=15 degrees, original bilateral contact, clearance>=60 mm, upright<=20 degrees, cup speed<0.1 m/s for25 consecutive actions",
            centered_full_pickup_contact_force="Original contact threshold only; phase minimum0.5 N is not added to this metric",
            centered_full_pickup_metric_changes_termination=False,
        )


class MovingGraspBootstrapEnv(_CenteredFullPickupMetric, CenteredGraspBootstrapEnv):
    """A secure centered moving grasp is a valid handoff to the lift specialist.

    Relative slip must be small. Upward motion of a cup moving with the gripper
    is permitted: an absolute table-height or cup-speed requirement would reject
    useful transfers. Full pickup is measured separately and is never inferred
    from the shorter phase endpoint.
    """

    version = MOVING_VERSION
    lessons = tuple(
        (
            replace(lesson, xy_tolerance=0.015, minimum_contact_force=0.5)
            for lesson in CenteredGraspBootstrapEnv.lessons
        )
    )

    def specification(self):
        spec = super().specification()
        spec.update(
            version=MOVING_VERSION,
            maximum_grasp_clearance_m=None,
            maximum_phase_cup_speed_m_s=None,
            minimum_phase_contact_force_n=0.5,
            phase_allows_secure_upward_motion=True,
            phase_success_is_full_pickup=False,
            centered_bilateral_quality="min(minimum contact force/0.5 N,1) * exp(-hypot(XY distance outside15 mm band/15 mm, Z distance outside0–15 mm band/15 mm))",
            premature_lift_penalty=False,
            **self._centered_full_pickup_specification(),
        )
        return spec

    def _goal(self, metrics):
        return (
            metrics["xy_error_m"] <= 0.015 + 1e-12
            and -1e-12 <= metrics["relative_z_m"] <= 0.015 + 1e-12
            and (metrics["downward"] >= math.cos(math.radians(15.0)) - 1e-12)
            and all(metrics["contacts"])
            and (min(metrics["contact_forces_n"]) >= 0.5)
            and (metrics["relative_speed_m_s"] <= self._lesson.maximum_speed)
            and (metrics["cup_upright"] >= math.cos(math.radians(20.0)))
        )

    def _costs(self, metrics):
        costs = super()._costs(metrics)
        force_quality = float(np.clip(min(metrics["contact_forces_n"]) / 0.5, 0.0, 1.0))
        costs["bilateral"] = 0.7 * (
            1.0 - force_quality * centered_contact_quality(metrics)
        )
        costs.pop("premature_lift")
        return costs

    def reset(self, *, seed=None, options=None):
        self._reset_centered_full_pickup()
        observation, info = super().reset(seed=seed, options=options)
        return (observation, self._centered_full_pickup_info(info))

    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)
        self._update_centered_full_pickup(info, info["phase_metrics"])
        return (
            observation,
            reward,
            done,
            truncated,
            self._centered_full_pickup_info(info),
        )


class FullPickupBootstrapEnv(_CenteredFullPickupMetric, CenteredGraspBootstrapEnv):
    """Reset-only evaluation with original full-pickup reward and termination.

    This environment does not train or terminate at a grasp phase boundary. It
    can verify full centered pickups from the easier resets without stepping
    beyond a terminated phase episode. A run ends on original full success,
    original physical failure, or the original 500-action deadline.
    """

    version = FULL_PICKUP_VERSION
    phase = "full_pickup"

    def specification(self):
        spec = super().specification()
        spec.update(
            version=FULL_PICKUP_VERSION,
            evaluation_only=True,
            success="Original complete pickup and stable hold",
            original_success_terminates=True,
            original_reward=True,
            phase_success_is_full_pickup=True,
            phase_reward="Original JointGraspEnv reward",
            phase_success_bonus=None,
            maximum_state_cost_rate=None,
            minimum_state_cost_rate=None,
            maximum_grasp_clearance_m=None,
            **self._centered_full_pickup_specification(),
        )
        spec.pop("grasp_is_not_lift_success", None)
        return spec

    def _full_pickup_info(self, info):
        success = bool(info.get("is_success", False))
        info = {
            **info,
            "phase": self.phase,
            "phase_success": success,
            "full_pickup_success": success,
            "full_pickup_success_ever": self._full_pickup_success_ever,
            "phase_stable_steps": self._hold_steps,
            "skill_stage": self.episode_skill_stage,
            "phase_metrics": self._phase_metrics(info),
        }
        info.pop("phase_goal_reached", None)
        return self._centered_full_pickup_info(info)

    def reset(self, *, seed=None, options=None):
        self._reset_centered_full_pickup()
        observation, info = super().reset(seed=seed, options=options)
        return (observation, self._full_pickup_info(info))

    def step(self, action):
        observation, reward, done, truncated, info = JointGraspEnv.step(self, action)
        self._full_pickup_success_ever |= bool(info["is_success"])
        info = self._generalization_info(info)
        self._update_centered_full_pickup(info, self._phase_metrics(info))
        return (observation, reward, done, truncated, self._full_pickup_info(info))
