"""Pure force-control PPO lessons with explicit approach/grasp handoff goals.

These environments change resets, rewards and subtask success only. The actor
still sees two RGB frames plus the original 18 proprioceptive values and emits
all five direct torque/force commands. Object geometry is reward/evaluation
information, never an additional actor input or a control intervention.
"""

from dataclasses import asdict, dataclass
import math
from numbers import Real
import mujoco
import numpy as np
from .generalization_env import GeneralizationGraspEnv
from .joint_env import JointGraspEnv
from .reverse_curriculum import ReverseGraspEnv, LEVELS

VERSION = "hierarchical-force-skills-v1"


@dataclass(frozen=True)
class ApproachLesson:
    xy_tolerance: float
    z_tolerance: float
    maximum_speed: float
    minimum_opening: float
    maximum_tilt_degrees: float
    stable_steps: int
    height_low: float = 0.025
    height_high: float = 0.14


@dataclass(frozen=True)
class GraspLesson:
    xy_tolerance: float
    z_low: float
    z_high: float
    maximum_speed: float
    minimum_contact_force: float
    stable_steps: int
    height_low: float
    height_high: float
    arrival_xy_speed: float = 0.0
    arrival_z_low: float = 0.0
    arrival_z_high: float = 0.0
    maximum_tilt_degrees: float = 15.0


APPROACH_STAGES = (
    ApproachLesson(0.02, 0.02, 0.15, 0.037, 25.0, 3),
    ApproachLesson(0.015, 0.012, 0.08, 0.039, 20.0, 5),
    ApproachLesson(0.012, 0.008, 0.05, 0.04, 15.0, 10),
)
GRASP_STAGES = (
    GraspLesson(0.015, 0.0, 0.015, 0.1, 0.2, 5, 0.0, 0.008),
    GraspLesson(0.015, 0.0, 0.015, 0.07, 0.35, 8, 0.0, 0.015),
    GraspLesson(0.012, 0.0, 0.015, 0.05, 0.5, 10, 0.0, 0.025),
    GraspLesson(0.012, 0.0, 0.015, 0.05, 0.5, 10, 0.0, 0.025, 0.03, -0.06, 0.03),
)


def phase_reward(
    costs,
    action,
    previous_action,
    *,
    dt,
    gamma,
    remaining_steps,
    success,
    failure,
    maximum_state_cost=2.0,
):
    """Negative state costs, a one-time goal bonus, and an absorbing failure tail.

    Every nonterminal tick is negative, including hovering at the goal. The
    tail charges the worst remaining per-tick cost, so a crash cannot avoid the
    rest of its attempt. Geometry components stay separately inspectable.
    """
    if success and failure:
        raise ValueError("A phase outcome cannot be both success and failure")
    if not 0.0 < gamma <= 1.0 or dt <= 0.0 or remaining_steps < 0:
        raise ValueError("Invalid reward discount, timestep or remaining horizon")
    if (
        any((not math.isfinite(value) or value < 0.0 for value in costs.values()))
        or sum(costs.values()) > maximum_state_cost + 1e-12
    ):
        raise ValueError("Phase state costs exceed the declared bound")
    terms = {f"phase_{name}_cost": -dt * value for name, value in costs.items()}
    terms.update(
        effort=-dt * 0.02 * float(np.square(action).mean()),
        force_change=-dt * 0.01 * float(np.square(action - previous_action).mean()),
        phase_bonus=100.0 if success else 0.0,
        phase_failure=0.0,
        remaining_phase_cost=0.0,
        crash=0.0,
    )
    if failure:
        future = (
            remaining_steps
            if gamma == 1.0
            else -math.expm1(remaining_steps * math.log(gamma)) / (1.0 - gamma)
        )
        terms["phase_failure"] = -25.0 * gamma**remaining_steps
        terms["remaining_phase_cost"] = (
            -dt * (maximum_state_cost + 0.02 + 0.04) * gamma * future
        )
        if remaining_steps:
            terms["crash"] = -5.0
    return terms


class _PhaseSkillEnv(GeneralizationGraspEnv):
    version = VERSION
    target_relative_z = 0.014
    lessons = ()
    phase = ""

    def __init__(self, *, skill_stage=0, **kwargs):
        self._explicit_height_bands = "height_bands" in kwargs
        self.set_skill_stage(skill_stage)
        lesson = self.lessons[self.skill_stage]
        kwargs.setdefault(
            "height_bands", ((lesson.height_low, lesson.height_high, 1.0),)
        )
        kwargs.setdefault("gamma", 0.999)
        super().__init__(**kwargs)

    def set_skill_stage(self, stage):
        if (
            isinstance(stage, (bool, np.bool_))
            or not isinstance(stage, (int, np.integer))
            or (not 0 <= stage < len(self.lessons))
        ):
            raise ValueError(f"Invalid {self.phase} skill stage")
        self.skill_stage = int(stage)
        if hasattr(self, "height_bands"):
            lesson = self.lessons[self.skill_stage]
            if not self._explicit_height_bands:
                self.height_bands = ((lesson.height_low, lesson.height_high, 1.0),)

    def specification(self):
        return {
            **super().specification(),
            "version": VERSION,
            "phase": self.phase,
            "skill_stage": self.skill_stage,
            "lessons": [asdict(lesson) for lesson in self.lessons],
            "target_relative_z": self.target_relative_z,
            "success": "Phase handoff success, reported separately from full pickup",
            "shaping": False,
            "phase_reward": "Negative additive state costs plus one-time phase bonus and worst-cost absorbing failure tail",
            "phase_success_bonus": 100.0,
            "maximum_state_cost_rate": 2.0,
            "minimum_state_cost_rate": 0.1,
            "original_success_terminates": False,
            "privileged_handoff_metrics": "Reward/evaluation only; no actor inputs or control overrides",
            "height_bands_explicit_override": self._explicit_height_bands,
        }

    def _contact_forces(self):
        forces = np.zeros(2)
        wrench = np.zeros(6)
        for index, contact in enumerate(self.data.contact):
            first, second = (int(contact.geom1), int(contact.geom2))
            for side, finger in enumerate(self._finger_geoms):
                if (
                    first in finger
                    and second in self._cup_geom_set
                    or (second in finger and first in self._cup_geom_set)
                ):
                    mujoco.mj_contactForce(self.model, self.data, index, wrench)
                    forces[side] += max(0.0, float(wrench[0]))
        return forces

    def _phase_metrics(self, info):
        error = (
            self.grasp_position - self.cup_position - [0.0, 0.0, self.target_relative_z]
        )
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacobian, None, self._grasp_sid)
        hand_velocity = jacobian @ self.data.qvel
        relative_velocity = (
            hand_velocity - self.data.qvel[self._cup_vadr : self._cup_vadr + 3]
        )
        axis = self.data.site_xmat[self._grasp_sid].reshape(3, 3)[:, 0]
        downward = float(-axis[2])
        xy_error, z_error = (float(np.linalg.norm(error[:2])), float(error[2]))
        opening = float(
            np.clip((np.min(self.data.qpos[4:6]) - 0.028) / 0.014, 0.0, 1.0)
        )
        return dict(
            x_error_m=float(error[0]),
            y_error_m=float(error[1]),
            xy_error_m=xy_error,
            z_error_m=z_error,
            relative_z_m=z_error + self.target_relative_z,
            jaw_gap_m=float(np.sum(self.data.qpos[4:6])),
            minimum_jaw_m=float(np.min(self.data.qpos[4:6])),
            relative_speed_m_s=float(np.linalg.norm(relative_velocity)),
            relative_velocity_m_s=relative_velocity.tolist(),
            hand_velocity_m_s=hand_velocity.tolist(),
            downward=downward,
            tilt_degrees=math.degrees(math.acos(float(np.clip(downward, -1.0, 1.0)))),
            openness_score=opening,
            xy_score=math.exp(-xy_error / 0.06),
            z_score=math.exp(-abs(z_error) / 0.1),
            orientation_score=float(np.clip(downward, 0.0, 1.0)) ** 4,
            speed_score=math.exp(-float(np.linalg.norm(relative_velocity)) / 0.1),
            contact_forces_n=self._contact_forces().tolist(),
            contacts=list(info["contacts"]),
            cup_speed_m_s=info["cup_speed"],
            cup_upright=info["upright"],
            clearance_m=info["clearance"],
        )

    def _skill_info(self, info, metrics, *, success=False):
        return {
            **info,
            "phase": self.phase,
            "phase_success": bool(success),
            "skill_stage": self.episode_skill_stage,
            "phase_stable_steps": self._phase_stable_steps,
            "phase_goal_reached": bool(self._goal(metrics)),
            "phase_metrics": metrics,
            "full_pickup_success_ever": self._full_pickup_success_ever,
            "reset_arrival_velocity_m_s": self._reset_arrival_velocity.tolist(),
        }

    def _install_arrival_velocity(self):
        self._reset_arrival_velocity = np.zeros(3)

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.episode_skill_stage = self.skill_stage
        self._lesson = self.lessons[self.episode_skill_stage]
        self._phase_stable_steps = 0
        self._phase_reward_totals = {}
        self._full_pickup_success_ever = False
        self._install_arrival_velocity()
        if np.any(self._reset_arrival_velocity):
            mujoco.mj_forward(self.model, self.data)
            self._previous_frame = None
            observation = self._observation()
            info = self._generalization_info(self._info())
        metrics = self._phase_metrics(info)
        return (
            observation,
            self._skill_info(
                {**info, "is_success": False, "full_pickup_success": False}, metrics
            ),
        )

    def step(self, action):
        previous_action = self.last_action.copy()
        observation, original_reward, _, truncated, info = super().step(action)
        full_success = bool(info["is_success"])
        self._full_pickup_success_ever |= full_success
        metrics = self._phase_metrics(info)
        reached = self._goal(metrics)
        self._phase_stable_steps = self._phase_stable_steps + 1 if reached else 0
        physical_failure = info["reason"] not in ("", "success", "timeout", "dropped")
        success = self._phase_stable_steps >= self._lesson.stable_steps and (
            not physical_failure
        )
        done = bool(success or physical_failure or self.step_count >= self.max_steps)
        reason = (
            info["reason"]
            if physical_failure
            else f"{self.phase}_success"
            if success
            else "timeout"
            if self.step_count >= self.max_steps
            else ""
        )
        terms = phase_reward(
            self._costs(metrics),
            self.last_action,
            previous_action,
            dt=self.control_dt,
            gamma=self.gamma,
            remaining_steps=max(0, self.max_steps - self.step_count),
            success=success,
            failure=done and (not success),
        )
        for name, value in terms.items():
            self._phase_reward_totals[name] = (
                self._phase_reward_totals.get(name, 0.0) + value
            )
        info = {
            **info,
            "is_success": bool(success),
            "reason": reason,
            "full_pickup_success": full_success,
            "original_is_success": full_success,
            "original_reward": original_reward,
            "original_reward_components": info["reward_components"].copy(),
            "reward_components": terms,
        }
        info.pop("episode_reward_components", None)
        if done:
            info["episode_reward_components"] = self._phase_reward_totals.copy()
        return (
            observation,
            float(sum(terms.values())),
            done,
            truncated,
            self._skill_info(info, metrics, success=success),
        )


class ApproachSkillEnv(_PhaseSkillEnv):
    """Reach a centered, open, downward, low-speed handoff from broad heights."""

    phase = "approach"
    lessons = APPROACH_STAGES

    def _goal(self, metrics):
        lesson = self._lesson
        return (
            metrics["xy_error_m"] <= lesson.xy_tolerance + 1e-12
            and abs(metrics["z_error_m"]) <= lesson.z_tolerance + 1e-12
            and (
                metrics["downward"]
                >= math.cos(math.radians(lesson.maximum_tilt_degrees)) - 1e-12
            )
            and (metrics["minimum_jaw_m"] >= lesson.minimum_opening)
            and (metrics["relative_speed_m_s"] <= lesson.maximum_speed)
            and (not any(metrics["contacts"]))
        )

    def _costs(self, metrics):
        proximity = math.exp(
            -metrics["xy_error_m"] / 0.04 - abs(metrics["z_error_m"]) / 0.04
        )
        return dict(
            time=0.1,
            xy=0.4 * (1.0 - metrics["xy_score"]),
            z=0.6 * (1.0 - metrics["z_score"]),
            openness=0.4 * (1.0 - metrics["openness_score"]),
            orientation=0.3 * (1.0 - metrics["orientation_score"]),
            speed=0.2 * proximity * (1.0 - metrics["speed_score"]),
        )


class GraspSkillEnv(_PhaseSkillEnv):
    """Learn centered bilateral contact at low speed while the cup remains low."""

    phase = "grasp"
    lessons = GRASP_STAGES

    def __init__(self, *, arrival_velocity_ranges=None, **kwargs):
        if arrival_velocity_ranges is not None:
            try:
                ranges = tuple((tuple(pair) for pair in arrival_velocity_ranges))
            except TypeError as error:
                raise ValueError(
                    "arrival_velocity_ranges requires three (low,high) pairs"
                ) from error
            if (
                len(ranges) != 3
                or any((len(pair) != 2 for pair in ranges))
                or any(
                    (
                        isinstance(value, (bool, np.bool_))
                        or not isinstance(value, Real)
                        or (not math.isfinite(value))
                        for pair in ranges
                        for value in pair
                    )
                )
                or any(
                    (
                        low > high or max(abs(low), abs(high)) > 0.15
                        for low, high in ranges
                    )
                )
            ):
                raise ValueError(
                    "Require three ordered finite velocity ranges within +/- .15 m/s"
                )
            self.arrival_velocity_ranges = ranges
        else:
            self.arrival_velocity_ranges = None
        super().__init__(**kwargs)

    def specification(self):
        return {
            **super().specification(),
            "arrival_velocity_ranges": self.arrival_velocity_ranges,
            "arrival_velocity_method": "Reset-only Jacobian inversion; no controller after reset",
            "maximum_grasp_clearance_m": 0.005,
            "grasp_is_not_lift_success": True,
        }

    def _install_arrival_velocity(self):
        lesson = self._lesson
        ranges = self.arrival_velocity_ranges or (
            (-lesson.arrival_xy_speed, lesson.arrival_xy_speed),
            (-lesson.arrival_xy_speed, lesson.arrival_xy_speed),
            (lesson.arrival_z_low, lesson.arrival_z_high),
        )
        velocity = np.array(
            [
                self.rng.uniform(low, high) if low != high else low
                for low, high in ranges
            ]
        )
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacobian, None, self._grasp_sid)
        matrix = np.vstack([jacobian[:, :4], [0.0, 1.0, 1.0, 1.0]])
        self.data.qvel[:4] = np.linalg.solve(matrix, np.r_[velocity, 0.0])
        self._reset_arrival_velocity = velocity

    def _goal(self, metrics):
        lesson = self._lesson
        return (
            metrics["xy_error_m"] <= lesson.xy_tolerance + 1e-12
            and lesson.z_low - 1e-12 <= metrics["relative_z_m"] <= lesson.z_high + 1e-12
            and (
                metrics["downward"]
                >= math.cos(math.radians(lesson.maximum_tilt_degrees)) - 1e-12
            )
            and (min(metrics["contact_forces_n"]) >= lesson.minimum_contact_force)
            and all(metrics["contacts"])
            and (metrics["relative_speed_m_s"] <= lesson.maximum_speed)
            and (metrics["cup_speed_m_s"] <= lesson.maximum_speed)
            and (metrics["cup_upright"] >= math.cos(math.radians(20.0)))
            and (metrics["clearance_m"] <= 0.005)
        )

    def _costs(self, metrics):
        force_quality = float(
            np.clip(
                np.min(metrics["contact_forces_n"])
                / self._lesson.minimum_contact_force,
                0.0,
                1.0,
            )
        )
        return dict(
            time=0.1,
            xy=0.3 * (1.0 - metrics["xy_score"]),
            z=0.4 * (1.0 - math.exp(-abs(metrics["z_error_m"]) / 0.025)),
            bilateral=0.7 * (1.0 - force_quality),
            orientation=0.2 * (1.0 - metrics["orientation_score"]),
            speed=0.2 * (1.0 - metrics["speed_score"]),
            premature_lift=0.1
            * float(np.clip((metrics["clearance_m"] - 0.005) / 0.04, 0.0, 1.0)),
        )


class LiftSkillEnv(ReverseGraspEnv):
    """Lift an initially grasped cup; final stage starts on the table, not aloft."""

    version = VERSION
    phase = "lift"
    lessons = (9, 10, 11)
    target_relative_z = 0.014
    _contact_forces = _PhaseSkillEnv._contact_forces
    _phase_metrics = _PhaseSkillEnv._phase_metrics

    def __init__(self, *, skill_stage=0, **kwargs):
        if any(
            (key in kwargs for key in ("stage", "curriculum_level", "replay_fraction"))
        ):
            raise ValueError(
                "LiftSkillEnv controls its reset curriculum; use skill_stage"
            )
        if kwargs.pop("max_steps", 500) != 500:
            raise ValueError("LiftSkillEnv retains the original 500-action deadline")
        self.set_skill_stage(skill_stage)
        kwargs.setdefault("gamma", 0.999)
        super().__init__(
            curriculum_level=self.lessons[self.skill_stage],
            replay_fraction=0.0,
            max_steps=500,
            **kwargs,
        )

    def set_skill_stage(self, stage):
        if (
            isinstance(stage, (bool, np.bool_))
            or not isinstance(stage, (int, np.integer))
            or (not 0 <= stage < len(self.lessons))
        ):
            raise ValueError("Invalid lift skill stage")
        self.skill_stage = int(stage)
        if hasattr(self, "curriculum_level"):
            self.curriculum_level = self.lessons[self.skill_stage]

    def specification(self):
        return dict(
            version=VERSION,
            phase=self.phase,
            skill_stage=self.skill_stage,
            lessons=[
                dict(reset_level=level, initial_clearance=LEVELS[level].clearance)
                for level in self.lessons
            ],
            max_steps=self.max_steps,
            hold_steps=self.hold_steps,
            gamma=self.gamma,
            control=dict(self.control_spec),
            observation=self.observation_mode,
            success="Original full pickup: >=60 mm clearance, bilateral, upright, low cup speed for 25 actions",
            shaping=False,
            runtime_controller=False,
            action_demonstrations=False,
            original_reward=True,
            fixed_nominal_appearance=True,
            cup_mass=0.08,
            grip_friction=1.0,
        )

    def _early_failure(self, info):
        return ""

    def _lift_info(self, info):
        success = bool(info.get("is_success", False))
        return {
            **info,
            "phase": self.phase,
            "phase_success": success,
            "full_pickup_success": success,
            "full_pickup_success_ever": self._full_pickup_success_ever,
            "skill_stage": self.episode_skill_stage,
            "phase_stable_steps": self._hold_steps,
            "phase_metrics": self._phase_metrics(info),
            "reset_curriculum_level": self.lessons[self.episode_skill_stage],
        }

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.episode_skill_stage = self.skill_stage
        self._full_pickup_success_ever = False
        return (observation, self._lift_info(info))

    def step(self, action):
        observation, reward, done, truncated, info = JointGraspEnv.step(self, action)
        self._full_pickup_success_ever |= bool(info["is_success"])
        return (observation, reward, done, truncated, self._lift_info(info))
