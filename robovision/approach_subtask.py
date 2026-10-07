"""Optional open-hand reaching lessons for direct-torque PPO.

Approach success is explicitly NOT a successful pickup. Evaluate the finished
agent in JointGraspEnv(stage=3), without these lessons or their reward.
"""

from dataclasses import dataclass
import numpy as np
import mujoco
from .approach_bridge import ApproachBridgeEnv
from .reverse_curriculum import ReverseGraspEnv

VERSION = "open-approach-subtask-v1"


@dataclass(frozen=True)
class ApproachLesson:
    target_height: float
    approach_fraction: float
    shaping_scale: float = 20.0


LESSONS = tuple(
    (
        ApproachLesson(*v)
        for v in [
            (0.05, 1.0),
            (0.035, 1.0),
            (0.02, 1.0),
            (0.0, 1.0),
            (0.0, 0.75),
            (0.0, 0.5),
            (0.0, 0.25),
            (0.0, 0.0, 10.0),
            (0.0, 0.0, 0.0),
        ]
    )
)


def approach_metrics(
    offset, downward, openings, speed, *, minimum_opening=0.038, maximum_speed=0.12
):
    """Geometry is reward-only; no extra privileged policy observations."""
    offset = np.asarray(offset)
    openness = float(np.clip((np.min(openings) - 0.028) / 0.012, 0.0, 1.0))
    quality = float(
        np.exp(-np.linalg.norm(offset / [0.025, 0.025, 0.035]))
        * np.clip(downward, 0.0, 1.0) ** 4
        * openness
    )
    reached = bool(
        np.linalg.norm(offset[:2]) < 0.012
        and abs(offset[2]) < 0.01
        and (downward > np.cos(np.deg2rad(15)))
        and (np.min(openings) >= minimum_opening)
        and (speed < maximum_speed)
    )
    return (quality, reached)


def approach_reward(quality, *, dt, gamma, remaining, success, failure):
    """Negative bounded distance cost; arrival pays once, hovering never pays.

    Failure charges a worst-case absorbing tail, avoiding a crash shortcut.
    This intentionally changes the training subtask, rather than claiming
    potential shaping alone changes optimal behavior.
    """
    terms = {
        "approach_cost": -dt * (0.2 + 4.0 * (1.0 - np.clip(quality, 0, 1))),
        "approach_success": 20.0 if success else 0.0,
        "approach_failure": 0.0,
    }
    if failure:
        future = remaining if gamma == 1 else (1 - gamma**remaining) / (1 - gamma)
        terms["approach_failure"] = -5.0 - 4.2 * dt * gamma * future
    return terms


class ApproachSubtaskEnv(ApproachBridgeEnv):
    """Drop-in bridge environment with separately reported reaching episodes.

    set_subtask_stage affects future resets only. Replay episodes preserve
    the previous pickup/hold task. Caller should gate approach and pickup
    evaluations separately, never count `is_success` as pickup without checking
    `pickup_success` or `approach_episode`.
    """

    lessons = LESSONS

    def __init__(
        self,
        *,
        subtask_stage=0,
        height_range=(0.07, 0.08),
        retention_levels=(5, 11, 17),
        **kwargs,
    ):
        self.set_subtask_stage(subtask_stage)
        if (
            len(height_range) != 2
            or not 0 <= height_range[0] <= height_range[1] <= 0.14
        ):
            raise ValueError("Require 0 <= height range <= .14")
        self.height_range = tuple(height_range)
        if not retention_levels or any(
            (
                isinstance(v, bool) or int(v) != v or (not 0 <= v <= 22)
                for v in retention_levels
            )
        ):
            raise ValueError("Retention levels must be nonempty integers from 0 to 22")
        self.retention_levels = tuple(map(int, retention_levels))
        super().__init__(**kwargs)

    def set_subtask_stage(self, stage):
        if (
            isinstance(stage, bool)
            or int(stage) != stage
            or (not 0 <= stage < len(self.lessons))
        ):
            raise ValueError("Invalid approach subtask stage")
        self.subtask_stage = int(stage)

    def _metrics(self):
        axis = self.data.site_xmat[self._grasp_sid].reshape(3, 3)[:, 0]
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self._grasp_sid)
        velocity = (
            jac @ self.data.qvel - self.data.qvel[self._cup_vadr : self._cup_vadr + 3]
        )
        offset = (
            self.grasp_position
            - self.cup_position
            - [0, 0, 0.014 + self._lesson.target_height]
        )
        return approach_metrics(
            offset,
            -axis[2],
            self.data.qpos[4:6],
            np.linalg.norm(velocity),
            minimum_opening=getattr(self._lesson, "minimum_opening", 0.038),
            maximum_speed=getattr(self._lesson, "maximum_speed", 0.12),
        )

    def _sample_approach_height(self):
        return float(
            self.fixed_height
            if self.fixed_height is not None
            else self.rng.uniform(*self.height_range)
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        replay = bool(self.replay_fraction and self.rng.random() < self.replay_fraction)
        frontier, fraction = (self.curriculum_level, self.replay_fraction)
        self.replay_fraction = 0.0
        try:
            if replay:
                self.curriculum_level = int(self.rng.choice(self.retention_levels))
                obs, info = ReverseGraspEnv.reset(self, seed=None, options=options)
                self.episode_frontier = frontier
                info = self._curriculum_info(info)
                self.episode_bridge_stage = self.bridge_stage
                self.approach_height = None
                self._approach_scale = self._approach_previous = 0.0
            else:
                obs, info = super().reset(seed=None, options=options)
        finally:
            self.curriculum_level, self.replay_fraction = (frontier, fraction)
        self.episode_subtask_stage = self.subtask_stage
        self._lesson = self.lessons[self.episode_subtask_stage]
        self.approach_episode = bool(
            not info["curriculum_replay"]
            and self.rng.random() < self._lesson.approach_fraction
        )
        if not info["curriculum_replay"]:
            self.approach_height = self._sample_approach_height()
            self.data.qpos[:4] = self.inverse_kinematics(
                self.cup_position + [0, 0, 0.014 + self.approach_height]
            )
            mujoco.mj_forward(self.model, self.data)
            self._previous_frame = None
            obs = self._observation()
            self._approach_scale = self._lesson.shaping_scale if self.shaping else 0.0
            self._approach_previous = self._approach_potential()
        self._approach_stable_steps = 0
        self._subtask_reward_totals = {}
        return (
            obs,
            {
                **info,
                "approach_height": self.approach_height,
                "approach_episode": self.approach_episode,
                "subtask_stage": self.episode_subtask_stage,
            },
        )

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        pickup_success = bool(info["is_success"])
        quality, reached = self._metrics()
        self._approach_stable_steps = self._approach_stable_steps + 1 if reached else 0
        approach_success = bool(
            self._approach_stable_steps >= getattr(self._lesson, "stable_steps", 3)
            and info["reason"] not in ("table_collision", "out_of_bounds", "grip_lost")
        )
        if self.approach_episode:
            success = approach_success
            done = bool(done or success)
            terms = approach_reward(
                quality,
                dt=self.control_dt,
                gamma=self.gamma,
                remaining=max(0, self.max_steps - self.step_count),
                success=success,
                failure=done and (not success),
            )
            reward = float(sum(terms.values()))
            for key, value in terms.items():
                self._subtask_reward_totals[key] = (
                    self._subtask_reward_totals.get(key, 0.0) + value
                )
            info.update(
                is_success=success,
                reward_components=terms,
                reason="approach_success" if success else info["reason"],
            )
            if done:
                info["episode_reward_components"] = self._subtask_reward_totals.copy()
        info.update(
            approach_episode=self.approach_episode,
            approach_success=approach_success,
            pickup_success=pickup_success,
            approach_quality=quality,
            approach_stable_steps=self._approach_stable_steps,
            subtask_stage=self.episode_subtask_stage,
        )
        return (obs, reward, done, truncated, info)
