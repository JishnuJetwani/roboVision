"""Open-hand hold bootstrap followed by small descent curriculum increments.

All control remains direct-torque PPO. Early successes are approach subtasks,
not pickups; only the last lessons require full pickup/hold. No task signal or
privileged geometry is added to policy observations.
"""

from dataclasses import asdict, dataclass
import numpy as np
from .approach_subtask import ApproachLesson, ApproachSubtaskEnv

VERSION = "open-hold-bootstrap-v1"


@dataclass(frozen=True)
class BootstrapLesson(ApproachLesson):
    stable_steps: int = 3
    minimum_opening: float = 0.038
    maximum_speed: float = 0.12
    reset_height: float | None = None


LESSONS = tuple(
    [
        BootstrapLesson(
            0.075,
            1.0,
            stable_steps=1,
            minimum_opening=0.032,
            maximum_speed=0.2,
            reset_height=0.075,
        ),
        BootstrapLesson(
            0.075,
            1.0,
            stable_steps=2,
            minimum_opening=0.035,
            maximum_speed=0.2,
            reset_height=0.075,
        ),
        BootstrapLesson(0.075, 1.0, stable_steps=3, reset_height=0.075),
    ]
    + [
        BootstrapLesson(h, 1.0)
        for h in (
            0.07,
            0.065,
            0.06,
            0.055,
            0.05,
            0.045,
            0.04,
            0.035,
            0.03,
            0.025,
            0.02,
            0.015,
            0.01,
            0.005,
            0.0,
        )
    ]
    + [
        BootstrapLesson(0.0, 0.75),
        BootstrapLesson(0.0, 0.5),
        BootstrapLesson(0.0, 0.25),
        BootstrapLesson(0.0, 0.0, 10.0),
        BootstrapLesson(0.0, 0.0, 0.0),
    ]
)
FINAL_LESSON = len(LESSONS) - 1
FULL_APPROACH_LESSON = next(
    (i for i, l in enumerate(LESSONS) if l.target_height == 0.0)
)
FULL_PICKUP_LESSON = next(
    (i for i, l in enumerate(LESSONS) if l.approach_fraction == 0.0)
)


class OpenBootstrapEnv(ApproachSubtaskEnv):
    lessons = LESSONS
    first_descent_lesson = 3
    fixed_descent_reset_lessons = (3, 4)

    def __init__(
        self,
        *,
        early_close_failure=False,
        strict_descent=False,
        bootstrap_height_jitter=0.0,
        **kwargs,
    ):
        self.early_close_failure = bool(early_close_failure)
        self.strict_descent = bool(strict_descent)
        if (
            not np.isfinite(bootstrap_height_jitter)
            or not 0 <= bootstrap_height_jitter <= 0.065
        ):
            raise ValueError(
                "bootstrap_height_jitter must be finite and in [0, .065] meters"
            )
        self.bootstrap_height_jitter = float(bootstrap_height_jitter)
        super().__init__(**kwargs)

    def _metrics(self):
        quality, reached = super()._metrics()
        if (
            self.strict_descent
            and self.episode_subtask_stage >= self.first_descent_lesson
        ):
            z_error = float(
                self.grasp_position[2]
                - self.cup_position[2]
                - 0.014
                - self._lesson.target_height
            )
            reached = bool(reached and -0.005 <= z_error <= 0.001)
        return (quality, reached)

    def _early_failure(self, info):
        previous = super()._early_failure(info)
        if previous:
            return previous
        if (
            self.early_close_failure
            and self.approach_episode
            and (np.min(self.data.qpos[4:6]) < 0.03)
        ):
            return "closed_before_approach"
        return ""

    def _sample_approach_height(self):
        if self.fixed_height is not None:
            return float(self.fixed_height)
        center = (
            0.075
            if self.strict_descent
            and self.episode_subtask_stage in self.fixed_descent_reset_lessons
            else self._lesson.reset_height
        )
        if center is not None:
            if self.bootstrap_height_jitter:
                return float(
                    self.rng.uniform(
                        max(0.0, center - self.bootstrap_height_jitter),
                        min(0.14, center + self.bootstrap_height_jitter),
                    )
                )
            return float(center)
        return super()._sample_approach_height()

    @staticmethod
    def specification():
        return dict(
            version=VERSION,
            lessons=[asdict(l) for l in LESSONS],
            full_approach_lesson=FULL_APPROACH_LESSON,
            full_pickup_lesson=FULL_PICKUP_LESSON,
            final_lesson=FINAL_LESSON,
            action_demonstrations=False,
            privileged_observations=False,
            final_evaluation="unmodified JointGraspEnv(stage=3)",
            optional_bootstrap_height_jitter=dict(
                default=0.0,
                units="meters",
                lessons=[0, 1, 2],
                strict_descent_additional_lessons=[3, 4],
                distribution="uniform(reset_height-jitter, reset_height+jitter)",
                maximum=0.065,
                explicit_evaluation_height_overrides=True,
            ),
            optional_strict_descent=dict(
                default=False,
                start_lesson=3,
                arrival_z_error_m=[-0.005, 0.001],
                fixed_training_reset_lessons=[3, 4],
                fixed_training_reset_height=0.075,
                explicit_evaluation_height_overrides=True,
            ),
        )
