"""Fine reset-height curriculum using only the original complete pickup task."""

from dataclasses import asdict
from numbers import Integral
import mujoco
from .open_bootstrap import BootstrapLesson
from .reverse_curriculum import ReverseGraspEnv, LEVELS as ORIGINAL_LEVELS

HEIGHTS = tuple((i / 1000.0 for i in range(10, 51))) + (
    0.055,
    0.06,
    0.065,
    0.07,
    0.075,
    0.08,
    0.09,
    0.1,
    0.12,
    0.14,
)
LESSONS = tuple(
    (BootstrapLesson(0.0, 0.0, shaping_scale=0.0, reset_height=h) for h in HEIGHTS)
)
FINAL_LESSON = len(LESSONS) - 1
VERSION = "near-pickup-height-v1"


class NearPickupEnv(ReverseGraspEnv):
    lessons = LESSONS
    heights = HEIGHTS
    version = VERSION

    def __init__(
        self,
        *,
        subtask_stage=0,
        replay_fraction=0.0,
        frontier_attempt_limit=None,
        **kwargs,
    ):
        if replay_fraction != 0:
            raise ValueError("Near pickup requires explicit separate retention workers")
        self.set_subtask_stage(subtask_stage)
        super().__init__(curriculum_level=19, replay_fraction=0.0, **kwargs)
        self.set_frontier_attempt_limit(frontier_attempt_limit)

    def set_frontier_attempt_limit(self, steps):
        """Optional training deadline; preserve the original observation clock."""
        if steps is not None:
            if (
                isinstance(steps, bool)
                or not isinstance(steps, Integral)
                or (not 1 <= steps <= self.max_steps)
            ):
                raise ValueError("Invalid frontier_attempt_limit")
            steps = int(steps)
        self.frontier_attempt_limit = steps

    def _early_failure(self, info):
        previous = super()._early_failure(info)
        if previous:
            return previous
        if (
            self.frontier_attempt_limit is not None
            and self.step_count >= self.frontier_attempt_limit
            and (self._hold_steps < self.hold_steps)
        ):
            return "frontier_attempt_limit"
        return ""

    def set_subtask_stage(self, stage):
        if (
            isinstance(stage, bool)
            or not isinstance(stage, int)
            or (not 0 <= stage < len(self.lessons))
        ):
            raise ValueError("Invalid near pickup stage")
        self.subtask_stage = stage

    def _near_info(self, info):
        return {
            **info,
            "subtask_stage": self.episode_subtask_stage,
            "near_pickup_height": self.heights[self.episode_subtask_stage],
            "approach_height": self.heights[self.episode_subtask_stage],
            "approach_episode": False,
            "pickup_success": bool(info.get("is_success", False)),
            "curriculum_kind": self.version,
            "curriculum_name": f"near-pickup-{self.heights[self.episode_subtask_stage]:.3f}",
        }

    def reset(self, *, seed=None, options=None):
        self.episode_subtask_stage = self.subtask_stage
        height = self.heights[self.episode_subtask_stage]
        exact = next(
            (i for i in range(18, 26) if ORIGINAL_LEVELS[i].approach_height == height),
            None,
        )
        self.curriculum_level = exact if exact is not None else 19
        obs, info = super().reset(seed=seed, options=options)
        if exact is not None:
            return (obs, self._near_info(info))
        self.data.qpos[:4] = self.inverse_kinematics(
            self.cup_position + [0, 0, 0.014 + self.heights[self.episode_subtask_stage]]
        )
        mujoco.mj_forward(self.model, self.data)
        self._previous_frame = None
        return (self._observation(), self._near_info({**info, **self._info()}))

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        return (obs, reward, done, truncated, self._near_info(info))

    @classmethod
    def specification(cls):
        return dict(
            version=cls.version,
            lessons=[asdict(l) for l in cls.lessons],
            heights=list(cls.heights),
            final_lesson=len(cls.lessons) - 1,
            success="Original full pickup and stable hold",
            shaping=False,
            action_demonstrations=False,
            final_evaluation="JointGraspEnv(stage=3)",
        )


FINE_HEIGHTS = tuple((mm / 1000.0 for mm in range(10, 81))) + (
    0.085,
    0.09,
    0.095,
    0.1,
    0.11,
    0.12,
    0.13,
    0.14,
)
FINE_LESSONS = tuple(
    (BootstrapLesson(0.0, 0.0, shaping_scale=0.0, reset_height=h) for h in FINE_HEIGHTS)
)


class FineNearPickupEnv(NearPickupEnv):
    heights = FINE_HEIGHTS
    lessons = FINE_LESSONS
    version = "fine-near-pickup-height-v1"


MILLIMETER_HEIGHTS = tuple((mm / 1000.0 for mm in range(10, 141)))
MILLIMETER_LESSONS = tuple(
    (
        BootstrapLesson(0.0, 0.0, shaping_scale=0.0, reset_height=h)
        for h in MILLIMETER_HEIGHTS
    )
)


class MillimeterNearPickupEnv(NearPickupEnv):
    """One-millimeter reset increments through the full 140 mm near-pickup range."""

    heights = MILLIMETER_HEIGHTS
    lessons = MILLIMETER_LESSONS
    version = "millimeter-near-pickup-height-v1"
