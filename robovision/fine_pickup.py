"""Millimeter finger-opening reset curriculum; full pickup task is unchanged."""

from collections import deque
from copy import deepcopy
from numbers import Integral
import mujoco
from .reverse_curriculum import ReverseGraspEnv

VERSION = "fine-pickup-openings-v1"
OPENINGS = tuple((mm / 1000.0 for mm in range(34, 46)))
FINAL_INDEX = len(OPENINGS) - 1


def integer(value, name, low, high=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < low
        or (high is not None and value > high)
    ):
        raise ValueError(f"Invalid {name}")
    return int(value)


class FinePickupEnv(ReverseGraspEnv):
    def __init__(
        self,
        *,
        state=None,
        initial_opening_index=0,
        minimum_episodes=16,
        window=12,
        required_successes=10,
        promotion_enabled=True,
        attempt_limit=None,
        **kwargs,
    ):
        if "curriculum_level" in kwargs or "replay_fraction" in kwargs:
            raise ValueError("FinePickupEnv owns reset level and disables replay")
        if state is None:
            state = dict(
                opening_index=integer(
                    initial_opening_index, "initial_opening_index", 0, FINAL_INDEX
                ),
                episodes_at_opening=0,
                total_episodes=0,
                recent=[],
                promotions=[],
            )
        else:
            state = deepcopy(state)
            if state.get("kind") != VERSION or state.get("openings") != list(OPENINGS):
                raise ValueError("Incompatible fine pickup state")
            minimum_episodes = state["minimum_episodes"]
            window = state["window"]
            required_successes = state["required_successes"]
            promotion_enabled = state.get("promotion_enabled", True)
            attempt_limit = state.get("attempt_limit", None)
        self.set_pickup_promotion_enabled(promotion_enabled)
        self.minimum_episodes = integer(minimum_episodes, "minimum_episodes", 1)
        self.window = integer(window, "window", 1, self.minimum_episodes)
        self.required_successes = integer(
            required_successes, "required_successes", 1, self.window
        )
        self.opening_index = integer(
            state["opening_index"], "opening_index", 0, FINAL_INDEX
        )
        self.episodes_at_opening = integer(
            state["episodes_at_opening"], "episodes_at_opening", 0
        )
        self.total_pickup_episodes = integer(
            state["total_episodes"], "total_episodes", self.episodes_at_opening
        )
        if (
            not isinstance(state["recent"], list)
            or len(state["recent"]) != min(self.window, self.episodes_at_opening)
            or any((type(v) is not bool for v in state["recent"]))
            or (not isinstance(state["promotions"], list))
        ):
            raise ValueError("Invalid fine pickup outcomes")
        self.pickup_recent = deque(state["recent"], maxlen=self.window)
        self.pickup_promotions = deepcopy(state["promotions"])
        self._pickup_episode_finished = False
        super().__init__(curriculum_level=17, replay_fraction=0.0, **kwargs)
        self.set_pickup_attempt_limit(attempt_limit)

    def set_pickup_attempt_limit(self, steps):
        if steps is not None:
            steps = integer(steps, "attempt_limit", 1, self.max_steps)
        self.attempt_limit = steps

    def _early_failure(self, info):
        previous = super()._early_failure(info)
        if previous:
            return previous
        if (
            self.attempt_limit is not None
            and self.step_count >= self.attempt_limit
            and (self._hold_steps < self.hold_steps)
        ):
            return "pickup_attempt_limit"
        return ""

    def set_pickup_promotion_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("promotion_enabled must be bool")
        self.promotion_enabled = enabled

    def get_pickup_curriculum_state(self):
        return dict(
            kind=VERSION,
            openings=list(OPENINGS),
            opening_index=self.opening_index,
            promotion_enabled=self.promotion_enabled,
            attempt_limit=self.attempt_limit,
            minimum_episodes=self.minimum_episodes,
            window=self.window,
            required_successes=self.required_successes,
            episodes_at_opening=self.episodes_at_opening,
            total_episodes=self.total_pickup_episodes,
            recent=list(self.pickup_recent),
            promotions=deepcopy(self.pickup_promotions),
        )

    def _pickup_info(self, info):
        return {
            **info,
            "pickup_opening_index": self.opening_index,
            "pickup_episode_opening_index": self.episode_opening_index,
            "pickup_episode_opening": OPENINGS[self.episode_opening_index],
            "pickup_curriculum_state": self.get_pickup_curriculum_state(),
        }

    def reset(self, *, seed=None, options=None):
        self._maybe_promote()
        self.episode_opening_index = self.opening_index
        self._pickup_episode_finished = False
        obs, info = super().reset(seed=seed, options=options)
        if self.episode_opening_index != FINAL_INDEX:
            self.data.qpos[4:6] = OPENINGS[self.episode_opening_index]
            mujoco.mj_forward(self.model, self.data)
            self._previous_frame = None
            obs = self._observation()
            info = {**info, **self._info()}
        return (obs, self._pickup_info(info))

    def _record_completed_pickup(self, success):
        self.episodes_at_opening += 1
        self.total_pickup_episodes += 1
        self.pickup_recent.append(bool(success))
        self._maybe_promote()

    def _maybe_promote(self):
        if (
            self.promotion_enabled
            and self.opening_index < FINAL_INDEX
            and (self.episodes_at_opening >= self.minimum_episodes)
            and (len(self.pickup_recent) == self.window)
            and (sum(self.pickup_recent) >= self.required_successes)
        ):
            self.pickup_promotions.append(
                dict(
                    from_index=self.opening_index,
                    to_index=self.opening_index + 1,
                    from_opening=OPENINGS[self.opening_index],
                    to_opening=OPENINGS[self.opening_index + 1],
                    episodes_at_opening=self.episodes_at_opening,
                    total_episodes=self.total_pickup_episodes,
                    successes=sum(self.pickup_recent),
                    window=self.window,
                )
            )
            self.opening_index += 1
            self.episodes_at_opening = 0
            self.pickup_recent.clear()

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if (done or truncated) and (not self._pickup_episode_finished):
            self._pickup_episode_finished = True
            self._record_completed_pickup(info["is_success"])
        return (obs, reward, done, truncated, self._pickup_info(info))
