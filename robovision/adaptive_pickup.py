"""Training-only finger-opening reset curriculum; original pickup rules persist.

No action labels, control intervention or observation changes. Evaluate with
ordinary ReverseGraspEnv(level=17) or JointGraspEnv(stage=3), never this adaptive
training distribution. Save `get_pickup_curriculum_state()` in checkpoint metadata.
"""

from collections import deque
from copy import deepcopy
from numbers import Integral
from .reverse_curriculum import ReverseGraspEnv

VERSION = "adaptive-pickup-openings-v1"
FIRST_LEVEL = 12
FINAL_LEVEL = 17


def _integer(value, name, minimum, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"Invalid {name}")
    return int(value)


class AdaptivePickupEnv(ReverseGraspEnv):
    def __init__(
        self,
        *,
        state=None,
        pickup_state=None,
        initial_level=12,
        minimum_episodes=16,
        window=12,
        required_successes=10,
        **kwargs,
    ):
        if state is not None:
            if pickup_state is not None:
                raise ValueError("Provide state or pickup_state, not both")
            pickup_state = state
        if "curriculum_level" in kwargs or "replay_fraction" in kwargs:
            raise ValueError(
                "AdaptivePickupEnv owns curriculum_level and disables replay"
            )
        if pickup_state is not None:
            state = deepcopy(pickup_state)
            if state.get("kind") != VERSION:
                raise ValueError("Incompatible adaptive pickup state")
            minimum_episodes = state["minimum_episodes"]
            window = state["window"]
            required_successes = state["required_successes"]
        else:
            state = dict(
                level=_integer(
                    initial_level, "initial_level", FIRST_LEVEL, FINAL_LEVEL
                ),
                episodes_at_level=0,
                total_episodes=0,
                recent=[],
                promotions=[],
            )
        self.minimum_episodes = _integer(minimum_episodes, "minimum_episodes", 1)
        self.window = _integer(window, "window", 1, self.minimum_episodes)
        self.required_successes = _integer(
            required_successes, "required_successes", 1, self.window
        )
        self.pickup_level = _integer(state["level"], "level", FIRST_LEVEL, FINAL_LEVEL)
        self.episodes_at_level = _integer(
            state["episodes_at_level"], "episodes_at_level", 0
        )
        self.total_pickup_episodes = _integer(
            state["total_episodes"], "total_episodes", self.episodes_at_level
        )
        recent = state["recent"]
        if (
            not isinstance(recent, list)
            or len(recent) != min(self.window, self.episodes_at_level)
            or any((type(v) is not bool for v in recent))
        ):
            raise ValueError("Invalid adaptive pickup recent outcomes")
        if not isinstance(state["promotions"], list):
            raise ValueError("Invalid adaptive pickup promotions")
        self.pickup_recent = deque(recent, maxlen=self.window)
        self.pickup_promotions = deepcopy(state["promotions"])
        self._pickup_episode_finished = False
        super().__init__(
            curriculum_level=self.pickup_level, replay_fraction=0.0, **kwargs
        )

    def get_pickup_curriculum_state(self):
        """JSON-ready independent copy, sufficient to resume promotion statistics."""
        return dict(
            kind=VERSION,
            level=self.pickup_level,
            minimum_episodes=self.minimum_episodes,
            window=self.window,
            required_successes=self.required_successes,
            episodes_at_level=self.episodes_at_level,
            total_episodes=self.total_pickup_episodes,
            recent=list(self.pickup_recent),
            promotions=deepcopy(self.pickup_promotions),
        )

    def _pickup_info(self, info):
        return {
            **info,
            "pickup_curriculum_level": self.pickup_level,
            "pickup_episode_level": self.episode_level,
            "pickup_curriculum_state": self.get_pickup_curriculum_state(),
        }

    def reset(self, *, seed=None, options=None):
        self.curriculum_level = self.pickup_level
        self._pickup_episode_finished = False
        obs, info = super().reset(seed=seed, options=options)
        return (obs, self._pickup_info(info))

    def _record_completed_pickup(self, success):
        self.episodes_at_level += 1
        self.total_pickup_episodes += 1
        self.pickup_recent.append(bool(success))
        if (
            self.pickup_level < FINAL_LEVEL
            and self.episodes_at_level >= self.minimum_episodes
            and (len(self.pickup_recent) == self.window)
            and (sum(self.pickup_recent) >= self.required_successes)
        ):
            self.pickup_promotions.append(
                dict(
                    from_level=self.pickup_level,
                    to_level=self.pickup_level + 1,
                    total_episodes=self.total_pickup_episodes,
                    episodes_at_level=self.episodes_at_level,
                    successes=sum(self.pickup_recent),
                    window=self.window,
                )
            )
            self.pickup_level += 1
            self.episodes_at_level = 0
            self.pickup_recent.clear()

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if (done or truncated) and (not self._pickup_episode_finished):
            self._pickup_episode_finished = True
            self._record_completed_pickup(info["is_success"])
        return (obs, reward, done, truncated, self._pickup_info(info))
