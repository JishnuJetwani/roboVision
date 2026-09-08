"""Explicit positive reward scaling for balancing interleaved training roles.

A constant positive scale preserves the ordering of returns within one task,
while changing its contribution when PPO trains on a mixture of task roles.
The wrapper never changes policy inputs, actions, physics, or success criteria.
"""

import math
import gymnasium as gym


class RoleRewardScale(gym.Wrapper):
    def __init__(self, env, scale):
        try:
            value = float(scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Reward scale must be positive and finite") from exc
        if isinstance(scale, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("Reward scale must be positive and finite")
        super().__init__(env)
        self.scale = value

    def _info(self, info):
        result = dict(info)
        for key in ("reward_components", "episode_reward_components"):
            if key in info:
                original = dict(info[key])
                result["raw_" + key] = original
                result[key] = {
                    name: value * self.scale for name, value in original.items()
                }
        result["training_reward_scale"] = self.scale
        return result

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return (observation, self._info(info))

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation,
            reward * self.scale,
            terminated,
            truncated,
            self._info(info),
        )
