"""Optional training-only geometry-gated closure potential for PPO retention.

Wrap a pickup training environment explicitly. Final-height evaluation should
remain unwrapped. This supplies intermediate closure feedback without changing
success, actions, observations, or the ordering of complete discounted returns.
"""

import gymnasium as gym
import numpy as np

VERSION = "grasp-closure-potential-v1"


def closure_quality(offset, downward, openings):
    offset = np.asarray(offset, dtype=float)
    openings = np.asarray(openings, dtype=float)
    if (
        offset.shape != (3,)
        or openings.shape != (2,)
        or (not np.isfinite(offset).all())
        or (not np.isfinite(openings).all())
        or (not np.isfinite(downward))
    ):
        raise ValueError(
            "Expected finite 3D offset, two openings and downward alignment"
        )
    centered = np.exp(-np.sum((offset / np.array([0.025, 0.025, 0.02])) ** 2))
    aperture = np.exp(-np.mean(((openings - 0.0279) / 0.015) ** 2))
    return float(centered * float(np.clip(downward, 0.0, 1.0)) ** 4 * aperture)


class GraspClosurePotential(gym.Wrapper):
    def __init__(self, env, scale=10.0):
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("Closure potential scale must be finite and nonnegative")
        super().__init__(env)
        self.scale = float(scale)
        self.previous_potential = 0.0
        self._closure_total = 0.0

    def potential(self):
        env = self.unwrapped
        axis = env.data.site_xmat[env._grasp_sid].reshape(3, 3)[:, 0]
        return self.scale * closure_quality(
            env.grasp_position - env.cup_position - [0, 0, 0.014],
            -axis[2],
            env.data.qpos[4:6],
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._closure_total = 0.0
        self.previous_potential = self.potential() if self.scale else 0.0
        if not self.scale:
            return (obs, info)
        return (
            obs,
            {
                **info,
                "closure_potential": self.previous_potential,
                "closure_potential_scale": self.scale,
            },
        )

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if not self.scale:
            return (obs, reward, done, truncated, info)
        current = 0.0 if done or truncated else self.potential()
        extra = self.unwrapped.gamma * current - self.previous_potential
        self.previous_potential = current
        self._closure_total += extra
        info = {
            **info,
            "closure_potential": current,
            "closure_potential_scale": self.scale,
        }
        info["reward_components"] = {
            **info.get("reward_components", {}),
            "closure_potential": extra,
        }
        if done or truncated:
            info["episode_reward_components"] = {
                **info.get("episode_reward_components", {}),
                "closure_potential": self._closure_total,
            }
        return (obs, float(reward + extra), done, truncated, info)


PickupPotential = GraspClosurePotential
