"""Optional first-achievement approach shaping for pure force-control PPO.

Unlike terminal-zero potential shaping, this intentionally changes the temporary
training objective: a failed attempt that actually approached the cup is better
than one that did nothing. A running maximum bounds the total undiscounted bonus
and prevents collecting reward by retreating and returning. Evaluation must use
shaping=False (or the original environment). No action, observation, termination,
physics, or success criterion is modified.
"""

import numpy as np
from .approach_bridge import ApproachBridgeEnv, approach_quality

VERSION = "first-achievement-approach-v1"


def best_progress(previous_best, quality, scale):
    """Return bounded positive improvement and new record; never terminal-cancel."""
    values = np.asarray([previous_best, quality, scale], dtype=float)
    if (
        not np.isfinite(values).all()
        or not 0 <= previous_best <= 1
        or (not 0 <= quality <= 1)
        or (scale < 0)
    ):
        raise ValueError("Expected finite qualities in [0, 1] and nonnegative scale")
    current_best = max(previous_best, quality)
    return (scale * (current_best - previous_best), current_best)


class DenseApproachEnv(ApproachBridgeEnv):
    """Same bridge resets/actions, with up to dense_scale extra reward per attempt.

    The record starts at reset quality, so resets cannot collect a free bonus.
    Replay receives unchanged rewards. Bilateral contact permits closing at the
    grasp location; otherwise opening contributes continuously to approach
    quality. A latched record means lifting cannot claw back earned approach
    reward. Shaping scale is fixed per episode and can be annealed on next reset.
    """

    def __init__(self, *, shaping=True, dense_scale=20.0, **kwargs):
        self.set_dense_scale(dense_scale)
        self.dense_shaping = bool(shaping)
        self._dense_best = 0.0
        self._dense_episode_scale = 0.0
        super().__init__(shaping=False, **kwargs)

    def set_dense_scale(self, scale):
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("dense_scale must be finite and nonnegative")
        self.dense_scale = float(scale)

    def _dense_quality(self, info):
        axis = self.data.site_xmat[self._grasp_sid].reshape(3, 3)[:, 0]
        return approach_quality(
            self.grasp_position - self.cup_position - [0, 0, 0.014],
            -axis[2],
            float(np.mean(self.data.qpos[4:6])),
            all(info["contacts"]),
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._dense_episode_scale = (
            self.dense_scale
            if self.dense_shaping and (not info["curriculum_replay"])
            else 0.0
        )
        self._dense_best = self._dense_quality(info)
        self._dense_initial = self._dense_best
        return (
            obs,
            {
                **info,
                "dense_approach_quality": self._dense_best,
                "dense_approach_scale": self._dense_episode_scale,
            },
        )

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        quality = self._dense_quality(info)
        extra, self._dense_best = best_progress(
            self._dense_best, quality, self._dense_episode_scale
        )
        reward += extra
        info["reward_components"]["dense_approach_progress"] = extra
        self._reward_totals["dense_approach_progress"] = (
            self._reward_totals.get("dense_approach_progress", 0.0) + extra
        )
        if done or truncated:
            info["episode_reward_components"] = self._reward_totals.copy()
        info.update(
            dense_approach_quality=quality,
            dense_approach_best=self._dense_best,
            dense_approach_initial=self._dense_initial,
            dense_approach_scale=self._dense_episode_scale,
        )
        return (obs, float(reward), done, truncated, info)
