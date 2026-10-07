"""Optional bounded credit for opposite-side contact around the cup's middle."""

import numpy as np
import mujoco
from .approach_bridge import ApproachBridgeEnv

VERSION = "middle-grasp-credit-v1"


def band_quality(left, right, width=0.02):
    """Cup-local, force-weighted contacts; prefer the middle, not the bottom.

    Contact points must be on opposite sides; both must lie near mid-height.
    Translation/lifting and world-frame rotation cannot change this score.
    """
    if left is None or right is None:
        return 0.0
    left, right = (np.asarray(left), np.asarray(right))
    denom = np.linalg.norm(left[:2]) * np.linalg.norm(right[:2])
    if denom < 1e-10:
        return 0.0
    cosine = np.dot(left[:2], right[:2]) / denom
    opposition = np.clip((-cosine - 0.5) / 0.5, 0.0, 1.0)
    heights = np.exp(-0.5 * (np.array([left[2], right[2]]) / width) ** 2)
    return float(opposition * min(heights))


def best_quality_credit(previous_best, quality, scale):
    best = max(previous_best, quality)
    return (scale * (best - previous_best), best)


class GraspBandEnv(ApproachBridgeEnv):
    def __init__(self, *, grasp_band_bonus=5.0, **kwargs):
        if not np.isfinite(grasp_band_bonus) or grasp_band_bonus < 0:
            raise ValueError("Grasp band bonus must be finite and nonnegative")
        self.grasp_band_bonus = float(grasp_band_bonus)
        self._best_band_quality = 0.0
        super().__init__(**kwargs)

    def grasp_band_metrics(self):
        sums = np.zeros((2, 3))
        forces = np.zeros(2)
        rotation = self.data.xmat[self._cup_bid].reshape(3, 3)
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            a, b = (int(c.geom1), int(c.geom2))
            for side, geoms in enumerate(self._finger_geoms):
                if (
                    a in geoms
                    and b in self._cup_geom_set
                    or (b in geoms and a in self._cup_geom_set)
                ):
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    force = max(0.0, float(wrench[0]))
                    local = rotation.T @ (c.pos - self.cup_position)
                    sums[side] += force * local
                    forces[side] += force
        points = [sums[i] / forces[i] if forces[i] > 0.05 else None for i in range(2)]
        quality = band_quality(*points)
        if self._info()["upright"] < np.cos(np.deg2rad(20)) or self._table_collision():
            quality = 0.0
        return dict(
            quality=quality,
            contact_heights=[None if p is None else float(p[2]) for p in points],
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._best_band_quality = self.grasp_band_metrics()["quality"]
        return (obs, info)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        metrics = self.grasp_band_metrics()
        extra, self._best_band_quality = best_quality_credit(
            self._best_band_quality, metrics["quality"], self.grasp_band_bonus
        )
        reward += extra
        info["reward_components"]["grasp_band"] = extra
        self._reward_totals["grasp_band"] = (
            self._reward_totals.get("grasp_band", 0.0) + extra
        )
        if done or truncated:
            info["episode_reward_components"] = self._reward_totals.copy()
        info.update(
            grasp_band_quality=metrics["quality"],
            grasp_contact_heights=metrics["contact_heights"],
            best_grasp_band_quality=self._best_band_quality,
        )
        return (obs, reward, done, truncated, info)
