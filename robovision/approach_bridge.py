"""An opt-in approach curriculum; direct torque actions and pickup success unchanged."""

from dataclasses import dataclass
import numpy as np
import mujoco
from .reverse_curriculum import ReverseGraspEnv

VERSION = "approach-bridge-v1"


@dataclass(frozen=True)
class BridgeStage:
    low: float
    high: float
    scale: float


STAGES = tuple(
    (
        BridgeStage(*v)
        for v in [
            (0.04, 0.055, 20.0),
            (0.05, 0.065, 20.0),
            (0.06, 0.07, 20.0),
            (0.065, 0.075, 20.0),
            (0.065, 0.08, 20.0),
            (0.05, 0.08, 10.0),
            (0.05, 0.08, 0.0),
        ]
    )
)


def approach_quality(offset, downward, opening, bilateral):
    """Bounded potential: close to nominal grasp, downward-facing, open or gripping."""
    proximity = np.exp(-np.linalg.norm(np.asarray(offset) / [0.035, 0.035, 0.06]))
    aperture = max(float(bilateral), float(np.clip(opening / 0.032, 0.0, 1.0)))
    return float(proximity * np.clip(downward, 0.0, 1.0) ** 2 * aperture)


def potential_increment(previous, current, gamma, terminated):
    return gamma * (0.0 if terminated else current) - previous


class ApproachBridgeEnv(ReverseGraspEnv):
    def __init__(
        self,
        *,
        bridge_stage=0,
        fixed_height=None,
        shaping=True,
        replay_fraction=0.2,
        **kwargs,
    ):
        self.set_bridge_stage(bridge_stage)
        if fixed_height is not None and (not 0 <= fixed_height <= 0.14):
            raise ValueError("Height must be between 0 and .14 m")
        self.fixed_height = fixed_height
        self.shaping = shaping
        self._approach_previous = 0.0
        super().__init__(curriculum_level=23, replay_fraction=replay_fraction, **kwargs)

    def set_bridge_stage(self, stage):
        if (
            isinstance(stage, bool)
            or int(stage) != stage
            or (not 0 <= stage < len(STAGES))
        ):
            raise ValueError("Invalid approach bridge stage")
        self.bridge_stage = int(stage)

    def _approach_potential(self):
        axis = self.data.site_xmat[self._grasp_sid].reshape(3, 3)[:, 0]
        info = self._info()
        return self._approach_scale * approach_quality(
            self.grasp_position - self.cup_position - [0, 0, 0.014],
            -axis[2],
            float(np.mean(self.data.qpos[4:6])),
            all(info["contacts"]),
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.episode_bridge_stage = self.bridge_stage
        stage = STAGES[self.bridge_stage]
        self._approach_scale = (
            stage.scale if self.shaping and (not info["curriculum_replay"]) else 0.0
        )
        self.approach_height = None
        if not info["curriculum_replay"]:
            self.approach_height = float(
                self.fixed_height
                if self.fixed_height is not None
                else self.rng.uniform(stage.low, stage.high)
            )
            target = self.cup_position + [0, 0, 0.014 + self.approach_height]
            self.data.qpos[:4] = self.inverse_kinematics(target)
            mujoco.mj_forward(self.model, self.data)
            self._previous_frame = None
            obs = self._observation()
            info = self._curriculum_info(self._info())
        self._approach_previous = self._approach_potential()
        return (
            obs,
            {
                **info,
                "bridge_stage": self.episode_bridge_stage,
                "approach_height": self.approach_height,
            },
        )

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        current = self._approach_potential()
        extra = potential_increment(
            self._approach_previous, current, self.gamma, done or truncated
        )
        self._approach_previous = 0.0 if done or truncated else current
        reward += extra
        info["reward_components"]["approach_potential"] = extra
        self._reward_totals["approach_potential"] = (
            self._reward_totals.get("approach_potential", 0.0) + extra
        )
        if done or truncated:
            info["episode_reward_components"] = self._reward_totals.copy()
        info.update(
            bridge_stage=self.episode_bridge_stage, approach_height=self.approach_height
        )
        return (obs, reward, done, truncated, info)


def passes_gate(rows):
    """Require complete pickups at every tested height and both action modes."""
    if not rows:
        return False
    for height in {r["height"] for r in rows}:
        for deterministic in (True, False):
            group = [
                r
                for r in rows
                if r["height"] == height and r["deterministic"] == deterministic
            ]
            if not group or sum((r["success"] for r in group)) / len(group) < 0.8:
                return False
    return True
