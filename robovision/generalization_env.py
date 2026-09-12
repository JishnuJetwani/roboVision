"""Height-only resets with the open hand directly above a fixed cup."""

from numbers import Real
import mujoco
import numpy as np
from .joint_env import JointGraspEnv, VERSION as JOINT_VERSION
from .reverse_curriculum import ReverseGraspEnv

VERSION = "centered-height-reset-v1"
FRONTIER_HEIGHT_BANDS = ((0.025, 0.055, 0.5), (0.055, 0.1, 0.4), (0.1, 0.14, 0.1))
PICKUP_HEIGHT_BANDS = ((0.0, 0.03, 1.0),)
NOMINAL_HAND_XY = (0.32, 0.0)
NOMINAL_GRASP_Z = 0.31
ORIGINAL_MAX_STEPS = 500


def _finite_number(value, name, *, nonnegative=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or (not np.isfinite(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def validate_height_bands(height_bands):
    bands = tuple((tuple(band) for band in height_bands))
    if not bands or any((len(band) != 3 for band in bands)):
        raise ValueError("Expected (low, high, probability) height bands")
    result = []
    for low, high, probability in bands:
        low, high, probability = (
            _finite_number(v, "height band", nonnegative=True)
            for v in (low, high, probability)
        )
        if not 0 <= low < high <= 0.14 or probability <= 0:
            raise ValueError("Height bands must lie between 0 and .140 metres")
        result.append((low, high, probability))
    if not np.isclose(sum((b[2] for b in result)), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Height probabilities must sum to one")
    if any((a[1] > b[0] for a, b in zip(result, result[1:]))):
        raise ValueError("Height bands must be ordered and non-overlapping")
    return tuple(result)


class GeneralizationGraspEnv(JointGraspEnv):
    """Vary only starting height; cup and initial hand share fixed world XY."""

    version = VERSION

    def __init__(
        self, *, fixed_height=None, height_bands=FRONTIER_HEIGHT_BANDS, **kwargs
    ):
        self.height_bands = validate_height_bands(height_bands)
        self.fixed_height = (
            None
            if fixed_height is None
            else _finite_number(fixed_height, "fixed_height", nonnegative=True)
        )
        if self.fixed_height is not None and self.fixed_height > 0.14:
            raise ValueError("Height must lie between 0 and .140 metres")
        if "stage" in kwargs:
            raise ValueError("Use fixed_height or height_bands")
        max_steps = kwargs.pop("max_steps", ORIGINAL_MAX_STEPS)
        if type(max_steps) is not int or max_steps != ORIGINAL_MAX_STEPS:
            raise ValueError("The task uses a 500-action deadline")
        super().__init__(stage=3, max_steps=max_steps, **kwargs)

    def specification(self):
        return dict(
            version=self.version,
            fixed_height=self.fixed_height,
            height_bands=[list(b) for b in self.height_bands],
            nominal_hand_xy=list(NOMINAL_HAND_XY),
            nominal_grasp_z=NOMINAL_GRASP_Z,
            cup_mass=0.08,
            grip_friction=1.0,
            max_steps=self.max_steps,
            hold_steps=self.hold_steps,
            gamma=self.gamma,
            observation=self.observation_mode,
            control=dict(self.control_spec),
            success="Complete pickup and stable hold",
            shaping=False,
            action_demonstrations=False,
            runtime_controller=False,
        )

    def _generalization_info(self, info):
        return {
            **info,
            "curriculum_kind": self.version,
            "reset_height": self.episode_reset_height,
            "height_band_index": self.episode_height_band_index,
            "cup_offset": [0.0, 0.0],
        }

    def reset(self, *, seed=None, options=None):
        render_images = self.render_images
        self.render_images = False
        try:
            super().reset(seed=seed, options=options)
        finally:
            self.render_images = render_images
        if self.fixed_height is None:
            index = int(
                self.rng.choice(
                    len(self.height_bands), p=[b[2] for b in self.height_bands]
                )
            )
            low, high, _ = self.height_bands[index]
            height = float(self.rng.uniform(low, high))
        else:
            index = None
            height = self.fixed_height
        self.episode_cup_offset = (0.0, 0.0)
        self.episode_reset_height = height
        self.episode_height_band_index = index
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._cup_qadr : self._cup_qadr + 3] = [
            *NOMINAL_HAND_XY,
            self.table_z + self.cup_height / 2 + 0.001,
        ]
        self.data.qpos[self._cup_qadr + 3 : self._cup_qadr + 7] = [1, 0, 0, 0]
        self.data.qpos[:4] = self.inverse_kinematics(
            [*NOMINAL_HAND_XY, NOMINAL_GRASP_Z + height]
        )
        self.data.qpos[4:6] = 0.045
        self.params.update(spawn_x=NOMINAL_HAND_XY[0], spawn_y=NOMINAL_HAND_XY[1])
        mujoco.mj_forward(self.model, self.data)
        self._check_simulation()
        self._previous_frame = None
        self._peak_clearance = max(0.0, self._clearance())
        return (
            self._observation(),
            self._generalization_info(
                {
                    **self._info(),
                    "env_version": JOINT_VERSION,
                    "stage": self.episode_stage,
                }
            ),
        )

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        return (obs, reward, done, truncated, self._generalization_info(info))


class OriginalHoldGraspEnv(ReverseGraspEnv):
    """Held-object retention reset with the original reward and termination rules.

    Reverse level 5 supplies the same physical reset as earlier retention checks.
    Its extra potential reward and prompt contact-loss termination are excluded.
    """

    version = "generalization-hold-reset-v1"

    def __init__(self, **kwargs):
        if any(
            (key in kwargs for key in ("stage", "curriculum_level", "replay_fraction"))
        ):
            raise ValueError("OriginalHoldGraspEnv uses only the reverse level 5 reset")
        max_steps = kwargs.pop("max_steps", ORIGINAL_MAX_STEPS)
        if isinstance(max_steps, (bool, np.bool_)) or max_steps != ORIGINAL_MAX_STEPS:
            raise ValueError(
                "OriginalHoldGraspEnv retains the original 500-action deadline"
            )
        super().__init__(
            curriculum_level=5,
            replay_fraction=0.0,
            max_steps=ORIGINAL_MAX_STEPS,
            **kwargs,
        )

    def specification(self):
        return dict(
            version=self.version,
            reset_curriculum_level=5,
            max_steps=self.max_steps,
            hold_steps=self.hold_steps,
            gamma=self.gamma,
            observation=self.observation_mode,
            control=dict(self.control_spec),
            fixed_nominal_appearance=True,
            cup_mass=0.08,
            grip_friction=1.0,
            success="Original full pickup and stable hold",
            shaping=False,
            action_demonstrations=False,
            runtime_controller=False,
        )

    def _hold_info(self, info):
        return {
            **info,
            "stage": None,
            "curriculum_kind": self.version,
            "reset_curriculum_level": 5,
            "hold_target_steps": self.hold_steps,
        }

    def reset(self, *, seed=None, options=None):
        observation, _ = super().reset(seed=seed, options=options)
        return (
            observation,
            self._hold_info({**self._info(), "env_version": JOINT_VERSION}),
        )

    def _early_failure(self, info):
        return ""

    def step(self, action):
        observation, reward, terminated, truncated, info = JointGraspEnv.step(
            self, action
        )
        return (observation, reward, terminated, truncated, self._hold_info(info))
