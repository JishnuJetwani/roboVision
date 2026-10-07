"""Hold-duration and reset curriculum, without action labels or controllers.

Initial hold exercises use potential shaping and prompt loss termination.
Later pickup levels and the terminal hard level retain the original task rules.
"""

from collections import deque
from dataclasses import asdict, dataclass
import mujoco
import numpy as np
from .joint_env import JointGraspEnv, VERSION

CURRICULUM_VERSION = "reverse-grasp-v2"
HOLD_TARGETS = (3, 5, 8, 12, 18, 25)
LOSS_GRACE_STEPS = 3
HOLD_POTENTIAL_SCALE = 10.0


@dataclass(frozen=True)
class ResetLevel:
    name: str
    clearance: float = 0.001
    opening: float = 0.045
    approach_height: float = 0.0
    hold_steps: int = 25
    hold_bootstrap: bool = False


LEVELS = tuple(
    [
        ResetLevel(
            f"hold-{steps}-steps",
            clearance=0.085,
            opening=0.0279,
            hold_steps=steps,
            hold_bootstrap=True,
        )
        for steps in HOLD_TARGETS
    ]
    + [
        ResetLevel(
            f"hold-lift-{height:.3f}",
            clearance=height,
            opening=0.0279,
            hold_bootstrap=height >= 0.065,
        )
        for height in (0.075, 0.065, 0.05, 0.035, 0.02, 0.001)
    ]
    + [
        ResetLevel(f"close-{opening:.3f}", opening=opening)
        for opening in (0.029, 0.031, 0.034, 0.038, 0.042, 0.045)
    ]
    + [
        ResetLevel(f"approach-{height:.3f}", approach_height=height)
        for height in (0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.14)
    ]
)
FINAL_LEVEL = len(LEVELS) - 1


def validate_level(level):
    if (
        isinstance(level, bool)
        or int(level) != level
        or (not 0 <= level <= FINAL_LEVEL)
    ):
        raise ValueError(f"Curriculum level must be an integer from 0 to {FINAL_LEVEL}")
    return int(level)


class ReverseGraspEnv(JointGraspEnv):
    def __init__(self, *, curriculum_level=0, replay_fraction=0.2, **kwargs):
        if not np.isfinite(replay_fraction) or not 0 <= replay_fraction < 1:
            raise ValueError("Replay fraction must be in [0, 1)")
        if "stage" in kwargs:
            raise ValueError(
                "Use curriculum_level for training; stage=3 remains the separate hard evaluation"
            )
        self.curriculum_level = validate_level(curriculum_level)
        self.replay_fraction = float(replay_fraction)
        super().__init__(stage=3, **kwargs)

    def set_curriculum_level(self, level):
        self.curriculum_level = validate_level(level)

    def _curriculum_info(self, info):
        return {
            **info,
            "stage": 3 if self.episode_level == FINAL_LEVEL else None,
            "curriculum_kind": CURRICULUM_VERSION,
            "curriculum_level": self.episode_level,
            "curriculum_frontier": self.episode_frontier,
            "curriculum_replay": self.episode_level < self.episode_frontier,
            "curriculum_name": LEVELS[self.episode_level].name,
            "hold_target_steps": self.hold_steps,
            "max_stable_hold_steps": self._max_hold_steps,
            "bilateral_contact_steps": self._bilateral_steps,
            "first_contact_loss_step": self._first_loss_step,
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_frontier = self.curriculum_level
        self.episode_level = self.curriculum_level
        if (
            self.curriculum_level
            and self.replay_fraction
            and (self.rng.random() < self.replay_fraction)
        ):
            self.episode_level = int(self.rng.integers(self.curriculum_level))
        self.hold_steps = LEVELS[self.episode_level].hold_steps
        self._lost_steps = self._max_hold_steps = self._bilateral_steps = 0
        self._first_loss_step = None
        self._hold_potential = 0.0
        render_images = self.render_images
        self.render_images = render_images and self.episode_level == FINAL_LEVEL
        try:
            observation, info = super().reset(seed=None, options=options)
        finally:
            self.render_images = render_images
        level = LEVELS[self.episode_level]
        xy = np.array([0.32, 0.0])
        mass = 0.08
        friction = 1.0
        self.model.body_mass[self._cup_bid] = mass
        self.model.body_inertia[self._cup_bid] = self._base_inertia * mass / 0.08
        self.model.pair_friction[:, :2] = friction
        for indices, base in (
            (self._cup_geoms, [0.1, 0.65, 0.72]),
            (self._table_id, [0.88, 0.9, 0.92]),
        ):
            self.model.geom_rgba[indices, :3] = np.array(base)
        self.model.light_diffuse[0] = 0.8
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        cup_z = self.table_z + self.cup_height / 2 + level.clearance
        self.data.qpos[self._cup_qadr : self._cup_qadr + 3] = [*xy, cup_z]
        self.data.qpos[self._cup_qadr + 3 : self._cup_qadr + 7] = [1, 0, 0, 0]
        target = np.r_[xy, cup_z + 0.014 + level.approach_height]
        self.data.qpos[:4] = self.inverse_kinematics(target)
        self.data.qpos[4:6] = level.opening
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        self.params.update(
            spawn_x=float(xy[0]),
            spawn_y=float(xy[1]),
            cup_mass=float(mass),
            grip_friction=float(friction),
        )
        self._previous_frame = None
        mujoco.mj_forward(self.model, self.data)
        self._check_simulation()
        self._peak_clearance = max(0.0, self._clearance())
        self._hold_potential = self._potential(self._info())
        return (
            self._observation(),
            self._curriculum_info(
                {**self._info(), "env_version": VERSION, "stage": self.episode_stage}
            ),
        )

    def _potential(self, info):
        if not LEVELS[self.episode_level].hold_bootstrap:
            return 0.0
        forces = np.zeros(2)
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            a, b = (int(contact.geom1), int(contact.geom2))
            for side, geoms in enumerate(self._finger_geoms):
                if (
                    a in geoms
                    and b in self._cup_geom_set
                    or (b in geoms and a in self._cup_geom_set)
                ):
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    forces[side] += max(0.0, float(wrench[0]))
        contact_quality = float(np.sqrt(np.prod(1.0 - np.exp(-forces / 0.2))))
        offset = self.grasp_position - self.cup_position - np.array([0.0, 0.0, 0.014])
        centered = np.exp(-np.sum((offset / 0.025) ** 2))
        aperture = np.exp(-np.mean(((self.data.qpos[4:6] - 0.0279) / 0.01) ** 2))
        upright = np.clip(info["upright"], 0.0, 1.0) ** 4
        clearance = np.clip(info["clearance"] / 0.06, 0.0, 1.0)
        still = np.exp(-((info["cup_speed"] / 0.15) ** 2))
        quality = (
            upright
            * clearance
            * (
                0.4 * centered * aperture
                + 0.3 * contact_quality
                + 0.3 * centered * aperture * still
            )
        )
        return float(HOLD_POTENTIAL_SCALE * quality)

    def _early_failure(self, info):
        self._max_hold_steps = max(self._max_hold_steps, self._hold_steps)
        bilateral = all(info["contacts"])
        self._bilateral_steps += int(bilateral)
        if not bilateral and self._first_loss_step is None:
            self._first_loss_step = self.step_count
        if not LEVELS[self.episode_level].hold_bootstrap:
            return ""
        self._lost_steps = 0 if bilateral else self._lost_steps + 1
        return (
            "grip_lost"
            if self._lost_steps >= LOSS_GRACE_STEPS or info["clearance"] < 0.04
            else ""
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if LEVELS[self.episode_level].hold_bootstrap:
            potential = 0.0 if terminated else self._potential(info)
            shaping = self.gamma * potential - self._hold_potential
            self._hold_potential = potential
            reward += shaping
            info["reward_components"]["hold_potential"] = shaping
            self._reward_totals["hold_potential"] = (
                self._reward_totals.get("hold_potential", 0.0) + shaping
            )
            if terminated:
                info["episode_reward_components"] = self._reward_totals.copy()
        return (observation, reward, terminated, truncated, self._curriculum_info(info))


class ReverseGraspCurriculum:
    def __init__(
        self, level=0, *, minimum_episodes=256, window=100, threshold=0.8, state=None
    ):
        if state is not None:
            if state.get("kind") != CURRICULUM_VERSION:
                raise ValueError("Incompatible reverse curriculum state")
            level = state["level"]
            minimum_episodes, window, threshold = (
                state[k] for k in ("minimum_episodes", "window", "threshold")
            )
        if (
            not isinstance(window, int)
            or not isinstance(minimum_episodes, int)
            or window < 1
            or (minimum_episodes < window)
        ):
            raise ValueError("Require a positive window and minimum episodes >= window")
        if not np.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("Promotion threshold must be in (0, 1]")
        self.level = validate_level(level)
        self.minimum_episodes, self.window, self.threshold = (
            minimum_episodes,
            window,
            threshold,
        )
        self.episodes = state["episodes"] if state else 0
        self.recent = deque(state["recent"] if state else [], maxlen=window)
        self.promotions = list(state["promotions"]) if state else []

    @property
    def stage(self):
        return self.level

    def observe(self, info, steps):
        if info["curriculum_level"] != self.level:
            return False
        self.episodes += 1
        self.recent.append(bool(info["is_success"]))
        if (
            self.level < FINAL_LEVEL
            and self.episodes >= self.minimum_episodes
            and (len(self.recent) == self.window)
            and (np.mean(self.recent) >= self.threshold)
        ):
            self.level += 1
            self.promotions.append(
                dict(
                    steps=steps,
                    level=self.level,
                    name=LEVELS[self.level].name,
                    success_rate=float(np.mean(self.recent)),
                    episodes=self.episodes,
                )
            )
            self.episodes = 0
            self.recent.clear()
            return True
        return False

    def state_dict(self):
        return dict(
            kind=CURRICULUM_VERSION,
            level=self.level,
            minimum_episodes=self.minimum_episodes,
            window=self.window,
            threshold=self.threshold,
            episodes=self.episodes,
            recent=list(self.recent),
            promotions=list(self.promotions),
        )

    @staticmethod
    def specification():
        return dict(
            version=CURRICULUM_VERSION,
            levels=[asdict(level) for level in LEVELS],
            final_level=FINAL_LEVEL,
            final_evaluation_stage=3,
            reset_only=False,
            action_demonstrations=False,
            hold_bootstrap=dict(
                targets=HOLD_TARGETS,
                loss_grace_steps=LOSS_GRACE_STEPS,
                potential_scale=HOLD_POTENTIAL_SCALE,
                shaping="gamma * Phi(next) - Phi(previous); terminal Phi=0",
            ),
        )
