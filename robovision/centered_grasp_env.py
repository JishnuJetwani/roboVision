"""Centered pickup objective with bounded, terminal-zero potential shaping.

The policy retains the original pixels,18 proprioceptive values and five direct
force actions. Only training rewards and the success condition change. Original
pickup success is still measured, but a shallow grasp cannot end this task.
Held-object retention should use OriginalHoldGraspEnv separately.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from numbers import Real
import numpy as np
from .generalization_env import GeneralizationGraspEnv
from .grasp_reward import GraspReward

VERSION = "centered-grasp-potential-v1"
POTENTIAL_KEYS = ("approach", "grasp", "lift")


@dataclass(frozen=True)
class CenteredGraspConfig:
    """Distances are metres; budgets are maximum potential magnitudes."""

    z_low: float = 0.0
    z_high: float = 0.015
    xy_radius: float = 0.015
    maximum_finger_tilt_degrees: float = 15.0
    broad_z_scale: float = 0.1
    broad_xy_scale: float = 0.06
    center_z_scale: float = 0.015
    center_xy_scale: float = 0.015
    closed_jaw: float = 0.028
    open_jaw: float = 0.042
    approach_budget: float = 10.0
    grasp_budget: float = 10.0
    lift_budget: float = 20.0
    success_bonus: float = 100.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or (not np.isfinite(value))
            ):
                raise ValueError(f"{name} must be a finite number")
        if not 0.0 <= self.z_low < self.z_high <= 0.045:
            raise ValueError("Center Z band must satisfy 0 <= low < high <= .045 m")
        if not 0.0 < self.xy_radius <= 0.025:
            raise ValueError("Center XY radius must be in (0,.025] m")
        if not 0.0 < self.maximum_finger_tilt_degrees <= 30.0:
            raise ValueError("Maximum finger tilt must be in (0,30] degrees")
        if any(
            (
                getattr(self, name) <= 0.0
                for name in (
                    "broad_z_scale",
                    "broad_xy_scale",
                    "center_z_scale",
                    "center_xy_scale",
                )
            )
        ):
            raise ValueError("Potential distance scales must be positive")
        if not 0.0 <= self.closed_jaw < self.open_jaw <= 0.045:
            raise ValueError("Jaw thresholds must satisfy 0 <= closed < open <= .045 m")
        if (
            any(
                (
                    getattr(self, name) < 0.0
                    for name in ("approach_budget", "grasp_budget", "lift_budget")
                )
            )
            or self.success_bonus <= 0.0
        ):
            raise ValueError(
                "Potential budgets must be nonnegative and success bonus positive"
            )

    @property
    def downward_minimum(self):
        return math.cos(math.radians(self.maximum_finger_tilt_degrees))


def potential_increment(previous, current, gamma, terminal):
    """Terminal potential is zero on both success and failure."""
    return gamma * (0.0 if terminal else current) - previous


def original_task_return_ordering(config, gamma, max_steps=500):
    """Conservative cross-reset bound for the nominal open-hand reset family.

    Its initial grasp/lift potentials are zero and approach potential is at most
    approach_budget. With proper shaping every completed trajectory receives
    exactly minus its initial potential in discounted shaping return.
    """
    reward = GraspReward(success_bonus=config.success_bonus)
    horizon_sum = (
        max_steps if gamma == 1.0 else (1.0 - gamma**max_steps) / (1.0 - gamma)
    )
    terminal_weight = gamma ** (max_steps - 1)
    success_lower = (
        config.success_bonus * terminal_weight
        - 0.02
        * (reward.time_cost + reward.effort_cost + 4.0 * reward.force_change_cost)
        * horizon_sum
        - config.approach_budget
    )
    failure_upper = (
        -0.02 * reward.time_cost * horizon_sum - reward.failure_cost * terminal_weight
    )
    return dict(
        success_lower_bound=success_lower,
        failure_upper_bound=failure_upper,
        minimum_gap=success_lower - failure_upper,
        scope="Original500-action nominal open-hand resets; any centered success versus any failure",
        proof="Discounted shaping telescopes to minus initial potential; no original positive quality rates are stacked",
    )


class CenteredGraspEnv(GeneralizationGraspEnv):
    """Train centered bilateral lift/hold; never installs a runtime controller."""

    version = VERSION

    def __init__(self, *, reward_config=CenteredGraspConfig(), **kwargs):
        if not isinstance(reward_config, CenteredGraspConfig):
            raise ValueError("reward_config must be a CenteredGraspConfig")
        self.reward_config = reward_config
        gamma = kwargs.get("gamma", 0.995)
        if (
            isinstance(gamma, (bool, np.bool_))
            or not isinstance(gamma, Real)
            or (not np.isfinite(gamma))
            or (not 0.0 < gamma <= 1.0)
        ):
            raise ValueError("Require a finite gamma in (0,1]")
        bounds = original_task_return_ordering(reward_config, float(gamma))
        if bounds["minimum_gap"] <= 0.0:
            raise ValueError(
                "Success bonus is too small for guaranteed success-over-failure ordering at this gamma and approach budget"
            )
        self.return_ordering_bounds = bounds
        super().__init__(**kwargs)
        self.centered_reward_function = GraspReward(
            success_bonus=reward_config.success_bonus
        )

    def specification(self):
        specification = super().specification()
        specification.update(
            version=self.version,
            reward_config=asdict(self.reward_config),
            success="Centered bilateral grasp, clearance>=.06 m, upright<=20 degrees, cup speed<.1 m/s for25 consecutive actions",
            original_success_terminates=False,
            shaping=True,
            shaping_formula="gamma * Phi(next) - Phi(previous); Phi=0 at every terminal state",
            potential_budget_sum=sum(
                (getattr(self.reward_config, f"{key}_budget") for key in POTENTIAL_KEYS)
            ),
            original_positive_quality_rewards_applied=False,
            centered_base_reward=self.centered_reward_function.specification(),
            centered_hold_steps=self.hold_steps,
            return_ordering_bounds=dict(self.return_ordering_bounds),
            hold_retention_environment="OriginalHoldGraspEnv, unchanged original reward/success",
        )
        return specification

    def _center_metrics(self, info):
        config = self.reward_config
        offset = self.grasp_position - self.cup_position
        xy_error = float(np.linalg.norm(offset[:2]))
        z_relative = float(offset[2])
        xy_outside = max(0.0, xy_error - config.xy_radius)
        z_outside = max(0.0, config.z_low - z_relative, z_relative - config.z_high)
        axis = self.data.site_xmat[self._grasp_sid].reshape(3, 3)[:, 0]
        downward = float(-axis[2])
        orientation = float(np.clip(downward / config.downward_minimum, 0.0, 1.0)) ** 4
        center_quality = (
            math.exp(
                -math.hypot(
                    xy_outside / config.center_xy_scale,
                    z_outside / config.center_z_scale,
                )
            )
            * orientation
        )
        broad_quality = (
            math.exp(
                -math.hypot(
                    xy_outside / config.broad_xy_scale, z_outside / config.broad_z_scale
                )
            )
            * orientation
        )
        opening = float(
            np.clip(
                (np.min(self.data.qpos[4:6]) - config.closed_jaw)
                / (config.open_jaw - config.closed_jaw),
                0.0,
                1.0,
            )
        )
        opening = opening * opening * (3.0 - 2.0 * opening)
        approach = broad_quality * (center_quality + (1.0 - center_quality) * opening)
        bilateral = bool(all(info["contacts"]))
        grasp = (
            center_quality
            * float(bilateral)
            * float(np.clip(info["upright"], 0.0, 1.0)) ** 4
        )
        lift = grasp * float(np.clip(info["clearance"] / 0.06, 0.0, 1.0))
        tolerance = 1e-12
        centered = bool(
            xy_error <= config.xy_radius + tolerance
            and config.z_low - tolerance <= z_relative <= config.z_high + tolerance
            and (downward >= config.downward_minimum - tolerance)
        )
        return dict(
            centered_geometry=centered,
            center_xy_error_m=xy_error,
            center_z_relative_m=z_relative,
            center_z_band_distance_m=z_outside,
            center_downward=downward,
            centered_geometry_quality=center_quality,
            centered_openness=opening,
            approach_score=approach,
            grasp_score=grasp,
            lift_score=lift,
        )

    def _potentials(self, metrics):
        return {
            key: getattr(self.reward_config, f"{key}_budget") * metrics[f"{key}_score"]
            for key in POTENTIAL_KEYS
        }

    def _center_info(self, info, metrics):
        return {
            **info,
            **metrics,
            "centered_success": bool(info.get("is_success", False)),
            "original_success_ever": self._original_success_ever,
            "centered_stable_hold_steps": self._centered_hold_steps,
            "centered_shaping_totals": self._centered_shaping_totals.copy(),
            "centered_initial_potential": self._centered_initial_potential,
            "centered_current_potentials": self._potentials(metrics),
            "centered_effective_next_potentials": self._centered_previous_potentials.copy(),
        }

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._original_success_ever = False
        self._centered_hold_steps = 0
        self._centered_reward_totals = {}
        self._centered_shaping_totals = {key: 0.0 for key in POTENTIAL_KEYS}
        metrics = self._center_metrics(info)
        self._centered_previous_potentials = self._potentials(metrics)
        self._centered_initial_potential = sum(
            self._centered_previous_potentials.values()
        )
        if (
            self._centered_initial_potential
            > self.reward_config.approach_budget + 1e-10
        ):
            raise RuntimeError(
                "Reset exceeds the initial potential bound used for reward ordering"
            )
        return (
            observation,
            self._center_info({**info, "original_is_success": False}, metrics),
        )

    def step(self, action):
        previous_action = self.last_action.copy()
        observation, original_reward, _, truncated, original_info = super().step(action)
        original_success = bool(original_info["is_success"])
        self._original_success_ever |= original_success
        metrics = self._center_metrics(original_info)
        stable = (
            metrics["centered_geometry"]
            and all(original_info["contacts"])
            and (original_info["clearance"] >= 0.06)
            and (original_info["upright"] >= math.cos(math.radians(20.0)))
            and (original_info["cup_speed"] < 0.1)
        )
        self._centered_hold_steps = self._centered_hold_steps + 1 if stable else 0
        physical_failure = original_info["reason"] not in (
            "",
            "success",
            "timeout",
            "dropped",
        )
        success = bool(
            self._centered_hold_steps >= self.hold_steps and (not physical_failure)
        )
        deadline = self.step_count >= self.max_steps
        terminated = bool(success or physical_failure or deadline)
        if physical_failure:
            reason = original_info["reason"]
        elif success:
            reason = "success"
        elif deadline:
            reason = (
                "dropped"
                if self._peak_clearance >= 0.06 and original_info["clearance"] < 0.02
                else "timeout"
            )
        else:
            reason = ""
        components = self.centered_reward_function.components(
            dict(reach=0.0, alignment=0.0, grip=0.0, lift=0.0),
            self.last_action,
            previous_action,
            dt=self.control_dt,
            gamma=self.gamma,
            remaining_steps=max(0, self.max_steps - self.step_count),
            success=success,
            failure=terminated and (not success),
        )
        potentials = self._potentials(metrics)
        for key in POTENTIAL_KEYS:
            extra = potential_increment(
                self._centered_previous_potentials[key],
                potentials[key],
                self.gamma,
                terminated,
            )
            components[f"centered_{key}_potential"] = extra
            self._centered_shaping_totals[key] += extra
        self._centered_previous_potentials = {
            key: 0.0 if terminated else potentials[key] for key in POTENTIAL_KEYS
        }
        for key, value in components.items():
            self._centered_reward_totals[key] = (
                self._centered_reward_totals.get(key, 0.0) + value
            )
        info = {
            **original_info,
            "is_success": success,
            "reason": reason,
            "original_is_success": original_success,
            "original_reward": original_reward,
            "original_reward_components": original_info["reward_components"].copy(),
            "reward_components": components,
        }
        info.pop("episode_reward_components", None)
        if terminated:
            info["episode_reward_components"] = self._centered_reward_totals.copy()
        return (
            observation,
            float(sum(components.values())),
            terminated,
            truncated,
            self._center_info(info, metrics),
        )
