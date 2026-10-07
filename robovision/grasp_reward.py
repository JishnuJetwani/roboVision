"""Finite-deadline grasp rewards, expressed as rates per simulated second.

All nonterminal rewards are negative, even at perfect grasp quality. Crashing
charges the discounted costs of the rest of a failed attempt, so ending an
episode early cannot escape its time/deadline bill. No policy control is here.
"""

from dataclasses import asdict, dataclass
import math
import numpy as np

VERSION = "timed-dense-grasp-v1"


@dataclass(frozen=True)
class GraspReward:
    time_cost: float = 0.5
    reach_rate: float = 0.1
    alignment_rate: float = 0.1
    grip_rate: float = 0.1
    lift_rate: float = 0.15
    effort_cost: float = 0.02
    force_change_cost: float = 0.01
    success_bonus: float = 50.0
    failure_cost: float = 25.0
    crash_cost: float = 5.0

    def __post_init__(self):
        if any(
            (not math.isfinite(value) or value < 0 for value in asdict(self).values())
        ):
            raise ValueError("Reward magnitudes must be finite and nonnegative")
        if (
            self.time_cost
            <= self.reach_rate + self.alignment_rate + self.grip_rate + self.lift_rate
        ):
            raise ValueError("Time cost must exceed all quality rates combined")

    def specification(self):
        return {"version": VERSION, **asdict(self)}

    def components(
        self,
        scores,
        action,
        previous_action,
        *,
        dt,
        gamma,
        remaining_steps,
        success=False,
        failure=False,
    ):
        if not 0 < gamma <= 1 or dt <= 0 or remaining_steps < 0:
            raise ValueError("Invalid discount, timestep, or remaining horizon")
        if success and failure:
            raise ValueError("An outcome cannot be both success and failure")
        quality = {
            key: float(np.clip(scores[key], 0, 1))
            for key in ("reach", "alignment", "grip", "lift")
        }
        terms = {
            key: dt * getattr(self, key + "_rate") * value
            for key, value in quality.items()
        }
        terms.update(
            time=-dt * self.time_cost,
            effort=-dt * self.effort_cost * float(np.square(action).mean()),
            force_change=-dt
            * self.force_change_cost
            * float(np.square(action - previous_action).mean()),
            success=self.success_bonus if success else 0.0,
            failure=0.0,
            remaining_time=0.0,
            crash=0.0,
        )
        if failure:
            discount = gamma**remaining_steps
            future_steps = (
                remaining_steps
                if gamma == 1
                else -math.expm1(remaining_steps * math.log(gamma)) / (1 - gamma)
            )
            terms["remaining_time"] = -dt * self.time_cost * gamma * future_steps
            terms["failure"] = -self.failure_cost * discount
            if remaining_steps:
                terms["crash"] = -self.crash_cost
        return terms
