"""Broad-height approach with a centered, low-speed grasp-compatible endpoint.

Only the reward target and phase goal change from ApproachSkillEnv. Both lessons
retain25–140 mm starts, fixed cup XY, all five original force actions,
pixels plus18 proprioceptive values, and the original500-action episode clock.
The final goal lies inside the moving-grasp specialist's centered geometry band.
"""

from dataclasses import asdict, dataclass
import math
from .hierarchical_skills import ApproachSkillEnv

VERSION = "centered-open-approach-skill-v1"


@dataclass(frozen=True)
class CenteredApproachLesson:
    xy_tolerance: float
    z_low: float
    z_high: float
    maximum_speed: float
    minimum_opening: float
    maximum_tilt_degrees: float
    stable_steps: int
    height_low: float = 0.025
    height_high: float = 0.14


CENTERED_APPROACH_STAGES = (
    CenteredApproachLesson(0.015, 0.0, 0.02, 0.08, 0.04, 15.0, 3),
    CenteredApproachLesson(0.01, 0.0025, 0.0125, 0.025, 0.04, 15.0, 10),
)


class CenteredApproachSkillEnv(ApproachSkillEnv):
    """One coarse promotion from nearby capture to a stable centered arrival."""

    version = VERSION
    target_relative_z = 0.0075
    lessons = CENTERED_APPROACH_STAGES

    def specification(self):
        spec = super().specification()
        spec.update(
            version=VERSION,
            lessons=[asdict(lesson) for lesson in self.lessons],
            goal_height_reference="Grasp-site world Z minus actual cup-center world Z",
            final_goal_within_moving_grasp_geometry=True,
            dense_reward_formula_unchanged=True,
            target_relative_z=self.target_relative_z,
            curriculum_changes_only_endpoint=True,
            phase_success_is_full_pickup=False,
        )
        return spec

    def _goal(self, metrics):
        lesson = self._lesson
        return (
            metrics["xy_error_m"] <= lesson.xy_tolerance + 1e-12
            and lesson.z_low - 1e-12 <= metrics["relative_z_m"] <= lesson.z_high + 1e-12
            and (
                metrics["downward"]
                >= math.cos(math.radians(lesson.maximum_tilt_degrees)) - 1e-12
            )
            and (metrics["minimum_jaw_m"] >= lesson.minimum_opening)
            and (metrics["relative_speed_m_s"] <= lesson.maximum_speed)
            and (not any(metrics["contacts"]))
        )
