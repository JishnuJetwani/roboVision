"""The task starts directly above the cup at every curriculum level."""

import dataclasses
import numpy as np
import pytest
from robovision.joint_env import JointGraspEnv
from robovision.reverse_curriculum import ReverseGraspEnv, LEVELS, FINAL_LEVEL
from robovision.generalization_env import GeneralizationGraspEnv
from robovision.centered_grasp_env import CenteredGraspEnv
from robovision.ordered_confirmation import confirmation_cases


@pytest.mark.parametrize("level", range(len(LEVELS)))
def test_every_reverse_lesson_has_fixed_xy_and_physics(level):
    env = ReverseGraspEnv(
        curriculum_level=level, replay_fraction=0.0, render_images=False
    )
    try:
        for seed in (0, 17, 1234):
            env.reset(seed=seed)
            np.testing.assert_allclose(env.grasp_position[:2], [0.32, 0.0], atol=1e-12)
            np.testing.assert_allclose(env.cup_position[:2], [0.32, 0.0], atol=1e-12)
            assert env.params["cup_mass"] == 0.08 and env.params["grip_friction"] == 1.0
            np.testing.assert_array_equal(env.data.ctrl, 0.0)
        assert LEVELS[-1].approach_height == 0.14
        assert FINAL_LEVEL == 25
    finally:
        env.close()


@pytest.mark.parametrize("cls", [GeneralizationGraspEnv, CenteredGraspEnv])
def test_height_resets_have_no_lateral_configuration(cls):
    for key, value in [
        ("fixed_cup_offset", (0.01, 0.0)),
        ("xy_half_range", 0.01),
        ("xy_half_ranges", (0.01, 0.01)),
    ]:
        with pytest.raises(TypeError):
            cls(**{key: value}, render_images=False)
    with pytest.raises(ValueError):
        cls(fixed_height=0.141, render_images=False)
    env = cls(render_images=False)
    try:
        heights = []
        for seed in range(30):
            _, info = env.reset(seed=seed)
            heights.append(info["reset_height"])
            np.testing.assert_allclose(
                env.grasp_position[:2], env.cup_position[:2], atol=1e-12
            )
        assert min(heights) < 0.055 and max(heights) > 0.1
    finally:
        env.close()


def test_evaluation_is_two_hundred_distinct_centered_heights():
    cases = confirmation_cases(741000000)
    assert len(cases) == len({c["height"] for c in cases}) == 200
    assert all(
        c["cup_offset"] == [0.0, 0.0] and 0.025 <= c["height"] <= 0.14 for c in cases
    )


@pytest.mark.parametrize(
    "case",
    [
        dict(height=0.1, offset=[0.01, 0.0]),
        dict(height=0.1, offset=[0.0, -0.01]),
        dict(height=0.15, offset=[0.0, 0.0]),
    ],
)
def test_ordered_cases_reject_unsupported_reset_before_loading_policies(
    monkeypatch, case
):
    from robovision import train_ordered_pickup as trainer

    monkeypatch.setattr(trainer, "validate_ordered_manager", lambda *args: None)
    with pytest.raises(ValueError, match="centered start"):
        trainer.evaluate_ordered(None, (), registered_cases=[case])
