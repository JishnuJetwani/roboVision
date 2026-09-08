import mujoco
import numpy as np
import pytest
from robovision.hover_bootstrap import (
    HoverBootstrapEnv,
    HOVER_STEPS,
    LESSONS,
    FIRST_DESCENT_LESSON,
    FULL_APPROACH_LESSON,
    FULL_PICKUP_LESSON,
    FINAL_LESSON,
)


def make(**kw):
    return HoverBootstrapEnv(
        replay_fraction=0, render_images=False, observation="state", **kw
    )


def test_hover_schedule_then_original_descent_pickup():
    assert [l.stable_steps for l in LESSONS[:6]] == list(HOVER_STEPS)
    assert FIRST_DESCENT_LESSON == 6 and FULL_APPROACH_LESSON == 20
    assert FULL_PICKUP_LESSON == 24 and FINAL_LESSON == 25
    assert LESSONS[6].target_height == 0.07 and LESSONS[20].target_height == 0
    assert LESSONS[-1].shaping_scale == 0


def test_hover_duration_enforced_without_controller(monkeypatch):
    env = make(subtask_stage=1)
    try:
        env.reset(seed=3)
        monkeypatch.setattr(env, "_metrics", lambda: (1.0, True))
        for i in range(5):
            _, _, done, _, info = env.step(np.zeros(5))
            assert done == (i == 4)
        assert info["approach_success"] and (not info["pickup_success"])
        np.testing.assert_array_equal(env.data.ctrl, np.zeros(6))
        assert not env.model.body_gravcomp.any()
    finally:
        env.close()


def test_strict_descent_does_not_tighten_hover_tolerance():
    env = make(strict_descent=True)
    try:
        for stage in range(6):
            env.set_subtask_stage(stage)
            env.fixed_height = 0.08
            env.reset(seed=4)
            assert env._metrics()[1]
        env.set_subtask_stage(6)
        env.reset(seed=4)
        assert not env._metrics()[1]
        for error, expected in [
            (0.0009, True),
            (0.0011, False),
            (-0.0049, True),
            (-0.0051, False),
        ]:
            env.data.qpos[:4] = env.inverse_kinematics(
                env.cup_position + [0, 0, 0.014 + 0.07 + error]
            )
            env.data.qvel[:] = 0
            mujoco.mj_forward(env.model, env.data)
            assert env._metrics()[1] == expected
    finally:
        env.close()


def test_jitter_covers_all_hover_and_fixed_descent_stages_and_eval_override():
    env = make(strict_descent=True, bootstrap_height_jitter=0.005)
    try:
        for stage in range(8):
            env.set_subtask_stage(stage)
            heights = []
            for seed in range(10):
                env.fixed_height = None
                _, info = env.reset(seed=seed)
                heights.append(info["approach_height"])
                env.fixed_height = 0.082
                _, info = env.reset(seed=seed)
                assert info["approach_height"] == 0.082
            assert len(set(heights)) == 10
            assert all((0.07 <= h <= 0.08 for h in heights))
        spec = env.specification()
        assert spec["optional_strict_descent"]["start_lesson"] == 6
        assert spec["optional_bootstrap_height_jitter"]["lessons"] == list(range(6))
    finally:
        env.close()
