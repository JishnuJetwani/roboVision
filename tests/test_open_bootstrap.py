import numpy as np
import pytest
from robovision.open_bootstrap import (
    OpenBootstrapEnv,
    LESSONS,
    FINAL_LESSON,
    FULL_APPROACH_LESSON,
)
from robovision.approach_subtask import ApproachSubtaskEnv, approach_metrics
from robovision.approach_bridge import ApproachBridgeEnv


def make(**kw):
    return OpenBootstrapEnv(
        replay_fraction=0, render_images=False, observation="state", **kw
    )


def test_schedule_has_small_monotonic_descents_then_pickup():
    heights = [s.target_height for s in LESSONS[: FULL_APPROACH_LESSON + 1]]
    assert heights[:3] == [0.075] * 3
    assert all((-0.0050001 <= b - a <= 0 for a, b in zip(heights, heights[1:])))
    assert heights[-1] == 0
    assert LESSONS[-1].approach_fraction == LESSONS[-1].shaping_scale == 0
    assert [l.stable_steps for l in LESSONS[:3]] == [1, 2, 3]


def test_first_three_reset_fixed_then_variation_and_explicit_override():
    env = make()
    fixed = make(fixed_height=0.08)
    try:
        for stage in range(4):
            env.set_subtask_stage(stage)
            fixed.set_subtask_stage(stage)
            heights = []
            for seed in range(4):
                _, info = env.reset(seed=seed)
                heights.append(info["approach_height"])
                _, info = fixed.reset(seed=seed)
                assert info["approach_height"] == 0.08
            if stage < 3:
                assert heights == [0.075] * 4
            else:
                assert len(set(heights)) == 4 and all(
                    (0.07 <= h <= 0.08 for h in heights)
                )
    finally:
        env.close()
        fixed.close()


def test_relaxed_open_thresholds_do_not_change_default_metrics():
    assert approach_metrics(
        [0, 0, 0], 1, [0.033, 0.033], 0.15, minimum_opening=0.032, maximum_speed=0.2
    )[1]
    assert not approach_metrics([0, 0, 0], 1, [0.033, 0.033], 0.15)[1]


def test_bootstrap_success_durations_and_no_mid_episode_change(monkeypatch):
    env = make()
    try:
        for stage in range(3):
            env.set_subtask_stage(stage)
            env.reset(seed=8)
            monkeypatch.setattr(env, "_metrics", lambda: (1.0, True))
            env.set_subtask_stage(FINAL_LESSON)
            for i in range(stage + 1):
                _, _, done, _, info = env.step(np.zeros(5))
                assert done == (i == stage)
            assert info["approach_success"] and (not info["pickup_success"])
            assert info["subtask_stage"] == stage
    finally:
        env.close()


def test_final_matches_unshaped_pickup():
    a = make(subtask_stage=FINAL_LESSON, fixed_height=0.075)
    b = ApproachBridgeEnv(
        replay_fraction=0,
        render_images=False,
        observation="state",
        shaping=False,
        fixed_height=0.075,
    )
    try:
        a.reset(seed=10)
        b.reset(seed=10)
        for _ in range(5):
            ar = a.step(np.zeros(5))
            br = b.step(np.zeros(5))
            assert ar[1:4] == br[1:4]
            np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
    finally:
        a.close()
        b.close()


def test_early_close_is_opt_in_and_charges_full_failure_tail():
    import mujoco
    from robovision.approach_subtask import approach_reward

    enabled = make(early_close_failure=True)
    disabled = make()
    try:
        for env in [enabled, disabled]:
            env.reset(seed=12)
            env.data.qpos[4:6] = 0.029
            mujoco.mj_forward(env.model, env.data)
        action = np.zeros(5)
        _, reward, done, _, info = enabled.step(action)
        _, _, other_done, _, other_info = disabled.step(action)
        assert done and (not other_done)
        assert info["reason"] == "closed_before_approach"
        assert not info["is_success"] and (not info["approach_success"])
        expected = approach_reward(
            info["approach_quality"],
            dt=enabled.control_dt,
            gamma=enabled.gamma,
            remaining=enabled.max_steps - 1,
            success=False,
            failure=True,
        )
        assert info["reward_components"] == expected
        assert reward == pytest.approx(sum(expected.values()))
        assert info["episode_reward_components"] == expected
        np.testing.assert_array_equal(enabled.data.qpos, disabled.data.qpos)
        assert other_info["reason"] == ""
    finally:
        enabled.close()
        disabled.close()


def test_early_close_never_changes_full_pickup_rules():
    import mujoco

    a = make(subtask_stage=FINAL_LESSON, early_close_failure=True)
    b = make(subtask_stage=FINAL_LESSON, early_close_failure=False)
    try:
        for env in [a, b]:
            env.reset(seed=9)
            env.data.qpos[4:6] = 0.029
            mujoco.mj_forward(env.model, env.data)
        for _ in range(5):
            ar = a.step(np.zeros(5))
            br = b.step(np.zeros(5))
            assert ar[1:] == br[1:]
            for key in ar[0]:
                np.testing.assert_array_equal(ar[0][key], br[0][key])
    finally:
        a.close()
        b.close()


def test_strict_descent_actual_pose_arrival_band_and_bootstrap_unchanged():
    import mujoco

    env = make(strict_descent=True, subtask_stage=3)
    try:
        env.reset(seed=3)
        for error, expected in [
            (-0.0051, False),
            (-0.0049, True),
            (0.0, True),
            (0.0009, True),
            (0.0011, False),
            (0.005, False),
        ]:
            target = env.cup_position + [
                0,
                0,
                0.014 + env._lesson.target_height + error,
            ]
            env.data.qpos[:4] = env.inverse_kinematics(target)
            env.data.qvel[:] = 0
            mujoco.mj_forward(env.model, env.data)
            assert env._metrics()[1] == expected
        env.set_subtask_stage(2)
        env.fixed_height = 0.08
        env.reset(seed=3)
        assert env._metrics()[1]
    finally:
        env.close()


def test_strict_descent_reset_schedule_and_explicit_eval_override():
    env = make(strict_descent=True)
    override = make(strict_descent=True, fixed_height=0.08)
    try:
        for stage in (3, 4, 5):
            env.set_subtask_stage(stage)
            override.set_subtask_stage(stage)
            heights = []
            for seed in range(4):
                _, info = env.reset(seed=seed)
                heights.append(info["approach_height"])
                _, explicit = override.reset(seed=seed)
                assert explicit["approach_height"] == 0.08
            if stage in (3, 4):
                assert heights == [0.075] * 4
            else:
                assert len(set(heights)) == 4
    finally:
        env.close()
        override.close()


def test_strict_descent_does_not_change_actions_or_observation_shape():
    a = make(strict_descent=True, subtask_stage=3, fixed_height=0.075)
    b = make(strict_descent=False, subtask_stage=3, fixed_height=0.075)
    try:
        oa, _ = a.reset(seed=2)
        ob, _ = b.reset(seed=2)
        for k in oa:
            np.testing.assert_array_equal(oa[k], ob[k])
        action = np.array([0.01, 0.02, 0.03, 0.04, 0.1])
        a.step(action)
        b.step(action)
        np.testing.assert_array_equal(a.data.ctrl, b.data.ctrl)
        np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
        spec = a.specification()["optional_strict_descent"]
        assert spec["default"] is False and spec["arrival_z_error_m"] == [-0.005, 0.001]
    finally:
        a.close()
        b.close()


def test_bootstrap_height_jitter_support_determinism_and_fixed_override():
    env = make(bootstrap_height_jitter=0.005)
    fixed = make(bootstrap_height_jitter=0.005, fixed_height=0.083)
    try:
        for stage in range(3):
            env.set_subtask_stage(stage)
            fixed.set_subtask_stage(stage)
            heights = []
            for seed in range(30):
                _, info = env.reset(seed=seed)
                h = info["approach_height"]
                heights.append(h)
                assert 0.07 <= h <= 0.08
                assert env.grasp_position[2] - env.cup_position[
                    2
                ] - 0.014 == pytest.approx(h)
                _, repeat = env.reset(seed=seed)
                assert repeat["approach_height"] == h
                _, override = fixed.reset(seed=seed)
                assert override["approach_height"] == 0.083
            assert len(set(heights)) == 30
            assert min(heights) < 0.072 and max(heights) > 0.078
        assert env.specification()["optional_bootstrap_height_jitter"]["default"] == 0.0
    finally:
        env.close()
        fixed.close()


def test_bootstrap_jitter_rejects_invalid_values_and_has_no_later_effect():
    for jitter in [-0.001, float("nan"), float("inf"), 0.066]:
        with pytest.raises(ValueError):
            make(bootstrap_height_jitter=jitter)
    a = make(bootstrap_height_jitter=0.005, subtask_stage=5)
    b = make(bootstrap_height_jitter=0.0, subtask_stage=5)
    try:
        oa, ia = a.reset(seed=7)
        ob, ib = b.reset(seed=7)
        assert ia == ib
        for key in oa:
            np.testing.assert_array_equal(oa[key], ob[key])
    finally:
        a.close()
        b.close()


def test_strict_descent_fixed_training_resets_also_receive_jitter():
    env = make(bootstrap_height_jitter=0.005, strict_descent=True)
    fixed = make(bootstrap_height_jitter=0.005, strict_descent=True, fixed_height=0.082)
    try:
        for stage in (3, 4):
            env.set_subtask_stage(stage)
            fixed.set_subtask_stage(stage)
            heights = []
            for seed in range(30):
                _, info = env.reset(seed=seed)
                heights.append(info["approach_height"])
                _, repeat = env.reset(seed=seed)
                assert info["approach_height"] == repeat["approach_height"]
                _, override = fixed.reset(seed=seed)
                assert override["approach_height"] == 0.082
            assert all((0.07 <= h <= 0.08 for h in heights))
            assert min(heights) < 0.072 and max(heights) > 0.078
    finally:
        env.close()
        fixed.close()
