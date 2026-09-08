import numpy as np
import pytest
from robovision.approach_subtask import (
    ApproachSubtaskEnv,
    approach_metrics,
    approach_reward,
)
from robovision.approach_bridge import ApproachBridgeEnv


def test_endpoint_requires_open_aligned_slow_pose():
    good = approach_metrics([0, 0, 0], 1, [0.045, 0.045], 0)
    assert good == (1.0, True)
    for args in [
        ([0, 0, 0.075], 1, [0.045, 0.045], 0),
        ([0, 0, 0], 1, [0.045, 0.025], 0),
        ([0, 0, 0], 0.8, [0.045, 0.045], 0),
        ([0, 0, 0], 1, [0.045, 0.045], 0.2),
    ]:
        assert not approach_metrics(*args)[1]


def test_hovering_never_pays_and_progress_reduces_cost():
    def reward(q, **kwargs):
        return sum(
            approach_reward(
                q,
                dt=0.02,
                gamma=0.995,
                remaining=100,
                success=False,
                failure=False,
                **kwargs,
            ).values()
        )

    assert reward(0) < reward(0.5) < reward(1) < 0
    crash = approach_reward(
        1, dt=0.02, gamma=0.995, remaining=100, success=False, failure=True
    )
    assert sum(crash.values()) < reward(0)


def make(**kwargs):
    return ApproachSubtaskEnv(
        observation="state", render_images=False, replay_fraction=0, **kwargs
    )


def test_no_controller_and_next_reset_stage_change():
    env = make(fixed_height=0.075)
    try:
        _, info = env.reset(seed=5)
        assert info["approach_episode"]
        assert env.grasp_position[2] - env.cup_position[2] - 0.014 == pytest.approx(
            0.075
        )
        env.set_subtask_stage(8)
        assert env.episode_subtask_stage == 0
        action = np.array([0.01, 0.02, 0.03, 0.04, -0.1])
        env.step(action)
        np.testing.assert_allclose(
            env.data.ctrl[:4], env.torque_limits * action[:4], rtol=1e-06
        )
        np.testing.assert_allclose(env.data.ctrl[4:], -1.0)
        assert not env.model.body_gravcomp.any()
        _, info = env.reset(seed=5)
        assert not info["approach_episode"]
    finally:
        env.close()


def test_approach_success_separate_from_pickup(monkeypatch):
    env = make()
    try:
        env.reset(seed=5)
        monkeypatch.setattr(env, "_metrics", lambda: (1.0, True))
        for _ in range(3):
            _, _, done, _, info = env.step(np.zeros(5))
        assert done and info["is_success"] and info["approach_success"]
        assert not info["pickup_success"]
        assert info["reason"] == "approach_success"
    finally:
        env.close()


def test_final_lesson_matches_unshaped_pickup():
    a = make(subtask_stage=8, fixed_height=0.075)
    b = ApproachBridgeEnv(
        fixed_height=0.075,
        shaping=False,
        replay_fraction=0,
        observation="state",
        render_images=False,
    )
    try:
        a.reset(seed=5)
        b.reset(seed=5)
        for _ in range(5):
            ar = a.step(np.zeros(5))
            br = b.step(np.zeros(5))
            assert ar[1:4] == br[1:4]
            np.testing.assert_allclose(a.data.qpos, b.data.qpos)
    finally:
        a.close()
        b.close()


def test_retention_samples_only_explicit_full_skills():
    env = ApproachSubtaskEnv(
        replay_fraction=0.999, observation="state", render_images=False
    )
    try:
        levels = set()
        for seed in range(30):
            _, info = env.reset(seed=seed)
            assert info["curriculum_replay"] and (not info["approach_episode"])
            assert info["curriculum_frontier"] == 23
            levels.add(info["curriculum_level"])
            assert env._approach_scale == 0
        assert levels == {5, 11, 17}
    finally:
        env.close()
