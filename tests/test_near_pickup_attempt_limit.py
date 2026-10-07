"""Training deadlines preserve torque physics and the original task's failure bill."""

import numpy as np
import pytest
from robovision.near_pickup import NearPickupEnv, FineNearPickupEnv
from robovision.reverse_curriculum import ReverseGraspEnv


def make(cls=FineNearPickupEnv, **kwargs):
    return cls(observation="state", render_images=False, **kwargs)


@pytest.mark.parametrize("cls", [NearPickupEnv, FineNearPickupEnv])
def test_disabled_matches_original_and_default(cls):
    default = make(cls)
    explicit = make(cls, frontier_attempt_limit=None)
    original = ReverseGraspEnv(
        curriculum_level=19, replay_fraction=0, observation="state", render_images=False
    )
    try:
        assert default.frontier_attempt_limit is None
        observations = [e.reset(seed=9)[0] for e in (default, explicit, original)]
        for key in observations[0]:
            for other in observations[1:]:
                np.testing.assert_array_equal(observations[0][key], other[key])
        for action in np.random.default_rng(8).uniform(-0.01, 0.01, (6, 5)):
            rows = [e.step(action) for e in (default, explicit, original)]
            for row in rows[1:]:
                assert rows[0][1:4] == row[1:4]
                assert rows[0][4]["reward_components"] == row[4]["reward_components"]
                for key in row[0]:
                    np.testing.assert_array_equal(rows[0][0][key], row[0][key])
            for other in (explicit, original):
                np.testing.assert_array_equal(default.data.qpos, other.data.qpos)
                np.testing.assert_array_equal(default.data.ctrl, other.data.ctrl)
    finally:
        for env in (default, explicit, original):
            env.close()


@pytest.mark.parametrize("cls", [NearPickupEnv, FineNearPickupEnv])
def test_limit_physics_observation_clock_and_exact_absorbing_failure_tail(cls):
    limited = make(cls, frontier_attempt_limit=3)
    original = make(cls)
    try:
        for env in (limited, original):
            obs, _ = env.reset(seed=5)
            assert obs["proprio"][-1] == 1
            assert env.max_steps == 500
        actions = (
            np.random.default_rng(11).uniform(-0.01, 0.01, (3, 5)).astype(np.float32)
        )
        for i, action in enumerate(actions):
            a = limited.step(action)
            b = original.step(action)
            for key in a[0]:
                np.testing.assert_array_equal(a[0][key], b[0][key])
            for name in ("qpos", "qvel", "ctrl"):
                np.testing.assert_array_equal(
                    getattr(limited.data, name), getattr(original.data, name)
                )
            if i < 2:
                assert a[1:4] == b[1:4]
                assert a[4]["reward_components"] == b[4]["reward_components"]
        assert a[2] and (not a[3]) and (not b[2])
        assert a[4]["reason"] == "frontier_attempt_limit"
        assert not a[4]["is_success"] and (not a[4]["pickup_success"])
        expected = limited.reward_function.components(
            a[4]["grasp_scores"],
            actions[-1],
            actions[-2],
            dt=0.02,
            gamma=limited.gamma,
            remaining_steps=497,
            success=False,
            failure=True,
        )
        assert a[4]["reward_components"] == expected
        assert a[1] == sum(expected.values())
        assert expected["crash"] == -5
        assert expected["remaining_time"] < 0 and expected["failure"] < 0
        assert a[0]["proprio"][-1] == pytest.approx(1 - 3 / 500)
        obs, _ = limited.reset(seed=5)
        assert limited.frontier_attempt_limit == 3 and limited.step_count == 0
        assert limited.max_steps == 500 and obs["proprio"][-1] == 1
    finally:
        limited.close()
        original.close()


def test_setter_validates_original_horizon_and_can_disable():
    env = make(frontier_attempt_limit=150)
    try:
        assert env.frontier_attempt_limit == 150
        for invalid in (True, False, 0, -1, 501, 1.0, 1.5, float("nan"), "150"):
            with pytest.raises(ValueError, match="frontier_attempt_limit"):
                env.set_frontier_attempt_limit(invalid)
            assert env.frontier_attempt_limit == 150
        env.set_frontier_attempt_limit(np.int64(500))
        assert env.frontier_attempt_limit == 500
        env.set_frontier_attempt_limit(1)
        env.set_frontier_attempt_limit(None)
        env.reset(seed=1)
        env.step_count = 150
        assert env._early_failure(env._info()) == ""
    finally:
        env.close()


@pytest.mark.parametrize("cls", [NearPickupEnv, FineNearPickupEnv])
def test_actual_step_preserves_completed_hold_at_boundary(cls, monkeypatch):
    env = make(cls, frontier_attempt_limit=1)
    try:
        env.reset(seed=3)
        env._hold_steps = env.hold_steps - 1
        physical_info = env._info()
        monkeypatch.setattr(
            env,
            "_info",
            lambda: {
                **physical_info,
                "contacts": [True, True],
                "clearance": 0.06,
                "upright": 1.0,
                "cup_speed": 0.0,
            },
        )
        monkeypatch.setattr(env, "_table_collision", lambda: False)
        _, _, done, truncated, info = env.step(np.zeros(5))
        assert done and (not truncated) and info["is_success"]
        assert info["reason"] == "success" and info["pickup_success"]
        assert info["reward_components"]["success"] == 50
        assert info["reward_components"]["failure"] == 0
        assert info["reward_components"]["crash"] == 0
    finally:
        env.close()
