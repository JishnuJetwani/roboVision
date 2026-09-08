import numpy as np
import pytest
from robovision.approach_bridge import ApproachBridgeEnv, approach_quality
from robovision.dense_approach import DenseApproachEnv, best_progress


def test_failed_approach_has_positive_return_without_cycle_farming():
    best = 0.2
    rewards = []
    for quality in [0.3, 0.7, 0.1, 0.7, 0.9, 0.2]:
        reward, best = best_progress(best, quality, 20.0)
        rewards.append(reward)
    assert sum(rewards) == pytest.approx(14.0)
    assert rewards[2:4] == [0.0, 0.0]
    assert rewards[-1] == 0.0
    assert sum((0.995**i * r for i, r in enumerate(rewards))) > 0


def test_total_bonus_bounded_even_for_random_loops():
    rng = np.random.default_rng(3)
    best = 0.1
    total = 0.0
    for value in rng.random(1000):
        reward, best = best_progress(best, value, 20.0)
        total += reward
    assert 0 < total <= 18.0
    assert total == pytest.approx(20 * (best - 0.1))


def test_open_descent_earns_more_than_closed_descent_or_upward_motion():
    start = approach_quality([0, 0, 0.075], 1, 0.045, False)
    lower = approach_quality([0, 0, 0.06], 1, 0.045, False)
    closed = approach_quality([0, 0, 0.06], 1, 0.015, False)
    higher = approach_quality([0, 0, 0.09], 1, 0.045, False)
    assert best_progress(start, lower, 20)[0] > 0
    assert best_progress(start, closed, 20)[0] == 0
    assert best_progress(start, higher, 20)[0] == 0


@pytest.mark.parametrize("scale", [-1, float("nan"), float("inf")])
def test_rejects_invalid_scale(scale):
    with pytest.raises(ValueError):
        DenseApproachEnv(dense_scale=scale)


def test_disabled_variant_exactly_preserves_original_reward_physics_and_observations():
    kwargs = dict(
        fixed_height=0.075, replay_fraction=0, observation="state", render_images=False
    )
    a = DenseApproachEnv(shaping=False, **kwargs)
    b = ApproachBridgeEnv(shaping=False, **kwargs)
    try:
        oa, _ = a.reset(seed=17)
        ob, _ = b.reset(seed=17)
        for key in oa:
            np.testing.assert_array_equal(oa[key], ob[key])
        rng = np.random.default_rng(5)
        for _ in range(30):
            action = rng.normal(0, 0.005, 5)
            ra, rb = (a.step(action), b.step(action))
            np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
            assert ra[1:4] == rb[1:4]
            assert ra[4]["reward_components"]["dense_approach_progress"] == 0
            if ra[2] or ra[3]:
                break
    finally:
        a.close()
        b.close()


def test_scale_updates_only_at_reset_and_replay_is_unmodified():
    env = DenseApproachEnv(
        replay_fraction=0.9999, observation="state", render_images=False
    )
    try:
        _, info = env.reset(seed=1)
        assert info["curriculum_replay"]
        assert info["dense_approach_scale"] == 0
        env.replay_fraction = 0
        _, info = env.reset(seed=2)
        assert info["dense_approach_scale"] == 20
        env.set_dense_scale(5)
        assert env._dense_episode_scale == 20
        _, info = env.reset(seed=2)
        assert info["dense_approach_scale"] == 5
        assert env._dense_best == env._dense_initial
        assert env.model.body_gravcomp.sum() == 0
    finally:
        env.close()
