import numpy as np
import pytest
from robovision.near_pickup import NearPickupEnv, HEIGHTS
from robovision.reverse_curriculum import ReverseGraspEnv
from robovision.train_skill_curriculum import (
    curriculum_definition,
    gate_configurations,
    resolve_source_lesson,
    make_env,
)


def test_original_full_pickup_physics_reward_observations_and_force_parity():
    for height, level in [(0.01, 19), (0.02, 20), (0.05, 22), (0.14, 25)]:
        a = NearPickupEnv(
            subtask_stage=HEIGHTS.index(height),
            observation="state",
            render_images=False,
        )
        b = ReverseGraspEnv(
            curriculum_level=level,
            replay_fraction=0,
            observation="state",
            render_images=False,
        )
        try:
            oa, _ = a.reset(seed=7)
            ob, _ = b.reset(seed=7)
            for k in oa:
                np.testing.assert_array_equal(oa[k], ob[k])
            for action in np.random.default_rng(7).uniform(-0.04, 0.04, (8, 5)):
                ar = a.step(action)
                br = b.step(action)
                assert ar[1:4] == br[1:4]
                assert (
                    ar[4]["is_success"]
                    == br[4]["is_success"]
                    == ar[4]["pickup_success"]
                )
                assert ar[4]["reward_components"] == br[4]["reward_components"]
                np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
                np.testing.assert_array_equal(a.data.ctrl, b.data.ctrl)
                assert ar[4]["approach_episode"] is False
        finally:
            a.close()
            b.close()


def test_curriculum_gates_and_crossvariant_start_are_honest():
    assert len(HEIGHTS) == 51 and HEIGHTS[0] == 0.01 and (HEIGHTS[-1] == 0.14)
    _, lessons, approach_last, pickup_first = curriculum_definition("near_pickup")
    assert approach_last == -1 and pickup_first == 0
    assert all((l.approach_fraction == l.shaping_scale == 0 for l in lessons))
    assert gate_configurations("near_pickup", 0) == [(0.01, 0, "pickup")]
    assert gate_configurations("near_pickup", 50) == [(0.14, 50, "pickup")]
    assert resolve_source_lesson(
        "near_pickup", {"variant": "fine_hover_bootstrap"}, {"lesson": 9}
    ) == (0, None)
    assert gate_configurations("fine_hover_bootstrap", 9) == [
        (0.07, 9, "approach"),
        (0.075, 9, "approach"),
        (0.08, 9, "approach"),
    ]


def test_frontier_next_reset_progression_and_unsupported_height_options():
    env = NearPickupEnv(observation="state", render_images=False)
    try:
        env.reset(seed=4)
        before = env.data.qpos.copy()
        env.set_subtask_stage(1)
        np.testing.assert_array_equal(env.data.qpos, before)
        _, info = env.reset(seed=4)
        assert info["near_pickup_height"] == 0.011
        assert env.grasp_position[2] - env.cup_position[2] - 0.014 == pytest.approx(
            0.011
        )
    finally:
        env.close()
    for kwargs in [
        dict(bootstrap_height_jitter=0.005),
        dict(frontier_fixed_height=0.08),
    ]:
        with pytest.raises(ValueError):
            make_env("near_pickup", 0, 0, **kwargs)
