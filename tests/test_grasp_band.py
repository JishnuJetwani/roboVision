import numpy as np
import pytest
from robovision.grasp_band import GraspBandEnv, band_quality, best_quality_credit
from robovision.approach_bridge import ApproachBridgeEnv


def test_middle_beats_rim_and_bottom_requires_both_sides():
    middle = band_quality([0.028, 0, 0], [-0.028, 0, 0])
    assert middle == pytest.approx(1.0)
    for z in [-0.04, 0.04]:
        assert band_quality([0.028, 0, z], [-0.028, 0, z]) < 0.15
    assert band_quality([0.028, 0, 0], [-0.028, 0, 0.04]) < 0.15
    assert band_quality([0.028, 0, 0], [0.028, 0, 0]) == 0
    assert band_quality([0.028, 0, 0], None) == 0


def test_credit_is_bounded_and_cannot_be_repeated():
    best = 0.0
    total = 0.0
    for q in [0.1, 0.5, 0.2, 0.5, 0.8, 0, 1, 0, 1]:
        credit, best = best_quality_credit(best, q, 5.0)
        total += credit
    assert total == pytest.approx(5.0)
    assert best_quality_credit(1.0, 1.0, 5.0) == (0.0, 1.0)


def test_bonus_disabled_preserves_physics_and_rewards():
    args = dict(
        fixed_height=0.05, replay_fraction=0, observation="state", render_images=False
    )
    a = GraspBandEnv(grasp_band_bonus=0, **args)
    b = ApproachBridgeEnv(**args)
    try:
        a.reset(seed=8)
        b.reset(seed=8)
        for _ in range(10):
            action = np.array([0, -0.04, -0.02, 0.01, -0.15])
            ra = a.step(action)
            rb = b.step(action)
            assert ra[1:4] == rb[1:4]
            np.testing.assert_allclose(a.data.qpos, b.data.qpos, atol=1e-10)
    finally:
        a.close()
        b.close()


def test_real_contact_score_unchanged_by_lifting():
    import mujoco

    env = GraspBandEnv(replay_fraction=0.999, observation="state", render_images=False)
    try:
        env.curriculum_level = 5
        env.reset(seed=1)
        before = env.grasp_band_metrics()
        assert before["quality"] > 0
        base_z = env.model.body_pos[1, 2]
        env.model.body_pos[1, 2] = base_z + 0.03
        env.data.qpos[env._cup_qadr + 2] += 0.03
        mujoco.mj_forward(env.model, env.data)
        after = env.grasp_band_metrics()
        assert after["quality"] == pytest.approx(before["quality"], abs=1e-06)
    finally:
        env.close()
