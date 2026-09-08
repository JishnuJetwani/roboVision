import numpy as np
import pytest
from robovision.approach_bridge import (
    ApproachBridgeEnv,
    approach_quality,
    potential_increment,
    passes_gate,
)
from robovision.reverse_curriculum import ReverseGraspEnv


def test_progress_prefers_open_aligned_approach():
    far = approach_quality([0, 0, 0.075], 1, 0.045, False)
    near = approach_quality([0, 0, 0.025], 1, 0.045, False)
    assert near > far
    assert approach_quality([0, 0, 0.025], 1, 0.02, False) < near
    assert approach_quality([0, 0, 0.025], 0, 0.045, False) == 0
    assert approach_quality([0, 0, 0.025], 1, 0.02, True) == near


def test_shaping_telescopes_including_terminal():
    gamma = 0.995
    for values in [[2, 9, 2, 9, 2], [2, 2, 2, 2, 2], [2, 15, 1, 3, 8]]:
        extra = [
            potential_increment(a, b, gamma, i == 3)
            for i, (a, b) in enumerate(zip(values, values[1:]))
        ]
        assert sum((gamma**i * r for i, r in enumerate(extra))) == pytest.approx(-2)


def test_reset_height_and_no_mid_episode_anneal():
    env = ApproachBridgeEnv(replay_fraction=0, observation="state", render_images=False)
    try:
        _, info = env.reset(seed=4)
        assert 0.04 <= info["approach_height"] <= 0.055
        assert env.grasp_position[2] - env.cup_position[2] - 0.014 == pytest.approx(
            info["approach_height"]
        )
        env.set_bridge_stage(6)
        assert env._approach_scale == 20
        env.reset(seed=4)
        assert env._approach_scale == 0
        assert env.model.body_gravcomp.sum() == 0
    finally:
        env.close()


def test_disabled_shaping_matches_original_level23():
    a = ApproachBridgeEnv(
        fixed_height=0.075,
        shaping=False,
        replay_fraction=0,
        observation="state",
        render_images=False,
    )
    b = ReverseGraspEnv(
        curriculum_level=23, replay_fraction=0, observation="state", render_images=False
    )
    try:
        oa, _ = a.reset(seed=7)
        ob, _ = b.reset(seed=7)
        for k in oa:
            np.testing.assert_allclose(oa[k], ob[k], atol=1e-06)
        for _ in range(5):
            ra = a.step(np.zeros(5))
            rb = b.step(np.zeros(5))
            np.testing.assert_allclose(a.data.qpos, b.data.qpos, atol=1e-09)
            assert ra[1:4] == rb[1:4]
    finally:
        a.close()
        b.close()


def test_gate_rejects_missing_modes_or_one_bad_height():
    rows = [
        dict(height=h, deterministic=d, success=True)
        for h in [0.05, 0.065, 0.075]
        for d in [True, False]
        for _ in range(5)
    ]
    assert passes_gate(rows)
    assert not passes_gate([])
    assert not passes_gate([r for r in rows if r["deterministic"]])
    for r in rows:
        if r["height"] == 0.075:
            r["success"] = False
    assert not passes_gate(rows)
