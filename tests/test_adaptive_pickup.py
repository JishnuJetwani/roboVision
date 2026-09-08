import json
import numpy as np
import pytest
from robovision.adaptive_pickup import AdaptivePickupEnv
from robovision.reverse_curriculum import ReverseGraspEnv, LEVELS


def make(**kw):
    return AdaptivePickupEnv(render_images=False, observation="state", **kw)


def test_actual_initial_reset_and_step_match_original_level12():
    a = make()
    b = ReverseGraspEnv(
        curriculum_level=12, replay_fraction=0, render_images=False, observation="state"
    )
    try:
        oa, _ = a.reset(seed=4)
        ob, _ = b.reset(seed=4)
        np.testing.assert_array_equal(a.data.qpos[4:6], [0.029, 0.029])
        for k in oa:
            np.testing.assert_array_equal(oa[k], ob[k])
        for _ in range(4):
            ar = a.step(np.zeros(5))
            br = b.step(np.zeros(5))
            assert ar[1:4] == br[1:4]
            for k in ar[0]:
                np.testing.assert_array_equal(ar[0][k], br[0][k])
    finally:
        a.close()
        b.close()


def test_minimum_window_threshold_and_final_cap():
    env = make()
    try:
        for _ in range(15):
            env._record_completed_pickup(True)
        assert env.pickup_level == 12
        env._record_completed_pickup(True)
        assert env.pickup_level == 13 and (not env.pickup_recent)
        for success in [True] * 6 + [False] * 3 + [True] * 7:
            env._record_completed_pickup(success)
        assert env.pickup_level == 13
        env._record_completed_pickup(True)
        assert env.pickup_level == 13
        env._record_completed_pickup(True)
        assert env.pickup_level == 13
        env._record_completed_pickup(True)
        assert env.pickup_level == 14
        for _ in range(100):
            env._record_completed_pickup(True)
        assert env.pickup_level == 17
        env.reset(seed=1)
        np.testing.assert_array_equal(env.data.qpos[4:6], [0.045, 0.045])
    finally:
        env.close()


def test_only_done_original_success_counts_and_promotion_next_reset(monkeypatch):
    env = make(minimum_episodes=1, window=1, required_successes=1)
    try:
        env.reset(seed=1)
        original = ReverseGraspEnv.step
        monkeypatch.setattr(
            ReverseGraspEnv,
            "step",
            lambda self, action: ({}, 0.0, False, False, {"is_success": True}),
        )
        env.step(np.zeros(5))
        assert env.total_pickup_episodes == 0
        monkeypatch.setattr(
            ReverseGraspEnv,
            "step",
            lambda self, action: ({}, 50.0, True, False, {"is_success": True}),
        )
        env.step(np.zeros(5))
        assert env.pickup_level == 13 and env.episode_level == 12
        env.step(np.zeros(5))
        assert env.total_pickup_episodes == 1
        monkeypatch.setattr(ReverseGraspEnv, "step", original)
        env.reset(seed=1)
        assert env.episode_level == 13
        np.testing.assert_array_equal(env.data.qpos[4:6], [LEVELS[13].opening] * 2)
        env.reset(seed=2)
        assert env.total_pickup_episodes == 1
    finally:
        env.close()


def test_state_round_trip_exact_and_independent():
    a = make()
    b = None
    try:
        for _ in range(18):
            a._record_completed_pickup(True)
        state = json.loads(json.dumps(a.get_pickup_curriculum_state()))
        b = make(pickup_state=state)
        assert b.get_pickup_curriculum_state() == state
        state["recent"].append(False)
        assert b.get_pickup_curriculum_state() == a.get_pickup_curriculum_state()
        for outcome in [False, True] * 12:
            a._record_completed_pickup(outcome)
            b._record_completed_pickup(outcome)
        assert a.get_pickup_curriculum_state() == b.get_pickup_curriculum_state()
    finally:
        a.close()
        if b:
            b.close()


def test_fresh_initial_level14_and_resume_state_authoritative():
    env = make(initial_level=14)
    resumed = None
    try:
        env.reset(seed=2)
        assert env.pickup_level == env.episode_level == 14
        np.testing.assert_array_equal(env.data.qpos[4:6], [0.034, 0.034])
        for _ in range(3):
            env._record_completed_pickup(True)
        state = json.loads(json.dumps(env.get_pickup_curriculum_state()))
        resumed = make(state=state, initial_level=12)
        assert resumed.get_pickup_curriculum_state() == state
        resumed.reset(seed=2)
        assert resumed.episode_level == 14
    finally:
        env.close()
        if resumed:
            resumed.close()
    for invalid in [11, 18, 14.5, True]:
        with pytest.raises(ValueError):
            make(initial_level=invalid)
