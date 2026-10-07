import json
import numpy as np
import pytest
from robovision.fine_pickup import FinePickupEnv, OPENINGS, FINAL_INDEX
from robovision.reverse_curriculum import ReverseGraspEnv


def make(**kw):
    return FinePickupEnv(render_images=False, observation="state", **kw)


def test_exact_millimeter_resets_and_final_original_parity():
    for index in [0, 1]:
        env = make(initial_opening_index=index)
        try:
            obs, info = env.reset(seed=3)
            np.testing.assert_array_equal(
                env.data.qpos[4:6], [0.034 + index * 0.001] * 2
            )
            assert info["pickup_episode_opening"] == OPENINGS[index]
        finally:
            env.close()
    a = make(initial_opening_index=FINAL_INDEX)
    b = ReverseGraspEnv(
        curriculum_level=17, replay_fraction=0, render_images=False, observation="state"
    )
    try:
        oa, _ = a.reset(seed=9)
        ob, _ = b.reset(seed=9)
        for k in oa:
            np.testing.assert_array_equal(oa[k], ob[k])
        for action in np.random.default_rng(9).uniform(-0.04, 0.04, (10, 5)):
            ar = a.step(action)
            br = b.step(action)
            assert ar[1:4] == br[1:4] and ar[4]["is_success"] == br[4]["is_success"]
            assert ar[4]["reward_components"] == br[4]["reward_components"]
            for k in ar[0]:
                np.testing.assert_array_equal(ar[0][k], br[0][k])
            np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
    finally:
        a.close()
        b.close()


def test_gate_and_next_reset_only_promotion(monkeypatch):
    env = make()
    try:
        env.reset(seed=1)
        for _ in range(15):
            env._record_completed_pickup(True)
        assert env.opening_index == 0
        before = env.data.qpos.copy()
        monkeypatch.setattr(
            ReverseGraspEnv,
            "step",
            lambda self, action: ({}, 50.0, True, False, {"is_success": True}),
        )
        env.step(np.zeros(5))
        assert env.opening_index == 1 and env.episode_opening_index == 0
        np.testing.assert_array_equal(env.data.qpos, before)
        env.step(np.zeros(5))
        assert env.total_pickup_episodes == 16
        env.reset(seed=1)
        assert env.episode_opening_index == 1
        np.testing.assert_array_equal(env.data.qpos[4:6], [0.035, 0.035])
    finally:
        env.close()


def test_exact_json_state_resume_and_failure_threshold():
    a = make()
    b = None
    try:
        for outcome in [True] * 15:
            a._record_completed_pickup(outcome)
        state = json.loads(json.dumps(a.get_pickup_curriculum_state()))
        b = make(state=state, initial_opening_index=9)
        assert b.get_pickup_curriculum_state() == state
        a._record_completed_pickup(True)
        b._record_completed_pickup(True)
        assert b.get_pickup_curriculum_state() == a.get_pickup_curriculum_state()
        for _ in range(16):
            b._record_completed_pickup(False)
        assert b.opening_index == 1
        for _ in range(10):
            b._record_completed_pickup(True)
        assert b.opening_index == 2
    finally:
        a.close()
        if b:
            b.close()
    for invalid in [-1, 12, True, 1.2]:
        with pytest.raises(ValueError):
            make(initial_opening_index=invalid)


def test_frozen_promotion_accumulates_resumes_and_reenables_on_reset():
    env = make(promotion_enabled=False)
    resumed = None
    try:
        env.reset(seed=1)
        for outcome in [False] * 6 + [True] * 10:
            env._record_completed_pickup(outcome)
        assert env.opening_index == 0 and env.episodes_at_opening == 16
        assert sum(env.pickup_recent) == 10
        state = json.loads(json.dumps(env.get_pickup_curriculum_state()))
        assert state["promotion_enabled"] is False
        resumed = make(state=state)
        resumed.reset(seed=1)
        assert resumed.promotion_enabled is False and resumed.opening_index == 0
        resumed._record_completed_pickup(True)
        assert resumed.episodes_at_opening == 17 and resumed.total_pickup_episodes == 17
        assert resumed.opening_index == 0
        resumed.set_pickup_promotion_enabled(True)
        assert resumed.opening_index == 0
        resumed.reset(seed=1)
        assert resumed.opening_index == 1 and resumed.episode_opening_index == 1
        assert resumed.total_pickup_episodes == 17 and resumed.episodes_at_opening == 0
        np.testing.assert_array_equal(resumed.data.qpos[4:6], [0.035, 0.035])
    finally:
        env.close()
        if resumed:
            resumed.close()


def test_older_state_defaults_enabled_and_flag_does_not_change_task():
    a = make(promotion_enabled=False)
    b = make()
    try:
        state = b.get_pickup_curriculum_state()
        state.pop("promotion_enabled")
        c = make(state=state)
        try:
            assert c.promotion_enabled is True
        finally:
            c.close()
        oa, _ = a.reset(seed=6)
        ob, _ = b.reset(seed=6)
        for k in oa:
            np.testing.assert_array_equal(oa[k], ob[k])
        ar = a.step(np.zeros(5))
        br = b.step(np.zeros(5))
        assert ar[1:4] == br[1:4]
        np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
        with pytest.raises(ValueError):
            a.set_pickup_promotion_enabled(1)
    finally:
        a.close()
        b.close()


def test_attempt_limit_preserves_clock_physics_then_charges_original_tail():
    a = make(attempt_limit=3, promotion_enabled=False)
    b = make(promotion_enabled=False)
    try:
        oa, _ = a.reset(seed=5)
        ob, _ = b.reset(seed=5)
        assert a.max_steps == b.max_steps == 500
        for k in oa:
            np.testing.assert_array_equal(oa[k], ob[k])
        for step in range(1, 4):
            ar = a.step(np.zeros(5))
            br = b.step(np.zeros(5))
            np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
            for k in ar[0]:
                np.testing.assert_array_equal(ar[0][k], br[0][k])
            if step < 3:
                assert ar[1:4] == br[1:4]
        assert ar[2] and (not ar[3]) and (not br[2])
        info = ar[4]
        assert info["reason"] == "pickup_attempt_limit" and (not info["is_success"])
        expected = a.reward_function.components(
            info["grasp_scores"],
            np.zeros(5),
            np.zeros(5),
            dt=a.control_dt,
            gamma=a.gamma,
            remaining_steps=497,
            success=False,
            failure=True,
        )
        assert info["reward_components"] == expected
        assert ar[1] == pytest.approx(sum(expected.values()))
        assert (
            expected["remaining_time"] < 0
            and expected["failure"] < 0
            and (expected["crash"] < 0)
        )
        assert ar[0]["proprio"][-1] == pytest.approx(1 - 3 / 500)
    finally:
        a.close()
        b.close()


def test_attempt_limit_state_inheritance_validation_and_boundary_success():
    env = make(attempt_limit=150)
    other = None
    try:
        state = json.loads(json.dumps(env.get_pickup_curriculum_state()))
        other = make(state=state, attempt_limit=2)
        assert other.attempt_limit == 150
        other.set_pickup_attempt_limit(None)
        assert other.get_pickup_curriculum_state()["attempt_limit"] is None
        for bad in [0, -1, 501, 1.5, True]:
            with pytest.raises(ValueError):
                other.set_pickup_attempt_limit(bad)
        other.set_pickup_attempt_limit(150)
        other.reset(seed=1)
        other.step_count = 150
        other._hold_steps = other.hold_steps
        assert other._early_failure(other._info()) == ""
        other._hold_steps = 0
        assert other._early_failure(other._info()) == "pickup_attempt_limit"
        state.pop("attempt_limit")
        legacy = make(state=state)
        try:
            assert legacy.attempt_limit is None
        finally:
            legacy.close()
    finally:
        env.close()
        if other:
            other.close()
