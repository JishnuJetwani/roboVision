"""Saved simulator states must preserve continuation, not just visible geometry."""

import copy
import pickle
import gymnasium as gym
import mujoco
import numpy as np
import pytest
from stable_baselines3.common.monitor import Monitor
from robovision.hierarchical_skills import ApproachSkillEnv, GraspSkillEnv, LiftSkillEnv
from robovision.joint_env import JointGraspEnv
from robovision.skill_state import (
    capture_skill_state,
    restore_skill_state,
    restore_skill_handoff,
)

ACTION = np.array([0.001, -0.03, -0.004, 0.005, -0.2], np.float32)


def arrays_equal(actual, expected):
    for name in (
        "qpos",
        "qvel",
        "qacc",
        "qacc_warmstart",
        "ctrl",
        "act",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "userdata",
        "qfrc_constraint",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.time == expected.time


@pytest.mark.parametrize("cls", [ApproachSkillEnv, GraspSkillEnv, LiftSkillEnv])
def test_full_snapshot_pickle_restores_exact_continuation_model_rng_and_history(
    cls, monkeypatch
):
    monkeypatch.setattr(
        JointGraspEnv,
        "render_camera",
        lambda self: np.full((96, 96, 3), self.step_count, np.uint8),
    )
    env = cls(render_images=True)
    try:
        observation, info = env.reset(seed=4)
        for _ in range(5):
            observation, _, _, _, info = env.step(ACTION)
        assert observation["image"][0, 0, 0] == 4
        assert observation["image"][3, 0, 0] == 5
        snapshot = pickle.loads(
            pickle.dumps(capture_skill_state(env, observation, info=info))
        )
        expected_rng = copy.deepcopy(env.get_rng_state())
        expected = []
        for _ in range(4):
            result = env.step(ACTION)
            expected.append((copy.copy(env.data), result))
        env.model.body_mass[env._cup_bid] = 0.119
        env.model.pair_friction[:] = 0.12
        env.model.geom_rgba[:] = 0.2
        env.model.opt.gravity[:] = [0.0, 0.0, -2.0]
        env.data.qfrc_applied[:] = 20.0
        env.data.ctrl[:] = 10.0
        env.last_action[:] = 1.0
        env._previous_frame[:] = 90
        env._phase_reward_totals = {"unexpected": 123.0}
        env.unexpected_new_history = 5
        env.rng.random(4)
        restored, restored_info = restore_skill_state(env, snapshot)
        for key in observation:
            np.testing.assert_array_equal(restored[key], observation[key])
        assert restored_info == info
        assert env.get_rng_state() == expected_rng
        assert not hasattr(env, "unexpected_new_history")
        for field in (
            "body_mass",
            "body_inertia",
            "pair_friction",
            "geom_rgba",
            "light_diffuse",
        ):
            np.testing.assert_array_equal(
                getattr(env.model, field), getattr(snapshot["model"], field)
            )
        np.testing.assert_array_equal(
            env.model.opt.gravity, snapshot["model"].opt.gravity
        )
        for saved_data, expected_result in expected:
            actual = env.step(ACTION)
            arrays_equal(env.data, saved_data)
            assert actual[1:] == expected_result[1:]
            for key in actual[0]:
                np.testing.assert_array_equal(actual[0][key], expected_result[0][key])
    finally:
        env.close()


def test_contact_solver_and_warmstart_are_preserved_exactly():
    env = LiftSkillEnv(skill_stage=2, render_images=False)
    try:
        observation, info = env.reset(seed=3)
        for _ in range(5):
            observation, _, _, _, info = env.step(ACTION)
        assert all(info["contacts"]) and np.linalg.norm(env.data.qacc_warmstart) > 0.0
        snapshot = capture_skill_state(env, observation, info=info)
        expected = env.step(ACTION)
        expected_data = copy.copy(env.data)
        restore_skill_state(env, snapshot)
        actual = env.step(ACTION)
        arrays_equal(env.data, expected_data)
        assert actual[1:] == expected[1:]
    finally:
        env.close()


def test_monitor_reward_and_episode_histories_survive_terminal_restore(monkeypatch):
    env = Monitor(
        ApproachSkillEnv(skill_stage=2, fixed_height=0.0, render_images=False)
    )
    try:
        observation, info = env.reset(seed=2)
        monkeypatch.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: None)
        for _ in range(8):
            observation, _, _, _, info = env.step(np.zeros(5))
        snapshot = capture_skill_state(env, observation, info=info)
        env.step(np.zeros(5))
        expected = env.step(np.zeros(5))
        assert expected[2] and env.needs_reset
        assert len(env.rewards) == 10 and len(env.episode_returns) == 1
        restored, _ = restore_skill_state(env, snapshot)
        assert len(env.rewards) == 8 and env.total_steps == 8
        assert not env.needs_reset and env.episode_returns == []
        np.testing.assert_array_equal(restored["proprio"], observation["proprio"])
        env.step(np.zeros(5))
        actual = env.step(np.zeros(5))
        assert actual[1:4] == expected[1:4]
        assert actual[4]["episode"]["r"] == expected[4]["episode"]["r"]
        assert actual[4]["episode"]["l"] == expected[4]["episode"]["l"] == 10
    finally:
        env.close()


def approach_arrival(monkeypatch):
    env = ApproachSkillEnv(fixed_height=0.0, render_images=False)
    observation, info = env.reset(seed=3)
    with monkeypatch.context() as context:
        context.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: None)
        for _ in range(3):
            observation, _, done, _, info = env.step(np.zeros(5))
        assert done and info["phase_success"]
    return (env, capture_skill_state(env, observation, info=info))


def test_approach_handoff_preserves_clock_force_velocity_and_both_images(monkeypatch):
    source, snapshot = approach_arrival(monkeypatch)
    target = Monitor(GraspSkillEnv(skill_stage=2, render_images=False))
    try:
        target.reset(seed=90)
        rng_before = copy.deepcopy(target.unwrapped.get_rng_state())
        observation, info = restore_skill_handoff(target, snapshot)
        arrays_equal(target.unwrapped.data, source.data)
        np.testing.assert_array_equal(target.unwrapped.last_action, source.last_action)
        np.testing.assert_array_equal(
            target.unwrapped._previous_frame, source._previous_frame
        )
        assert target.unwrapped.step_count == 3
        assert info["handoff_remaining_steps"] == 497
        assert not info["handoff_clock_reset"] and (not info["handoff_velocity_reset"])
        assert target.unwrapped.get_rng_state() == rng_before
        assert target.unwrapped.skill_stage == 2
        assert target.unwrapped._phase_stable_steps == 0
        assert target.unwrapped._phase_reward_totals == {}
        assert target.rewards == [] and target.total_steps == 0
        assert not info["phase_success"] and info["phase"] == "grasp"
        for key in observation:
            np.testing.assert_array_equal(
                observation[key], snapshot["observation"][key]
            )
        source_result = source.step(ACTION)
        target_result = target.step(ACTION)
        arrays_equal(target.unwrapped.data, source.data)
        for key in source_result[0]:
            np.testing.assert_array_equal(target_result[0][key], source_result[0][key])
    finally:
        source.close()
        target.close()


def test_dynamic_handoff_never_zeros_arrival_velocity_or_action():
    source, target = (
        ApproachSkillEnv(render_images=False),
        GraspSkillEnv(render_images=False),
    )
    try:
        observation, info = source.reset(seed=4)
        for _ in range(7):
            observation, _, _, _, info = source.step(ACTION)
        assert (
            np.linalg.norm(source.data.qvel) > 0.0
            and np.linalg.norm(source.last_action) > 0.0
        )
        snapshot = capture_skill_state(source, observation, info=info)
        target.reset(seed=6)
        restored, _ = restore_skill_handoff(target, snapshot, require_success=False)
        arrays_equal(target.data, source.data)
        np.testing.assert_array_equal(target.last_action, source.last_action)
        np.testing.assert_array_equal(restored["proprio"], observation["proprio"])
        assert target.step_count == 7 and target.data.time == source.data.time
    finally:
        source.close()
        target.close()


def test_grasp_to_lift_preserves_table_contact_and_original_hold_history(monkeypatch):
    source, target = (
        GraspSkillEnv(fixed_height=0.0, render_images=False),
        LiftSkillEnv(skill_stage=2, render_images=False),
    )
    try:
        observation, info = source.reset(seed=4)
        for _ in range(4):
            observation, _, _, _, info = source.step(ACTION)
        snapshot = capture_skill_state(source, observation, info=info)
        target.reset(seed=3)
        restored, info = restore_skill_handoff(target, snapshot, require_success=False)
        arrays_equal(target.data, source.data)
        assert target._hold_steps == source._hold_steps
        assert target._reward_totals == source._reward_totals
        assert info["phase"] == "lift" and info["handoff_remaining_steps"] == 496
        assert info["phase_success"] == info["full_pickup_success"] == False
        np.testing.assert_array_equal(
            restored["proprio"], snapshot["observation"]["proprio"]
        )
    finally:
        source.close()
        target.close()


def test_stale_observation_info_mutated_state_and_wrong_restore_class_are_rejected():
    source, target = (
        ApproachSkillEnv(render_images=False),
        GraspSkillEnv(render_images=False),
    )
    try:
        old, old_info = source.reset(seed=1)
        observation, _, _, _, info = source.step(ACTION)
        with pytest.raises(ValueError, match="stale"):
            capture_skill_state(source, old)
        with pytest.raises(ValueError, match="current environment step"):
            capture_skill_state(source, observation, info=old_info)
        snapshot = capture_skill_state(source, observation, info=info)
        target.reset(seed=2)
        with pytest.raises(ValueError, match="same environment class"):
            restore_skill_state(target, snapshot)
        with pytest.raises(ValueError, match="phase_success evidence"):
            restore_skill_handoff(target, snapshot)
        snapshot["data"].qvel[0] += 0.1
        with pytest.raises(ValueError, match="modified after capture"):
            restore_skill_state(source, snapshot)
    finally:
        source.close()
        target.close()


def test_handoff_rejects_expired_clock_active_target_and_unknown_wrappers(monkeypatch):
    source, snapshot = approach_arrival(monkeypatch)
    target = GraspSkillEnv(render_images=False)
    try:
        target.reset(seed=3)
        target.step(ACTION)
        with pytest.raises(ValueError, match="Reset the receiving"):
            restore_skill_handoff(target, snapshot)
        target.reset(seed=3)
        source.step_count = 500
        observation = source._observation()
        info = {**snapshot["info"], "step": 500}
        expired = capture_skill_state(source, observation, info=info)
        with pytest.raises(ValueError, match="No original episode time"):
            restore_skill_handoff(target, expired)
        wrapper = gym.wrappers.TimeLimit(source, max_episode_steps=500)
        with pytest.raises(ValueError, match="Unsupported stateful wrapper"):
            capture_skill_state(wrapper, snapshot["observation"])
    finally:
        source.close()
        target.close()
