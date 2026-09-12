"""Independent phase goals retain original physics, observations and force control."""

import json
import mujoco
import numpy as np
import pytest
from robovision.generalization_env import GeneralizationGraspEnv
from robovision.hierarchical_skills import (
    ApproachSkillEnv,
    GraspSkillEnv,
    LiftSkillEnv,
    phase_reward,
)
from robovision.joint_env import JointGraspEnv
from robovision.reverse_curriculum import ReverseGraspEnv


@pytest.mark.parametrize("cls", [ApproachSkillEnv, GraspSkillEnv, LiftSkillEnv])
def test_specs_and_actor_interface_are_serializable_and_original(cls):
    env = cls(render_images=False)
    try:
        obs, info = env.reset(seed=2)
        assert obs["image"].shape == (6, 96, 96)
        assert obs["proprio"].shape == (18,)
        assert env.action_space.shape == (5,)
        assert env.max_steps == 500 and env.control_dt == 0.02
        assert not env.model.body_gravcomp.any()
        spec = json.loads(json.dumps(env.specification()))
        assert spec["control"] == JointGraspEnv.control_spec
        assert not spec["runtime_controller"] and (not spec["action_demonstrations"])
        assert not info["phase_success"] and (not info["full_pickup_success"])
        for stage in [-1, 100, True, 1.5]:
            with pytest.raises(ValueError):
                env.set_skill_stage(stage)
    finally:
        env.close()


@pytest.mark.parametrize("cls", [ApproachSkillEnv, GraspSkillEnv])
def test_original_physics_and_observations_under_identical_force_commands(cls):
    kwargs = dict(fixed_height=0.04, render_images=False, gamma=0.999)
    phase, original = (cls(**kwargs), GeneralizationGraspEnv(**kwargs))
    try:
        obs_phase, _ = phase.reset(seed=5)
        obs_original, _ = original.reset(seed=5)
        for key in obs_phase:
            np.testing.assert_array_equal(obs_phase[key], obs_original[key])
        for action in np.random.default_rng(4).uniform(-0.1, 0.1, (12, 5)):
            a, b = (phase.step(action), original.step(action))
            np.testing.assert_array_equal(phase.data.qpos, original.data.qpos)
            np.testing.assert_array_equal(phase.data.qvel, original.data.qvel)
            np.testing.assert_array_equal(phase.data.ctrl, original.data.ctrl)
            for key in a[0]:
                np.testing.assert_array_equal(a[0][key], b[0][key])
            assert a[4]["original_reward"] == b[1]
            assert a[4]["full_pickup_success"] == b[4]["is_success"]
    finally:
        phase.close()
        original.close()


def test_broad_approach_reset_distribution_and_fixed_cup_position():
    env = ApproachSkillEnv(render_images=False)
    try:
        heights, offsets = ([], [])
        for seed in range(60):
            _, info = env.reset(seed=seed)
            heights.append(info["reset_height"])
            offsets.append(info["cup_offset"])
            np.testing.assert_allclose(env.grasp_position[:2], [0.32, 0.0], atol=1e-12)
            assert 0.025 <= info["reset_height"] <= 0.14
            assert not env._table_collision()
        assert min(heights) < 0.04 and max(heights) > 0.13
        np.testing.assert_array_equal(offsets, 0.0)
        env.set_skill_stage(2)
        assert env._lesson == env.lessons[0]
        _, info = env.reset(seed=4)
        assert info["skill_stage"] == 2 and env._lesson == env.lessons[2]
        assert env.height_bands == ((0.025, 0.14, 1.0),)
    finally:
        env.close()
    first = ApproachSkillEnv(fixed_height=0.1, render_images=False)
    second = ApproachSkillEnv(fixed_height=0.1, render_images=False)
    try:
        a, _ = first.reset(seed=2)
        b, _ = second.reset(seed=2)
        np.testing.assert_array_equal(a["proprio"], b["proprio"])
    finally:
        first.close()
        second.close()


def test_explicit_reset_support_survives_stage_change():
    env = ApproachSkillEnv(height_bands=((0.07, 0.08, 1.0),), render_images=False)
    try:
        env.set_skill_stage(2)
        assert env.height_bands == ((0.07, 0.08, 1.0),)
    finally:
        env.close()


def test_approach_goal_requires_open_centered_still_downward_and_no_contact(
    monkeypatch,
):
    env = ApproachSkillEnv(skill_stage=2, fixed_height=0.0, render_images=False)
    try:
        _, info = env.reset(seed=2)
        good = info["phase_metrics"]
        assert env._goal(good)
        for update in [
            dict(xy_error_m=0.02),
            dict(z_error_m=0.015),
            dict(minimum_jaw_m=0.035),
            dict(relative_speed_m_s=0.1),
            dict(downward=0.9),
            dict(contacts=[True, False]),
        ]:
            assert not env._goal({**good, **update})
        monkeypatch.setattr(env, "_phase_metrics", lambda _: good)
        monkeypatch.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: None)
        for step in range(10):
            _, _, done, _, info = env.step(np.zeros(5))
            assert done == info["phase_success"] == (step == 9)
            assert not info["full_pickup_success"]
        assert info["reason"] == "approach_success"
        assert info["reward_components"]["phase_bonus"] == 100.0
    finally:
        env.close()


def test_additive_reward_has_descent_gradient_even_with_closed_gripper():
    env = ApproachSkillEnv(fixed_height=0.14, render_images=False)
    try:
        _, info = env.reset(seed=2)
        high = info["phase_metrics"]
        closed = {**high, "openness_score": 0.0}
        lower = {**closed, "z_error_m": 0.13, "z_score": np.exp(-0.13 / 0.1)}
        assert (
            sum(env._costs(closed).values()) - sum(env._costs(lower).values()) > 0.014
        )
        assert env._costs(closed)["xy"] == env._costs(high)["xy"]
        assert env._costs(closed)["z"] == env._costs(high)["z"]
        assert env._costs(closed)["openness"] > env._costs(high)["openness"]
        reward = phase_reward(
            env._costs(high),
            np.zeros(5),
            np.zeros(5),
            dt=0.02,
            gamma=0.999,
            remaining_steps=499,
            success=False,
            failure=False,
        )
        assert all((value <= 0.0 for value in reward.values()))
        assert sum(reward.values()) < 0.0
    finally:
        env.close()


@pytest.mark.parametrize("gamma", [0.995, 0.999, 1.0])
def test_failure_tail_cannot_make_crashing_escape_future_worst_costs(gamma):
    early = phase_reward(
        dict(time=2.0),
        np.ones(5),
        -np.ones(5),
        dt=0.02,
        gamma=gamma,
        remaining_steps=99,
        success=False,
        failure=True,
    )
    continued = (
        -0.02 * 2.06 * sum((gamma**step for step in range(100))) - 25.0 * gamma**99
    )
    assert sum(early.values()) == pytest.approx(continued - 5.0)
    geometric = sum((gamma**step for step in range(500)))
    assert (
        100.0 * gamma**499 - 0.02 * 2.06 * geometric
        > -0.02 * 0.1 * geometric - 25.0 * gamma**499
    )


def test_grasp_requires_actual_centered_bilateral_contact_before_lift():
    env = GraspSkillEnv(fixed_height=0.0, render_images=False)
    try:
        _, info = env.reset(seed=3)
        metrics = info["phase_metrics"]
        assert not env._goal(metrics)
        good = {**metrics, "contacts": [True, True], "contact_forces_n": [0.5, 0.5]}
        assert env._goal(good)
        for stage in range(len(env.lessons)):
            env._lesson = env.lessons[stage]
            assert env._goal(good)
            for update in [
                dict(contacts=[True, False]),
                dict(contact_forces_n=[0.01, 1.0]),
                dict(relative_z_m=0.016),
                dict(clearance_m=0.006),
                dict(relative_speed_m_s=0.2),
                dict(cup_speed_m_s=0.2),
            ]:
                assert not env._goal({**good, **update})
    finally:
        env.close()


def test_grasp_stability_is_consecutive_and_separate_from_full_pickup(monkeypatch):
    env = GraspSkillEnv(skill_stage=2, fixed_height=0.0, render_images=False)
    try:
        _, info = env.reset(seed=1)
        measured = {
            **info["phase_metrics"],
            "contacts": [True, True],
            "contact_forces_n": [0.6, 0.6],
        }
        monkeypatch.setattr(env, "_phase_metrics", lambda _: measured)
        monkeypatch.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: None)
        for _ in range(9):
            assert not env.step(np.zeros(5))[2]
        measured["contact_forces_n"] = [0.1, 0.6]
        result = env.step(np.zeros(5))
        assert not result[2] and result[4]["phase_stable_steps"] == 0
        measured["contact_forces_n"] = [0.6, 0.6]
        for step in range(10):
            result = env.step(np.zeros(5))
            assert result[2] == result[4]["phase_success"] == (step == 9)
            assert not result[4]["full_pickup_success"]
    finally:
        env.close()


def test_original_pickup_does_not_replace_open_approach_goal_and_deadline(monkeypatch):
    env = ApproachSkillEnv(fixed_height=0.1, render_images=False)
    try:
        env.reset(seed=4)
        baseline = env._info()
        monkeypatch.setattr(
            env,
            "_info",
            lambda: {
                **baseline,
                "contacts": [True, True],
                "clearance": 0.08,
                "upright": 1.0,
                "cup_speed": 0.01,
            },
        )
        monkeypatch.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: None)
        env._hold_steps = 24
        env.step_count = 498
        result = env.step(np.zeros(5))
        assert (
            result[4]["full_pickup_success"] and result[4]["full_pickup_success_ever"]
        )
        assert not result[2] and (not result[4]["phase_success"])
        result = env.step(np.zeros(5))
        assert result[2] and result[4]["reason"] == "timeout" and (not result[3])
        assert (
            not result[4]["phase_success"]
            and result[4]["reward_components"]["phase_bonus"] == 0.0
        )
    finally:
        env.close()


def test_grasp_arrival_velocity_is_reset_only_reproducible_and_physically_valid():
    env = GraspSkillEnv(skill_stage=3, render_images=False)
    try:
        first, info = env.reset(seed=10)
        target = info["reset_arrival_velocity_m_s"]
        np.testing.assert_allclose(
            info["phase_metrics"]["relative_velocity_m_s"], target, atol=1e-12
        )
        assert np.linalg.norm(target) > 0.0
        assert env.data.ctrl.tolist() == [0.0] * 6
        assert env.last_action.tolist() == [0.0] * 5
        assert env.data.time == 0.0 and env.step_count == 0
        second, repeated = env.reset(seed=10)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        assert repeated["reset_arrival_velocity_m_s"] == target
        for seed in range(30):
            env.reset(seed=seed)
            assert not env._table_collision()
            for contact in env.data.contact:
                pair = {int(contact.geom1), int(contact.geom2)}
                if pair & env._cup_geom_set and pair & set().union(*env._finger_geoms):
                    assert contact.dist >= -1e-08
        action = np.array([0.1, -0.1, 0.2, -0.2, -0.3])
        env.step(action)
        np.testing.assert_allclose(
            env.data.ctrl[:4], env.torque_limits * action[:4], rtol=1e-07
        )
        np.testing.assert_allclose(env.data.ctrl[4:6], 10.0 * action[-1], rtol=1e-07)
    finally:
        env.close()


@pytest.mark.parametrize("stage,clearance", [(0, 0.035), (1, 0.02), (2, 0.001)])
def test_lift_starts_below_success_height_and_keeps_original_reward(
    stage, clearance, monkeypatch
):
    env = LiftSkillEnv(skill_stage=stage, render_images=False)
    original = ReverseGraspEnv(
        curriculum_level=(9, 10, 11)[stage],
        replay_fraction=0.0,
        gamma=0.999,
        render_images=False,
    )
    try:
        obs, info = env.reset(seed=8)
        other, _ = original.reset(seed=8)
        assert info["clearance"] == pytest.approx(clearance)
        assert info["clearance"] < 0.06
        for key in obs:
            np.testing.assert_array_equal(obs[key], other[key])
        for _ in range(3):
            a = env.step(np.zeros(5))
            b = JointGraspEnv.step(original, np.zeros(5))
            assert a[1] == b[1] and a[2] == b[2]
            assert a[4]["reward_components"] == b[4]["reward_components"]
            assert "hold_potential" not in a[4]["reward_components"]
            np.testing.assert_array_equal(env.data.qpos, original.data.qpos)
        env.reset(seed=8)
        monkeypatch.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: None)
        for _ in range(25):
            _, _, _, _, info = env.step(np.zeros(5))
            assert not info["phase_success"] and (not info["full_pickup_success"])
    finally:
        env.close()
        original.close()
