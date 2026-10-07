"""Physical resets, full-pickup equivalence, rehearsal and promotion safety."""

from argparse import Namespace
import copy
import json
import mujoco
import numpy as np
import pytest
from robovision.joint_env import JointGraspEnv, VERSION
from robovision.reverse_curriculum import (
    ReverseGraspEnv,
    ReverseGraspCurriculum,
    LEVELS,
    FINAL_LEVEL,
)


def test_all_reset_levels_are_physical_and_release_to_direct_control(monkeypatch):
    env = ReverseGraspEnv(render_images=False, replay_fraction=0)
    try:
        for level in range(len(LEVELS)):
            env.set_curriculum_level(level)
            for seed in (0, 41, 120001, 91003):
                obs, info = env.reset(seed=seed)
                assert info["env_version"] == VERSION
                assert info["curriculum_level"] == level
                assert info["stage"] == (3 if level == FINAL_LEVEL else None)
                assert not info["is_success"] and env._hold_steps == 0
                assert set(obs) == {"image", "proprio"}
                assert env.observation_space.contains(obs)
                assert np.all(env.data.qpos[:6] >= env._joint_low)
                assert np.all(env.data.qpos[:6] <= env._joint_high)
                assert not env._table_collision()
                np.testing.assert_array_equal(env.data.qvel, 0)
                np.testing.assert_array_equal(env.data.ctrl, 0)
                np.testing.assert_array_equal(env.data.qfrc_applied, 0)
                np.testing.assert_array_equal(env.data.xfrc_applied, 0)
                assert not env.model.body_gravcomp.any()
                np.testing.assert_allclose(env.model.opt.gravity, [0, 0, -9.81])
                for _ in range(8):
                    _, reward, done, _, _ = env.step(np.zeros(5))
                    assert np.isfinite(reward)
                    if done:
                        break
        env.set_curriculum_level(0)
        env.reset(seed=9)

        def forbidden(*args):
            raise AssertionError("No reset pose controller during an episode")

        monkeypatch.setattr(env, "inverse_kinematics", forbidden)
        env.step([0.5, -0.5, 0.25, -0.25, -1])
        np.testing.assert_allclose(
            env.data.actuator_force, [20, -27.5, 11.25, -5, -10, -10]
        )
    finally:
        env.close()


def test_lifted_start_has_real_contact_but_zero_force_cannot_win():
    env = ReverseGraspEnv(render_images=False, replay_fraction=0)
    try:
        _, info = env.reset(seed=1)
        assert all(info["contacts"]) and info["clearance"] > 0.06
        assert env.model.neq == 0
        initial = env.cup_position.copy()
        for _ in range(60):
            _, _, done, _, info = env.step(np.zeros(5))
            assert not info["is_success"]
            if done:
                break
        assert info["reason"] == "grip_lost"
        assert env.step_count <= 3
        assert env.cup_position[2] < initial[2]
        env.reset(seed=1)
        jacobian = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, env.data, jacobian, None, env._grasp_sid)
        force = env.data.qfrc_bias[:4] + jacobian[:, :4].T @ np.array(
            [0, 0, 0.08 * 9.81]
        )
        action = np.r_[force / env.torque_limits, -0.2]
        for index in range(env.hold_steps):
            _, _, done, _, info = env.step(action)
            assert done == (index == env.hold_steps - 1)
        assert info["is_success"]
    finally:
        env.close()


@pytest.mark.parametrize("render_images", [False, True])
def test_final_level_exactly_matches_hard_reset_and_trajectory(render_images):
    env = ReverseGraspEnv(
        curriculum_level=FINAL_LEVEL, replay_fraction=0, render_images=render_images
    )
    hard = JointGraspEnv(stage=3, render_images=render_images)
    try:
        for seed in (1, 120009, 1100000):
            actual, _ = env.reset(seed=seed)
            expected, _ = hard.reset(seed=seed)
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])
            for name in ("qpos", "qvel", "ctrl"):
                np.testing.assert_array_equal(
                    getattr(env.data, name), getattr(hard.data, name)
                )
            for name in (
                "body_mass",
                "body_inertia",
                "pair_friction",
                "geom_rgba",
                "light_diffuse",
            ):
                np.testing.assert_array_equal(
                    getattr(env.model, name), getattr(hard.model, name)
                )
            for action in np.random.default_rng(seed).uniform(-0.1, 0.1, (8, 5)):
                a, b = (env.step(action), hard.step(action))
                assert a[1:4] == b[1:4]
                assert a[4]["reward_components"] == b[4]["reward_components"]
                for key in a[0]:
                    np.testing.assert_array_equal(a[0][key], b[0][key])
                if a[2]:
                    break
        env.set_curriculum_level(0)
        env.reset(seed=5)
        env.set_curriculum_level(FINAL_LEVEL)
        a, _ = env.reset(seed=1)
        b, _ = hard.reset(seed=1)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    finally:
        env.close()
        hard.close()


def test_rehearsal_is_seeded_reset_only_and_rng_restorable():
    env = ReverseGraspEnv(curriculum_level=8, replay_fraction=0.5, render_images=False)
    try:
        _, first = env.reset(seed=12)
        pose = env.data.qpos.copy()
        env.set_curriculum_level(10)
        np.testing.assert_array_equal(pose, env.data.qpos)
        assert env.step(np.zeros(5))[4]["curriculum_level"] == first["curriculum_level"]
        env.reset(seed=17)
        state = copy.deepcopy(env.get_rng_state())
        outcomes = [env.reset()[1]["curriculum_level"] for _ in range(40)]
        assert 10 in outcomes and any((level < 10 for level in outcomes))
        assert all((0 <= level <= 10 for level in outcomes))
        env.set_rng_state(state)
        assert [env.reset()[1]["curriculum_level"] for _ in range(40)] == outcomes
        for invalid in (-0.1, 1, float("nan")):
            with pytest.raises(ValueError, match="Replay"):
                ReverseGraspEnv(replay_fraction=invalid)
    finally:
        env.close()


def test_promotion_uses_current_level_only_and_survives_resume():
    curriculum = ReverseGraspCurriculum(
        level=2, minimum_episodes=5, window=4, threshold=0.75
    )
    for _ in range(100):
        assert not curriculum.observe(dict(curriculum_level=0, is_success=True), 1)
    assert curriculum.episodes == 0
    for success in (False, False, True, True):
        assert not curriculum.observe(dict(curriculum_level=2, is_success=success), 2)
    restored = ReverseGraspCurriculum(
        state=json.loads(json.dumps(curriculum.state_dict()))
    )
    assert restored.state_dict() == curriculum.state_dict()
    assert restored.observe(dict(curriculum_level=2, is_success=True), 10)
    assert restored.level == 3 and restored.episodes == 0 and (not restored.recent)
    assert not restored.observe(dict(curriculum_level=2, is_success=True), 11)
    final = ReverseGraspCurriculum(level=FINAL_LEVEL, minimum_episodes=1, window=1)
    assert not final.observe(dict(curriculum_level=FINAL_LEVEL, is_success=True), 12)


def test_reverse_training_rejects_imitation_initialization():
    from robovision.train_joint import train

    for field in ("initial_model", "demo_anchor", "reference_model"):
        args = Namespace(curriculum=True, stage=0, **{field: "not-used.zip"})
        with pytest.raises(ValueError, match="from scratch"):
            train(args)


def _constant_hold_action(env):
    jacobian = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jacobian, None, env._grasp_sid)
    force = env.data.qfrc_bias[:4] + jacobian[:, :4].T @ np.array([0, 0, 0.08 * 9.81])
    return np.r_[force / env.torque_limits, -0.2]


@pytest.mark.parametrize("level,target", list(enumerate((3, 5, 8, 12, 18, 25))))
def test_hold_duration_milestones_require_real_consecutive_control(level, target):
    env = ReverseGraspEnv(
        curriculum_level=level, replay_fraction=0, render_images=False
    )
    try:
        env.reset(seed=1)
        action = _constant_hold_action(env)
        initial_potential = env._hold_potential
        shaping_return = 0.0
        for step in range(target):
            _, reward, done, _, info = env.step(action)
            assert done == (step == target - 1)
            assert reward == pytest.approx(sum(info["reward_components"].values()))
            shaping_return += (
                env.gamma**step * info["reward_components"]["hold_potential"]
            )
        assert info["is_success"] and info["max_stable_hold_steps"] == target
        assert info["hold_target_steps"] == target
        assert shaping_return == pytest.approx(-initial_potential)
        assert sum(info["episode_reward_components"].values()) > 30.0
    finally:
        env.close()


@pytest.mark.parametrize("hold_before_release", [0, 5, 15])
def test_lost_grip_resets_promptly_without_erasing_failure_bill(hold_before_release):
    env = ReverseGraspEnv(curriculum_level=5, replay_fraction=0, render_images=False)
    try:
        env.reset(seed=1)
        hold = _constant_hold_action(env)
        initial_potential = env._hold_potential
        shaping_return = 0.0
        for step in range(100):
            _, _, done, _, info = env.step(
                hold if step < hold_before_release else np.zeros(5)
            )
            shaping_return += (
                env.gamma**step * info["reward_components"]["hold_potential"]
            )
            if done:
                break
        assert info["reason"] == "grip_lost"
        assert info["max_stable_hold_steps"] == hold_before_release
        assert env.step_count <= hold_before_release + 4
        assert shaping_return == pytest.approx(-initial_potential)
        costs = info["episode_reward_components"]
        assert (
            costs["remaining_time"] < 0
            and costs["failure"] < 0
            and (costs["crash"] == -5.0)
        )
        discounted_failure = env.gamma**step * costs["failure"]
        assert discounted_failure == pytest.approx(
            -25 * env.gamma ** (env.max_steps - 1)
        )
    finally:
        env.close()


def test_potential_feedback_rewards_preservation_and_is_smooth_near_grasp():
    env = ReverseGraspEnv(curriculum_level=5, replay_fraction=0, render_images=False)
    try:
        env.reset(seed=1)
        _, good, _, _, kept = env.step(_constant_hold_action(env))
        env.reset(seed=1)
        _, bad, _, _, lost = env.step(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
        assert good > bad + 0.1
        assert (
            kept["reward_components"]["hold_potential"]
            > lost["reward_components"]["hold_potential"] + 0.1
        )
        info = env._info()
        phi = env._potential(info)
        assert env._potential(dict(info, cup_speed=info["cup_speed"] + 0.001)) <= phi
        assert env._potential(dict(info, clearance=0)) == 0
        env.set_curriculum_level(8)
        env.reset(seed=1)
        assert not LEVELS[8].hold_bootstrap
        for _ in range(3):
            _, _, _, _, info = env.step(np.zeros(5))
            assert info["reason"] != "grip_lost"
            assert "hold_potential" not in info["reward_components"]
    finally:
        env.close()
