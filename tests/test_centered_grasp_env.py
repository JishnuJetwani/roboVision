from dataclasses import FrozenInstanceError
import json
import mujoco
import numpy as np
import pytest
from robovision.centered_grasp_env import (
    CenteredGraspConfig,
    CenteredGraspEnv,
    original_task_return_ordering,
    potential_increment,
)
from robovision.generalization_env import GeneralizationGraspEnv


def set_pose(env, *, relative_z=0.014, cup_z=0.381, xy=(0.32, 0.0), jaws=0.028):
    env.data.qpos[env._cup_qadr : env._cup_qadr + 3] = [0.32, 0.0, cup_z]
    env.data.qpos[:4] = env.inverse_kinematics([*xy, cup_z + relative_z])
    env.data.qpos[4:6] = jaws
    mujoco.mj_forward(env.model, env.data)


def test_configuration_is_immutable_serializable_and_rejects_invalid_targets():
    config = CenteredGraspConfig()
    with pytest.raises(FrozenInstanceError):
        config.z_high = 0.08
    for kwargs in [
        dict(z_low=0.02),
        dict(xy_radius=0.0),
        dict(open_jaw=0.05),
        dict(broad_z_scale=0.0),
        dict(grasp_budget=-1.0),
        dict(success_bonus=np.nan),
        dict(z_high=".015"),
        dict(z_high=True),
    ]:
        with pytest.raises(ValueError):
            CenteredGraspConfig(**kwargs)
    env = CenteredGraspEnv(render_images=False)
    try:
        spec = json.loads(json.dumps(env.specification()))
        assert spec["reward_config"]["success_bonus"] == 100.0
        assert spec["potential_budget_sum"] == 40.0
        assert spec["return_ordering_bounds"]["minimum_gap"] > 0.0
        assert not spec["original_positive_quality_rewards_applied"]
    finally:
        env.close()
    with pytest.raises(ValueError, match="too small"):
        CenteredGraspEnv(
            reward_config=CenteredGraspConfig(success_bonus=50.0), render_images=False
        )


@pytest.mark.parametrize("gamma", [0.995, 0.999, 1.0])
def test_potential_hover_and_repeated_progress_cannot_farm_discounted_reward(gamma):
    for cycles in [1, 5, 50]:
        sequence = [2.0] + [10.0, 4.0, 10.0, 2.0] * cycles + [0.0]
        rewards = [
            potential_increment(previous, current, gamma, index == len(sequence) - 2)
            for index, (previous, current) in enumerate(zip(sequence, sequence[1:]))
        ]
        assert sum(
            (gamma**index * reward for index, reward in enumerate(rewards))
        ) == pytest.approx(-2.0, abs=1e-12)
    assert potential_increment(10.0, 10.0, gamma, False) <= 0.0
    bounds = original_task_return_ordering(CenteredGraspConfig(), gamma)
    assert bounds["success_lower_bound"] > bounds["failure_upper_bound"]


def test_far_height_potential_has_useful_gradient_and_requires_open_approach():
    env = CenteredGraspEnv(fixed_height=0.14, render_images=False)
    try:
        env.reset(seed=4)
        info = env._info()
        high = env._center_metrics(info)
        set_pose(env, relative_z=0.144, cup_z=0.296, jaws=0.045)
        lower = env._center_metrics(env._info())
        assert lower["approach_score"] - high["approach_score"] > 0.02
        env.data.qpos[4:6] = 0.02
        mujoco.mj_forward(env.model, env.data)
        closed = env._center_metrics(env._info())
        assert closed["approach_score"] < 0.001 * lower["approach_score"]
        set_pose(env, relative_z=0.014, cup_z=0.296, jaws=0.045)
        opened_inside = env._center_metrics(env._info())
        env.data.qpos[4:6] = 0.028
        mujoco.mj_forward(env.model, env.data)
        closed_inside = env._center_metrics(env._info())
        assert opened_inside["approach_score"] == pytest.approx(
            closed_inside["approach_score"]
        )
    finally:
        env.close()


def test_center_band_is_physically_reachable_and_deep_grasp_scores_above_rim_grasp():
    env = CenteredGraspEnv(fixed_height=0.0, render_images=False)
    try:
        _, info = env.reset(seed=3)
        assert info["centered_geometry"]
        assert not env._table_collision()
        for height in [0.296, 0.31, 0.311]:
            set_pose(env, relative_z=height - 0.296, cup_z=0.296, jaws=0.045)
            assert env._center_metrics(env._info())["centered_geometry"]
            assert not env._table_collision()
            for geom in env._pad_ids:
                bottom = (
                    env.data.geom_xpos[geom, 2]
                    - np.abs(env.data.geom_xmat[geom].reshape(3, 3)[2])
                    @ env.model.geom_size[geom]
                )
                assert bottom - env.table_z >= 0.011 - 1e-12
        measured = dict(
            contacts=[True, True], upright=1.0, clearance=0.08, cup_speed=0.0
        )
        set_pose(env)
        deep = env._center_metrics(measured)
        set_pose(env, relative_z=0.06)
        shallow = env._center_metrics(measured)
        assert deep["centered_geometry"] and (not shallow["centered_geometry"])
        assert deep["grasp_score"] > 10 * shallow["grasp_score"]
        assert deep["lift_score"] > 10 * shallow["lift_score"]
    finally:
        env.close()


def test_shallow_original_success_continues_until_centered_success(monkeypatch):
    env = CenteredGraspEnv(render_images=False)
    try:
        env.reset(seed=4)
        set_pose(env, relative_z=0.06)
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
        for _ in range(25):
            _, _, done, truncated, info = env.step(np.zeros(5))
            assert not done and (not truncated)
            assert not info["centered_success"]
            assert info["reward_components"]["success"] == 0.0
        assert info["original_is_success"] and info["original_success_ever"]
        assert info["original_reward_components"]["success"] == 50.0
        assert "episode_reward_components" not in info
        set_pose(env, relative_z=0.014)
        for index in range(25):
            _, reward, done, truncated, info = env.step(np.zeros(5))
            assert done == info["centered_success"] == (index == 24)
            assert not truncated
        assert info["reward_components"]["success"] == 100.0
        assert info["centered_stable_hold_steps"] == 25
        assert all(
            (
                value == 0.0
                for value in info["centered_effective_next_potentials"].values()
            )
        )
        assert info["reason"] == "success" and reward > 50.0
    finally:
        env.close()


def test_original_physics_observations_and_original_reward_remain_identical():
    kwargs = dict(fixed_height=0.1, render_images=False)
    centered, original = (CenteredGraspEnv(**kwargs), GeneralizationGraspEnv(**kwargs))
    try:
        oa, _ = centered.reset(seed=8)
        ob, _ = original.reset(seed=8)
        for key in oa:
            np.testing.assert_array_equal(oa[key], ob[key])
        for action in np.random.default_rng(4).uniform(-0.03, 0.03, (8, 5)):
            a = centered.step(action)
            b = original.step(action)
            np.testing.assert_array_equal(centered.data.qpos, original.data.qpos)
            np.testing.assert_array_equal(centered.data.ctrl, original.data.ctrl)
            for key in a[0]:
                np.testing.assert_array_equal(a[0][key], b[0][key])
            assert a[4]["original_reward"] == b[1]
            assert a[4]["original_reward_components"] == b[4]["reward_components"]
            assert a[4]["original_is_success"] == b[4]["is_success"]
            assert centered.max_steps == 500 and (
                not centered.model.body_gravcomp.any()
            )
    finally:
        centered.close()
        original.close()


def test_centered_hold_requires_consecutive_valid_steps(monkeypatch):
    env = CenteredGraspEnv(render_images=False)
    try:
        env.reset(seed=4)
        set_pose(env)
        baseline = env._info()
        conditions = dict(
            contacts=[True, True], clearance=0.08, upright=1.0, cup_speed=0.01
        )
        monkeypatch.setattr(env, "_info", lambda: {**baseline, **conditions})
        monkeypatch.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: None)
        for _ in range(24):
            assert not env.step(np.zeros(5))[2]
        assert env._centered_hold_steps == 24
        conditions["cup_speed"] = 0.101
        result = env.step(np.zeros(5))
        assert not result[2] and result[4]["centered_stable_hold_steps"] == 0
        conditions["cup_speed"] = 0.01
        for index in range(25):
            result = env.step(np.zeros(5))
            assert result[2] == (index == 24)
    finally:
        env.close()


def test_shallow_original_success_does_not_bypass_centered_deadline(monkeypatch):
    env = CenteredGraspEnv(render_images=False)
    try:
        env.reset(seed=4)
        set_pose(env, relative_z=0.06)
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
        env.step_count = 498
        env._hold_steps = 24
        _, _, done, _, info = env.step(np.zeros(5))
        assert not done and info["original_is_success"] and (not info["is_success"])
        _, _, done, truncated, info = env.step(np.zeros(5))
        assert done and (not truncated) and (info["reason"] == "timeout")
        assert info["original_is_success"] and (not info["centered_success"])
        assert info["reward_components"]["failure"] == -25.0
        assert info["reward_components"]["success"] == 0.0
        assert "episode_reward_components" in info
        assert all(
            (
                value == 0.0
                for value in info["centered_effective_next_potentials"].values()
            )
        )
    finally:
        env.close()


@pytest.mark.parametrize("gamma", [0.995, 0.999])
def test_failure_tail_uses_requested_gamma_and_terminal_potential_is_zero(
    monkeypatch, gamma
):
    env = CenteredGraspEnv(fixed_height=0.1, gamma=gamma, render_images=False)
    try:
        _, reset_info = env.reset(seed=4)
        previous = env._centered_previous_potentials.copy()
        env.step_count = 99
        monkeypatch.setattr(env, "_table_collision", lambda: True)
        monkeypatch.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: None)
        _, _, done, truncated, info = env.step(np.zeros(5))
        assert done and (not truncated) and (info["reason"] == "table_collision")
        components = info["reward_components"]
        assert components["success"] == 0.0 and components["crash"] == -5.0
        assert components["failure"] == pytest.approx(-25.0 * gamma**400)
        assert components["remaining_time"] == pytest.approx(
            -0.01 * gamma * (1.0 - gamma**400) / (1.0 - gamma)
        )
        for key in previous:
            assert components[f"centered_{key}_potential"] == -previous[key]
        assert all(
            (
                value == 0.0
                for value in info["centered_effective_next_potentials"].values()
            )
        )
        assert reset_info["centered_initial_potential"] <= 10.0
    finally:
        env.close()
