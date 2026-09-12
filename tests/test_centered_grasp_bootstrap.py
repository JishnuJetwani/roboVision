"""Bootstrap changes starting geometry, never the torque dynamics or goal band."""

import json
import mujoco
import numpy as np
import pytest
from robovision.centered_grasp_bootstrap import (
    BOOTSTRAP_RESETS,
    CenteredGraspBootstrapEnv,
    MovingGraspBootstrapEnv,
    FullPickupBootstrapEnv,
    centered_contact_quality,
)
from robovision.hierarchical_skills import GraspSkillEnv
from robovision.joint_env import JointGraspEnv


@pytest.mark.parametrize("stage", range(4))
def test_coarse_resets_are_nonpenetrating_centered_reproducible_and_unassisted(stage):
    env = CenteredGraspBootstrapEnv(skill_stage=stage, render_images=False)
    try:
        spec = json.loads(json.dumps(env.specification()))
        assert spec["target_relative_z"] == 0.0075 and (not spec["initial_grasp"])
        assert spec["control"] == JointGraspEnv.control_spec
        lesson = BOOTSTRAP_RESETS[stage]
        heights, offsets = ([], [])
        for seed in range(40):
            obs, info = env.reset(seed=seed)
            heights.append(info["reset_relative_z"])
            offsets.append(info["cup_offset"])
            assert lesson.relative_z_low <= heights[-1] <= lesson.relative_z_high
            assert max((abs(v) for v in info["cup_offset"])) <= 0.0
            assert not any(info["contacts"]) and (not info["phase_success"])
            assert not env._table_collision()
            np.testing.assert_allclose(env.data.qpos[4:6], lesson.opening)
            np.testing.assert_array_equal(env.data.ctrl, 0.0)
            np.testing.assert_array_equal(env.last_action, 0.0)
            assert env.data.time == 0.0 and env.step_count == 0
            assert obs["proprio"].shape == (18,) and obs["image"].shape == (6, 96, 96)
            assert info["reset_height"] == pytest.approx(env.grasp_position[2] - 0.31)
        assert max(heights) - min(heights) > 0.8 * (
            lesson.relative_z_high - lesson.relative_z_low
        )
        np.testing.assert_array_equal(offsets, 0.0)
        first, info = env.reset(seed=9)
        second, repeated = env.reset(seed=9)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        assert info == repeated
    finally:
        env.close()


def test_midpoint_pose_has_safe_table_margin_and_goal_remains_centered_actual_grasp():
    env = CenteredGraspBootstrapEnv(fixed_relative_z=0.0075, render_images=False)
    try:
        _, info = env.reset(seed=1)
        assert info["phase_metrics"]["z_error_m"] == pytest.approx(0.0, abs=1e-12)
        for gid in env._pad_ids:
            bottom = (
                env.data.geom_xpos[gid, 2]
                - np.abs(env.data.geom_xmat[gid].reshape(3, 3)[2])
                @ env.model.geom_size[gid]
            )
            assert bottom - env.table_z == pytest.approx(0.0185)
        good = {
            **info["phase_metrics"],
            "contacts": [True, True],
            "contact_forces_n": [0.6, 0.6],
        }
        assert env._goal(good)
        assert not env._goal(info["phase_metrics"])
        assert not env._goal({**good, "relative_z_m": 0.016})
        assert not env._goal({**good, "clearance_m": 0.006})
    finally:
        env.close()


def test_original_fixed_scene_overrides_preserve_comparable_physics_and_pixels():
    kwargs = dict(fixed_height=0.01, render_images=False, gamma=0.999)
    bootstrap = CenteredGraspBootstrapEnv(fixed_opening=0.045, **kwargs)
    original = GraspSkillEnv(**kwargs)
    try:
        a, info = bootstrap.reset(seed=5)
        b, _ = original.reset(seed=5)
        assert info["reset_relative_z"] == pytest.approx(0.024)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        for action in np.random.default_rng(3).uniform(-0.05, 0.05, (5, 5)):
            a, b = (bootstrap.step(action), original.step(action))
            np.testing.assert_array_equal(bootstrap.data.qpos, original.data.qpos)
            np.testing.assert_array_equal(bootstrap.data.qvel, original.data.qvel)
            np.testing.assert_array_equal(bootstrap.data.ctrl, original.data.ctrl)
            for key in a[0]:
                np.testing.assert_array_equal(a[0][key], b[0][key])
            assert a[4]["original_reward"] == b[4]["original_reward"]
    finally:
        bootstrap.close()
        original.close()


def test_stage_changes_only_future_resets_and_closed_reset_is_not_free_success():
    env = CenteredGraspBootstrapEnv(fixed_relative_z=0.0075, render_images=False)
    try:
        env.reset(seed=2)
        env.set_skill_stage(2)
        assert env.episode_skill_stage == 0 and env._lesson == env.lessons[0]
        assert env.data.qpos[4] == 0.029
        success = False
        for _ in range(20):
            _, _, done, _, info = env.step(np.zeros(5))
            success |= info["phase_success"]
            if done:
                break
        assert not success
        _, info = env.reset(seed=2)
        assert env.episode_skill_stage == 2 and info["reset_opening"] == 0.045
    finally:
        env.close()


def test_invalid_reset_options_are_rejected():
    with pytest.raises(ValueError):
        CenteredGraspBootstrapEnv(
            fixed_relative_z=0.01, fixed_height=0.03, render_images=False
        )
    with pytest.raises(TypeError):
        CenteredGraspBootstrapEnv(fixed_cup_offset=(0.02, 0.0), render_images=False)


def test_moving_secure_centered_grasp_allowed_but_shallow_unilateral_and_slip_rejected():
    moving = MovingGraspBootstrapEnv(render_images=False)
    stationary = CenteredGraspBootstrapEnv(render_images=False)
    try:
        _, info = moving.reset(seed=2)
        stationary.reset(seed=2)
        good = {
            **info["phase_metrics"],
            "xy_error_m": 0.007,
            "relative_z_m": 0.011,
            "downward": np.cos(np.deg2rad(9.0)),
            "contacts": [True, True],
            "contact_forces_n": [0.6, 0.7],
            "relative_speed_m_s": 0.01,
            "cup_speed_m_s": 0.2,
            "clearance_m": 0.04,
            "cup_upright": 0.99,
        }
        assert moving._goal(good)
        assert not stationary._goal(good)
        for stage in range(4):
            moving._lesson = moving.lessons[stage]
            assert moving._goal(good)
            for update in [
                dict(relative_z_m=0.016),
                dict(relative_z_m=-0.001),
                dict(xy_error_m=0.016),
                dict(downward=0.9),
                dict(contacts=[True, False]),
                dict(contact_forces_n=[0.49, 0.8]),
                dict(relative_speed_m_s=0.11),
                dict(cup_upright=0.9),
            ]:
                assert not moving._goal({**good, **update})
        spec = json.loads(json.dumps(moving.specification()))
        assert spec["version"] != stationary.specification()["version"]
        assert spec["minimum_phase_contact_force_n"] == 0.5
        assert spec["maximum_grasp_clearance_m"] is None
        assert not spec["phase_success_is_full_pickup"]
    finally:
        moving.close()
        stationary.close()


def test_moving_bilateral_reward_is_center_gated_and_has_no_lift_penalty():
    env = MovingGraspBootstrapEnv(render_images=False)
    try:
        _, info = env.reset(seed=2)
        centered = {
            **info["phase_metrics"],
            "xy_error_m": 0.0,
            "relative_z_m": 0.0075,
            "contact_forces_n": [1.0, 1.0],
            "contacts": [True, True],
        }
        shallow = {**centered, "relative_z_m": 0.045, "z_error_m": 0.0375}
        assert centered_contact_quality(centered) == 1.0
        assert centered_contact_quality(shallow) == pytest.approx(np.exp(-2.0))
        assert env._costs(centered)["bilateral"] == 0.0
        assert env._costs(shallow)["bilateral"] > 0.6
        assert "premature_lift" not in env._costs(centered)
        assert env._costs(centered) == env._costs(
            {**centered, "clearance_m": 0.2, "cup_speed_m_s": 0.3}
        )
        assert (
            env._costs({**centered, "contact_forces_n": [0.1, 0.1]})["bilateral"] > 0.0
        )
    finally:
        env.close()


def test_moving_phase_terminates_normally_without_claiming_full_pickup(monkeypatch):
    env = MovingGraspBootstrapEnv(render_images=False)
    try:
        _, info = env.reset(seed=2)
        moving = {
            **info["phase_metrics"],
            "contacts": [True, True],
            "contact_forces_n": [0.6, 0.6],
            "relative_z_m": 0.011,
            "xy_error_m": 0.005,
            "relative_speed_m_s": 0.01,
            "cup_speed_m_s": 0.2,
            "clearance_m": 0.04,
            "cup_upright": 1.0,
        }
        monkeypatch.setattr(env, "_phase_metrics", lambda _: moving)
        monkeypatch.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: None)
        for index in range(5):
            _, _, done, _, info = env.step(np.zeros(5))
            assert done == (index == 4)
            assert not info["centered_full_pickup_success"]
            assert not info["full_pickup_success"]
        assert info["phase_success"] and info["phase_stable_steps"] == 5
    finally:
        env.close()


def test_full_pickup_diagnostic_uses_original_termination_reward_and_separate_centered_counter(
    monkeypatch,
):
    env = FullPickupBootstrapEnv(fixed_relative_z=0.0075, render_images=False)
    try:
        env.reset(seed=3)
        env.data.qpos[env._cup_qadr : env._cup_qadr + 3] = [0.32, 0.0, 0.38]
        env.data.qpos[:4] = env.inverse_kinematics([0.32, 0.0, 0.3875])
        env.data.qpos[4:6] = 0.028
        mujoco.mj_forward(env.model, env.data)
        measured = env._info()
        assert all(measured["contacts"]) and measured["clearance"] > 0.06
        assert min(env._contact_forces()) < 0.5
        monkeypatch.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: None)
        for index in range(25):
            _, _, done, _, info = env.step(np.zeros(5))
            assert done == info["full_pickup_success"] == (index == 24)
            assert info["centered_full_pickup_stable_steps"] == index + 1
            assert info["centered_full_pickup_success"] == (index == 24)
            assert "phase_bonus" not in info["reward_components"]
        assert info["reward_components"]["success"] == 50.0
        assert info["centered_full_pickup_success_ever"]
        assert info["reason"] == "success"
        spec = json.loads(json.dumps(env.specification()))
        assert spec["evaluation_only"] and spec["original_success_terminates"]
        assert spec["phase_success_bonus"] is None
    finally:
        env.close()


def test_centered_full_pickup_counter_resets_when_geometry_or_absolute_speed_breaks():
    env = FullPickupBootstrapEnv(render_images=False)
    try:
        _, reset = env.reset(seed=3)
        info = dict(
            contacts=[True, True],
            clearance=0.07,
            upright=1.0,
            cup_speed=0.01,
            reason="",
        )
        metrics = {
            **reset["phase_metrics"],
            "xy_error_m": 0.01,
            "relative_z_m": 0.01,
            "downward": 1.0,
        }
        for _ in range(24):
            env._update_centered_full_pickup(info, metrics)
        assert env._centered_full_pickup_steps == 24
        env._update_centered_full_pickup({**info, "cup_speed": 0.1}, metrics)
        assert env._centered_full_pickup_steps == 0
        for _ in range(24):
            env._update_centered_full_pickup(info, metrics)
        env._update_centered_full_pickup(info, {**metrics, "relative_z_m": 0.016})
        assert env._centered_full_pickup_steps == 0
        for _ in range(25):
            env._update_centered_full_pickup(info, metrics)
        assert env._centered_full_pickup_info({})["centered_full_pickup_success"]
        env._update_centered_full_pickup({**info, "reason": "table_collision"}, metrics)
        report = env._centered_full_pickup_info({})
        assert (
            not report["centered_full_pickup_success"]
            and report["centered_full_pickup_success_ever"]
        )
    finally:
        env.close()


def test_full_pickup_diagnostic_reset_physics_matches_existing_bootstrap():
    kwargs = dict(fixed_opening=0.033, fixed_relative_z=0.0075, render_images=False)
    full, phase = (
        FullPickupBootstrapEnv(**kwargs),
        CenteredGraspBootstrapEnv(**kwargs),
    )
    try:
        a, _ = full.reset(seed=9)
        b, _ = phase.reset(seed=9)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        for action in np.random.default_rng(3).uniform(-0.03, 0.03, (6, 5)):
            observed = full.step(action)
            expected = JointGraspEnv.step(phase, action)
            assert observed[1:4] == expected[1:4]
            assert observed[4]["reward_components"] == expected[4]["reward_components"]
            np.testing.assert_array_equal(full.data.qpos, phase.data.qpos)
            for key in observed[0]:
                np.testing.assert_array_equal(observed[0][key], expected[0][key])
    finally:
        full.close()
        phase.close()
