"""Centered approach endpoint changes must preserve pure force-controlled physics."""

import json
import mujoco
import numpy as np
import pytest
from robovision.centered_approach_skill import CenteredApproachSkillEnv
from robovision.hierarchical_skills import ApproachSkillEnv


def install_static_pose(env, *, z=0.0075, xy=(0.0, 0.0), jaw=0.045, tilt=0.0):
    """Reset-only kinematics for checking whether the declared target is feasible."""
    target = env.cup_position + [xy[0], xy[1], z]
    q = env.inverse_kinematics(target)
    pitch = np.pi / 2 + np.deg2rad(tilt)
    for _ in range(15):
        env.data.qpos[:4] = q
        env.data.qpos[4:6] = jaw
        mujoco.mj_forward(env.model, env.data)
        error = np.r_[target - env.grasp_position, pitch - np.sum(q[1:])]
        if np.max(np.abs(error)) < 1e-11:
            break
        jacobian = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, env.data, jacobian, None, env._grasp_sid)
        q += np.linalg.solve(np.vstack([jacobian[:, :4], [0.0, 1.0, 1.0, 1.0]]), error)
    np.testing.assert_allclose(env.grasp_position, target, atol=1e-10)
    return env._phase_metrics(env._info())


@pytest.mark.parametrize("stage", [0, 1])
def test_both_lessons_retain_height_distribution_and_fixed_xy(stage):
    env = CenteredApproachSkillEnv(skill_stage=stage, render_images=False)
    try:
        spec = json.loads(json.dumps(env.specification()))
        assert spec["target_relative_z"] == 0.0075
        assert spec["height_bands"] == [[0.025, 0.14, 1.0]]
        assert spec["nominal_hand_xy"] == [0.32, 0.0]
        assert spec["control"] == ApproachSkillEnv.control_spec
        assert spec["max_steps"] == 500
        heights = []
        for seed in range(40):
            obs, info = env.reset(seed=seed)
            heights.append(info["reset_height"])
            np.testing.assert_allclose(env.grasp_position[:2], [0.32, 0.0], atol=1e-12)
            assert info["cup_offset"] == [0.0, 0.0]
            assert obs["proprio"].shape == (18,) and obs["image"].shape == (6, 96, 96)
        assert min(heights) < 0.035 and max(heights) > 0.13
    finally:
        env.close()


def test_stage_promotion_only_changes_next_episode_goal_and_preserves_explicit_resets():
    env = CenteredApproachSkillEnv(fixed_height=0.1, render_images=False)
    try:
        env.reset(seed=1)
        env.set_skill_stage(1)
        assert env._lesson == env.lessons[0]
        _, info = env.reset(seed=1)
        assert env._lesson == env.lessons[1]
        assert info["reset_height"] == 0.1 and info["cup_offset"] == [0.0, 0.0]
    finally:
        env.close()


def test_final_goal_is_inside_centered_grasp_band_and_rejects_old_overshoot_endpoint():
    env = CenteredApproachSkillEnv(skill_stage=1, fixed_height=0.0, render_images=False)
    try:
        env.reset(seed=1)
        good = install_static_pose(env)
        assert env._goal(good)
        for bad in [
            dict(relative_z_m=0.022),
            dict(relative_z_m=0.001),
            dict(xy_error_m=0.012),
            dict(relative_speed_m_s=0.026),
            dict(minimum_jaw_m=0.039),
            dict(downward=np.cos(np.deg2rad(16.0))),
            dict(contacts=[True, False]),
        ]:
            assert not env._goal({**good, **bad})
        lesson = env.lessons[1]
        assert 0.0 <= lesson.z_low < lesson.z_high <= 0.015
        assert lesson.xy_tolerance <= 0.015 and lesson.maximum_tilt_degrees <= 15.0
        assert (
            lesson.maximum_speed * 0.1
            <= min(lesson.z_low, 0.015 - lesson.z_high) + 1e-12
        )
    finally:
        env.close()


def test_final_goal_boundary_is_static_reachable_contact_free_and_above_table():
    env = CenteredApproachSkillEnv(skill_stage=1, fixed_height=0.0, render_images=False)
    try:
        env.reset(seed=2)
        for height in [0.0025, 0.0075, 0.0125]:
            for tilt in [-15.0, 0.0, 15.0]:
                for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                    xy = 0.01 * np.array([np.cos(angle), np.sin(angle)])
                    metrics = install_static_pose(
                        env, z=height, xy=xy, jaw=0.04, tilt=tilt
                    )
                    assert not any(metrics["contacts"]) and (not env._table_collision())
                    assert env._goal(metrics)
                    for gid in env._pad_ids:
                        bottom = (
                            env.data.geom_xpos[gid, 2]
                            - np.abs(env.data.geom_xmat[gid].reshape(3, 3)[2])
                            @ env.model.geom_size[gid]
                        )
                        assert bottom - env.table_z > 0.008
    finally:
        env.close()


def test_original_physics_and_actor_observations_remain_identical():
    kwargs = dict(fixed_height=0.14, render_images=False)
    centered, original = (
        CenteredApproachSkillEnv(**kwargs),
        ApproachSkillEnv(**kwargs),
    )
    try:
        first, _ = centered.reset(seed=5)
        second, _ = original.reset(seed=5)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        for action in np.random.default_rng(3).uniform(-0.04, 0.04, (10, 5)):
            a, b = (centered.step(action), original.step(action))
            np.testing.assert_array_equal(centered.data.qpos, original.data.qpos)
            np.testing.assert_array_equal(centered.data.qvel, original.data.qvel)
            np.testing.assert_array_equal(centered.data.ctrl, original.data.ctrl)
            for key in a[0]:
                np.testing.assert_array_equal(a[0][key], b[0][key])
            assert a[4]["original_reward"] == b[4]["original_reward"]
    finally:
        centered.close()
        original.close()


def test_final_goal_requires_ten_stable_frames_without_claiming_pickup(monkeypatch):
    env = CenteredApproachSkillEnv(skill_stage=1, fixed_height=0.0, render_images=False)
    try:
        env.reset(seed=2)
        install_static_pose(env)
        monkeypatch.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: None)
        for index in range(10):
            _, _, done, _, info = env.step(np.zeros(5))
            assert done == info["phase_success"] == (index == 9)
            assert not info["full_pickup_success"]
        assert info["phase_stable_steps"] == 10
    finally:
        env.close()
