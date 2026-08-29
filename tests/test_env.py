import mujoco
import numpy as np
import pytest

from robovision.env import VisionCupEnv


@pytest.fixture
def env():
    environment = VisionCupEnv(render_images=False)
    environment.reset(seed=81000)
    yield environment
    environment.close()


def test_moving_cup_changes_camera_but_not_proprioception(env):
    env.render_images = True
    before = env._observation()
    env.data.qpos[env._cup_qadr:env._cup_qadr + 2] = [.38, -.08]
    mujoco.mj_forward(env.model, env.data)
    after = env._observation()
    assert after['image'].shape == (6, 96, 96)
    assert after['image'].dtype == np.uint8
    assert not np.array_equal(before['image'][3:], after['image'][3:])
    np.testing.assert_array_equal(after['image'][:3], before['image'][3:])
    np.testing.assert_array_equal(before['proprio'], after['proprio'])


def test_inverse_kinematics_reaches_workspace_corners(env):
    for x in [.22, .42]:
        for y in [-.12, .12]:
            for z in [.29, .51]:
                target = np.array([x, y, z])
                joints = env.inverse_kinematics(target)
                assert np.all(joints >= env.model.jnt_range[:4, 0])
                assert np.all(joints <= env.model.jnt_range[:4, 1])
                env.data.qpos[:4] = joints
                mujoco.mj_forward(env.model, env.data)
                np.testing.assert_allclose(env.grasp_position, target, atol=1e-8)


def test_tossing_cup_does_not_count_as_grasp(env):
    env.data.qvel[env._cup_vadr + 2] = 2
    rows = []
    for _ in range(35):
        _, _, terminated, _, info = env.step([0, 0, 0, 1])
        rows.append(info)
        if terminated:
            break
    assert max(row['clearance'] for row in rows) >= .06
    assert not any(row['is_success'] for row in rows)


def test_success_requires_unbroken_stable_hold(env, monkeypatch):
    stable = dict(contacts=[True, True], clearance=.07, upright=1., cup_speed=0.)
    state = stable.copy()
    monkeypatch.setattr(env, '_info', lambda: state.copy())
    for failure in [dict(contacts=[True, False]), dict(clearance=.01),
                    dict(upright=.7), dict(cup_speed=.2)]:
        state = stable.copy()
        for _ in range(9):
            assert not env.step([0, 0, 0, 1])[-1]['is_success']
        state.update(failure)
        assert not env.step([0, 0, 0, 1])[-1]['is_success']
    state = stable.copy()
    for _ in range(9):
        assert not env.step([0, 0, 0, 1])[-1]['is_success']
    assert env.step([0, 0, 0, 1])[-1]['is_success']
