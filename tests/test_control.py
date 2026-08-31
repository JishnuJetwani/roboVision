import mujoco
import numpy as np
import torch

from robovision.control import actuator_action, robot_features, run_neural_grasp
from robovision.env import VisionCupEnv
from robovision.models import FEATURES, GRASP_VERSION, GraspNetwork, load_grasp


def test_cup_state_does_not_change_robot_inputs():
    env = VisionCupEnv(render_images=False)
    try:
        observation, _ = env.reset(seed=41)
        target = np.array([.31, .015])
        features = robot_features(env, target)
        env.data.qpos[env._cup_qadr:env._cup_qadr + 3] = [.38, -.08, .5]
        env.data.qvel[env._cup_vadr:env._cup_vadr + 6] = 2
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_array_equal(robot_features(env, target), features)
        np.testing.assert_array_equal(env._observation()['proprio'], observation['proprio'])
    finally:
        env.close()


def test_servo_tracks_commanded_xy_and_clips_actions():
    class Robot:
        grasp_position = np.array([.31, .02, .44])
        action_delta = .008

    action = actuator_action(Robot(), [.314, .004], [.2, -.6])
    np.testing.assert_allclose(action, [.5, -1, .2, -.6], atol=1e-6)
    action = actuator_action(Robot(), [.302, .024], [-2, 2])
    np.testing.assert_allclose(action, [-1, .5, -1, 1], atol=1e-6)


def test_loaded_grasp_network_controls_vertical_motion_and_jaws(tmp_path):
    network = GraspNetwork()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.layers[-1].bias.copy_(torch.tensor([-.35, .72]))
    path = tmp_path / 'grasp.pt'
    torch.save({'version': GRASP_VERSION, 'features': FEATURES,
                'weights': network.state_dict()}, path)
    env = VisionCupEnv(render_images=False)
    try:
        env.reset(seed=42)
        result = run_neural_grasp(env, env.grasp_position[:2], load_grasp(path),
                                  max_steps=3, record=True)
        assert result['neural_steps'] == 3
        for row in result['trace']:
            np.testing.assert_allclose(row['executed_action'][2:], [-.35, .72], atol=1e-7)
    finally:
        env.close()
