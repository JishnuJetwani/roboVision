"""Execute learned targets and grasp actions through robot kinematics."""
import mujoco
import numpy as np
import torch


def robot_features(env, target_xy):
    """Robot measurements relative to the commanded XY target."""
    jacobian = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jacobian, None, env._grasp_sid)
    velocity = jacobian[:, :6] @ env.data.qvel[:6]
    hand = env.grasp_position
    error = np.asarray(target_xy) - hand[:2]
    return np.asarray([
        (hand[2] - .35) / .1, velocity[2] / .2,
        env.data.qpos[4] / .045, env.data.qpos[5] / .045,
        env.data.qvel[4] / .1, env.data.qvel[5] / .1,
        env.last_action[2], env.last_action[3], error[0] / .01, error[1] / .01,
        velocity[0] / .2, velocity[1] / .2,
    ], dtype=np.float32)


def action_to_target(action):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError("Expected finite XY action")
    action = np.clip(action, -1, 1)
    return np.array([.32 + .075 * action[0], .095 * action[1]], dtype=np.float64)


def execute_approach(env, target_xy, frame_callback=None):
    """Move to the predicted XY target with the gripper open."""
    env.data.ctrl[:4] = env.inverse_kinematics(np.r_[target_xy, .45])
    env.data.ctrl[4:6] = .045
    for _ in range(16):
        mujoco.mj_step(env.model, env.data, nstep=25)
        if frame_callback is not None:
            frame_callback(env, "approach")
    env._check_simulation()
    env.last_action[:] = [0, 0, 0, 1]
    env._previous_frame = None
    env._last_potential = env._potential(env._info())


def actuator_action(env, target_xy, predicted):
    """XY servo tracks the target; neural output controls height and jaws."""
    predicted = np.asarray(predicted, dtype=np.float32)
    if predicted.shape != (2,) or not np.isfinite(predicted).all():
        raise ValueError("Expected finite Z/jaw action")
    xy = np.clip((np.asarray(target_xy) - env.grasp_position[:2]) / env.action_delta, -1, 1)
    return np.r_[xy, np.clip(predicted, -1, 1)].astype(np.float32)


def run_neural_grasp(env, target_xy, network, *, max_steps=160, frame_callback=None, record=False):
    if isinstance(max_steps, bool) or int(max_steps) != max_steps or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    rows = []
    for index in range(max_steps):
        features = robot_features(env, target_xy)
        with torch.no_grad():
            predicted = network(torch.from_numpy(features)[None])[0].numpy()
        action = actuator_action(env, target_xy, predicted)
        _, _, terminated, truncated, info = env.step(action)
        if record:
            rows.append({"features": features.tolist(), "neural_action": predicted.tolist(),
                         "executed_action": action.tolist()})
        if frame_callback is not None:
            frame_callback(env, "learned grasp")
        if terminated or truncated:
            break
    return {**info, "neural_steps": index + 1, "trace": rows}
