import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from robovision.context_exploration import (
    ContextExplorationPPO,
    training_action_std_scales,
    set_runtime_arm_scales,
)
from robovision.policy_loading import load_grasp_policy


class Task(gym.Env):
    observation_space = gym.spaces.Dict(
        {"proprio": gym.spaces.Box(-10, 10, (18,), np.float32)}
    )
    action_space = gym.spaces.Box(-1, 1, (5,), np.float32)

    def reset(self, *, seed=None, options=None):
        return ({"proprio": np.zeros(18, np.float32)}, {})

    def step(self, action):
        return (
            {"proprio": np.zeros(18, np.float32)},
            float(action.sum()),
            True,
            False,
            {},
        )


def factory(env):
    return ContextExplorationPPO(
        "MultiInputPolicy",
        env,
        std_scales=(1, 5, 1, 1),
        n_steps=4,
        batch_size=8,
        n_epochs=1,
        policy_kwargs=dict(net_arch=[8], share_features_extractor=False),
        device="cpu",
        seed=2,
    )


def test_arm_multiplier_changes_only_requested_worker_arm_scales():
    observation = {"proprio": torch.ones(4, 18)}
    before = training_action_std_scales(observation, [1, 5, 1, 1], 5, [1, 1, 1, 100])
    after = training_action_std_scales(
        observation, [1, 5, 1, 1], 5, [1, 1, 1, 100], runtime_arm_scales=[1, 6, 1, 1]
    )
    torch.testing.assert_close(after[1, :4], torch.full((4,), 30.0))
    torch.testing.assert_close(before[:, 4], after[:, 4])
    torch.testing.assert_close(before[[0, 2, 3]], after[[0, 2, 3]])


def test_runtime_density_capture_prediction_and_weights_unchanged():
    env = DummyVecEnv([Task] * 4)
    try:
        model = factory(env)
        weights = {
            key: value.clone() for key, value in model.policy.state_dict().items()
        }
        obs = env.reset()
        torch.manual_seed(123)
        before = model.predict(obs, deterministic=False)[0]
        set_runtime_arm_scales(model, (1, 6, 1, 1))
        torch.manual_seed(123)
        after = model.predict(obs, deterministic=False)[0]
        np.testing.assert_array_equal(before, after)
        for key, value in weights.items():
            assert torch.equal(value, model.policy.state_dict()[key])
        _, callback = model._setup_learn(16)
        model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=4)
        for batch in model.rollout_buffer.get(5):
            current = model._actor_distribution(batch).log_prob(batch.actions)
            torch.testing.assert_close(
                torch.exp(current - batch.old_log_prob),
                torch.ones_like(current),
                atol=1e-06,
                rtol=1e-06,
            )
            for i, context in enumerate(batch.context_ids.tolist()):
                torch.testing.assert_close(
                    batch.action_std_scales[i, :4],
                    torch.ones(4) * [1, 30, 1, 1][context],
                )
                assert batch.action_std_scales[i, 4] == [1, 5, 1, 1][context]
        with pytest.raises(RuntimeError, match="unconsumed"):
            set_runtime_arm_scales(model, (1, 2, 1, 1))
        model.train()
        assert set_runtime_arm_scales(model, (1, 2, 1, 1)) == (1, 2, 1, 1)
    finally:
        env.close()


def test_runtime_save_resume_and_validation(tmp_path):
    env = DummyVecEnv([Task] * 4)
    try:
        model = factory(env)
        for invalid in [
            (1, 1, 1),
            (1, 0, 1, 1),
            (1, float("nan"), 1, 1),
            (1, 101, 1, 1),
            (1, True, 1, 1),
        ]:
            with pytest.raises(ValueError):
                set_runtime_arm_scales(model, invalid)
        model._collecting_context_rollout = True
        with pytest.raises(RuntimeError, match="between completed"):
            set_runtime_arm_scales(model, (1, 2, 1, 1))
        model._collecting_context_rollout = False
        set_runtime_arm_scales(model, (1, 6, 1, 1))
        model.learn(16)
        model.save(tmp_path / "runtime")
        resumed = load_grasp_policy(tmp_path / "runtime", env=env, device="cpu")
        assert resumed.runtime_arm_scales == (1, 6, 1, 1)
        assert resumed.policy.runtime_arm_scales == (1, 6, 1, 1)
        assert resumed.algorithm_metadata["runtime_arm_scales"] == [1, 6, 1, 1]
        set_runtime_arm_scales(resumed, (1, 3, 1, 1))
        resumed.learn(16)
        assert resumed.policy.runtime_arm_scales == (1, 3, 1, 1)
    finally:
        env.close()
