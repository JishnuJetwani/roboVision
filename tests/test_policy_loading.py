import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from robovision.decoupled_ppo import DecoupledPPO
from robovision.policy_loading import load_grasp_policy


class Task(gym.Env):
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-1, 1, (2,), np.float32)}
    )
    action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        return ({"state": np.array([0.1, 0.2], np.float32)}, {})

    def step(self, action):
        return (
            {"state": np.array([0.1, 0.2], np.float32)},
            1.0 + float(action[0]),
            True,
            False,
            {},
        )


def identical(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            identical(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("algorithm", [PPO, DecoupledPPO])
def test_generic_load_prediction_and_optimizer_restoration(tmp_path, algorithm):
    env = DummyVecEnv([Task] * 2)
    try:
        model = algorithm(
            "MultiInputPolicy",
            env,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            policy_kwargs=dict(net_arch=[8], share_features_extractor=False),
            device="cpu",
            seed=2,
        )
        model.learn(8)
        model.save(tmp_path / "checkpoint")
        frozen = load_grasp_policy(tmp_path / "checkpoint", device="cpu")
        resumed = load_grasp_policy(tmp_path / "checkpoint.zip", env=env, device="cpu")
        assert type(frozen) is algorithm
        assert type(resumed) is algorithm
        observation = env.reset()
        action = model.predict(observation, deterministic=True)[0]
        np.testing.assert_array_equal(
            action, frozen.predict(observation, deterministic=True)[0]
        )
        np.testing.assert_array_equal(
            action, resumed.predict(observation, deterministic=True)[0]
        )
        identical(model.policy.state_dict(), resumed.policy.state_dict())
        identical(
            model.policy.optimizer.state_dict(), resumed.policy.optimizer.state_dict()
        )
        if algorithm is DecoupledPPO:
            identical(
                model.critic_optimizer.state_dict(),
                resumed.critic_optimizer.state_dict(),
            )
        resumed.learn(8)
    finally:
        env.close()


def test_generic_context_resume_preserves_algorithm_likelihood_and_optimizers(tmp_path):
    from robovision.context_exploration import ContextExplorationPPO
    from robovision.decoupled_ppo import load_decoupled_ppo
    from robovision.policy_loading import checkpoint_algorithm

    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO(
            "MultiInputPolicy",
            env,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            policy_kwargs=dict(net_arch=[8], share_features_extractor=False),
            device="cpu",
            seed=2,
        )
        model.learn(16)
        model.save(tmp_path / "context")
        assert checkpoint_algorithm(tmp_path / "context") == "context-decoupled-ppo-v1"
        loaded = load_grasp_policy(tmp_path / "context", env=env, device="cpu")
        assert type(loaded) is ContextExplorationPPO
        identical(model.policy.state_dict(), loaded.policy.state_dict())
        identical(
            model.policy.optimizer.state_dict(), loaded.policy.optimizer.state_dict()
        )
        identical(
            model.critic_optimizer.state_dict(), loaded.critic_optimizer.state_dict()
        )
        _, callback = loaded._setup_learn(16)
        loaded.collect_rollouts(env, callback, loaded.rollout_buffer, n_rollout_steps=4)
        for batch in loaded.rollout_buffer.get(5):
            current = loaded._actor_distribution(batch).log_prob(batch.actions)
            torch.testing.assert_close(
                torch.exp(current - batch.old_log_prob),
                torch.ones_like(current),
                atol=1e-06,
                rtol=1e-06,
            )
        loaded.train()
        assert loaded.context_update_stats
        with pytest.raises(ValueError, match="Context checkpoint requires"):
            load_decoupled_ppo(tmp_path / "context", env, "cpu")
    finally:
        env.close()
