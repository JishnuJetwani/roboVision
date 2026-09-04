import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from robovision.decoupled_ppo import DecoupledPPO, load_decoupled_ppo


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
            10.0 + float(action[0]),
            True,
            False,
            {},
        )


def configuration():
    return dict(
        n_steps=4,
        batch_size=4,
        n_epochs=2,
        seed=1,
        device="cpu",
        target_kl=0.03,
        policy_kwargs=dict(net_arch=[8], share_features_extractor=False),
    )


def test_kl_stop_keeps_critic_learning_and_actor_bit_identical(tmp_path):
    env = DummyVecEnv([Task] * 2)
    try:
        model = DecoupledPPO("MultiInputPolicy", env, **configuration())
        model.learn(8)
        actor = [p.detach().clone() for p in model.actor_parameters]
        critic = [p.detach().clone() for p in model.critic_parameters]
        model.rollout_buffer.log_probs -= 5
        model.train()
        assert model.last_update_stats["actor_updates"] == 0
        assert model.last_update_stats["critic_updates"] == 4
        assert model.last_update_stats["actor_kl_stopped"]
        assert all((torch.equal(a, b) for a, b in zip(actor, model.actor_parameters)))
        assert any(
            (not torch.equal(a, b) for a, b in zip(critic, model.critic_parameters))
        )
        model.save(tmp_path / "split")
        loaded = DecoupledPPO.load(tmp_path / "split", env=env, device="cpu")
        for key, value in model.policy.state_dict().items():
            assert torch.equal(value, loaded.policy.state_dict()[key])
        assert len(loaded.critic_optimizer.state) == len(model.critic_optimizer.state)
        assert loaded.critic_learning_rate == 0.0003
        loaded.learn(8)
        assert loaded.last_update_stats["critic_updates"] == 4
    finally:
        env.close()


def test_conversion_preserves_weights_and_both_optimizer_histories(tmp_path):
    env = DummyVecEnv([Task] * 2)
    try:
        original = PPO("MultiInputPolicy", env, **configuration())
        original.learn(8)
        original.save(tmp_path / "ordinary")
        converted = load_decoupled_ppo(tmp_path / "ordinary", env, "cpu")
        for key, value in original.policy.state_dict().items():
            assert torch.equal(value, converted.policy.state_dict()[key])
        old_by_name = {
            name: original.policy.optimizer.state.get(parameter)
            for name, parameter in original.policy.named_parameters()
        }
        for name, parameter in converted.policy.named_parameters():
            optimizer = (
                converted.policy.optimizer
                if parameter in converted.policy.optimizer.state
                else converted.critic_optimizer
            )
            for key, value in old_by_name[name].items():
                assert torch.equal(value, optimizer.state[parameter][key])
        assert converted.optimizer_migration["mode"] == "per-parameter-state-copy"
        assert converted.n_steps == original.n_steps
        converted.learn(8)
    finally:
        env.close()


def test_shared_features_rejected():
    env = DummyVecEnv([Task])
    try:
        with pytest.raises(ValueError, match="share_features_extractor"):
            DecoupledPPO("MultiInputPolicy", env, n_steps=4, batch_size=4)
    finally:
        env.close()


def test_helper_resumes_decoupled_both_optimizers_and_applies_critic_lr(tmp_path):
    env = DummyVecEnv([Task] * 2)
    try:
        original = DecoupledPPO("MultiInputPolicy", env, **configuration())
        original.learn(8)
        original.save(tmp_path / "split_resume")
        resumed = load_decoupled_ppo(
            tmp_path / "split_resume", env, "cpu", critic_learning_rate=0.0001
        )
        for key, value in original.policy.state_dict().items():
            assert torch.equal(value, resumed.policy.state_dict()[key])
        for left, right in [
            (original.policy.optimizer, resumed.policy.optimizer),
            (original.critic_optimizer, resumed.critic_optimizer),
        ]:
            a, b = (left.state_dict()["state"], right.state_dict()["state"])
            assert a.keys() == b.keys()
            for parameter in a:
                for key in a[parameter]:
                    assert torch.equal(a[parameter][key], b[parameter][key])
        assert resumed.critic_learning_rate == 0.0001
        assert all(
            (group["lr"] == 0.0001 for group in resumed.critic_optimizer.param_groups)
        )
        assert resumed.algorithm_metadata["name"] == "decoupled-ppo-v1"
        assert resumed.optimizer_migration["mode"] == "resumed-both-optimizer-states"
        resumed.learn(8)
    finally:
        env.close()


def test_zero_value_coefficient_leaves_critic_unchanged_with_existing_momentum():
    env = DummyVecEnv([Task] * 2)
    try:
        model = DecoupledPPO("MultiInputPolicy", env, **configuration())
        model.learn(8)
        model.vf_coef = 0
        before = [p.detach().clone() for p in model.critic_parameters]
        model.learn(8)
        assert all((torch.equal(a, b) for a, b in zip(before, model.critic_parameters)))
        assert model.last_update_stats["actor_updates"] > 0
        assert model.last_update_stats["critic_updates"] == 0
    finally:
        env.close()
