import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Normal
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.vec_env import DummyVecEnv
from robovision.context_exploration import (
    ContextExplorationPPO,
    ContextDictRolloutBuffer,
    scaled_gaussian,
    load_context_exploration_ppo,
)
from robovision.decoupled_ppo import DecoupledPPO


class Task(gym.Env):
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-100, 100, (1,), np.float32)}
    )
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        return ({"state": np.array([0.1], np.float32)}, {})

    def step(self, action):
        assert np.abs(action).max() <= 1
        return (
            {"state": np.array([0.1], np.float32)},
            float(action.sum()),
            True,
            False,
            {},
        )


def config():
    return dict(
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        seed=2,
        device="cpu",
        policy_kwargs=dict(net_arch=[8], share_features_extractor=False),
    )


def test_direct_normal_match_means_unchanged_and_scale_one_equivalence():
    mean = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    logstd = torch.tensor([-0.2, -0.4])
    base = DiagGaussianDistribution(2).proba_distribution(mean, logstd)
    scaled = scaled_gaussian(base, [10, 25])
    expected = Normal(mean, torch.exp(logstd) * torch.tensor([[10], [25]]))
    actions = torch.tensor([[0.5, 1.0], [-2.0, 0.8]])
    assert torch.equal(scaled.distribution.mean, base.distribution.mean)
    torch.testing.assert_close(
        scaled.log_prob(actions), expected.log_prob(actions).sum(-1)
    )
    torch.testing.assert_close(scaled.entropy(), expected.entropy().sum(-1))
    unit = scaled_gaussian(base, [1, 1])
    assert torch.equal(unit.log_prob(actions), base.log_prob(actions))
    assert torch.equal(unit.entropy(), base.entropy())


def test_rollout_likelihood_ratio_one_before_update_and_raw_actions_preserved():
    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO("MultiInputPolicy", env, **config())
        _, callback = model._setup_learn(16)
        model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=4)
        assert np.abs(model.rollout_buffer.actions).max() > 1
        for batch in model.rollout_buffer.get(5):
            current = model._actor_distribution(batch).log_prob(batch.actions)
            torch.testing.assert_close(
                torch.exp(current - batch.old_log_prob),
                torch.ones_like(current),
                atol=1e-06,
                rtol=1e-06,
            )
            torch.testing.assert_close(
                batch.std_scales,
                torch.tensor([10.0, 25.0, 1.0, 1.0])[batch.context_ids],
            )
        obs = model.policy.obs_to_tensor(env.reset())[0]
        base = model.policy.get_distribution(obs)
        contextual = scaled_gaussian(base, [10, 25, 1, 1])
        assert torch.equal(base.distribution.mean, contextual.distribution.mean)
        torch.testing.assert_close(
            base.distribution.stddev, model.policy.log_std.exp().expand(4, -1)
        )
    finally:
        env.close()


def test_context_shuffle_stays_paired_with_observations_and_returns():
    buffer = ContextDictRolloutBuffer(
        3, Task.observation_space, Task.action_space, n_envs=4, device="cpu"
    )
    ids = np.arange(12, dtype=np.float32).reshape(3, 4)
    buffer.observations["state"] = ids[..., None]
    buffer.actions = np.repeat(ids[..., None], 2, axis=2)
    buffer.returns = ids + 100
    buffer.full = True
    for _ in range(2):
        seen = []
        for batch in buffer.get(5):
            identity = batch.observations["state"].flatten().long()
            seen.extend(identity.tolist())
            assert torch.equal(batch.context_ids, identity % 4)
            torch.testing.assert_close(
                batch.std_scales, torch.tensor([10.0, 25.0, 1.0, 1.0])[identity % 4]
            )
            torch.testing.assert_close(batch.returns, identity.float() + 100)
        assert sorted(seen) == list(range(12))


def test_conversion_save_resume_preserves_weights_optimizer_and_contexts(tmp_path):
    env = DummyVecEnv([Task] * 4)
    try:
        source = DecoupledPPO("MultiInputPolicy", env, **config())
        source.learn(16)
        source.save(tmp_path / "source")
        model = load_context_exploration_ppo(tmp_path / "source", env, "cpu")
        for key, value in source.policy.state_dict().items():
            assert torch.equal(value, model.policy.state_dict()[key])
        for old, new in [
            (source.policy.optimizer, model.policy.optimizer),
            (source.critic_optimizer, model.critic_optimizer),
        ]:
            old_state, new_state = (
                old.state_dict()["state"],
                new.state_dict()["state"],
            )
            for param, values in old_state.items():
                for key, value in values.items():
                    assert torch.equal(value, new_state[param][key])
        model.learn(16)
        assert model.context_update_stats
        model.save(tmp_path / "context")
        resumed = load_context_exploration_ppo(tmp_path / "context", env, "cpu")
        assert tuple(resumed.std_scales) == (10.0, 25.0, 1.0, 1.0)
        assert tuple(resumed.policy.std_scales) == (10.0, 25.0, 1.0, 1.0)
        for key, value in model.policy.state_dict().items():
            assert torch.equal(value, resumed.policy.state_dict()[key])
        obs = env.reset()
        np.testing.assert_array_equal(
            model.predict(obs, deterministic=True)[0],
            resumed.predict(obs, deterministic=True)[0],
        )
        resumed.learn(16)
        with pytest.raises(ValueError, match="fixed saved"):
            load_context_exploration_ppo(
                tmp_path / "context", env, "cpu", std_scales=(1, 1, 1, 1)
            )
    finally:
        env.close()


def test_role_normalization_preserves_returns_and_context_pairing():
    buffer = ContextDictRolloutBuffer(
        4,
        Task.observation_space,
        Task.action_space,
        n_envs=4,
        device="cpu",
        role_normalization=True,
    )
    ids = np.arange(16, dtype=np.float32).reshape(4, 4)
    buffer.observations["state"] = ids[..., None].copy()
    buffer.actions = np.repeat(ids[..., None], 2, axis=2)
    buffer.log_probs = ids + 200
    buffer.returns = ids + 100
    buffer.advantages = ids + np.array([0, 0, 100, -50])
    buffer.full = True
    for _ in range(2):
        gathered = {0: [], 1: [], 2: []}
        for batch in buffer.get(5):
            identity = batch.observations["state"].flatten().long()
            assert torch.equal(batch.context_ids, identity % 4)
            torch.testing.assert_close(batch.returns, identity.float() + 100)
            torch.testing.assert_close(batch.old_log_prob, identity.float() + 200)
            torch.testing.assert_close(batch.actions[:, 0], identity.float())
            torch.testing.assert_close(
                batch.std_scales, torch.tensor([10.0, 25.0, 1.0, 1.0])[identity % 4]
            )
            for context, advantage in zip(
                batch.context_ids.tolist(), batch.advantages.tolist()
            ):
                gathered[[0, 0, 1, 2][context]].append(advantage)
        for values in gathered.values():
            assert np.mean(values) == pytest.approx(0, abs=1e-06)
            assert np.std(values) == pytest.approx(1, abs=1e-06)
    assert buffer.raw_role_advantage_stats["0"]["count"] == 8


def test_role_normalized_context_save_resume_and_explicit_reconfiguration(tmp_path):
    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO(
            "MultiInputPolicy", env, role_normalization=True, **config()
        )
        model.learn(16)
        assert not model.normalize_advantage
        assert model.rollout_buffer.raw_role_advantage_stats
        model.save(tmp_path / "role_context")
        resumed = load_context_exploration_ppo(tmp_path / "role_context", env, "cpu")
        assert resumed.role_normalization
        assert not resumed.normalize_advantage
        resumed.learn(16)
        with pytest.raises(ValueError, match="allow_context_reconfigure"):
            load_context_exploration_ppo(
                tmp_path / "role_context", env, "cpu", role_normalization=False
            )
        with pytest.raises(ValueError, match="allow_context_reconfigure"):
            load_context_exploration_ppo(
                tmp_path / "role_context", env, "cpu", std_scales=(5, 10, 1, 1)
            )
        changed = load_context_exploration_ppo(
            tmp_path / "role_context",
            env,
            "cpu",
            std_scales=(5, 10, 1, 1),
            role_normalization=False,
            allow_context_reconfigure=True,
        )
        assert not changed.rollout_buffer.full
        assert changed.rollout_buffer.pos == 0
        assert changed.policy.std_scales == (5, 10, 1, 1)
        assert changed.rollout_buffer.worker_std_scales == (5, 10, 1, 1)
        assert changed.context_reconfiguration["old_std_scales"] == [10, 25, 1, 1]
        assert changed.context_reconfiguration["new_std_scales"] == [5, 10, 1, 1]
        assert not changed.role_normalization
        for key, value in model.policy.state_dict().items():
            assert torch.equal(value, changed.policy.state_dict()[key])
        for old, new in [
            (model.policy.optimizer, changed.policy.optimizer),
            (model.critic_optimizer, changed.critic_optimizer),
        ]:
            for parameter, state in old.state_dict()["state"].items():
                for key, value in state.items():
                    assert torch.equal(value, new.state_dict()["state"][parameter][key])
        changed.learn(16)
    finally:
        env.close()
