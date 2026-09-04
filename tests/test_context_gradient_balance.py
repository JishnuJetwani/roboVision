import numpy as np
import pytest
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from robovision.context_exploration import (
    ContextExplorationPPO,
    load_context_exploration_ppo,
    role_gradient_weights,
)


class Task(gym.Env):
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-1, 1, (1,), np.float32)}
    )
    action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        return ({"state": np.array([0.1], np.float32)}, {})

    def step(self, action):
        return (
            {"state": np.array([0.1], np.float32)},
            1.0 + float(action[0]),
            True,
            False,
            {},
        )


def test_inverse_norm_weights_preserve_sample_mass_and_are_bounded():
    norms = np.array([1.0, 2.0, 4.0])
    masses = np.array([0.5, 0.25, 0.25])
    weights = role_gradient_weights(norms, masses)
    np.testing.assert_allclose(weights * norms, np.full(3, weights[0]))
    contributions = masses * weights * norms
    assert contributions[0] == pytest.approx(2 * contributions[1])
    assert contributions[1] == pytest.approx(contributions[2])
    bounded = role_gradient_weights([1e-06, 1000000.0], [0.5, 0.5], 3)
    np.testing.assert_allclose(bounded, [3, 1 / 3])
    np.testing.assert_array_equal(role_gradient_weights([0, 0], [0.5, 0.5]), [1, 1])
    np.testing.assert_array_equal(role_gradient_weights(norms, masses, 1), [1, 1, 1])


def test_balancing_unit_bound_matches_original_clipped_objective_and_gradient():
    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO(
            "MultiInputPolicy",
            env,
            n_steps=4,
            batch_size=16,
            n_epochs=1,
            device="cpu",
            policy_kwargs=dict(net_arch=[8], share_features_extractor=False),
            seed=2,
        )
        _, callback = model._setup_learn(16)
        model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=4)
        batch = next(model.rollout_buffer.get(16))
        distribution = model._actor_distribution(batch)
        log_prob = distribution.log_prob(batch.actions)
        ratio = torch.exp(log_prob - batch.old_log_prob)
        original = model._actor_losses(
            batch, batch.advantages, ratio, distribution.entropy(), log_prob, 0.2
        )
        original_grad = torch.autograd.grad(
            original[0], model.actor_parameters, retain_graph=True, allow_unused=True
        )
        model.role_gradient_balance = True
        model.gradient_balance_max_weight = 1
        model.role_gradient_stats = {}
        balanced = model._actor_losses(
            batch, batch.advantages, ratio, distribution.entropy(), log_prob, 0.2
        )
        balanced_grad = torch.autograd.grad(
            balanced[0], model.actor_parameters, retain_graph=True, allow_unused=True
        )
        torch.testing.assert_close(original[0], balanced[0])
        torch.testing.assert_close(original[1], balanced[1])
        for a, b in zip(original_grad, balanced_grad):
            if a is not None:
                torch.testing.assert_close(a, b, atol=1e-06, rtol=1e-05)
        assert set(model.role_gradient_stats) == {"0", "1", "2"}
    finally:
        env.close()


def test_balanced_training_save_resume_and_explicit_toggle(tmp_path):
    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO(
            "MultiInputPolicy",
            env,
            n_steps=4,
            batch_size=16,
            n_epochs=1,
            device="cpu",
            role_gradient_balance=True,
            gradient_balance_max_weight=3,
            policy_kwargs=dict(net_arch=[8], share_features_extractor=False),
            seed=2,
        )
        model.learn(16)
        assert model.role_gradient_stats
        assert all(
            (1 / 3 <= r["weight"] <= 3 for r in model.role_gradient_stats.values())
        )
        model.save(tmp_path / "balanced")
        resumed = load_context_exploration_ppo(tmp_path / "balanced", env, "cpu")
        assert resumed.role_gradient_balance
        assert resumed.gradient_balance_max_weight == 3
        resumed.learn(16)
        with pytest.raises(ValueError, match="allow_context_reconfigure"):
            load_context_exploration_ppo(
                tmp_path / "balanced", env, "cpu", role_gradient_balance=False
            )
        changed = load_context_exploration_ppo(
            tmp_path / "balanced",
            env,
            "cpu",
            role_gradient_balance=False,
            allow_context_reconfigure=True,
        )
        assert not changed.role_gradient_balance
        assert changed.context_reconfiguration["old_role_gradient_balance"]
        assert not changed.context_reconfiguration["new_role_gradient_balance"]
        for key, value in model.policy.state_dict().items():
            assert torch.equal(value, changed.policy.state_dict()[key])
        assert not changed.rollout_buffer.full
    finally:
        env.close()
