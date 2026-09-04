import copy
import json
import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from robovision.context_exploration import ContextExplorationPPO
from robovision.structured_exploration import (
    ALGORITHM,
    StructuredPPO,
    UnitLatentNoiseDistribution,
    convert_to_structured,
    exploration_report,
    unit_noise_features,
)


class Toy(gym.Env):
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-1, 1, (2,), np.float32)}
    )
    action_space = gym.spaces.Box(-1, 1, (5,), np.float32)

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        return ({"state": np.array([0.1, 0.2], np.float32)}, {})

    def step(self, action):
        self.steps += 1
        return (
            {"state": np.array([0.1, 0.2], np.float32)},
            float(action.sum()),
            self.steps >= 16,
            False,
            {},
        )


def parent(env):
    torch.set_num_threads(1)
    return ContextExplorationPPO(
        "MultiInputPolicy",
        env,
        n_steps=128,
        batch_size=128,
        n_epochs=1,
        device="cpu",
        seed=11,
        std_scales=(1.0,) * 4,
        policy_kwargs=dict(net_arch=[8], share_features_extractor=False),
    )


def test_normalized_noise_matches_sampled_variance_and_is_temporally_persistent():
    torch.manual_seed(3)
    latent = torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    torch.testing.assert_close(
        unit_noise_features(latent).square().sum(-1), torch.ones(2)
    )
    dist = UnitLatentNoiseDistribution(5)
    dist.latent_sde_dim = 3
    log_std = torch.tensor([-7.0, -6.0, -5.0, -4.0, -3.0]).repeat(3, 1)
    means = torch.zeros(2, 5)
    dist.sample_weights(log_std, batch_size=2)
    dist.proba_distribution(means, log_std, latent)
    torch.testing.assert_close(dist.distribution.stddev, log_std[0].exp().expand(2, 5))
    first = dist.sample()
    second = dist.sample()
    assert torch.equal(first, second)
    dist.sample_weights(log_std, batch_size=2)
    assert not torch.equal(first, dist.sample())
    samples = []
    for _ in range(3000):
        dist.sample_weights(log_std, batch_size=2)
        samples.append(dist.sample())
    empirical = torch.stack(samples).std(0)
    torch.testing.assert_close(
        empirical, dist.distribution.stddev, rtol=0.07, atol=1e-06
    )


def test_migration_preserves_all_weights_means_initial_std_rng_and_fresh_optimizers():
    env = DummyVecEnv([Toy] * 4)
    try:
        old = parent(env)
        old.num_timesteps = 123456
        obs = {"state": torch.randn(4, 2)}
        with torch.no_grad():
            expected = old.policy.get_distribution(obs).distribution
            expected_mean = expected.mean.clone()
            expected_std = expected.stddev.clone()
        torch.manual_seed(98)
        rng = torch.get_rng_state().clone()
        converted = convert_to_structured(old)
        assert torch.equal(rng, torch.get_rng_state())
        assert converted.num_timesteps == 123456
        assert not converted.policy.optimizer.state and (
            not converted.critic_optimizer.state
        )
        assert converted.checkpoint_algorithm == ALGORITHM
        for name, value in old.policy.state_dict().items():
            if name == "log_std":
                torch.testing.assert_close(
                    converted.policy.log_std,
                    value[None, :].expand_as(converted.policy.log_std),
                )
            else:
                assert torch.equal(value, converted.policy.state_dict()[name])
        actual = converted.policy.get_distribution(obs).distribution
        torch.testing.assert_close(actual.mean, expected_mean, rtol=0, atol=0)
        torch.testing.assert_close(actual.stddev, expected_std)
        report = exploration_report(converted)
        assert (
            report["physical_marginal_std_lower_bound"]
            == report["physical_marginal_std_upper_bound"]
        )
        json.dumps(report)
    finally:
        env.close()


def test_rollout_marginal_likelihood_reconstruction_training_save_reload_and_deployment(
    tmp_path,
):
    env = DummyVecEnv([Toy] * 4)
    try:
        converted = convert_to_structured(parent(env), resample_steps=25)
        _, callback = converted._setup_learn(512, reset_num_timesteps=False)
        converted.collect_rollouts(
            env, callback, converted.rollout_buffer, n_rollout_steps=128
        )
        for batch in converted.rollout_buffer.get(128):
            logs = converted._actor_distribution(batch).log_prob(batch.actions)
            torch.testing.assert_close(
                torch.exp(logs - batch.old_log_prob),
                torch.ones_like(logs),
                rtol=1e-05,
                atol=1e-05,
            )
        converted.train()
        converted.save(tmp_path / "structured.zip")
        loaded = StructuredPPO.load(tmp_path / "structured.zip", env=env, device="cpu")
        for name, value in converted.policy.state_dict().items():
            assert torch.equal(value, loaded.policy.state_dict()[name])
        assert loaded.sde_sample_freq == 25 and loaded.checkpoint_algorithm == ALGORITHM
        obs = env.reset()
        a, _ = loaded.predict(obs, deterministic=True)
        b, _ = loaded.predict(obs, deterministic=True)
        np.testing.assert_array_equal(a, b)
        a, _ = loaded.predict(obs, deterministic=False)
        b, _ = loaded.predict(obs, deterministic=False)
        assert not np.array_equal(a, b)
        loaded.learn(512, reset_num_timesteps=False)
        assert loaded.last_update_stats["actor_updates"] > 0
    finally:
        env.close()


def test_real_visual_residual_mean_parity_with_nonzero_residual_heads():
    from robovision.cnn import NormalizedGraspCNN

    class Visual(Toy):
        observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, (6, 96, 96), np.uint8),
                "proprio": gym.spaces.Box(-1, 1, (18,), np.float32),
            }
        )

    env = DummyVecEnv([Visual] * 4)
    try:
        old = ContextExplorationPPO(
            "MultiInputPolicy",
            env,
            n_steps=128,
            batch_size=128,
            n_epochs=1,
            device="cpu",
            seed=12,
            std_scales=(1.0,) * 4,
            wide_finger_residual=True,
            arm_residual_mode="wide",
            policy_kwargs=dict(
                features_extractor_class=NormalizedGraspCNN,
                net_arch=[128, 128],
                share_features_extractor=False,
            ),
        )
        with torch.no_grad():
            for head in (old.policy.wide_finger_head, old.policy.arm_residual_head):
                head.weight.normal_(0, 0.01)
                head.bias.fill_(0.02)
            old.policy.log_std.copy_(torch.tensor([-7.0, -7.0, -7.0, -7.0, -5.0]))
        obs = {
            "image": torch.randint(0, 256, (4, 6, 96, 96), dtype=torch.uint8),
            "proprio": torch.zeros(4, 18),
        }
        obs["proprio"][:, 4:6] = torch.tensor(
            [[-1.0, -1.0], [0.5, 0.5], [0.9, 0.9], [1.0, 1.0]]
        )
        with torch.no_grad():
            baseline = old.policy.get_distribution(obs).distribution
            mean = baseline.mean.clone()
            std = baseline.stddev.clone()
            converted = convert_to_structured(old)
            actual = converted.policy.get_distribution(obs).distribution
            torch.testing.assert_close(actual.mean, mean, rtol=0, atol=0)
            torch.testing.assert_close(actual.stddev, std, rtol=1e-05, atol=1e-08)
    finally:
        env.close()
