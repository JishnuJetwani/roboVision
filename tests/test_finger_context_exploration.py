import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.distributions import Normal
from robovision.context_exploration import (
    ContextExplorationPPO,
    load_context_exploration_ppo,
    training_action_std_scales,
    scaled_gaussian,
)


def observation(jaws):
    proprio = torch.zeros((len(jaws), 18))
    proprio[:, 4:6] = torch.tensor(jaws)[:, None] / 0.045 * 2 - 1
    return {"proprio": proprio}


class Task(gym.Env):
    observation_space = gym.spaces.Dict(
        {"proprio": gym.spaces.Box(-10, 10, (18,), np.float32)}
    )
    action_space = gym.spaces.Box(-1, 1, (5,), np.float32)

    def obs(self):
        result = observation([[0.045, 0.038, 0.034, 0.03][self.step_count % 4]])
        return {k: v.numpy()[0] for k, v in result.items()}

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        return (self.obs(), {})

    def step(self, action):
        assert np.max(np.abs(action)) <= 1
        self.step_count += 1
        return (self.obs(), float(action.sum()), self.step_count == 4, False, {})


def configuration():
    return dict(
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        seed=2,
        device="cpu",
        policy_kwargs=dict(net_arch=[8], share_features_extractor=False),
    )


def test_observed_jaw_smooth_gate_only_changes_enabled_finger():
    for jaw, expected in [
        (0.045, 100),
        (0.04, 100),
        (0.037, 50.5),
        (0.034, 1),
        (0.03, 1),
    ]:
        result = training_action_std_scales(
            observation([jaw] * 4), (10, 25, 1, 1), 5, (1, 1, 1, 100)
        )
        torch.testing.assert_close(
            result[:3], torch.tensor([[10.0] * 5, [25.0] * 5, [1.0] * 5])
        )
        torch.testing.assert_close(result[3, :4], torch.ones(4))
        assert result[3, 4].item() == pytest.approx(expected, abs=0.001)
    with pytest.raises(ValueError, match="5 direct-force"):
        training_action_std_scales(
            observation([0.045] * 4), (10, 25, 1, 1), 2, (1, 1, 1, 100)
        )


def test_actual_sampler_scales_density_shuffle_and_base_prediction():
    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO(
            "MultiInputPolicy",
            env,
            finger_exploration_max=(1, 1, 1, 100),
            **configuration(),
        )
        _, callback = model._setup_learn(16)
        model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=4)
        pickup_scales = model.rollout_buffer.action_std_scales[:, 3, 4].copy()
        np.testing.assert_allclose(pickup_scales, [100, 74.33333, 1, 1], atol=0.001)
        for batch in model.rollout_buffer.get(5):
            distribution = model._actor_distribution(batch)
            base = model.policy.get_distribution(batch.observations)
            expected = Normal(
                base.distribution.mean,
                base.distribution.stddev * batch.action_std_scales,
            )
            torch.testing.assert_close(
                distribution.log_prob(batch.actions),
                expected.log_prob(batch.actions).sum(-1),
            )
            torch.testing.assert_close(
                distribution.entropy(), expected.entropy().sum(-1)
            )
            ratio = torch.exp(distribution.log_prob(batch.actions) - batch.old_log_prob)
            torch.testing.assert_close(
                ratio, torch.ones_like(ratio), atol=1e-05, rtol=1e-05
            )
            assert torch.equal(distribution.distribution.mean, base.distribution.mean)
            for i, context in enumerate(batch.context_ids.tolist()):
                jaw = (batch.observations["proprio"][i, 4].item() + 1) * 0.5 * 0.045
                if context == 3 and jaw <= 0.034001:
                    assert batch.action_std_scales[i, -1] == 1
                if context != 3:
                    torch.testing.assert_close(
                        batch.action_std_scales[i],
                        torch.ones(5) * [10, 25, 1, 1][context],
                    )
        obs = env.reset()
        tensor = model.policy.obs_to_tensor(obs)[0]
        torch.manual_seed(11)
        raw = model.policy.get_distribution(tensor).sample().detach().numpy()
        torch.manual_seed(11)
        deployment = model.predict(obs, deterministic=False)[0]
        np.testing.assert_array_equal(deployment, np.clip(raw, -1, 1))
    finally:
        env.close()


def test_finger_schedule_resume_reconfigure_and_training(tmp_path):
    env = DummyVecEnv([Task] * 4)
    try:
        source = ContextExplorationPPO("MultiInputPolicy", env, **configuration())
        source.learn(16)
        source.save(tmp_path / "source")
        with pytest.raises(ValueError, match="allow_context_reconfigure"):
            load_context_exploration_ppo(
                tmp_path / "source", env, "cpu", finger_exploration_max=(1, 1, 1, 100)
            )
        model = load_context_exploration_ppo(
            tmp_path / "source",
            env,
            "cpu",
            finger_exploration_max=(1, 1, 1, 100),
            allow_context_reconfigure=True,
        )
        for key, value in source.policy.state_dict().items():
            assert torch.equal(value, model.policy.state_dict()[key])
        for a, b in [
            (source.policy.optimizer, model.policy.optimizer),
            (source.critic_optimizer, model.critic_optimizer),
        ]:
            for key, state in a.state_dict()["state"].items():
                for name, value in state.items():
                    assert torch.equal(value, b.state_dict()["state"][key][name])
        assert model.context_reconfiguration["old_finger_exploration_max"] == [1] * 4
        assert model.context_reconfiguration["new_finger_exploration_max"] == [
            1,
            1,
            1,
            100,
        ]
        model.learn(16)
        stats = model.finger_exploration_stats["3"]
        assert stats["minimum"] == 1
        assert stats["maximum"] == 100
        assert stats["base_fraction"] == 0.5
        assert stats["mean"] == pytest.approx((100 + 74.33333 + 1 + 1) / 4, abs=0.001)
        model.save(tmp_path / "conditional")
        resumed = load_context_exploration_ppo(tmp_path / "conditional", env, "cpu")
        assert resumed.finger_exploration_max == (1, 1, 1, 100)
        assert resumed.policy.finger_exploration_max == (1, 1, 1, 100)
        assert resumed.algorithm_metadata["finger_exploration_quiet_m"] == 0.034
        assert resumed.algorithm_metadata["finger_exploration_full_m"] == 0.04
        resumed.learn(16)
    finally:
        env.close()
