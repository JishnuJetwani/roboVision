import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from robovision.role_ppo import (
    RoleDictRolloutBuffer,
    RoleNormalizedPPO,
    normalize_by_role,
)


def test_normalizes_roles_independently_not_by_reward_magnitude():
    roles = np.array([[0, 0, 1, 2]] * 4)
    advantages = np.array(
        [[1, 2, 100, -10], [3, 4, 200, -9], [5, 6, 300, -8], [7, 8, 400, -7]],
        np.float32,
    )
    result = normalize_by_role(advantages, roles)
    for role in (0, 1, 2):
        assert result[roles == role].mean() == pytest.approx(0, abs=1e-06)
        assert result[roles == role].std() == pytest.approx(1, abs=1e-06)
    np.testing.assert_array_equal(advantages[:, 2], [100, 200, 300, 400])
    np.testing.assert_array_equal(
        normalize_by_role(np.ones((2, 1)) * 50, np.zeros((2, 1))), 0.0
    )


def test_shuffling_preserves_roles_and_trajectory_pairing_and_returns():
    observation_space = gym.spaces.Dict(
        {"id": gym.spaces.Box(0, 100, (1,), np.float32)}
    )
    action_space = gym.spaces.Box(-100, 100, (1,), np.float32)
    buffer = RoleDictRolloutBuffer(
        4, observation_space, action_space, n_envs=4, device="cpu"
    )
    ids = np.arange(16, dtype=np.float32).reshape(4, 4)
    buffer.observations["id"] = ids[..., None].copy()
    buffer.actions = ids[..., None].copy()
    buffer.values = ids + 20
    buffer.log_probs = ids + 30
    buffer.advantages = ids + np.array([0, 0, 100, -100])
    buffer.returns = ids + 1000
    original_returns = buffer.returns.copy()
    expected = normalize_by_role(buffer.advantages, buffer.role_ids)
    buffer.full = True
    for _ in range(2):
        seen = []
        for batch in buffer.get(3):
            flat_ids = batch.observations["id"].numpy().ravel().astype(int)
            seen.extend(flat_ids.tolist())
            np.testing.assert_array_equal(batch.actions.numpy().ravel(), flat_ids)
            np.testing.assert_array_equal(batch.returns.numpy(), flat_ids + 1000)
            np.testing.assert_array_equal(batch.old_values.numpy(), flat_ids + 20)
            np.testing.assert_array_equal(batch.old_log_prob.numpy(), flat_ids + 30)
            np.testing.assert_array_equal(
                batch.role_ids.numpy(), np.array([0, 0, 1, 2])[flat_ids % 4]
            )
            np.testing.assert_allclose(
                batch.advantages.numpy(), expected.ravel()[flat_ids]
            )
        assert sorted(seen) == list(range(16))
    np.testing.assert_array_equal(
        buffer.returns, buffer.swap_and_flatten(original_returns)
    )
    buffer.reset()
    assert buffer.role_ids.shape == (4, 4)
    assert not buffer.roles_normalized


class TinyTask(gym.Env):
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-1, 1, (1,), np.float32)}
    )
    action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        return ({"state": np.zeros(1, np.float32)}, {})

    def step(self, action):
        return ({"state": np.zeros(1, np.float32)}, float(action[0]), True, False, {})


def test_actual_ppo_update_uses_buffer_and_survives_save_load(tmp_path):
    env = DummyVecEnv([TinyTask] * 4)
    try:
        model = RoleNormalizedPPO(
            "MultiInputPolicy",
            env,
            n_steps=4,
            batch_size=8,
            n_epochs=1,
            policy_kwargs={"net_arch": [8]},
            device="cpu",
            seed=2,
        )
        model.learn(16)
        assert model._n_updates == 1
        assert not model.normalize_advantage
        assert model.rollout_buffer.roles_normalized
        model.save(tmp_path / "policy")
        loaded = RoleNormalizedPPO.load(tmp_path / "policy", env=env, device="cpu")
        loaded.learn(16)
        assert isinstance(loaded.rollout_buffer, RoleDictRolloutBuffer)
        assert not loaded.normalize_advantage
    finally:
        env.close()


def test_conversion_preserves_exact_actor_optimizer_and_rollout_hyperparameters(
    tmp_path,
):
    from stable_baselines3 import PPO
    from robovision.role_ppo import load_role_normalized_ppo

    def assert_identical(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                assert_identical(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for left, right in zip(a, b):
                assert_identical(left, right)
        else:
            assert a == b

    env = DummyVecEnv([TinyTask] * 4)
    try:
        original = PPO(
            "MultiInputPolicy",
            env,
            n_steps=8,
            batch_size=8,
            n_epochs=2,
            gamma=0.98,
            gae_lambda=0.91,
            target_kl=0.04,
            learning_rate=0.0001,
            policy_kwargs={"net_arch": [8]},
            seed=17,
        )
        original.learn(32)
        original.save(tmp_path / "ordinary")
        converted = load_role_normalized_ppo(tmp_path / "ordinary", env, "cpu")
        assert isinstance(converted, RoleNormalizedPPO)
        assert_identical(original.policy.state_dict(), converted.policy.state_dict())
        assert_identical(
            original.policy.optimizer.state_dict(),
            converted.policy.optimizer.state_dict(),
        )
        for name in [
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "target_kl",
            "num_timesteps",
            "_n_updates",
        ]:
            assert getattr(original, name) == getattr(converted, name)
        assert converted.rollout_buffer.gamma == 0.98
        assert converted.rollout_buffer.gae_lambda == 0.91
        assert converted.rollout_buffer.buffer_size == 8
        assert converted.role_normalization["role_ids"] == [0, 0, 1, 2]
        assert not converted.normalize_advantage
        converted.learn(32)
        stats = converted.rollout_buffer.raw_role_advantage_stats
        assert {k: v["count"] for k, v in stats.items()} == {"0": 16, "1": 8, "2": 8}
        assert all((np.isfinite(v["mean"]) for v in stats.values()))
    finally:
        env.close()
