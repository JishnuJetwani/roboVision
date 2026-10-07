import numpy as np
import pytest
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from robovision.context_exploration import ContextExplorationPPO
from robovision.structured_exploration import convert_to_structured, StructuredPPO
from robovision.train_hierarchical import SkillRecipe, initialize_skill


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


@pytest.mark.parametrize(
    "profile,reward",
    [
        ("nominal", "original"),
        ("xy-half", "original"),
        ("xy-half-high", "original"),
        ("xy-half-high", "clearance-guard-v1"),
    ],
)
def test_structured_successor_preserves_learned_means_and_noise_but_starts_fresh_adam(
    tmp_path, profile, reward
):
    torch.set_num_threads(1)
    env = DummyVecEnv([Toy] * 4)
    try:
        source = ContextExplorationPPO(
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
        source = convert_to_structured(source)
        source.learn(512)
        assert source.policy.optimizer.state and source.critic_optimizer.state
        source.save(tmp_path / "source.zip")
        expected = {k: v.clone() for k, v in source.policy.state_dict().items()}
        recipe = SkillRecipe(
            "approach",
            centered_approach=True,
            structured=True,
            noise_scale=1.0,
            learning_rate=1e-06,
            approach_reward=reward,
        )
        successor, report = initialize_skill(
            tmp_path / "source.zip", env, recipe, 317500, device="cpu"
        )
        assert isinstance(successor, StructuredPPO)
        assert successor.num_timesteps == 512 and successor.seed == 317500
        assert successor.n_steps == successor.rollout_buffer.buffer_size == 512
        assert successor.sde_sample_freq == 25
        assert successor.actor_update_scope == "all"
        assert not successor.policy.optimizer.state and (
            not successor.critic_optimizer.state
        )
        assert successor._last_obs is None and successor.rollout_buffer.pos == 0
        assert report["inherited_structured"]
        for name, value in successor.policy.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        successor.learn(2048, reset_num_timesteps=False)
        assert successor.num_timesteps == 2560
        assert successor.last_update_stats["actor_updates"] > 0
        assert any(
            (
                not torch.equal(v, expected[k])
                for k, v in successor.policy.state_dict().items()
            )
        )
        with pytest.raises(ValueError, match="explicitly structured"):
            initialize_skill(
                tmp_path / "source.zip",
                env,
                SkillRecipe("approach"),
                317501,
                device="cpu",
            )
    finally:
        env.close()
