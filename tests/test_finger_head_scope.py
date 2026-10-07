import copy
import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from robovision.decoupled_ppo import DecoupledPPO, load_decoupled_ppo
from robovision.context_exploration import (
    ContextExplorationPPO,
    load_context_exploration_ppo,
)


class Task(gym.Env):
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-1, 1, (2,), np.float32)}
    )
    action_space = gym.spaces.Box(-1, 1, (5,), np.float32)

    def reset(self, *, seed=None, options=None):
        return ({"state": np.array([0.1, 0.2], np.float32)}, {})

    def step(self, action):
        return (
            {"state": np.array([0.1, 0.2], np.float32)},
            float(3 + action.sum()),
            True,
            False,
            {},
        )


def configuration():
    return dict(
        n_steps=4,
        batch_size=4,
        n_epochs=2,
        seed=4,
        device="cpu",
        target_kl=None,
        policy_kwargs=dict(net_arch=[8], share_features_extractor=False),
    )


def snapshot(model):
    return {
        n: (p.detach().clone(), copy.deepcopy(model.policy.optimizer.state.get(p, {})))
        for n, p in model.policy.named_parameters()
        if id(p) in {id(q) for q in model.actor_parameters}
    }


def assert_frozen(model, before):
    changed = False
    for name, p in model.policy.named_parameters():
        if name not in before:
            continue
        old, state = before[name]
        now = model.policy.optimizer.state.get(p, {})
        if name.startswith("action_net."):
            assert torch.equal(old[:-1], p[:-1])
            changed |= not torch.equal(old[-1], p[-1])
            for key, value in state.items():
                if torch.is_tensor(value) and value.shape == p.shape:
                    assert torch.equal(value[:-1], now[key][:-1])
                elif key == "step":
                    assert now[key] > value
                else:
                    assert torch.equal(value, now[key])
        else:
            assert torch.equal(old, p), name
            assert state.keys() == now.keys()
            for key, value in state.items():
                assert torch.equal(value, now[key]), (name, key)
    assert changed, "Actual finger update required"


@pytest.mark.parametrize("contextual", [False, True])
def test_finger_training_freezes_actor_arm_rows_and_inactive_adam_histories(contextual):
    env = DummyVecEnv([Task] * 4)
    try:
        cls = ContextExplorationPPO if contextual else DecoupledPPO
        model = cls("MultiInputPolicy", env, **configuration())
        model.learn(32)
        before = snapshot(model)
        critic = [p.detach().clone() for p in model.critic_parameters]
        flags = [p.requires_grad for p in model.actor_parameters]
        obs = model.policy.obs_to_tensor(env.reset())[0]
        means = model.policy.get_distribution(obs).distribution.mean.detach().clone()
        model.set_actor_update_scope("finger_head")
        model.learn(32)
        assert_frozen(model, before)
        assert flags == [p.requires_grad for p in model.actor_parameters]
        after = model.policy.get_distribution(obs).distribution.mean.detach()
        assert torch.equal(means[:, :4], after[:, :4])
        assert not torch.equal(means[:, 4], after[:, 4])
        assert any(
            (not torch.equal(a, b) for a, b in zip(critic, model.critic_parameters))
        )
    finally:
        env.close()


def test_scope_clips_only_finger_gradients_and_restores_flags_on_exception():
    env = DummyVecEnv([Task] * 4)
    try:
        model = DecoupledPPO("MultiInputPolicy", env, **configuration())
        model.set_actor_update_scope("finger_head")
        model.actor_parameters[0].requires_grad_(False)
        flags = [p.requires_grad for p in model.actor_parameters]
        with model._actor_scope():
            model.policy.optimizer.zero_grad()
            coefficients = torch.ones_like(model.policy.action_net.weight)
            coefficients[:-1] = 1000000000.0
            (model.policy.action_net.weight * coefficients).sum().backward()
            assert torch.count_nonzero(model.policy.action_net.weight.grad[:-1]) == 0
            model._step_actor()
            assert (
                model.policy.action_net.weight.grad.norm()
                <= model.max_grad_norm + 1e-06
            )
            assert model.policy.action_net.weight.grad[-1].norm() > 0.1
        with pytest.raises(RuntimeError, match="intentional"):
            with model._actor_scope():
                raise RuntimeError("intentional")
        assert flags == [p.requires_grad for p in model.actor_parameters]
    finally:
        env.close()


def test_context_checkpoint_inherits_scope_requires_explicit_change_and_restores_joint(
    tmp_path,
):
    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO("MultiInputPolicy", env, **configuration())
        model.learn(16)
        model.set_actor_update_scope("finger_head")
        model.learn(16)
        model.save(tmp_path / "finger")
        loaded = load_context_exploration_ppo(tmp_path / "finger", env, "cpu")
        assert loaded.actor_update_scope == "finger_head"
        assert loaded.algorithm_metadata["actor_update_scope"] == "finger_head"
        old = snapshot(loaded)
        loaded.learn(16)
        assert_frozen(loaded, old)
        with pytest.raises(ValueError, match="allow_context_reconfigure"):
            load_context_exploration_ppo(
                tmp_path / "finger", env, "cpu", actor_update_scope="all"
            )
        joint = load_context_exploration_ppo(
            tmp_path / "finger",
            env,
            "cpu",
            actor_update_scope="all",
            allow_context_reconfigure=True,
        )
        assert joint.actor_update_scope == "all"
        assert joint.context_reconfiguration["old_actor_update_scope"] == "finger_head"
        arms = joint.policy.action_net.weight[:4].detach().clone()
        old_std = joint.policy.log_std.detach().clone()
        joint.learn(32)
        assert not torch.equal(arms, joint.policy.action_net.weight[:4])
        assert not torch.equal(old_std, joint.policy.log_std)
    finally:
        env.close()


def test_decoupled_scope_resume_and_guard(tmp_path):
    env = DummyVecEnv([Task] * 4)
    try:
        model = DecoupledPPO(
            "MultiInputPolicy", env, actor_update_scope="finger_head", **configuration()
        )
        model.learn(16)
        model.save(tmp_path / "finger")
        resumed = load_decoupled_ppo(tmp_path / "finger", env, "cpu")
        assert resumed.actor_update_scope == "finger_head"
        with pytest.raises(ValueError, match="allow_actor_reconfigure"):
            load_decoupled_ppo(
                tmp_path / "finger", env, "cpu", actor_update_scope="all"
            )
        joint = load_decoupled_ppo(
            tmp_path / "finger",
            env,
            "cpu",
            actor_update_scope="all",
            allow_actor_reconfigure=True,
        )
        assert joint.actor_update_scope == "all"
    finally:
        env.close()


def test_finger_scope_with_role_balancing_uses_only_active_gradients():
    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO(
            "MultiInputPolicy", env, role_gradient_balance=True, **configuration()
        )
        model.learn(16)
        before = snapshot(model)
        model.set_actor_update_scope("finger_head")
        model.learn(16)
        assert_frozen(model, before)
        assert model.role_gradient_stats
    finally:
        env.close()


def test_checkpoint_without_new_scope_field_defaults_to_all(tmp_path):
    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO("MultiInputPolicy", env, **configuration())
        model.learn(16)
        del model.actor_update_scope
        model.algorithm_metadata.pop("actor_update_scope", None)
        model.save(tmp_path / "legacy")
        loaded = load_context_exploration_ppo(tmp_path / "legacy", env, "cpu")
        assert loaded.actor_update_scope == "all"
        loaded.learn(16)
    finally:
        env.close()
