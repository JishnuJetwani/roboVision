"""Joint residual PPO changes both learned corrections, never the base actor."""

import copy
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from robovision.context_exploration import (
    ContextExplorationPPO,
    load_context_exploration_ppo,
)
from robovision.policy_state import tree_hash
from test_arm_residual import VisualTask, real_configuration
from test_finger_context_exploration import Task, configuration, observation


@pytest.mark.parametrize("visual", [False, True])
def test_both_heads_update_base_and_moments_frozen_quiet_means_resume(tmp_path, visual):
    env = DummyVecEnv([VisualTask if visual else Task] * 4)
    try:
        source = ContextExplorationPPO(
            "MultiInputPolicy",
            env,
            wide_finger_residual=True,
            arm_residual_mode="wide",
            **real_configuration() if visual else configuration(),
        )
        source.learn(32)
        source.save(tmp_path / "source")
        with pytest.raises(ValueError, match="allow_context_reconfigure"):
            load_context_exploration_ppo(
                tmp_path / "source", env, "cpu", actor_update_scope="grasp_residual"
            )
        model = load_context_exploration_ppo(
            tmp_path / "source",
            env,
            "cpu",
            actor_update_scope="grasp_residual",
            allow_context_reconfigure=True,
        )
        assert tree_hash(source.policy.state_dict()) == tree_hash(
            model.policy.state_dict()
        )
        assert tree_hash(source.policy.optimizer.state_dict()) == tree_hash(
            model.policy.optimizer.state_dict()
        )
        assert tree_hash(source.critic_optimizer.state_dict()) == tree_hash(
            model.critic_optimizer.state_dict()
        )
        head_ids = {
            id(p)
            for head in (model.policy.arm_residual_head, model.policy.wide_finger_head)
            for p in head.parameters()
        }
        assert {id(p) for p in model._active_actor_parameters()} == head_ids
        actor_ids = {id(p) for p in model.actor_parameters}
        base = {
            n: p
            for n, p in model.policy.named_parameters()
            if id(p) in actor_ids - head_ids
        }
        weights = {n: p.detach().clone() for n, p in base.items()}
        states = {
            n: copy.deepcopy(model.policy.optimizer.state[p]) for n, p in base.items()
        }
        heads = {
            name: tree_hash(getattr(model.policy, name).state_dict())
            for name in ("arm_residual_head", "wide_finger_head")
        }
        old_steps = {
            id(p): model.policy.optimizer.state[p]["step"].item()
            for p in model._active_actor_parameters()
        }
        flags = [p.requires_grad for p in model.actor_parameters]
        critic = tree_hash(model.critic_optimizer.state_dict())
        quiet, wide = (observation([0.03] * 4), observation([0.045] * 4))
        if visual:
            for obs in (quiet, wide):
                obs["image"] = torch.zeros((4, 6, 96, 96), dtype=torch.uint8)
        q0 = model.policy.get_distribution(quiet).distribution.mean.detach().clone()
        w0 = model.policy.get_distribution(wide).distribution.mean.detach().clone()
        model.learn(32)
        assert flags == [p.requires_grad for p in model.actor_parameters]
        for n, p in base.items():
            assert torch.equal(weights[n], p), n
            assert tree_hash(states[n]) == tree_hash(model.policy.optimizer.state[p]), n
        for name in heads:
            assert heads[name] != tree_hash(getattr(model.policy, name).state_dict())
        for p in model._active_actor_parameters():
            assert model.policy.optimizer.state[p]["step"].item() > old_steps[id(p)]
        assert critic != tree_hash(model.critic_optimizer.state_dict())
        q1 = model.policy.get_distribution(quiet).distribution.mean.detach().clone()
        w1 = model.policy.get_distribution(wide).distribution.mean.detach().clone()
        assert torch.equal(q0, q1)
        assert not torch.equal(w0[:, :4], w1[:, :4])
        assert not torch.equal(w0[:, 4], w1[:, 4])
        model.save(tmp_path / "grasp")
        loaded = load_context_exploration_ppo(tmp_path / "grasp", env, "cpu")
        assert loaded.actor_update_scope == "grasp_residual"
        assert tree_hash(loaded.policy.state_dict()) == tree_hash(
            model.policy.state_dict()
        )
        assert tree_hash(loaded.policy.optimizer.state_dict()) == tree_hash(
            model.policy.optimizer.state_dict()
        )
        assert tree_hash(loaded.critic_optimizer.state_dict()) == tree_hash(
            model.critic_optimizer.state_dict()
        )
        loaded.learn(16)
    finally:
        env.close()


@pytest.mark.parametrize("arm,finger", [(None, False), ("wide", False), (None, True)])
def test_requires_both_existing_heads(arm, finger):
    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO(
            "MultiInputPolicy",
            env,
            arm_residual_mode=arm,
            wide_finger_residual=finger,
            **configuration(),
        )
        with pytest.raises(ValueError, match="requires both"):
            model.set_actor_update_scope("grasp_residual")
        assert model.actor_update_scope == "all"
    finally:
        env.close()


def test_both_residuals_sample_and_recompute_same_likelihood_and_restore_flags():
    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO(
            "MultiInputPolicy",
            env,
            arm_residual_mode="wide",
            wide_finger_residual=True,
            actor_update_scope="grasp_residual",
            finger_exploration_max=(1, 1, 1, 100),
            **configuration(),
        )
        with torch.no_grad():
            model.policy.arm_residual_head.bias.fill_(0.03)
            model.policy.wide_finger_head.bias.fill_(-0.02)
        _, callback = model._setup_learn(16)
        model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=4)
        for batch in model.rollout_buffer.get(5):
            distribution = model._actor_distribution(batch)
            torch.testing.assert_close(
                distribution.log_prob(batch.actions),
                batch.old_log_prob,
                atol=1e-05,
                rtol=1e-05,
            )
            base = model.policy.get_distribution(batch.observations)
            _, logs, entropy = model.policy.evaluate_actions(
                batch.observations, batch.actions
            )
            torch.testing.assert_close(logs, base.log_prob(batch.actions))
            torch.testing.assert_close(entropy, base.entropy())
        model.policy.wide_finger_head.bias.requires_grad_(False)
        flags = [p.requires_grad for p in model.actor_parameters]
        with pytest.raises(RuntimeError, match="test exception"):
            with model._actor_scope():
                assert not model.policy.log_std.requires_grad
                assert model.policy.arm_residual_head.weight.requires_grad
                assert not model.policy.wide_finger_head.bias.requires_grad
                raise RuntimeError("test exception")
        assert flags == [p.requires_grad for p in model.actor_parameters]
    finally:
        env.close()
