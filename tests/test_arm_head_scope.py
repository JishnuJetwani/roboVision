"""Arm-only updates preserve learned finger means, including the gated residual."""

import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from robovision.decoupled_ppo import DecoupledPPO, load_decoupled_ppo
from robovision.context_exploration import (
    ContextExplorationPPO,
    load_context_exploration_ppo,
)
from test_finger_context_exploration import Task, configuration, observation
from test_finger_head_scope import snapshot


def assert_arm_update(model, before):
    changed = False
    for name, parameter in model.policy.named_parameters():
        if name not in before:
            continue
        old, state = before[name]
        current = model.policy.optimizer.state.get(parameter, {})
        assert state.keys() == current.keys()
        if name.startswith("action_net."):
            assert torch.equal(old[-1:], parameter[-1:])
            changed |= not torch.equal(old[:4], parameter[:4])
            for key, value in state.items():
                if value.shape == parameter.shape:
                    assert torch.equal(value[-1:], current[key][-1:])
                elif key == "step":
                    assert current[key] > value
                else:
                    assert torch.equal(value, current[key])
        else:
            assert torch.equal(old, parameter), name
            for key, value in state.items():
                assert torch.equal(value, current[key]), (name, key)
    assert changed, "At least one arm row must actually update"


@pytest.mark.parametrize("contextual", [False, True])
def test_actual_arm_learning_preserves_finger_function_and_optimizer_histories(
    contextual,
):
    env = DummyVecEnv([Task] * 4)
    try:
        cls = ContextExplorationPPO if contextual else DecoupledPPO
        options = {"wide_finger_residual": True} if contextual else {}
        model = cls("MultiInputPolicy", env, **options, **configuration())
        model.learn(32)
        before = snapshot(model)
        inputs = observation([0.03, 0.034, 0.038, 0.045])
        means = model.policy.get_distribution(inputs).distribution.mean.detach().clone()
        critic = [p.detach().clone() for p in model.critic_parameters]
        flags = [p.requires_grad for p in model.actor_parameters]
        model.set_actor_update_scope("arm_head")
        model.learn(32)
        assert_arm_update(model, before)
        after = model.policy.get_distribution(inputs).distribution.mean.detach()
        assert torch.equal(means[:, 4], after[:, 4])
        assert not torch.equal(means[:, :4], after[:, :4])
        assert flags == [p.requires_grad for p in model.actor_parameters]
        assert any(
            (not torch.equal(a, b) for a, b in zip(critic, model.critic_parameters))
        )
    finally:
        env.close()


def test_finger_gradient_mask_precedes_clipping_and_flags_restore_after_exception():
    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO(
            "MultiInputPolicy",
            env,
            wide_finger_residual=True,
            actor_update_scope="arm_head",
            **configuration(),
        )
        model.actor_parameters[0].requires_grad_(False)
        flags = [p.requires_grad for p in model.actor_parameters]
        with model._actor_scope():
            assert all(
                (
                    not p.requires_grad
                    for p in model.policy.wide_finger_head.parameters()
                )
            )
            model.policy.optimizer.zero_grad()
            coefficients = torch.ones_like(model.policy.action_net.weight)
            coefficients[-1] = 1000000000.0
            (model.policy.action_net.weight * coefficients).sum().backward()
            assert torch.count_nonzero(model.policy.action_net.weight.grad[-1]) == 0
            model._step_actor()
            gradient = model.policy.action_net.weight.grad
            assert 0.1 < gradient[:4].norm() <= model.max_grad_norm + 1e-06
        with pytest.raises(RuntimeError, match="intentional"):
            with model._actor_scope():
                raise RuntimeError("intentional")
        assert flags == [p.requires_grad for p in model.actor_parameters]
    finally:
        env.close()


@pytest.mark.parametrize("contextual", [False, True])
def test_scope_resume_preserves_histories_and_explicit_joint_reconfigure(
    tmp_path, contextual
):
    env = DummyVecEnv([Task] * 4)
    try:
        cls = ContextExplorationPPO if contextual else DecoupledPPO
        loader = load_context_exploration_ppo if contextual else load_decoupled_ppo
        permission = (
            "allow_context_reconfigure" if contextual else "allow_actor_reconfigure"
        )
        options = {"wide_finger_residual": True} if contextual else {}
        model = cls("MultiInputPolicy", env, **options, **configuration())
        model.learn(32)
        model.set_actor_update_scope("arm_head")
        model.learn(16)
        model.save(tmp_path / "arms")
        resumed = loader(tmp_path / "arms", env, "cpu")
        assert resumed.actor_update_scope == "arm_head"
        assert resumed.algorithm_metadata["actor_update_scope"] == "arm_head"
        original = snapshot(model)
        for name, (weight, state) in snapshot(resumed).items():
            assert torch.equal(weight, original[name][0])
            for key, value in state.items():
                assert torch.equal(value, original[name][1][key])
        before = snapshot(resumed)
        resumed.learn(16)
        assert_arm_update(resumed, before)
        with pytest.raises(ValueError, match=permission):
            loader(tmp_path / "arms", env, "cpu", actor_update_scope="all")
        joint = loader(
            tmp_path / "arms",
            env,
            "cpu",
            actor_update_scope="all",
            **{permission: True},
        )
        finger = joint.policy.action_net.weight[-1].detach().clone()
        joint.learn(32)
        assert not torch.equal(finger, joint.policy.action_net.weight[-1])
    finally:
        env.close()
