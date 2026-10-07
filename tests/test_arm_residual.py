"""Learned arm corrections preserve the original CNN policy and Adam histories."""

import copy
import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from robovision.cnn import NormalizedGraspCNN
from robovision.context_exploration import (
    ContextExplorationPPO,
    load_context_exploration_ppo,
)
from test_finger_context_exploration import Task, configuration, observation
from robovision.policy_state import tree_hash


class VisualTask(Task):
    observation_space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(0, 255, (6, 96, 96), np.uint8),
            "proprio": Task.observation_space["proprio"],
        }
    )

    def obs(self):
        return {
            **super().obs(),
            "image": np.full((6, 96, 96), self.step_count, np.uint8),
        }


def real_configuration():
    return {
        **configuration(),
        "policy_kwargs": dict(
            features_extractor_class=NormalizedGraspCNN,
            share_features_extractor=False,
            net_arch=[128, 128],
        ),
    }


def resume(path, env, **kwargs):
    return load_context_exploration_ppo(
        path, env, "cpu", allow_context_reconfigure=True, **kwargs
    )


@pytest.mark.parametrize("mode", ["wide", "always"])
def test_real_cnn_zero_identity_rng_and_exact_optimizer_migration(tmp_path, mode):
    env = DummyVecEnv([VisualTask] * 4)
    try:
        source = ContextExplorationPPO(
            "MultiInputPolicy", env, wide_finger_residual=True, **real_configuration()
        )
        source.learn(16)
        source.save(tmp_path / "source")
        obs = env.reset()
        before = source.predict(obs, deterministic=True)[0]
        model = resume(
            tmp_path / "source",
            env,
            arm_residual_mode=mode,
            actor_update_scope="arm_residual",
        )
        np.testing.assert_array_equal(before, model.predict(obs, deterministic=True)[0])
        for k, v in source.policy.state_dict().items():
            assert torch.equal(v, model.policy.state_dict()[k]), k
        assert tree_hash(source.policy.optimizer.state_dict()["state"]) == tree_hash(
            model.policy.optimizer.state_dict()["state"]
        )
        assert tree_hash(source.critic_optimizer.state_dict()) == tree_hash(
            model.critic_optimizer.state_dict()
        )
        assert len(model.policy.optimizer.param_groups) == 3
        assert all(
            (
                p not in model.policy.optimizer.state
                for p in model.policy.arm_residual_head.parameters()
            )
        )
        rng = torch.get_rng_state().clone()
        source.policy.enable_arm_residual(mode)
        assert torch.equal(rng, torch.get_rng_state())
        calls = []
        hook = model.policy.pi_features_extractor.register_forward_hook(
            lambda *a: calls.append(1)
        )
        model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
        hook.remove()
        assert len(calls) == 1
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["wide", "always"])
@pytest.mark.parametrize("visual", [False, True])
def test_updates_only_new_head_freeze_old_moments_finger_and_closed_gate(
    tmp_path, mode, visual
):
    env = DummyVecEnv([VisualTask if visual else Task] * 4)
    try:
        source = ContextExplorationPPO(
            "MultiInputPolicy",
            env,
            wide_finger_residual=True,
            **real_configuration() if visual else configuration(),
        )
        source.learn(32)
        source.save(tmp_path / "source")
        model = resume(
            tmp_path / "source",
            env,
            arm_residual_mode=mode,
            actor_update_scope="arm_residual",
        )
        old = {
            n: p
            for n, p in model.policy.named_parameters()
            if id(p) in {id(x) for x in model.actor_parameters}
            and (not n.startswith("arm_residual_head."))
        }
        weights = {n: p.detach().clone() for n, p in old.items()}
        states = {
            n: copy.deepcopy(model.policy.optimizer.state.get(p, {}))
            for n, p in old.items()
        }
        critic = tree_hash(model.critic_optimizer.state_dict())
        flags = [p.requires_grad for p in model.actor_parameters]
        quiet = observation([0.03] * 4)
        wide = observation([0.045] * 4)
        if visual:
            quiet["image"] = torch.zeros((4, 6, 96, 96), dtype=torch.uint8)
            wide["image"] = quiet["image"].clone()
        q0 = model.policy.get_distribution(quiet).distribution.mean.detach().clone()
        w0 = model.policy.get_distribution(wide).distribution.mean.detach().clone()
        model.learn(32)
        assert flags == [p.requires_grad for p in model.actor_parameters]
        for n, p in old.items():
            assert torch.equal(weights[n], p), n
            assert tree_hash(states[n]) == tree_hash(
                model.policy.optimizer.state.get(p, {})
            ), n
        assert critic != tree_hash(model.critic_optimizer.state_dict())
        q1 = model.policy.get_distribution(quiet).distribution.mean.detach().clone()
        w1 = model.policy.get_distribution(wide).distribution.mean.detach().clone()
        assert torch.equal(q0[:, 4], q1[:, 4]) and torch.equal(w0[:, 4], w1[:, 4])
        assert not torch.equal(w0[:, :4], w1[:, :4])
        if mode == "wide":
            assert torch.equal(q0, q1)
        else:
            assert not torch.equal(q0[:, :4], q1[:, :4])
        model.save(tmp_path / "arm")
        restored = load_context_exploration_ppo(tmp_path / "arm", env, "cpu")
        assert restored.arm_residual_mode == restored.policy.arm_residual_mode == mode
        assert restored.actor_update_scope == "arm_residual"
        assert tree_hash(restored.policy.state_dict()) == tree_hash(
            model.policy.state_dict()
        )
        assert tree_hash(restored.policy.optimizer.state_dict()) == tree_hash(
            model.policy.optimizer.state_dict()
        )
        assert tree_hash(restored.critic_optimizer.state_dict()) == tree_hash(
            model.critic_optimizer.state_dict()
        )
        restored.learn(16)
        joint = resume(tmp_path / "arm", env, actor_update_scope="all")
        before = joint.policy.action_net.weight.detach().clone()
        joint.learn(32)
        assert not torch.equal(before, joint.policy.action_net.weight)
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["wide", "always"])
def test_sampling_stored_likelihood_evaluation_prediction_share_residual(mode):
    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO(
            "MultiInputPolicy",
            env,
            arm_residual_mode=mode,
            actor_update_scope="arm_residual",
            finger_exploration_max=(1, 1, 1, 100),
            **configuration(),
        )
        with torch.no_grad():
            model.policy.arm_residual_head.bias.fill_(0.03)
        _, callback = model._setup_learn(16)
        model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=4)
        for batch in model.rollout_buffer.get(5):
            actual = model._actor_distribution(batch)
            torch.testing.assert_close(
                actual.log_prob(batch.actions),
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
            torch.testing.assert_close(
                model.policy._predict(batch.observations, deterministic=True),
                base.distribution.mean,
            )
    finally:
        env.close()


def test_explicit_reconfiguration_single_head_and_reverse_enable_group_order(tmp_path):
    env = DummyVecEnv([Task] * 4)
    try:
        source = ContextExplorationPPO("MultiInputPolicy", env, **configuration())
        source.learn(16)
        source.save(tmp_path / "old")
        with pytest.raises(ValueError, match="allow_context_reconfigure"):
            load_context_exploration_ppo(
                tmp_path / "old", env, "cpu", arm_residual_mode="wide"
            )
        with pytest.raises(ValueError, match="requires"):
            source.set_actor_update_scope("arm_residual")
        with pytest.raises(ValueError, match="arm_residual_mode"):
            resume(tmp_path / "old", env, arm_residual_mode="off")
        arm = resume(
            tmp_path / "old",
            env,
            arm_residual_mode="wide",
            actor_update_scope="arm_residual",
        )
        arm.learn(16)
        arm.save(tmp_path / "arm")
        with pytest.raises(ValueError, match="allow_context_reconfigure"):
            load_context_exploration_ppo(
                tmp_path / "arm", env, "cpu", arm_residual_mode="always"
            )
        both = resume(
            tmp_path / "arm", env, wide_finger_residual=True, arm_residual_mode="always"
        )
        assert len(both.policy.optimizer.param_groups) == 3
        assert torch.equal(
            arm.policy.arm_residual_head.weight, both.policy.arm_residual_head.weight
        )
        both.save(tmp_path / "both")
        loaded = load_context_exploration_ppo(tmp_path / "both", env, "cpu")
        assert tree_hash(both.policy.optimizer.state_dict()) == tree_hash(
            loaded.policy.optimizer.state_dict()
        )
        assert loaded.arm_residual_mode == "always"
        assert loaded.algorithm_metadata["arm_residual_mode"] == "always"
        head = loaded.policy.arm_residual_head
        loaded.policy.enable_arm_residual("wide")
        assert head is loaded.policy.arm_residual_head
    finally:
        env.close()


def test_checkpoint_without_new_metadata_loads_disabled(tmp_path):
    env = DummyVecEnv([Task] * 4)
    try:
        old = ContextExplorationPPO(
            "MultiInputPolicy", env, wide_finger_residual=True, **configuration()
        )
        old.learn(16)
        del old.arm_residual_mode
        old.policy_kwargs.pop("arm_residual_mode")
        old.algorithm_metadata.pop("arm_residual_mode")
        old.save(tmp_path / "legacy")
        loaded = load_context_exploration_ppo(tmp_path / "legacy", env, "cpu")
        assert (
            loaded.arm_residual_mode is None and loaded.policy.arm_residual_head is None
        )
        assert tree_hash(old.policy.state_dict()) == tree_hash(
            loaded.policy.state_dict()
        )
        assert tree_hash(old.policy.optimizer.state_dict()) == tree_hash(
            loaded.policy.optimizer.state_dict()
        )
    finally:
        env.close()
