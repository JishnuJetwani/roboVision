import copy
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from robovision.context_exploration import (
    ContextExplorationPPO,
    load_context_exploration_ppo,
)
from test_finger_context_exploration import Task, configuration, observation


def optimizer_states(model):
    return {
        n: copy.deepcopy(model.policy.optimizer.state.get(p, {}))
        for n, p in model.policy.named_parameters()
    }


def test_enable_zero_identity_rng_and_optimizer_groups(tmp_path):
    env = DummyVecEnv([Task] * 4)
    try:
        original = ContextExplorationPPO("MultiInputPolicy", env, **configuration())
        original.learn(16)
        original.save(tmp_path / "old")
        obs = env.reset()
        before = original.predict(obs, deterministic=True)[0]
        with pytest.raises(ValueError, match="allow_context_reconfigure"):
            load_context_exploration_ppo(
                tmp_path / "old",
                env,
                "cpu",
                wide_finger_residual=True,
                actor_update_scope="wide_finger",
            )
        model = load_context_exploration_ppo(
            tmp_path / "old",
            env,
            "cpu",
            wide_finger_residual=True,
            actor_update_scope="wide_finger",
            allow_context_reconfigure=True,
        )
        np.testing.assert_array_equal(before, model.predict(obs, deterministic=True)[0])
        for k, v in original.policy.state_dict().items():
            assert torch.equal(v, model.policy.state_dict()[k])
        for left, right in [
            (original.policy.optimizer, model.policy.optimizer),
            (original.critic_optimizer, model.critic_optimizer),
        ]:
            a, b = (left.state_dict()["state"], right.state_dict()["state"])
            for k in a:
                for field in a[k]:
                    assert torch.equal(a[k][field], b[k][field])
        assert len(model.policy.optimizer.param_groups) == 2
        plain = ContextExplorationPPO("MultiInputPolicy", env, **configuration())
        rng = torch.get_rng_state().clone()
        plain.policy.enable_wide_finger_residual()
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        env.close()


def test_wide_training_exactly_freezes_original_actor_and_adam_and_quiet_means(
    tmp_path,
):
    env = DummyVecEnv([Task] * 4)
    try:
        source = ContextExplorationPPO("MultiInputPolicy", env, **configuration())
        source.learn(16)
        source.save(tmp_path / "source")
        model = load_context_exploration_ppo(
            tmp_path / "source",
            env,
            "cpu",
            wide_finger_residual=True,
            actor_update_scope="wide_finger",
            allow_context_reconfigure=True,
        )
        actor_ids = {id(p) for p in model.actor_parameters}
        before = {
            n: p.detach().clone()
            for n, p in model.policy.named_parameters()
            if id(p) in actor_ids and (not n.startswith("wide_finger_head."))
        }
        states = optimizer_states(model)
        quiet = observation([0.03] * 4)
        wide = observation([0.045] * 4)
        q0 = model.policy.get_distribution(quiet).distribution.mean.detach().clone()
        w0 = model.policy.get_distribution(wide).distribution.mean.detach().clone()
        model.learn(32)
        for n, p in model.policy.named_parameters():
            if n in before:
                assert torch.equal(before[n], p), n
                for key, v in states[n].items():
                    assert torch.equal(v, model.policy.optimizer.state[p][key]), (
                        n,
                        key,
                    )
        q1 = model.policy.get_distribution(quiet).distribution.mean.detach()
        w1 = model.policy.get_distribution(wide).distribution.mean.detach()
        assert torch.equal(q0, q1)
        assert torch.equal(w0[:, :4], w1[:, :4])
        assert not torch.equal(w0[:, 4], w1[:, 4])
        model.save(tmp_path / "residual")
        resumed = load_context_exploration_ppo(tmp_path / "residual", env, "cpu")
        assert resumed.actor_update_scope == "wide_finger"
        assert resumed.wide_finger_residual and resumed.policy.wide_finger_residual
        assert len(resumed.policy.optimizer.param_groups) == 2
        for k, v in model.policy.state_dict().items():
            assert torch.equal(v, resumed.policy.state_dict()[k])
        for k, v in model.policy.optimizer.state_dict()["state"].items():
            for field, a in v.items():
                assert torch.equal(
                    a, resumed.policy.optimizer.state_dict()["state"][k][field]
                )
        resumed.learn(16)
        with pytest.raises(ValueError, match="Cannot disable"):
            load_context_exploration_ppo(
                tmp_path / "residual",
                env,
                "cpu",
                wide_finger_residual=False,
                allow_context_reconfigure=True,
            )
        joint = load_context_exploration_ppo(
            tmp_path / "residual",
            env,
            "cpu",
            actor_update_scope="all",
            allow_context_reconfigure=True,
        )
        arms = joint.policy.action_net.weight[:4].detach().clone()
        joint.learn(32)
        assert not torch.equal(arms, joint.policy.action_net.weight[:4])
    finally:
        env.close()


def test_enabled_prediction_evaluation_and_stored_likelihood_share_one_forward():
    env = DummyVecEnv([Task] * 4)
    try:
        model = ContextExplorationPPO(
            "MultiInputPolicy",
            env,
            wide_finger_residual=True,
            actor_update_scope="wide_finger",
            finger_exploration_max=(1, 1, 1, 100),
            **configuration(),
        )
        with torch.no_grad():
            model.policy.wide_finger_head.bias.fill_(0.03)
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
        calls = []
        hook = model.policy.pi_features_extractor.register_forward_hook(
            lambda *args: calls.append(1)
        )
        model.policy.get_distribution(observation([0.045] * 4))
        hook.remove()
        assert len(calls) == 1
    finally:
        env.close()
