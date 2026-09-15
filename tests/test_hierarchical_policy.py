"""Physical-return equivalence, frozen experts and deployable manager persistence."""

from types import SimpleNamespace
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from robovision.hierarchical_policy import (
    HierarchicalPolicy,
    OptionsEnv,
    assert_parameter_independence,
    freeze_experts,
)
from robovision.policy_state import policy_state_hash


def observation_space():
    return spaces.Dict(
        dict(
            image=spaces.Box(0, 255, (6, 96, 96), np.uint8),
            proprio=spaces.Box(-np.inf, np.inf, (18,), np.float32),
        )
    )


def observation(step=0):
    proprio = np.zeros(18, np.float32)
    proprio[0] = step
    return dict(image=np.full((6, 96, 96), step % 256, np.uint8), proprio=proprio)


class FakeExpert:
    def __init__(self, index=0):
        self.index = index
        self.policy = torch.nn.Linear(1, 1)
        self.observation_space = observation_space()
        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.calls = []
        self.return_state = None

    def predict(self, obs, deterministic=True):
        step = float(obs["proprio"][0])
        self.calls.append(
            dict(
                step=step,
                deterministic=deterministic,
                training=self.policy.training,
                grad_enabled=torch.is_grad_enabled(),
            )
        )
        return (
            np.array([-0.6, -0.3, 0.0, 0.3, 0.6], np.float32)
            + self.index * 0.05
            + step * 0.0001,
            self.return_state,
        )


def experts():
    return tuple((FakeExpert(index) for index in range(3)))


class FakeManager:
    def __init__(self, choices=(2, 0, 1)):
        self.policy = torch.nn.Linear(1, 1)
        self.observation_space = observation_space()
        self.action_space = spaces.Discrete(3)
        self.choices = choices
        self.calls = []
        self.return_state = None

    def predict(self, obs, deterministic=True):
        index = len(self.calls)
        self.calls.append((float(obs["proprio"][0]), deterministic))
        return (np.asarray(self.choices[index % len(self.choices)]), self.return_state)


class PhysicalEnv(gym.Env):
    control_dt = 0.02

    def __init__(self, horizon=13, gamma=0.9, truncated=False, both=False):
        self.gamma = gamma
        self.horizon = horizon
        self.truncation = truncated
        self.both = both
        self.observation_space = observation_space()
        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return (observation(), dict(physical_reset=True))

    def step(self, action):
        self.actions.append(np.array(action).copy())
        self.step_count += 1
        done = self.step_count == self.horizon
        return (
            observation(self.step_count),
            float(self.step_count),
            bool(done and (not self.truncation or self.both)),
            bool(done and self.truncation),
            dict(
                physical_step=self.step_count,
                is_success=done,
                reason="success" if done else "",
            ),
        )


def test_options_discounted_return_exactly_matches_every_physical_reward_and_terminal_break():
    base = PhysicalEnv(horizon=13, gamma=0.9)
    children = experts()
    env = OptionsEnv(base, children)
    obs, info = env.reset(seed=1)
    assert env.action_space == spaces.Discrete(3)
    assert env.observation_space is base.observation_space
    assert env.manager_gamma == pytest.approx(0.9**5)
    assert env.control_dt == pytest.approx(0.1)
    option_rewards = []
    for index, count in enumerate((5, 5, 3)):
        obs, reward, terminated, truncated, info = env.step(index)
        option_rewards.append(reward)
        assert info["option_steps_executed"] == count
        assert info["option_discount"] == pytest.approx(0.9**count)
        assert terminated is (index == 2)
        assert not truncated
        assert info["option_bootstrap_discount"] == pytest.approx(
            0.0 if terminated else 0.9**5
        )
    expected = sum((0.9**index * (index + 1) for index in range(13)))
    assert sum(
        (env.manager_gamma**index * value for index, value in enumerate(option_rewards))
    ) == pytest.approx(expected)
    assert info["episode_discounted_physical_return"] == pytest.approx(expected)
    assert info["episode_physical_return"] == sum(range(1, 14))
    assert (
        info["episode_physics_steps"]
        == info["total_physics_steps"]
        == len(base.actions)
        == 13
    )
    assert info["episode_manager_steps"] == 3
    assert info["expert_action_counts"] == [5, 5, 3]
    assert info["elapsed_control_time_s"] == pytest.approx(0.26)
    assert [call["step"] for call in children[0].calls] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [call["step"] for call in children[1].calls] == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert [call["step"] for call in children[2].calls] == [10.0, 11.0, 12.0]
    assert not any(
        (
            call["training"] or call["grad_enabled"]
            for child in children
            for call in child.calls
        )
    )
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(0)
    _, reset_info = env.reset()
    assert (
        reset_info["episode_physics_steps"] == 0
        and reset_info["total_physics_steps"] == 13
    )


def test_entire_500_action_deadline_is_100_manager_decisions_and_no_reduced_motor_authority():
    base = PhysicalEnv(horizon=500, gamma=0.995)
    children = experts()
    full_action = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], np.float32)
    children[0].predict = lambda *_args, **_kwargs: (full_action.copy(), None)
    env = OptionsEnv(base, children)
    env.reset()
    for _ in range(100):
        _, _, done, truncated, info = env.step(0)
    assert done and (not truncated)
    assert info["episode_manager_steps"] == 100 and info["episode_physics_steps"] == 500
    assert info["expert_action_counts"] == [500, 0, 0]
    np.testing.assert_array_equal(base.actions, np.tile(full_action, (500, 1)))


def test_partial_truncation_is_rejected_but_full_duration_truncation_bootstraps_correctly():
    env = OptionsEnv(PhysicalEnv(horizon=3, truncated=True), experts())
    env.reset()
    with pytest.raises(RuntimeError, match="duration-aware bootstrapping"):
        env.step(0)
    assert env.total_physics_steps == 3
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(0)
    full = OptionsEnv(PhysicalEnv(horizon=5, truncated=True), experts())
    full.reset()
    _, _, terminated, truncated, info = full.step(0)
    assert truncated and (not terminated)
    assert info["option_bootstrap_discount"] == pytest.approx(full.manager_gamma)
    both = OptionsEnv(PhysicalEnv(horizon=3, truncated=True, both=True), experts())
    both.reset()
    _, _, terminated, truncated, info = both.step(0)
    assert terminated and truncated and (info["option_bootstrap_discount"] == 0.0)


@pytest.mark.parametrize("action", [-1, 3, True, 1.0, np.array([1, 2]), "1"])
def test_invalid_manager_action_is_rejected_before_any_physical_step(action):
    env = OptionsEnv(PhysicalEnv(), experts())
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.total_physics_steps == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(option_steps=0),
        dict(option_steps=True),
        dict(option_steps=2.5),
        dict(physical_gamma=0.995),
        dict(physical_gamma=0.0),
        dict(physical_gamma=True),
        dict(expert_deterministic=None),
    ],
)
def test_invalid_or_mismatched_temporal_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        OptionsEnv(PhysicalEnv(), experts(), **kwargs)


def test_experts_freeze_without_changing_values_and_equal_independent_parameters_are_allowed():
    children = experts()
    for child in children[1:]:
        child.policy.load_state_dict(children[0].policy.state_dict())
    before = [policy_state_hash(child) for child in children]
    assert len(set(before)) == 1
    assert_parameter_independence(children)
    assert freeze_experts(children) == children
    assert [policy_state_hash(child) for child in children] == before
    assert all((not child.policy.training for child in children))
    assert all(
        (
            not parameter.requires_grad
            for child in children
            for parameter in child.policy.parameters()
        )
    )


def test_shared_module_and_shared_parameter_storage_are_rejected_unless_explicitly_allowed():
    children = list(experts())
    children[1].policy = children[0].policy
    with pytest.raises(ValueError, match="share"):
        freeze_experts(children)
    assert_parameter_independence(children, allow_shared_parameters=True)
    children = list(experts())
    children[1].policy.weight = torch.nn.Parameter(children[0].policy.weight.detach())
    with pytest.raises(ValueError, match="share"):
        assert_parameter_independence(children)


def test_manager_cannot_share_backbone_with_frozen_expert():
    children = experts()
    manager = FakeManager()
    manager.policy = children[0].policy
    with pytest.raises(ValueError, match="share"):
        HierarchicalPolicy(manager, children)
    assert all((parameter.requires_grad for parameter in manager.policy.parameters()))


def test_deployed_stack_reselects_every_five_actions_and_resets_explicitly():
    children, manager = (experts(), FakeManager())
    stack = HierarchicalPolicy(manager, children)
    before = policy_state_hash(stack)
    for step in range(12):
        action, state = stack.predict(observation(step))
        assert action.shape == (5,) and state is None
    assert manager.calls == [(0.0, True), (5.0, True), (10.0, True)]
    assert stack.expert_action_counts == [5, 2, 5]
    assert stack.manager_decisions == 3 and stack.physics_actions == 12
    assert stack.actions_remaining == 3 and stack.current_option == 1
    assert [call["step"] for call in children[2].calls] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [call["step"] for call in children[0].calls] == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert policy_state_hash(stack) == before
    stack.predict(observation(), episode_start=np.array([True]))
    assert stack.physics_actions == stack.manager_decisions == 1
    assert stack.option_history[0]["physics_action"] == 0
    stack.reset()
    assert (
        stack.current_option is None
        and stack.physics_actions == stack.manager_decisions == 0
    )


@pytest.mark.parametrize(
    "expert_mode,manager_mode,predict_mode,expected_expert,expected_manager",
    [
        (True, None, False, True, False),
        (None, None, False, False, False),
        (False, True, False, False, True),
        (None, None, True, True, True),
    ],
)
def test_expert_and_supervisor_stochasticity_are_separate_and_explicit(
    expert_mode, manager_mode, predict_mode, expected_expert, expected_manager
):
    children, manager = (experts(), FakeManager())
    stack = HierarchicalPolicy(
        manager,
        children,
        expert_deterministic=expert_mode,
        manager_deterministic=manager_mode,
    )
    stack.predict(observation(), deterministic=predict_mode)
    assert manager.calls[0][1] is expected_manager
    assert children[2].calls[0]["deterministic"] is expected_expert


def test_composite_training_mode_never_unfreezes_or_trains_experts():
    children, manager = (experts(), FakeManager())
    stack = HierarchicalPolicy(manager, children)
    stack.policy.eval()
    assert not manager.policy.training
    stack.policy.train(True)
    assert manager.policy.training and stack.policy.training
    assert all((not child.policy.training for child in children))
    assert all(
        (
            not parameter.requires_grad
            for child in children
            for parameter in child.policy.parameters()
        )
    )


def test_runtime_rejects_hidden_state_or_geometry_and_does_not_silently_drop_recurrence():
    children, manager = (experts(), FakeManager())
    stack = HierarchicalPolicy(manager, children)
    privileged = {**observation(), "cup_xyz": np.zeros(3)}
    with pytest.raises(ValueError, match="one simulator"):
        stack.predict(privileged)
    with pytest.raises(ValueError, match="state=None"):
        stack.predict(observation(), state=np.zeros(1))
    with pytest.raises(ValueError, match="episode_start"):
        stack.predict(observation(), episode_start=[False, False])
    manager.return_state = (np.zeros(1),)
    with pytest.raises(ValueError, match="nonrecurrent"):
        stack.predict(observation())
    manager.return_state = None
    children[0].return_state = np.zeros(1)
    env = OptionsEnv(PhysicalEnv(), children)
    env.reset()
    with pytest.raises(ValueError, match="nonrecurrent"):
        env.step(0)


def test_one_options_env_matches_deployed_stack_for_identical_learned_choices():
    option_children, deployed_children = (experts(), experts())
    env = OptionsEnv(PhysicalEnv(horizon=13), option_children)
    env.reset()
    for choice in (2, 0, 1):
        env.step(choice)
    base = PhysicalEnv(horizon=13)
    obs, _ = base.reset()
    stack = HierarchicalPolicy(FakeManager(), deployed_children)
    for _ in range(13):
        action, _ = stack.predict(obs)
        obs, _, _, _, _ = base.step(action)
    np.testing.assert_array_equal(base.actions, env.env.actions)
    assert stack.expert_action_counts == env.expert_action_counts


def test_real_ppo_discrete_manager_can_learn_without_modifying_experts():
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class TinyImageProprio(BaseFeaturesExtractor):
        def __init__(self, space):
            super().__init__(space, features_dim=8)
            self.net = torch.nn.Sequential(torch.nn.Linear(24, 8), torch.nn.Tanh())

        def forward(self, observations):
            pixels = observations["image"].mean(dim=(2, 3))
            return self.net(torch.cat((pixels, observations["proprio"]), dim=1))

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        children = experts()
        frozen_before = [policy_state_hash(child) for child in children]
        env = OptionsEnv(PhysicalEnv(), children)
        manager = PPO(
            "MultiInputPolicy",
            env,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            gamma=env.manager_gamma,
            learning_rate=0.001,
            device="cpu",
            seed=13,
            policy_kwargs=dict(features_extractor_class=TinyImageProprio, net_arch=[8]),
        )
        before = policy_state_hash(manager)
        manager.learn(total_timesteps=8)
        assert manager.num_timesteps == 8
        assert env.total_physics_steps == 36
        assert policy_state_hash(manager) != before
        assert [policy_state_hash(child) for child in children] == frozen_before
        stack = HierarchicalPolicy(manager, children)
        action, _ = stack.predict(observation(), episode_start=True)
        assert action.shape == (5,)
    finally:
        torch.set_num_threads(previous_threads)
