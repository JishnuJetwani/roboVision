"""Exact stopping-task returns, preserved controls/history and terminal closure."""

import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from robovision.ordered_pickup_policy import (
    OrderedPickupEnv,
    OrderedPickupPolicy,
    freeze_pickup_experts,
)
from robovision.policy_state import policy_state_hash


def observation_space():
    return spaces.Dict(
        image=spaces.Box(0, 255, (6, 96, 96), np.uint8),
        proprio=spaces.Box(-np.inf, np.inf, (18,), np.float32),
    )


def observation(step=0, previous=None):
    proprio = np.zeros(18, np.float32)
    proprio[0] = step
    proprio[12:17] = np.zeros(5) if previous is None else previous
    proprio[-1] = 1.0 - step / 500.0
    image = np.empty((6, 96, 96), np.uint8)
    image[:3] = max(0, step - 1) % 256
    image[3:] = step % 256
    return dict(image=image, proprio=proprio)


class Expert:
    def __init__(self, index):
        self.index = index
        self.policy = torch.nn.Linear(1, 1)
        self.observation_space = observation_space()
        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.calls = []
        self.return_state = None

    def predict(self, obs, deterministic=True):
        self.calls.append(
            dict(
                step=float(obs["proprio"][0]),
                proprio=obs["proprio"].copy(),
                pixels=obs["image"][[0, 3], 0, 0].copy(),
                deterministic=deterministic,
                training=self.policy.training,
                grad=torch.is_grad_enabled(),
            )
        )
        action = (
            np.array([-0.05, 0.04, -0.03, 0.02, -0.01], np.float32)
            + 0.02 * self.index
            + obs["proprio"][0] * 1e-05
            + 0.01 * obs["proprio"][12:17]
        )
        return (action, self.return_state)


def experts():
    return (Expert(0), Expert(1))


class Manager:
    def __init__(self, choices=(0, 0, 1)):
        self.policy = torch.nn.Linear(1, 1)
        self.observation_space = observation_space()
        self.action_space = spaces.Discrete(2)
        self.choices = choices
        self.calls = []
        self.return_state = None

    def predict(self, obs, deterministic=True):
        index = len(self.calls)
        self.calls.append(
            dict(
                step=float(obs["proprio"][0]),
                deterministic=deterministic,
                training=self.policy.training,
                grad=torch.is_grad_enabled(),
            )
        )
        return (np.asarray(self.choices[index % len(self.choices)]), self.return_state)


class PhysicalEnv(gym.Env):
    max_steps = 500
    control_dt = 0.02

    def __init__(self, horizon=37, gamma=0.999, truncated=False, both=False, start=0):
        self.horizon, self.gamma, self.truncation, self.both, self.start = (
            horizon,
            gamma,
            truncated,
            both,
            start,
        )
        self.observation_space = observation_space()
        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.actions, self.rewards = ([], [])
        self.reset_calls = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_calls += 1
        self.step_count = self.start
        self.ended = False
        self.last_action = np.array([0.12, -0.11, 0.1, -0.09, 0.08], np.float32)
        return (
            observation(self.step_count, self.last_action),
            dict(reset_token="unchanged"),
        )

    def step(self, action):
        assert not self.ended, "Terminal overstep"
        self.actions.append(np.array(action).copy())
        self.step_count += 1
        self.last_action = np.asarray(action).copy()
        self.ended = self.step_count == self.horizon
        terminated = self.ended and (not self.truncation or self.both)
        reward = (
            0.1 * self.step_count + 0.3 * float(np.sum(action)) + 100.0 * terminated
        )
        self.rewards.append(reward)
        info = dict(
            step=self.step_count,
            is_success=terminated,
            centered_success=terminated,
            original_success_ever=terminated,
            reason="success" if terminated else "",
        )
        return (
            observation(self.step_count, self.last_action),
            reward,
            terminated,
            self.ended and self.truncation,
            info,
        )


@pytest.mark.parametrize("duration", [5, 25])
def test_terminal_commit_exactly_matches_direct_physical_return_control_and_history(
    duration,
):
    base = PhysicalEnv(horizon=2 * duration + 7, gamma=0.97)
    pair = experts()
    env = OrderedPickupEnv(base, pair, option_steps=duration, physical_gamma=0.97)
    direct, direct_pair = (PhysicalEnv(horizon=base.horizon, gamma=0.97), experts())
    obs, info = env.reset(seed=19)
    direct_obs, _ = direct.reset(seed=19)
    assert info["reset_token"] == "unchanged"
    assert (
        env.observation_space is base.observation_space
        and env.action_space == spaces.Discrete(2)
    )
    assert env.manager_gamma == pytest.approx(0.97**duration)
    manager_rewards = []
    for decision, count in zip((0, 0, 1), (duration, duration, 7)):
        obs, reward, terminated, truncated, info = env.step(decision)
        manager_rewards.append(reward)
        for _ in range(count):
            action, _ = direct_pair[decision].predict(direct_obs)
            direct_obs, _, direct_ended, _, _ = direct.step(action)
        assert (
            terminated is bool(decision)
            and terminated == direct_ended
            and (not truncated)
        )
        assert info["option_steps_executed"] == count
        assert info["option_bootstrap_discount"] == pytest.approx(
            0.0 if decision else env.manager_gamma
        )
        for key in obs:
            np.testing.assert_array_equal(obs[key], direct_obs[key])
    expected = sum((0.97**step * reward for step, reward in enumerate(base.rewards)))
    assert sum(
        (
            env.manager_gamma**step * reward
            for step, reward in enumerate(manager_rewards)
        )
    ) == pytest.approx(expected)
    assert info["episode_discounted_physical_return"] == pytest.approx(expected)
    assert info["episode_physical_return"] == pytest.approx(sum(base.rewards))
    np.testing.assert_array_equal(base.actions, direct.actions)
    assert base.reset_calls == 1
    assert info["expert_action_counts"] == [duration * 2, 7]
    assert (
        info["committed"]
        and info["commit_global_step"] == info["commit_episode_step"] == duration * 2
    )
    assert info["episode_manager_steps"] == 3
    assert info["episode_physics_steps"] == info["total_physics_steps"] == base.horizon
    assert (
        info["is_success"]
        and info["centered_success"]
        and info["original_success_ever"]
    )
    assert [call["step"] for call in pair[1].calls] == list(
        range(duration * 2, base.horizon)
    )
    boundary = pair[1].calls[0]
    np.testing.assert_array_equal(
        boundary["proprio"][12:17], base.actions[duration * 2 - 1]
    )
    np.testing.assert_array_equal(boundary["pixels"], [duration * 2 - 1, duration * 2])
    assert not any(
        (call["training"] or call["grad"] for expert in pair for call in expert.calls)
    )
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(0)
    assert len(base.actions) == base.horizon


def test_commit_at_reset_keeps_arrival_clock_and_includes_remaining_terminal_tail_once():
    base = PhysicalEnv(horizon=500, start=493)
    env = OrderedPickupEnv(base, experts())
    obs, _ = env.reset()
    original_obs = copy.deepcopy(obs)
    obs, reward, done, truncated, info = env.step(1)
    assert done and (not truncated) and (base.step_count == 500)
    assert info["option_steps_requested"] == info["option_steps_executed"] == 7
    assert info["commit_global_step"] == 493 and info["commit_episode_step"] == 0
    assert info["episode_start_step"] == 493 and info["episode_physics_steps"] == 7
    assert info["episode_manager_steps"] == 1 and info["expert_action_counts"] == [0, 7]
    np.testing.assert_array_equal(
        env.experts[1].calls[0]["proprio"], original_obs["proprio"]
    )
    assert reward == pytest.approx(
        sum((0.999**i * r for i, r in enumerate(base.rewards)))
    )
    assert reward != pytest.approx(sum(base.rewards))
    assert info["option_bootstrap_discount"] == 0.0


@pytest.mark.parametrize("duration", [5, 25])
def test_all_continue_preserves_full_500_deadline_and_five_force_authority(duration):
    pair, base = (experts(), PhysicalEnv(horizon=500))
    forces = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], np.float32)
    pair[0].predict = lambda *_args, **_kwargs: (forces.copy(), None)
    env = OrderedPickupEnv(base, pair, option_steps=duration)
    env.reset()
    for _ in range(500 // duration):
        _, _, done, truncated, info = env.step(0)
    assert done and (not truncated) and (not info["committed"])
    assert info["expert_action_counts"] == [500, 0]
    assert info["episode_manager_steps"] == 500 // duration
    np.testing.assert_array_equal(base.actions, np.tile(forces, (500, 1)))
    _, reset_info = env.reset()
    assert (
        reset_info["total_physics_steps"] == 500
        and reset_info["episode_physics_steps"] == 0
    )


def test_continue_early_terminal_is_exact_and_never_calls_pickup():
    base = PhysicalEnv(horizon=3)
    pair = experts()
    env = OrderedPickupEnv(base, pair)
    env.reset()
    _, reward, done, _, info = env.step(0)
    assert (
        done
        and info["option_steps_executed"] == 3
        and (info["option_bootstrap_discount"] == 0.0)
    )
    assert reward == pytest.approx(
        sum((0.999**i * r for i, r in enumerate(base.rewards)))
    )
    assert not pair[1].calls


@pytest.mark.parametrize("action,horizon", [(0, 3), (1, 3), (1, 5), (1, 25)])
def test_partial_continue_or_any_pure_commit_truncation_is_rejected(action, horizon):
    base = PhysicalEnv(horizon=horizon, truncated=True)
    env = OrderedPickupEnv(base, experts())
    env.reset()
    with pytest.raises(RuntimeError, match="Truncation requires"):
        env.step(action)
    assert len(base.actions) == horizon
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(0)


def test_full_continue_truncation_bootstraps_but_true_terminal_commit_does_not():
    env = OrderedPickupEnv(PhysicalEnv(horizon=5, truncated=True), experts())
    env.reset()
    _, _, terminated, truncated, info = env.step(0)
    assert truncated and (not terminated)
    assert info["option_bootstrap_discount"] == pytest.approx(0.999**5)
    env = OrderedPickupEnv(PhysicalEnv(horizon=3, truncated=True, both=True), experts())
    env.reset()
    _, _, terminated, truncated, info = env.step(1)
    assert terminated and truncated and (info["option_bootstrap_discount"] == 0.0)


@pytest.mark.parametrize("action", [0, 1])
def test_missing_base_deadline_termination_raises_without_501st_action(action):
    base = PhysicalEnv(horizon=501, start=498)
    env = OrderedPickupEnv(base, experts())
    env.reset()
    with pytest.raises(RuntimeError, match="original deadline"):
        env.step(action)
    assert base.step_count == 500 and len(base.actions) == 2
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(1)


@pytest.mark.parametrize("action", [-1, 2, True, 1.0, "1", np.array([0, 1])])
def test_invalid_binary_action_is_rejected_without_control(action):
    env = OrderedPickupEnv(PhysicalEnv(), experts())
    env.reset()
    with pytest.raises(ValueError, match="Manager action"):
        env.step(action)
    assert not env.env.actions


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(option_steps=0),
        dict(option_steps=10),
        dict(option_steps=True),
        dict(option_steps=5.0),
        dict(physical_gamma=0.9),
        dict(physical_gamma=0.0),
        dict(physical_gamma=float("nan")),
        dict(expert_deterministic=None),
    ],
)
def test_invalid_environment_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        OrderedPickupEnv(PhysicalEnv(), experts(), **kwargs)


def test_two_experts_are_independent_frozen_and_value_preserving():
    pair = experts()
    pair[1].policy.load_state_dict(pair[0].policy.state_dict())
    before = [policy_state_hash(expert) for expert in pair]
    assert before[0] == before[1]
    assert freeze_pickup_experts(pair) == pair
    assert [policy_state_hash(expert) for expert in pair] == before
    assert all(
        (
            not expert.policy.training
            and (not any((p.requires_grad for p in expert.policy.parameters())))
            for expert in pair
        )
    )
    pair = experts()
    pair[1].policy.weight = torch.nn.Parameter(pair[0].policy.weight.detach())
    with pytest.raises(ValueError, match="share"):
        freeze_pickup_experts(pair)
    with pytest.raises(ValueError, match="exactly two"):
        freeze_pickup_experts([Expert(0)])


@pytest.mark.parametrize("duration", [5, 25])
def test_deployment_matches_training_actions_and_never_queries_manager_after_commit(
    duration,
):
    base, deployed_base = (
        PhysicalEnv(horizon=duration * 2 + 9),
        PhysicalEnv(horizon=duration * 2 + 9),
    )
    env_pair, policy_pair, manager = (experts(), experts(), Manager())
    env = OrderedPickupEnv(base, env_pair, option_steps=duration)
    stack = OrderedPickupPolicy(manager, policy_pair, option_steps=duration)
    before = policy_state_hash(stack)
    env.reset()
    for action in (0, 0, 1):
        env.step(action)
    obs, _ = deployed_base.reset()
    done = False
    while not done:
        forces, state = stack.predict(obs)
        assert state is None
        obs, _, done, _, _ = deployed_base.step(forces)
    np.testing.assert_array_equal(base.actions, deployed_base.actions)
    assert [call["step"] for call in manager.calls] == [0, duration, duration * 2]
    assert stack.committed and stack.commit_episode_step == duration * 2
    assert stack.expert_action_counts == [duration * 2, 9]
    assert stack.manager_decisions == 3 and stack.physics_actions == base.horizon
    assert policy_state_hash(stack) == before
    assert not any((call["grad"] or call["training"] for call in manager.calls))
    stack.reset()
    assert not stack.committed and stack.commit_episode_step is None
    assert (
        stack.physics_actions == stack.manager_decisions == stack.actions_remaining == 0
    )
    stack.predict(observation(), episode_start=np.array([True]))
    assert stack.physics_actions == stack.manager_decisions == 1


@pytest.mark.parametrize(
    "expert_mode,manager_mode,predict_mode,want_expert,want_manager",
    [
        (True, None, False, True, False),
        (None, None, False, False, False),
        (False, True, False, False, True),
        (None, None, True, True, True),
    ],
)
def test_stochastic_modes_are_explicit_and_no_later_manager_decision_is_sampled(
    expert_mode, manager_mode, predict_mode, want_expert, want_manager
):
    manager, pair = (Manager((1, 0)), experts())
    stack = OrderedPickupPolicy(
        manager,
        pair,
        expert_deterministic=expert_mode,
        manager_deterministic=manager_mode,
    )
    for step in range(9):
        stack.predict(observation(step), deterministic=predict_mode)
    assert len(manager.calls) == 1 and manager.calls[0]["deterministic"] is want_manager
    assert not pair[0].calls and len(pair[1].calls) == 9
    assert all((call["deterministic"] is want_expert for call in pair[1].calls))


def test_composite_modes_freeze_experts_without_freezing_manager_or_sharing_weights():
    manager, pair = (Manager(), experts())
    stack = OrderedPickupPolicy(manager, pair)
    stack.policy.train(True)
    assert manager.policy.training and all(
        (p.requires_grad for p in manager.policy.parameters())
    )
    assert all((not expert.policy.training for expert in pair))
    assert all(
        (not p.requires_grad for expert in pair for p in expert.policy.parameters())
    )
    manager, pair = (Manager(), experts())
    manager.policy = pair[0].policy
    with pytest.raises(ValueError, match="share"):
        OrderedPickupPolicy(manager, pair)
    assert all((p.requires_grad for p in manager.policy.parameters()))


def test_no_recurrent_state_or_nonbinary_manager_is_silently_accepted():
    manager = Manager()
    manager.action_space = spaces.Discrete(3)
    with pytest.raises(ValueError, match="two discrete"):
        OrderedPickupPolicy(manager, experts())
    manager = Manager()
    manager.return_state = np.zeros(1)
    stack = OrderedPickupPolicy(manager, experts())
    with pytest.raises(ValueError, match="nonrecurrent"):
        stack.predict(observation())
    env = OrderedPickupEnv(PhysicalEnv(), experts())
    env.experts[1].return_state = np.zeros(1)
    env.reset()
    with pytest.raises(ValueError, match="nonrecurrent"):
        env.step(1)
    assert not env.env.actions


def test_real_mujoco_wrapper_preserves_every_action_physics_pixel_history_and_terminal(
    monkeypatch,
):
    from robovision.generalization_env import GeneralizationGraspEnv
    from robovision.joint_env import JointGraspEnv

    monkeypatch.setattr(
        JointGraspEnv,
        "render_camera",
        lambda self: np.full((96, 96, 3), self.step_count % 256, np.uint8),
    )
    kwargs = dict(fixed_height=0.1, observation="pixels", gamma=0.999)
    wrapped_base, direct = (
        GeneralizationGraspEnv(**kwargs),
        GeneralizationGraspEnv(**kwargs),
    )
    pair, direct_pair = (experts(), experts())
    env = OrderedPickupEnv(wrapped_base, pair)
    try:
        obs, _ = env.reset(seed=41)
        direct_obs, _ = direct.reset(seed=41)
        for key in obs:
            np.testing.assert_array_equal(obs[key], direct_obs[key])
        for decision in (0, 1):
            obs, reward, terminated, truncated, info = env.step(decision)
            physical_rewards = []
            for _ in range(info["option_steps_executed"]):
                action, _ = direct_pair[decision].predict(direct_obs)
                direct_obs, step_reward, direct_done, direct_truncated, direct_info = (
                    direct.step(action)
                )
                physical_rewards.append(step_reward)
            assert terminated == direct_done and truncated == direct_truncated
            assert reward == pytest.approx(
                sum((0.999**i * r for i, r in enumerate(physical_rewards)))
            )
            for key in obs:
                np.testing.assert_array_equal(obs[key], direct_obs[key])
            for name in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
                np.testing.assert_array_equal(
                    getattr(wrapped_base.data, name), getattr(direct.data, name)
                )
            np.testing.assert_array_equal(wrapped_base.last_action, direct.last_action)
            assert wrapped_base.step_count == direct.step_count
            assert wrapped_base.data.time == direct.data.time
            assert info["reason"] == direct_info["reason"]
            if terminated:
                break
        assert terminated and env.committed
        terminal_step = wrapped_base.step_count
        with pytest.raises(RuntimeError, match="Reset"):
            env.step(0)
        assert wrapped_base.step_count == terminal_step
    finally:
        env.close()
        direct.close()


def test_specification_states_irreversible_architecture_and_unvalidated_continuation_prerequisite():
    env = OrderedPickupEnv(PhysicalEnv(), experts())
    stack = OrderedPickupPolicy(Manager(), experts())
    for spec in (env.specification(), stack.specification()):
        assert not spec["recovery_after_commit"]
        assert spec["manager_observes_only_approach_decisions"]
        assert not spec["runtime_geometric_gate"] and (
            not spec["runtime_physical_controller"]
        )
        assert not spec["reduced_expert_action_space"] and (
            not spec["deployment_validated"]
        )
        assert spec["physical_frequency_hz"] == 50.0
        assert (
            spec["prerequisite"]
            == "Frozen pickup continuation from actual approach arrivals"
        )
