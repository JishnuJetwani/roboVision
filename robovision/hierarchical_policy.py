"""A learned pixel supervisor over three frozen direct-force PPO specialists.

The supervisor chooses an expert every five physical actions by default. The
chosen expert observes the current image/proprio at *every* 50 Hz action; no
force is held or synthesized by this module. There are no geometric phase tests.

Ordinary fixed-discount PPO is correct at manager boundaries when the manager
uses gamma**option_steps and each option returns its internally discounted
physical rewards. A true terminal option may end early. A partial *truncation*
would need a duration-aware bootstrap and is rejected instead of silently using
the wrong discount. The original finite 500-action task uses true termination.
"""

from __future__ import annotations
from numbers import Integral, Real
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn

VERSION = "learned-three-expert-direct-force-stack-v1"
EXPERT_NAMES = ("approach", "grasp", "lift")


def _module(policy):
    module = (
        policy if isinstance(policy, nn.Module) else getattr(policy, "policy", None)
    )
    if not isinstance(module, nn.Module):
        raise TypeError(
            "Policies must expose their torch policy module for freezing and independence checks"
        )
    return module


def assert_parameter_independence(policies, *, allow_shared_parameters=False):
    """Reject shared parameter objects or storage across policy networks.

    Equal values copied into independent networks are allowed. Explicitly
    sharing parameters is opt-in; freezing an expert also freezes any parameter
    it shares, so shared backbones cannot simultaneously be freely fine-tuned.
    """
    if type(allow_shared_parameters) is not bool:
        raise ValueError("allow_shared_parameters must be a boolean")
    policies = tuple(policies)
    owners = {}
    shared = []
    tensors = 0
    module_ids = set()
    for index, policy in enumerate(policies):
        module = _module(policy)
        if id(module) in module_ids:
            shared.append(dict(policy_index=index, reason="same policy module"))
        module_ids.add(id(module))
        for name, parameter in module.named_parameters():
            tensors += 1
            if not parameter.numel():
                continue
            key = (str(parameter.device), parameter.untyped_storage().data_ptr())
            previous = owners.get(key)
            if previous is not None and previous[0] != index:
                shared.append(
                    dict(
                        policy_index=index,
                        parameter=name,
                        other_policy_index=previous[0],
                        other_parameter=previous[1],
                    )
                )
            owners.setdefault(key, (index, name))
    if shared and (not allow_shared_parameters):
        raise ValueError(f"Policy networks share parameters/backbones: {shared[0]}")
    return dict(
        policy_count=len(policies),
        parameter_tensors=tensors,
        shared_parameters_allowed=allow_shared_parameters,
        shared_parameters=shared,
    )


def _observation_space(space):
    if (
        not isinstance(space, spaces.Dict)
        or set(space.spaces) != {"image", "proprio"}
        or space["image"].shape != (6, 96, 96)
        or (space["proprio"].shape != (18,))
    ):
        raise ValueError(
            "Policies and environment require only two RGB frames and 18 proprioceptive values"
        )


def _observation(observation):
    if (
        not isinstance(observation, dict)
        or set(observation) != {"image", "proprio"}
        or np.shape(observation["image"]) != (6, 96, 96)
        or (np.shape(observation["proprio"]) != (18,))
    ):
        raise ValueError(
            "Use one simulator per stack: only two RGB frames and 18 proprioceptive values"
        )
    return observation


def _force_space(space):
    if (
        not isinstance(space, spaces.Box)
        or space.shape != (5,)
        or (not np.all(space.low == -1.0))
        or (not np.all(space.high == 1.0))
    ):
        raise ValueError(
            "Each expert must retain all five normalized direct-force actions in [-1,1]"
        )


def _expert_tuple(experts):
    experts = tuple(experts)
    if len(experts) != len(EXPERT_NAMES):
        raise ValueError(
            "Provide exactly three independent approach, grasp and lift experts"
        )
    for expert in experts:
        if not callable(getattr(expert, "predict", None)):
            raise TypeError(
                "Each expert must expose predict(observation, deterministic=...)"
            )
        _module(expert)
        _observation_space(expert.observation_space)
        _force_space(expert.action_space)
    return experts


def freeze_experts(experts, *, allow_shared_parameters=False):
    """Keep each independently stored specialist in evaluation mode and frozen."""
    experts = _expert_tuple(experts)
    assert_parameter_independence(
        experts, allow_shared_parameters=allow_shared_parameters
    )
    for expert in experts:
        _module(expert).requires_grad_(False)
        _module(expert).eval()
    return experts


def _duration(option_steps):
    if (
        isinstance(option_steps, (bool, np.bool_))
        or not isinstance(option_steps, Integral)
        or option_steps < 1
    ):
        raise ValueError("option_steps must be a positive integer")
    return int(option_steps)


def _option(action):
    value = np.asarray(action)
    if value.shape not in ((), (1,)) or value.dtype.kind not in "iu":
        raise ValueError("Supervisor action must be one discrete expert index")
    index = int(value.item())
    if not 0 <= index < len(EXPERT_NAMES):
        raise ValueError("Supervisor expert index must be 0, 1 or 2")
    return index


def _expert_action(expert, observation, deterministic):
    _observation(observation)
    _module(expert).eval()
    with torch.no_grad():
        action, state = expert.predict(observation, deterministic=deterministic)
    if state is not None:
        raise ValueError(
            "This stack requires nonrecurrent specialists; recurrent state is not discarded silently"
        )
    action = np.asarray(action)
    if (
        action.shape != (5,)
        or not np.isfinite(action).all()
        or np.any(np.abs(action) > 1.0)
    ):
        raise ValueError(
            "Expert must emit five finite normalized direct forces without wrapper correction"
        )
    return action


class OptionsEnv(gym.Wrapper):
    """Train a discrete PPO supervisor while frozen experts act at base frequency.

    Construct PPO with ``gamma=env.manager_gamma``. The wrapper's ``gamma`` and
    ``control_dt`` describe manager decisions; physical values are explicit.
    Reset distribution and full-pickup reward belong to the wrapped base env.
    Put Monitor outside this wrapper to monitor manager episodes, and use
    ``info['option_steps_executed']`` to count actual physical interactions.
    """

    def __init__(
        self,
        base_env,
        experts,
        *,
        option_steps=5,
        physical_gamma=None,
        expert_deterministic=True,
        allow_shared_parameters=False,
    ):
        super().__init__(base_env)
        self.option_steps = _duration(option_steps)
        _observation_space(base_env.observation_space)
        _force_space(base_env.action_space)
        if type(expert_deterministic) is not bool:
            raise ValueError(
                "Training experts require an explicit boolean expert_deterministic"
            )
        base_gamma = getattr(base_env.unwrapped, "gamma", None)
        if physical_gamma is None:
            physical_gamma = base_gamma
        if (
            isinstance(physical_gamma, (bool, np.bool_))
            or not isinstance(physical_gamma, Real)
            or (not np.isfinite(physical_gamma))
            or (not 0 < physical_gamma <= 1)
        ):
            raise ValueError("A finite physical gamma in (0,1] is required")
        if base_gamma is not None and (
            not np.isclose(float(base_gamma), physical_gamma, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(
                "Option discount must match the wrapped physical reward's gamma"
            )
        self.physical_gamma = float(physical_gamma)
        self.manager_gamma = self.physical_gamma**self.option_steps
        self.gamma = self.manager_gamma
        self.physical_control_dt = float(
            getattr(base_env.unwrapped, "control_dt", 0.02)
        )
        if not np.isclose(self.physical_control_dt, 0.02, rtol=0.0, atol=1e-12):
            raise ValueError(
                "Specialists must act at the original 50 Hz physical control frequency"
            )
        self.control_dt = self.physical_control_dt * self.option_steps
        self.experts = freeze_experts(
            experts, allow_shared_parameters=allow_shared_parameters
        )
        self.expert_deterministic = expert_deterministic
        self.action_space = spaces.Discrete(len(self.experts))
        self.total_physics_steps = 0
        self._observation = None
        self._ended = True
        self._reset_counters()

    def _reset_counters(self):
        self.episode_physics_steps = 0
        self.episode_manager_steps = 0
        self.episode_physical_return = 0.0
        self.episode_discounted_physical_return = 0.0
        self.expert_action_counts = [0] * len(EXPERT_NAMES)
        self._last_option = None

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._observation = _observation(observation)
        self._ended = False
        self._reset_counters()
        return (
            observation,
            {
                **info,
                "manager_gamma": self.manager_gamma,
                "physical_gamma": self.physical_gamma,
                "option_steps_requested": self.option_steps,
                "episode_physics_steps": 0,
                "episode_manager_steps": 0,
                "total_physics_steps": self.total_physics_steps,
            },
        )

    def step(self, action):
        if self._ended or self._observation is None:
            raise RuntimeError(
                "Reset OptionsEnv before stepping or after an episode ends"
            )
        index = _option(action)
        rewards = []
        terminated = truncated = False
        info = {}
        for _ in range(self.option_steps):
            forces = _expert_action(
                self.experts[index], self._observation, self.expert_deterministic
            )
            observation, reward, terminated, truncated, info = self.env.step(forces)
            self._observation = _observation(observation)
            reward = float(reward)
            if not np.isfinite(reward):
                raise ValueError("Physical reward must be finite")
            rewards.append(reward)
            self.episode_discounted_physical_return += (
                self.physical_gamma**self.episode_physics_steps * reward
            )
            self.episode_physical_return += reward
            self.episode_physics_steps += 1
            self.total_physics_steps += 1
            self.expert_action_counts[index] += 1
            if terminated or truncated:
                break
        executed = len(rewards)
        self.episode_manager_steps += 1
        self._ended = bool(terminated or truncated)
        if truncated and (not terminated) and (executed != self.option_steps):
            raise RuntimeError(
                "Partial option truncation needs duration-aware bootstrapping; fixed-gamma PPO is unsupported"
            )
        reward = sum(
            (self.physical_gamma**step * value for step, value in enumerate(rewards))
        )
        info = {
            **info,
            "option_index": index,
            "option_name": EXPERT_NAMES[index],
            "option_switched": self._last_option is not None
            and self._last_option != index,
            "option_steps_requested": self.option_steps,
            "option_steps_executed": executed,
            "option_discount": self.physical_gamma**executed,
            "option_bootstrap_discount": 0.0
            if terminated
            else self.physical_gamma**executed,
            "manager_gamma": self.manager_gamma,
            "physical_gamma": self.physical_gamma,
            "option_physical_rewards": rewards,
            "option_undiscounted_reward": sum(rewards),
            "option_discounted_reward": reward,
            "episode_physics_steps": self.episode_physics_steps,
            "episode_manager_steps": self.episode_manager_steps,
            "total_physics_steps": self.total_physics_steps,
            "episode_physical_return": self.episode_physical_return,
            "episode_discounted_physical_return": self.episode_discounted_physical_return,
            "expert_action_counts": self.expert_action_counts.copy(),
            "elapsed_control_time_s": self.episode_physics_steps
            * self.physical_control_dt,
            "expert_deterministic": self.expert_deterministic,
        }
        self._last_option = index
        return (
            self._observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def specification(self):
        base = self.env.unwrapped
        return dict(
            version=VERSION,
            expert_names=list(EXPERT_NAMES),
            observation="Two RGB frames plus original 18 proprioceptive values",
            option_steps=self.option_steps,
            physical_frequency_hz=50.0,
            manager_frequency_hz=1.0 / self.control_dt,
            physical_gamma=self.physical_gamma,
            manager_gamma=self.manager_gamma,
            expert_deterministic=self.expert_deterministic,
            experts_frozen=True,
            runtime_geometric_gate=False,
            reduced_expert_action_space=False,
            reward="Sum gamma**j times each physical reward inside the option",
            partial_truncation="Rejected; true terminal early breaks have no bootstrap",
            base_environment=base.specification()
            if hasattr(base, "specification")
            else None,
        )


class _StackModules(nn.Module):
    """Expose all tensors for frozen audit while keeping expert modes frozen."""

    def __init__(self, manager, experts):
        super().__init__()
        training = _module(manager).training
        self.manager = _module(manager)
        self.experts = nn.ModuleList([_module(expert) for expert in experts])
        self.train(training)

    def train(self, mode=True):
        super().train(mode)
        for expert in self.experts:
            expert.eval()
        return self


class HierarchicalPolicy:
    """Deploy the learned supervisor with persistent five-action selections.

    One instance controls one simulator. Call reset() at *every* episode boundary
    or pass episode_start=True to predict(). No physics or geometry is inspected.
    The manager and specialists retain separate original checkpoint objects.

    By default predict(deterministic=False) samples supervisor choices while the
    frozen specialists remain deterministic. Set expert_deterministic=None to
    inherit predict's flag, or explicitly False to sample expert actions as well.
    """

    def __init__(
        self,
        manager,
        experts,
        *,
        option_steps=5,
        expert_deterministic=True,
        manager_deterministic=None,
        allow_shared_parameters=False,
    ):
        self.option_steps = _duration(option_steps)
        for value in (expert_deterministic, manager_deterministic):
            if value is not None and type(value) is not bool:
                raise ValueError(
                    "Stochasticity modes must be explicit booleans or None to inherit predict"
                )
        if not callable(getattr(manager, "predict", None)):
            raise TypeError("Manager must expose a learned discrete predict method")
        _observation_space(manager.observation_space)
        if (
            not isinstance(manager.action_space, spaces.Discrete)
            or manager.action_space.n != 3
        ):
            raise ValueError("Manager must select one of exactly three experts")
        candidates = _expert_tuple(experts)
        self.independence = assert_parameter_independence(
            (manager, *candidates), allow_shared_parameters=allow_shared_parameters
        )
        self.manager = manager
        self.experts = freeze_experts(
            candidates, allow_shared_parameters=allow_shared_parameters
        )
        self.expert_deterministic = expert_deterministic
        self.manager_deterministic = manager_deterministic
        self.observation_space = manager.observation_space
        self.action_space = self.experts[0].action_space
        self.policy = _StackModules(manager, self.experts)
        self.reset()

    def reset(self):
        self.current_option = None
        self.actions_remaining = 0
        self.physics_actions = 0
        self.manager_decisions = 0
        self.expert_action_counts = [0] * len(EXPERT_NAMES)
        self.option_history = []

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        if state is not None:
            raise ValueError(
                "This nonrecurrent stack manages its own option state; pass state=None"
            )
        if type(deterministic) is not bool:
            raise ValueError("deterministic must be a boolean")
        if episode_start is not None:
            starts = np.asarray(episode_start)
            if starts.shape not in ((), (1,)) or starts.dtype.kind != "b":
                raise ValueError(
                    "episode_start must describe one simulator with a boolean"
                )
            if bool(starts.item()):
                self.reset()
        _observation(observation)
        manager_mode = (
            deterministic
            if self.manager_deterministic is None
            else self.manager_deterministic
        )
        expert_mode = (
            deterministic
            if self.expert_deterministic is None
            else self.expert_deterministic
        )
        if self.actions_remaining == 0:
            _module(self.manager).eval()
            with torch.no_grad():
                choice, manager_state = self.manager.predict(
                    observation, deterministic=manager_mode
                )
            if manager_state is not None:
                raise ValueError(
                    "This supervisor is nonrecurrent; recurrent state is not discarded silently"
                )
            self.current_option = _option(choice)
            self.actions_remaining = self.option_steps
            self.manager_decisions += 1
            self.option_history.append(
                dict(
                    physics_action=self.physics_actions,
                    option_index=self.current_option,
                    option_name=EXPERT_NAMES[self.current_option],
                )
            )
        action = _expert_action(
            self.experts[self.current_option], observation, expert_mode
        )
        self.actions_remaining -= 1
        self.physics_actions += 1
        self.expert_action_counts[self.current_option] += 1
        return (action, None)

    def specification(self):
        return dict(
            version=VERSION,
            expert_names=list(EXPERT_NAMES),
            option_steps=self.option_steps,
            physical_frequency_hz=50.0,
            manager_frequency_hz=50.0 / self.option_steps,
            manager_deterministic=self.manager_deterministic,
            expert_deterministic=self.expert_deterministic,
            inherited_mode="None means inherit predict(deterministic=...)",
            experts_frozen=True,
            independent_parameters=self.independence,
            episode_reset_required=True,
            runtime_geometric_gate=False,
            reduced_expert_action_space=False,
        )
