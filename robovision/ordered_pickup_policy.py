"""A learned irreversible approach-to-pickup decision over frozen force experts.

The manager observes only approach decision states. Continue runs approach for
one fixed block; commit runs pickup until the *base task* really terminates.
The latter is one terminal manager transition, with all physical rewards
internally discounted. No manager decision or value bootstrap is needed after
commitment. This architectural prior removes recovery after a mistaken switch.

The manager learns an explicit approach-to-pickup handoff.
There are no geometric gates, action labels, motor masks or physical controllers.
"""

from __future__ import annotations
from numbers import Integral, Real
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from .hierarchical_policy import (
    _StackModules,
    _expert_action,
    _force_space,
    _module,
    _observation,
    _observation_space,
    assert_parameter_independence,
)

VERSION = "learned-ordered-approach-pickup-v1"
EXPERT_NAMES = ("approach", "pickup")
MAX_PHYSICAL_ACTIONS = 500


def _duration(value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value not in (5, 25)
    ):
        raise ValueError("Ordered pickup option_steps must be 5 or 25")
    return int(value)


def _choice(action):
    value = np.asarray(action)
    if (
        value.shape not in ((), (1,))
        or value.dtype.kind not in "iu"
        or int(value.item()) not in (0, 1)
    ):
        raise ValueError("Manager action must be integer 0 (continue) or 1 (commit)")
    return int(value.item())


def _experts_pair(experts):
    pair = tuple(experts)
    if len(pair) != 2:
        raise ValueError("Provide exactly two independent approach and pickup experts")
    for expert in pair:
        if not callable(getattr(expert, "predict", None)):
            raise TypeError(
                "Each expert must expose predict(observation, deterministic=...)"
            )
        _module(expert)
        _observation_space(expert.observation_space)
        _force_space(expert.action_space)
    assert_parameter_independence(pair)
    return pair


def freeze_pickup_experts(experts):
    """Freeze two independent networks without changing any parameter values."""
    pair = _experts_pair(experts)
    for expert in pair:
        _module(expert).requires_grad_(False)
        _module(expert).eval()
    return pair


def _architecture():
    return dict(
        version=VERSION,
        expert_names=list(EXPERT_NAMES),
        observation="Two RGB frames plus original 18 proprioceptive values",
        architectural_prior="One irreversible learned approach-to-pickup switch",
        recovery_after_commit=False,
        manager_observes_only_approach_decisions=True,
        runtime_geometric_gate=False,
        action_demonstrations=False,
        runtime_physical_controller=False,
        reduced_expert_action_space=False,
        physical_frequency_hz=50.0,
        experts_frozen=True,
        deployment_validated=False,
        prerequisite="Frozen pickup continuation from actual approach arrivals",
    )


class OrderedPickupEnv(gym.Wrapper):
    """Discrete PPO stopping task over the original finite-horizon pickup env.

    Use ``gamma=env.manager_gamma`` in PPO. Action 0 performs ``option_steps``
    fresh approach predictions. Action 1 performs fresh pickup predictions
    through true base termination, including its terminal reward exactly once.
    Never reset physics, observations or the episode clock at commitment.

    The base env must have the original 500-action true-terminal deadline and
    expose its current step_count. It may reset to an actual arrival with less
    time left. Pure truncation is allowed only after a full continue block;
    commitment cannot bootstrap an approach-state value from a pickup state.
    """

    def __init__(
        self,
        base_env,
        experts_pair,
        *,
        option_steps=5,
        physical_gamma=0.999,
        expert_deterministic=True,
    ):
        super().__init__(base_env)
        self.option_steps = _duration(option_steps)
        _observation_space(base_env.observation_space)
        _force_space(base_env.action_space)
        if type(expert_deterministic) is not bool:
            raise ValueError("expert_deterministic must be an explicit boolean")
        if (
            isinstance(physical_gamma, (bool, np.bool_))
            or not isinstance(physical_gamma, Real)
            or (not np.isfinite(physical_gamma))
            or (not 0.0 < physical_gamma <= 1.0)
        ):
            raise ValueError("physical_gamma must be finite and in (0, 1]")
        base = base_env.unwrapped
        if getattr(base, "max_steps", None) != MAX_PHYSICAL_ACTIONS or not np.isclose(
            getattr(base, "control_dt", np.nan), 0.02, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "Base environment must retain the original 500-action deadline and 50 Hz control"
            )
        if not np.isclose(
            getattr(base, "gamma", np.nan), physical_gamma, rtol=0.0, atol=1e-12
        ):
            raise ValueError("Physical discount must match the base reward discount")
        self.physical_gamma = float(physical_gamma)
        self.manager_gamma = self.physical_gamma**self.option_steps
        self.gamma = self.manager_gamma
        self.physical_control_dt = 0.02
        self.control_dt = self.option_steps * self.physical_control_dt
        self.experts = freeze_pickup_experts(experts_pair)
        self.expert_deterministic = expert_deterministic
        self.action_space = spaces.Discrete(2)
        self.total_physics_steps = 0
        self._observation = None
        self._ended = True
        self._reset_counters()

    def _reset_counters(self):
        self.committed = False
        self.commit_global_step = None
        self.commit_episode_step = None
        self.episode_start_step = None
        self.episode_physics_steps = 0
        self.episode_manager_steps = 0
        self.episode_physical_return = 0.0
        self.episode_discounted_physical_return = 0.0
        self.expert_action_counts = [0, 0]

    def _global_step(self):
        step = getattr(self.env.unwrapped, "step_count", None)
        if (
            isinstance(step, (bool, np.bool_))
            or not isinstance(step, Integral)
            or (not 0 <= step <= MAX_PHYSICAL_ACTIONS)
        ):
            raise RuntimeError(
                "Base environment must expose its original physical step_count"
            )
        return int(step)

    def reset(self, *, seed=None, options=None):
        self._ended = True
        observation, info = self.env.reset(seed=seed, options=options)
        self._observation = _observation(observation)
        self._reset_counters()
        self.episode_start_step = self._global_step()
        if self.episode_start_step >= MAX_PHYSICAL_ACTIONS:
            raise RuntimeError(
                "Reset must leave time in the original physical deadline"
            )
        self._ended = False
        return (
            observation,
            {
                **info,
                "manager_gamma": self.manager_gamma,
                "physical_gamma": self.physical_gamma,
                "option_steps_requested": self.option_steps,
                "episode_start_step": self.episode_start_step,
                "episode_physics_steps": 0,
                "episode_manager_steps": 0,
                "total_physics_steps": self.total_physics_steps,
                "committed": False,
            },
        )

    def step(self, action):
        if self._ended or self._observation is None:
            raise RuntimeError(
                "Reset OrderedPickupEnv before stepping or after an episode ends"
            )
        index = _choice(action)
        start = self._global_step()
        if start != self.episode_start_step + self.episode_physics_steps:
            raise RuntimeError("The physical clock changed outside the ordered wrapper")
        self.committed = bool(index)
        if self.committed:
            self.commit_global_step = start
            self.commit_episode_step = self.episode_physics_steps
        requested = (
            MAX_PHYSICAL_ACTIONS - start if self.committed else self.option_steps
        )
        rewards = []
        terminated = truncated = False
        info = {}
        for _ in range(requested):
            if self._global_step() >= MAX_PHYSICAL_ACTIONS:
                self._ended = True
                raise RuntimeError(
                    "Base environment did not terminate at its original deadline"
                )
            forces = _expert_action(
                self.experts[index], self._observation, self.expert_deterministic
            )
            observation, reward, terminated, truncated, info = self.env.step(forces)
            self._observation = _observation(observation)
            reward = float(reward)
            if not np.isfinite(reward):
                self._ended = True
                raise ValueError("Physical reward must be finite")
            rewards.append(reward)
            self.episode_discounted_physical_return += (
                self.physical_gamma**self.episode_physics_steps * reward
            )
            self.episode_physical_return += reward
            self.episode_physics_steps += 1
            self.total_physics_steps += 1
            self.expert_action_counts[index] += 1
            if (
                self._global_step()
                != self.episode_start_step + self.episode_physics_steps
            ):
                self._ended = True
                raise RuntimeError(
                    "Each expert prediction must advance exactly one physical action"
                )
            if terminated or truncated:
                break
        executed = len(rewards)
        self.episode_manager_steps += 1
        self._ended = bool(terminated or truncated)
        if (
            truncated
            and (not terminated)
            and (self.committed or executed != self.option_steps)
        ):
            raise RuntimeError(
                "Truncation requires a duration/phase-aware bootstrap; commit requires true terminal closure"
            )
        if (self.committed or self._global_step() == MAX_PHYSICAL_ACTIONS) and (
            not terminated
        ):
            self._ended = True
            raise RuntimeError(
                "Base environment did not provide true terminal closure at its original deadline"
            )
        reward = sum(
            (self.physical_gamma**step * value for step, value in enumerate(rewards))
        )
        info = {
            **info,
            "manager_action": index,
            "option_index": index,
            "option_name": EXPERT_NAMES[index],
            "committed": self.committed,
            "commit_global_step": self.commit_global_step,
            "commit_episode_step": self.commit_episode_step,
            "option_steps_requested": requested,
            "option_steps_executed": executed,
            "continue_option_steps": self.option_steps,
            "option_discount": self.physical_gamma**executed,
            "option_bootstrap_discount": 0.0 if terminated else self.manager_gamma,
            "manager_gamma": self.manager_gamma,
            "physical_gamma": self.physical_gamma,
            "option_physical_rewards": rewards,
            "option_undiscounted_reward": sum(rewards),
            "option_discounted_reward": reward,
            "episode_start_step": self.episode_start_step,
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
            **_architecture(),
            option_steps=self.option_steps,
            manager_frequency_before_commit_hz=1.0 / self.control_dt,
            physical_gamma=self.physical_gamma,
            manager_gamma=self.manager_gamma,
            expert_deterministic=self.expert_deterministic,
            actions={
                "0": "Continue approach for one fixed block",
                "1": "Commit pickup through true base terminal",
            },
            reward="Sum gamma**j physical rewards for the entire block or terminal commitment",
            commit_bootstrap=0.0,
            terminal_reward_included_once=True,
            partial_continue_truncation="Rejected",
            pure_commit_truncation="Rejected",
            base_environment=base.specification()
            if hasattr(base, "specification")
            else None,
        )


class OrderedPickupPolicy:
    """Deploy the learned stopping decision without any simulator-state access.

    Call reset() at every episode boundary, or pass episode_start=True. Until
    commitment, the manager runs every option_steps actions. After commitment
    only pickup runs, freshly observing every physical action. By default only
    manager choices become stochastic with predict(deterministic=False); use
    expert_deterministic=None to inherit the flag or False to sample forces.
    """

    def __init__(
        self,
        manager,
        experts_pair,
        *,
        option_steps=5,
        expert_deterministic=True,
        manager_deterministic=None,
    ):
        self.option_steps = _duration(option_steps)
        for mode in (expert_deterministic, manager_deterministic):
            if mode is not None and type(mode) is not bool:
                raise ValueError(
                    "Stochasticity modes must be booleans or None to inherit predict"
                )
        if not callable(getattr(manager, "predict", None)):
            raise TypeError("Manager must expose a learned discrete predict method")
        _observation_space(manager.observation_space)
        if (
            not isinstance(manager.action_space, spaces.Discrete)
            or manager.action_space.n != 2
            or manager.action_space.start != 0
        ):
            raise ValueError("Ordered manager must have exactly two discrete actions")
        candidates = _experts_pair(experts_pair)
        self.independence = assert_parameter_independence((manager, *candidates))
        self.manager = manager
        self.experts = freeze_pickup_experts(candidates)
        self.expert_deterministic = expert_deterministic
        self.manager_deterministic = manager_deterministic
        self.observation_space = manager.observation_space
        self.action_space = self.experts[0].action_space
        self.policy = _StackModules(manager, self.experts)
        self.reset()

    def reset(self):
        self.committed = False
        self.commit_episode_step = None
        self.actions_remaining = 0
        self.current_option = None
        self.physics_actions = 0
        self.manager_decisions = 0
        self.expert_action_counts = [0, 0]
        self.option_history = []

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        if state is not None:
            raise ValueError(
                "Ordered stack stores its own commitment state; pass state=None"
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
        if not self.committed and self.actions_remaining == 0:
            _module(self.manager).eval()
            with torch.no_grad():
                action, manager_state = self.manager.predict(
                    observation, deterministic=manager_mode
                )
            if manager_state is not None:
                raise ValueError(
                    "Ordered manager must be nonrecurrent; returned state cannot be discarded"
                )
            self.current_option = _choice(action)
            self.committed = bool(self.current_option)
            self.manager_decisions += 1
            self.option_history.append(
                dict(
                    physics_action=self.physics_actions,
                    option_index=self.current_option,
                    option_name=EXPERT_NAMES[self.current_option],
                )
            )
            if self.committed:
                self.commit_episode_step = self.physics_actions
            else:
                self.actions_remaining = self.option_steps
        # Commitment is irreversible: the pickup expert owns the rest of the episode.
        index = 1 if self.committed else 0
        forces = _expert_action(self.experts[index], observation, expert_mode)
        if not self.committed:
            self.actions_remaining -= 1
        self.physics_actions += 1
        self.expert_action_counts[index] += 1
        return (forces, None)

    def specification(self):
        return dict(
            **_architecture(),
            option_steps=self.option_steps,
            manager_frequency_before_commit_hz=50.0 / self.option_steps,
            manager_deterministic=self.manager_deterministic,
            expert_deterministic=self.expert_deterministic,
            inherited_mode="None means inherit predict(deterministic=...)",
            independent_parameters=self.independence,
            episode_reset_required=True,
        )
