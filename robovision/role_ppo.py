"""Opt-in on-policy PPO with rollout-wide per-role advantage normalization.

Roles are fixed vector-environment slots, defaulting to frontier/frontier/hold/
pickup. Normalize after GAE has computed returns, before minibatch shuffling;
never recompute critic targets from normalized advantages. SB3's PPO loss,
clipping, entropy, value loss, and KL stopping are otherwise used unchanged.

This is a weighting experiment, not a guarantee against forgetting. Centering
constant advantages removes that role's actor signal, and each role's variance
estimate can be noisy. Critic loss remains sensitive to absolute reward scale.
"""

from typing import NamedTuple
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer


class RoleRolloutSamples(NamedTuple):
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    role_ids: torch.Tensor


def normalize_by_role(advantages, role_ids, epsilon=1e-08):
    """Population-standardize each role across its entire collected rollout."""
    advantages = np.asarray(advantages)
    role_ids = np.asarray(role_ids)
    if advantages.shape != role_ids.shape or not np.isfinite(advantages).all():
        raise ValueError("Finite advantages and role IDs must have matching shapes")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    result = np.empty_like(advantages, dtype=np.float32)
    for role in np.unique(role_ids):
        selected = role_ids == role
        values = advantages[selected].astype(np.float64)
        result[selected] = (values - values.mean()) / (values.std(ddof=0) + epsilon)
    return result


class RoleDictRolloutBuffer(DictRolloutBuffer):
    def __init__(self, *args, role_ids=(0, 0, 1, 2), **kwargs):
        roles = np.asarray(role_ids)
        if (
            roles.ndim != 1
            or not np.issubdtype(roles.dtype, np.integer)
            or (roles < 0).any()
        ):
            raise ValueError(
                "role_ids must be a one-dimensional sequence of nonnegative integers"
            )
        self.env_role_ids = roles.astype(np.int64).copy()
        super().__init__(*args, **kwargs)
        if len(self.env_role_ids) != self.n_envs:
            raise ValueError("Provide one fixed role ID per vector environment")

    def reset(self):
        super().reset()
        self.role_ids = np.broadcast_to(
            self.env_role_ids, (self.buffer_size, len(self.env_role_ids))
        ).copy()
        self.roles_normalized = False
        self.raw_role_advantage_stats = {}

    def get(self, batch_size=None):
        if not self.full:
            raise RuntimeError("Cannot normalize an incomplete rollout")
        if not self.roles_normalized:
            self.raw_role_advantage_stats = {
                str(int(role)): dict(
                    count=int(np.sum(self.role_ids == role)),
                    mean=float(self.advantages[self.role_ids == role].mean()),
                    std=float(self.advantages[self.role_ids == role].std()),
                    minimum=float(self.advantages[self.role_ids == role].min()),
                    maximum=float(self.advantages[self.role_ids == role].max()),
                    positive_fraction=float(
                        np.mean(self.advantages[self.role_ids == role] > 0)
                    ),
                )
                for role in np.unique(self.role_ids)
            }
            self.advantages = normalize_by_role(self.advantages, self.role_ids)
            self.role_ids = self.swap_and_flatten(self.role_ids).flatten()
            self.roles_normalized = True
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env=None):
        sample = super()._get_samples(batch_inds, env)
        return RoleRolloutSamples(
            *sample, role_ids=self.to_torch(self.role_ids[batch_inds])
        )


class RoleNormalizedPPO(PPO):
    """PPO for Dict observations and fixed per-environment curriculum roles.

    Do not combine with SB3's minibatch advantage normalization: it would mix
    role scales again. These roles are training metadata, never actor inputs.
    """

    def __init__(self, policy, env, *, role_ids=(0, 0, 1, 2), **kwargs):
        if kwargs.pop("normalize_advantage", False):
            raise ValueError("Per-role normalization replaces minibatch normalization")
        buffer_class = kwargs.pop("rollout_buffer_class", RoleDictRolloutBuffer)
        if buffer_class is not RoleDictRolloutBuffer:
            raise ValueError("RoleNormalizedPPO requires RoleDictRolloutBuffer")
        buffer_kwargs = dict(kwargs.pop("rollout_buffer_kwargs", {}) or {})
        if "role_ids" in buffer_kwargs and tuple(buffer_kwargs["role_ids"]) != tuple(
            role_ids
        ):
            raise ValueError("Conflicting role IDs")
        buffer_kwargs["role_ids"] = tuple(role_ids)
        super().__init__(
            policy,
            env,
            normalize_advantage=False,
            rollout_buffer_class=RoleDictRolloutBuffer,
            rollout_buffer_kwargs=buffer_kwargs,
            **kwargs,
        )

    def train(self):
        if self.normalize_advantage:
            raise RuntimeError("Minibatch normalization must remain disabled")
        super().train()


def load_role_normalized_ppo(path, env, device="auto", *, role_ids=(0, 0, 1, 2)):
    """Convert an ordinary saved PPO checkpoint without rebuilding its actor.

    SB3 load applies these three explicit overrides before constructing the
    rollout buffer, then restores saved policy and optimizer state normally.
    All other saved PPO hyperparameters are retained. Partial rollouts are never
    reused: this starts a fresh on-policy rollout under the loaded actor.
    """
    model = RoleNormalizedPPO.load(
        path,
        env=env,
        device=device,
        normalize_advantage=False,
        rollout_buffer_class=RoleDictRolloutBuffer,
        rollout_buffer_kwargs={"role_ids": tuple(role_ids)},
    )
    model.role_normalization = {
        "version": "rollout-role-population-std-v1",
        "role_ids": list(role_ids),
        "scope": "entire_rollout_per_role",
        "critic_returns_changed": False,
    }
    return model
