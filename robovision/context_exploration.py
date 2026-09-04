"""Worker-conditioned Gaussian exploration for pure on-policy PPO.

Context never enters the actor mean network. Multipliers affect STANDARD DEVIATION. The raw pre-clipping action and its contextual log density
are stored, and the same multiplier is used for PPO likelihoods and entropy.
Deployment predict() deliberately retains the learned base Gaussian. Optional
finger-only exploration tapers with observed jaw aperture; it does not choose
action means, opening direction, or grasp timing.

Optional role_gradient_balance reweights the existing clipped PPO losses using
detached inverse gradient norms bounded by gradient_balance_max_weight. Original
role sample fractions remain in the aggregate. This adds backward-pass cost and
does not remove opposing gradient directions or equalize Adam-preconditioned
updates. Global clipping remains after aggregation.
"""

from typing import NamedTuple
import numpy as np
import torch as th
from torch.distributions import Normal
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy, BasePolicy
from .decoupled_ppo import DecoupledPPO, load_decoupled_ppo
from .role_ppo import normalize_by_role


def validate_std_scales(values):
    values = tuple((float(v) for v in values))
    if not values or any((not np.isfinite(v) or v <= 0 or v > 100 for v in values)):
        raise ValueError("Standard deviation scales must be finite and in (0, 100]")
    return values


def scaled_gaussian(base, std_scales):
    if not isinstance(base, DiagGaussianDistribution):
        raise ValueError(
            "Context exploration requires unsquashed diagonal Gaussian actions"
        )
    mean, std = (base.distribution.mean, base.distribution.stddev)
    factors = th.as_tensor(std_scales, dtype=mean.dtype, device=mean.device)
    if factors.ndim == 1:
        factors = factors[:, None]
    if (
        factors.ndim != 2
        or factors.shape[0] != mean.shape[0]
        or factors.shape[1] not in (1, mean.shape[1])
        or (not th.isfinite(factors).all())
        or (factors <= 0).any()
    ):
        raise ValueError(
            "One positive finite standard deviation scale required per sample"
        )
    result = DiagGaussianDistribution(mean.shape[-1])
    result.distribution = Normal(mean, std * factors)
    return result


def role_gradient_weights(norms, masses, max_weight=10.0, epsilon=1e-08):
    """Detached inverse-norm weights, bounded reciprocally, preserving sample mass.

    Geometric target keeps scale comparable to observed role gradients. Zero
    gradients receive weight one rather than amplified numerical noise. The
    caller still multiplies each role loss by its original sample fraction.
    """
    norms, masses = (np.asarray(norms, float), np.asarray(masses, float))
    if norms.shape != masses.shape or norms.ndim != 1 or (not len(norms)):
        raise ValueError("Matching nonempty norm/mass vectors required")
    if (
        not np.isfinite(norms).all()
        or not np.isfinite(masses).all()
        or (norms < 0).any()
        or (masses <= 0).any()
    ):
        raise ValueError("Finite nonnegative norms and positive masses required")
    if not np.isfinite(max_weight) or max_weight < 1:
        raise ValueError("max_weight must be finite and at least one")
    active = norms > epsilon
    weights = np.ones_like(norms)
    if active.any():
        target = np.exp(np.average(np.log(norms[active]), weights=masses[active]))
        weights[active] = np.clip(target / norms[active], 1 / max_weight, max_weight)
    return weights


FINGER_EXPLORATION_QUIET_M = 0.034
FINGER_EXPLORATION_FULL_M = 0.04


def validate_finger_exploration(values, n_workers):
    values = tuple([1.0] * n_workers) if values is None else validate_std_scales(values)
    if len(values) != n_workers or any((value < 1 for value in values)):
        raise ValueError(
            "One finger exploration maximum in [1,100] required per worker"
        )
    return values


def wide_finger_gate(obs):
    """Existing jaw proprioception only; no target or action-direction labels."""
    if "proprio" not in obs or obs["proprio"].shape[-1] != 18:
        raise ValueError("Wide finger gate requires original 18-value proprioception")
    proprio = obs["proprio"]
    proprio = (
        proprio
        if isinstance(proprio, th.Tensor)
        else th.as_tensor(proprio, dtype=th.float32)
    )
    jaw = ((proprio[:, 4:6] + 1) * 0.5 * 0.045).mean(-1)
    fraction = th.clamp(
        (jaw - FINGER_EXPLORATION_QUIET_M)
        / (FINGER_EXPLORATION_FULL_M - FINGER_EXPLORATION_QUIET_M),
        0,
        1,
    )
    return fraction.square() * (3 - 2 * fraction)


def validate_runtime_arm_scales(values, n_workers):
    if values is not None and any(
        (isinstance(value, (bool, np.bool_)) for value in values)
    ):
        raise ValueError("Runtime arm scales must be numeric, not Boolean")
    values = tuple([1.0] * n_workers) if values is None else validate_std_scales(values)
    if len(values) != n_workers:
        raise ValueError("One positive bounded runtime arm scale required per worker")
    return values


def training_action_std_scales(
    obs, std_scales, action_dim, finger_exploration_max=None, runtime_arm_scales=None
):
    """Observed jaw aperture changes exploration spread, never action direction.

    q4/q5 are normalized physical jaw positions from the existing proprio vector;
    the robot asset defines their range as [0,.045] m. Smoothly taper exploration
    to its base Gaussian on entering the known successful 34 mm reset region.
    """
    reference = next(iter(obs.values()))
    device = reference.device if isinstance(reference, th.Tensor) else "cpu"
    contexts = th.as_tensor(std_scales, dtype=th.float32, device=device)
    maxima = validate_finger_exploration(finger_exploration_max, len(contexts))
    scales = contexts[:, None].expand(-1, action_dim).clone()
    arm_scales = validate_runtime_arm_scales(runtime_arm_scales, len(contexts))
    if any((value != 1 for value in arm_scales)):
        if action_dim != 5:
            raise ValueError(
                "Arm exploration requires four arm torques plus one finger action"
            )
        scales[:, :4] *= th.as_tensor(arm_scales, dtype=th.float32, device=device)[
            :, None
        ]
    if any((value > 1 for value in maxima)):
        if action_dim != 5 or "proprio" not in obs or obs["proprio"].shape[-1] != 18:
            raise ValueError(
                "Finger exploration requires 5 direct-force actions and original 18-value proprioception"
            )
        proprio = th.as_tensor(obs["proprio"], dtype=th.float32, device=device)
        if proprio.shape[0] != len(contexts):
            raise ValueError(
                "Rollout worker count does not match finger exploration context"
            )
        smooth = wide_finger_gate({"proprio": proprio})
        maximum = th.as_tensor(maxima, dtype=th.float32, device=device)
        enabled = maximum > 1
        scales[enabled, -1] = (1 + (maximum - 1) * smooth)[enabled]
    return scales


def validate_arm_residual_mode(mode):
    if mode not in (None, "wide", "always"):
        raise ValueError("arm_residual_mode must be None, 'wide', or 'always'")
    return mode


class ContextActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self,
        *args,
        std_scales=(10.0, 25.0, 1.0, 1.0),
        finger_exploration_max=None,
        wide_finger_residual=False,
        runtime_arm_scales=None,
        arm_residual_mode=None,
        **kwargs,
    ):
        self.std_scales = validate_std_scales(std_scales)
        self.runtime_arm_scales = validate_runtime_arm_scales(
            runtime_arm_scales, len(self.std_scales)
        )
        self.finger_exploration_max = validate_finger_exploration(
            finger_exploration_max, len(self.std_scales)
        )
        self._last_action_std_scales = None
        super().__init__(*args, **kwargs)
        if self.use_sde or self.squash_output:
            raise ValueError(
                "Context exploration requires unsquashed non-SDE Gaussian PPO"
            )
        self.wide_finger_residual = False
        self.wide_finger_head = None
        self.arm_residual_mode = None
        self.arm_residual_head = None
        if wide_finger_residual:
            self.enable_wide_finger_residual()
        if validate_arm_residual_mode(arm_residual_mode) is not None:
            self.enable_arm_residual(arm_residual_mode)

    def enable_arm_residual(self, mode):
        mode = validate_arm_residual_mode(mode)
        if mode is None:
            raise ValueError("Enabling an arm residual requires an explicit mode")
        if (
            self.action_space.shape != (5,)
            or "proprio" not in self.observation_space.spaces
            or self.observation_space["proprio"].shape != (18,)
        ):
            raise ValueError(
                "Arm residual requires five forces and original proprioception"
            )
        if self.arm_residual_head is None:
            with th.random.fork_rng(devices=[]):
                head = th.nn.Linear(self.action_net.in_features, 4)
                th.nn.init.zeros_(head.weight)
                th.nn.init.zeros_(head.bias)
            self.arm_residual_head = head.to(
                device=self.action_net.weight.device, dtype=self.action_net.weight.dtype
            )
        self.arm_residual_mode = mode

    def enable_wide_finger_residual(self):
        if self.wide_finger_head is not None:
            return
        if (
            self.action_space.shape != (5,)
            or "proprio" not in self.observation_space.spaces
            or self.observation_space["proprio"].shape != (18,)
        ):
            raise ValueError(
                "Wide finger residual requires five forces and original proprioception"
            )
        with th.random.fork_rng(devices=[]):
            head = th.nn.Linear(self.action_net.in_features, 1)
            th.nn.init.zeros_(head.weight)
            th.nn.init.zeros_(head.bias)
        self.wide_finger_head = head.to(
            device=self.action_net.weight.device, dtype=self.action_net.weight.dtype
        )
        self.wide_finger_residual = True

    def get_distribution(self, obs):
        if not self.wide_finger_residual and self.arm_residual_mode is None:
            return super().get_distribution(obs)
        features = BasePolicy.extract_features(self, obs, self.pi_features_extractor)
        latent = self.mlp_extractor.forward_actor(features)
        means = self.action_net(latent)
        if self.wide_finger_residual:
            correction = wide_finger_gate(obs)[:, None] * self.wide_finger_head(latent)
            means = th.cat((means[:, :4], means[:, 4:5] + correction), dim=1)
        if self.arm_residual_mode is not None:
            correction = self.arm_residual_head(latent)
            if self.arm_residual_mode == "wide":
                correction = wide_finger_gate(obs)[:, None] * correction
            means = th.cat((means[:, :4] + correction, means[:, 4:5]), dim=1)
        return self.action_dist.proba_distribution(means, self.log_std)

    def evaluate_actions(self, obs, actions):
        if not self.wide_finger_residual and self.arm_residual_mode is None:
            return super().evaluate_actions(obs, actions)
        distribution = self.get_distribution(obs)
        return (
            self.predict_values(obs),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def forward(self, obs, deterministic=False):
        scales = training_action_std_scales(
            obs,
            self.std_scales,
            self.action_space.shape[0],
            self.finger_exploration_max,
            self.runtime_arm_scales,
        )
        self._last_action_std_scales = scales.detach().cpu().numpy().copy()
        distribution = scaled_gaussian(self.get_distribution(obs), scales)
        actions = distribution.get_actions(deterministic=deterministic)
        return (
            actions.reshape((-1, *self.action_space.shape)),
            self.predict_values(obs),
            distribution.log_prob(actions),
        )


class ContextSamples(NamedTuple):
    observations: dict[str, th.Tensor]
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    context_ids: th.Tensor
    std_scales: th.Tensor
    action_std_scales: th.Tensor


class ContextDictRolloutBuffer(DictRolloutBuffer):
    def __init__(
        self,
        *args,
        std_scales=(10.0, 25.0, 1.0, 1.0),
        role_normalization=False,
        finger_exploration_max=None,
        **kwargs,
    ):
        self.role_normalization = bool(role_normalization)
        self.worker_std_scales = validate_std_scales(std_scales)
        self.finger_exploration_max = validate_finger_exploration(
            finger_exploration_max, len(self.worker_std_scales)
        )
        self.scale_provider = None
        if self.role_normalization and len(self.worker_std_scales) != 4:
            raise ValueError(
                "Role normalization requires four workers: frontier/frontier/hold/pickup"
            )
        super().__init__(*args, **kwargs)
        if self.n_envs != len(self.worker_std_scales):
            raise ValueError(
                "One fixed standard deviation scale required per vector worker"
            )

    def reset(self):
        super().reset()
        self.raw_role_advantage_stats = {}
        self.context_ids = np.broadcast_to(
            np.arange(len(self.worker_std_scales)),
            (self.buffer_size, len(self.worker_std_scales)),
        ).copy()
        self.std_scales = np.broadcast_to(
            np.asarray(self.worker_std_scales, np.float32), self.context_ids.shape
        ).copy()
        self.action_std_scales = np.repeat(
            self.std_scales[..., None], self.action_dim, axis=-1
        )

    def add(self, obs, action, reward, episode_start, value, log_prob):
        if self.scale_provider is not None:
            scales = np.asarray(self.scale_provider(), dtype=np.float32)
        elif any((value > 1 for value in self.finger_exploration_max)):
            raise RuntimeError(
                "Conditional exploration requires actual sampler scales provider"
            )
        else:
            scales = np.broadcast_to(
                np.asarray(self.worker_std_scales, np.float32)[:, None],
                (self.n_envs, self.action_dim),
            )
        if (
            scales.shape != (self.n_envs, self.action_dim)
            or not np.isfinite(scales).all()
            or (scales <= 0).any()
        ):
            raise ValueError("Actual sampler scales must match each worker/action")
        self.action_std_scales[self.pos] = scales
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size=None):
        if not self.full:
            raise RuntimeError("Incomplete rollout")
        if not self.generator_ready:
            if self.role_normalization:
                roles = np.array([0, 0, 1, 2])[self.context_ids]
                self.raw_role_advantage_stats = {
                    str(role): dict(
                        count=int((roles == role).sum()),
                        mean=float(self.advantages[roles == role].mean()),
                        std=float(self.advantages[roles == role].std()),
                    )
                    for role in (0, 1, 2)
                }
                self.advantages = normalize_by_role(self.advantages, roles)
            self.context_ids = self.swap_and_flatten(self.context_ids).flatten()
            self.std_scales = self.swap_and_flatten(self.std_scales).flatten()
            self.action_std_scales = self.swap_and_flatten(self.action_std_scales)
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env=None):
        base = super()._get_samples(batch_inds, env)
        return ContextSamples(
            *base,
            self.to_torch(self.context_ids[batch_inds]),
            self.to_torch(self.std_scales[batch_inds]),
            self.to_torch(self.action_std_scales[batch_inds]),
        )


class ContextExplorationPPO(DecoupledPPO):
    def __init__(
        self,
        policy,
        env,
        *,
        std_scales=(10.0, 25.0, 1.0, 1.0),
        role_normalization=False,
        role_gradient_balance=False,
        gradient_balance_max_weight=10.0,
        finger_exploration_max=None,
        wide_finger_residual=False,
        runtime_arm_scales=None,
        arm_residual_mode=None,
        **kwargs,
    ):
        self.arm_residual_mode = validate_arm_residual_mode(arm_residual_mode)
        self.wide_finger_residual = bool(wide_finger_residual)
        self.role_gradient_balance = bool(role_gradient_balance)
        self.gradient_balance_max_weight = float(gradient_balance_max_weight)
        role_gradient_weights([1.0], [1.0], self.gradient_balance_max_weight)
        scales = validate_std_scales(std_scales)
        self.runtime_arm_scales = validate_runtime_arm_scales(
            runtime_arm_scales, len(scales)
        )
        self.finger_exploration_max = validate_finger_exploration(
            finger_exploration_max, len(scales)
        )
        if policy not in ("MultiInputPolicy", ContextActorCriticPolicy):
            raise ValueError("Context exploration requires its MultiInput policy class")
        policy_kwargs = dict(kwargs.pop("policy_kwargs", {}) or {})
        policy_kwargs["std_scales"] = scales
        policy_kwargs["runtime_arm_scales"] = self.runtime_arm_scales
        policy_kwargs["finger_exploration_max"] = self.finger_exploration_max
        policy_kwargs["wide_finger_residual"] = self.wide_finger_residual
        policy_kwargs["arm_residual_mode"] = self.arm_residual_mode
        kwargs["rollout_buffer_class"] = ContextDictRolloutBuffer
        self.role_normalization = bool(role_normalization)
        self.base_normalize_advantage = kwargs.get("normalize_advantage", True)
        if self.role_normalization:
            kwargs["normalize_advantage"] = False
        kwargs["rollout_buffer_kwargs"] = {
            "std_scales": scales,
            "role_normalization": self.role_normalization,
            "finger_exploration_max": self.finger_exploration_max,
        }
        self.std_scales = scales
        self.checkpoint_algorithm = "context-decoupled-ppo-v1"
        super().__init__(
            ContextActorCriticPolicy, env, policy_kwargs=policy_kwargs, **kwargs
        )

    def _setup_model(self):
        self._collecting_context_rollout = False
        self._context_update_active = False
        self._context_rollout_pending = False
        self.runtime_arm_scales = validate_runtime_arm_scales(
            getattr(self, "runtime_arm_scales", None), len(self.std_scales)
        )
        self.policy_kwargs["runtime_arm_scales"] = self.runtime_arm_scales
        self.finger_exploration_max = validate_finger_exploration(
            self.finger_exploration_max, len(self.std_scales)
        )
        super()._setup_model()
        self.rollout_buffer.scale_provider = lambda: self.policy._last_action_std_scales
        self.algorithm_metadata = {
            **self.algorithm_metadata,
            "name": "context-decoupled-ppo-v1",
            "std_scales": list(self.std_scales),
            "deployment_std_scale": 1.0,
            "role_normalization": self.role_normalization,
            "role_gradient_balance": self.role_gradient_balance,
            "gradient_balance_max_weight": self.gradient_balance_max_weight,
            "finger_exploration_max": list(self.finger_exploration_max),
            "finger_exploration_quiet_m": FINGER_EXPLORATION_QUIET_M,
            "finger_exploration_full_m": FINGER_EXPLORATION_FULL_M,
            "wide_finger_residual": self.wide_finger_residual,
            "arm_residual_mode": self.arm_residual_mode,
            "runtime_arm_scales": list(self.runtime_arm_scales),
        }

    def _actor_distribution(self, batch):
        return scaled_gaussian(
            self.policy.get_distribution(batch.observations), batch.action_std_scales
        )

    def _actor_losses(self, batch, advantages, ratio, entropy, log_prob, clip_range):
        if not self.role_gradient_balance:
            return super()._actor_losses(
                batch, advantages, ratio, entropy, log_prob, clip_range
            )
        if len(self.std_scales) != 4:
            raise ValueError(
                "Role gradient balancing requires frontier/frontier/hold/pickup workers"
            )
        roles = th.tensor([0, 0, 1, 2], device=batch.context_ids.device)[
            batch.context_ids
        ]
        surrogate = -th.min(
            advantages * ratio,
            advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
        )
        entropy_terms = -entropy if entropy is not None else log_prob
        records = []
        for role in roles.unique():
            selected = roles == role
            actor, exploration = (
                surrogate[selected].mean(),
                entropy_terms[selected].mean(),
            )
            gradients = th.autograd.grad(
                actor + self.ent_coef * exploration,
                self._active_actor_parameters(),
                retain_graph=True,
                allow_unused=True,
            )
            norm = float(
                th.sqrt(
                    sum(
                        (
                            g.detach().double().square().sum()
                            for g in gradients
                            if g is not None
                        ),
                        th.zeros((), device=ratio.device, dtype=th.float64),
                    )
                ).cpu()
            )
            records.append(
                (int(role), actor, exploration, norm, float(selected.float().mean()))
            )
        weights = role_gradient_weights(
            [r[3] for r in records],
            [r[4] for r in records],
            self.gradient_balance_max_weight,
        )
        actor_loss, entropy_loss = (0.0, 0.0)
        for (role, actor, exploration, norm, mass), weight in zip(records, weights):
            actor_loss += mass * float(weight) * actor
            entropy_loss += mass * float(weight) * exploration
            record = self.role_gradient_stats.setdefault(
                str(role),
                dict(
                    batches=0,
                    norm_sum=0.0,
                    weight_sum=0.0,
                    mass_sum=0.0,
                    weighted_norm_sum=0.0,
                ),
            )
            record["batches"] += 1
            record["norm_sum"] += norm
            record["weight_sum"] += float(weight)
            record["mass_sum"] += mass
            record["weighted_norm_sum"] += mass * float(weight) * norm
        return (actor_loss, entropy_loss)

    def _actor_distribution_diagnostics(self, batch, ratio, log_ratio, clip_range):
        with th.no_grad():
            for context in batch.context_ids.unique():
                selected = batch.context_ids == context
                key = str(int(context))
                record = self.context_update_stats.setdefault(
                    key, dict(samples=0, kl_sum=0.0, clipped=0)
                )
                record["samples"] += int(selected.sum())
                record["kl_sum"] += float(
                    (th.exp(log_ratio[selected]) - 1 - log_ratio[selected]).sum().cpu()
                )
                record["clipped"] += int(
                    (th.abs(ratio[selected] - 1) > clip_range).sum()
                )

    def collect_rollouts(self, *args, **kwargs):
        self._collecting_context_rollout = True
        self._context_rollout_pending = True
        try:
            return super().collect_rollouts(*args, **kwargs)
        finally:
            self._collecting_context_rollout = False

    def train(self):
        self._context_update_active = True
        try:
            result = self._train_context()
            self._context_rollout_pending = False
            return result
        finally:
            self._context_update_active = False

    def _train_context(self):
        if self.role_normalization and self.normalize_advantage:
            raise RuntimeError(
                "Role normalization cannot be followed by mixed minibatch normalization"
            )
        self.finger_exploration_stats = {}
        contexts = self.rollout_buffer.context_ids.reshape(-1)
        factors = self.rollout_buffer.action_std_scales.reshape(
            -1, self.rollout_buffer.action_dim
        )[:, -1]
        for context in np.unique(contexts):
            values = factors[contexts == context]
            record = dict(
                minimum=float(values.min()),
                mean=float(values.mean()),
                maximum=float(values.max()),
                base_fraction=float(np.mean(values == 1.0)),
                samples=int(len(values)),
            )
            self.finger_exploration_stats[str(int(context))] = record
            for metric, value in record.items():
                self.logger.record(
                    "train/context_"
                    + str(int(context))
                    + "_finger_std_scale_"
                    + metric,
                    value,
                )
        self.context_update_stats = {}
        self.role_gradient_stats = {}
        super().train()
        for key, record in self.role_gradient_stats.items():
            for metric in ("norm", "weight", "mass", "weighted_norm"):
                record[metric] = record[metric + "_sum"] / record["batches"]
                self.logger.record(
                    "train/role_" + key + "_gradient_" + metric, record[metric]
                )
        for key, record in self.context_update_stats.items():
            record["approx_kl"] = record["kl_sum"] / record["samples"]
            record["clip_fraction"] = record["clipped"] / record["samples"]
            self.logger.record(
                "train/context_" + key + "_approx_kl", record["approx_kl"]
            )
            self.logger.record(
                "train/context_" + key + "_clip_fraction", record["clip_fraction"]
            )


def _enable_wide_finger(model):
    if getattr(model.policy, "wide_finger_residual", False):
        return
    if not hasattr(model.policy, "wide_finger_head"):
        model.policy.wide_finger_head = None
        model.policy.wide_finger_residual = False
    model.policy.enable_wide_finger_residual()
    parameters = list(model.policy.wide_finger_head.parameters())
    group = {
        key: value
        for key, value in model.policy.optimizer.param_groups[0].items()
        if key != "params"
    }
    model.policy.optimizer.add_param_group(dict(group, params=parameters))
    if getattr(model.policy, "arm_residual_head", None) is not None:
        model.policy.optimizer.param_groups.insert(
            1, model.policy.optimizer.param_groups.pop()
        )
    model.actor_parameters.extend(parameters)
    model.wide_finger_residual = True
    model.policy_kwargs = dict(model.policy_kwargs, wide_finger_residual=True)
    model.algorithm_metadata["wide_finger_residual"] = True


def _enable_arm_residual(model, mode):
    existing = getattr(model.policy, "arm_residual_head", None)
    if not hasattr(model.policy, "arm_residual_head"):
        model.policy.arm_residual_head = None
        model.policy.arm_residual_mode = None
    model.policy.enable_arm_residual(mode)
    if existing is None:
        parameters = list(model.policy.arm_residual_head.parameters())
        group = {
            key: value
            for key, value in model.policy.optimizer.param_groups[0].items()
            if key != "params"
        }
        model.policy.optimizer.add_param_group(dict(group, params=parameters))
        model.actor_parameters.extend(parameters)
    model.arm_residual_mode = mode
    model.policy_kwargs = dict(model.policy_kwargs, arm_residual_mode=mode)
    model.algorithm_metadata["arm_residual_mode"] = mode


def _configure_context(model, scales, role_normalization):
    model.std_scales = scales
    model.role_normalization = bool(role_normalization)
    model.normalize_advantage = (
        False if role_normalization else model.base_normalize_advantage
    )
    model.checkpoint_algorithm = "context-decoupled-ppo-v1"
    model.runtime_arm_scales = validate_runtime_arm_scales(
        getattr(model, "runtime_arm_scales", None), len(scales)
    )
    model.policy.runtime_arm_scales = model.runtime_arm_scales
    model.policy.std_scales = scales
    model.policy.finger_exploration_max = model.finger_exploration_max
    model.policy_kwargs = dict(
        model.policy_kwargs,
        std_scales=scales,
        finger_exploration_max=model.finger_exploration_max,
        runtime_arm_scales=model.runtime_arm_scales,
    )
    model.rollout_buffer_class = ContextDictRolloutBuffer
    model.rollout_buffer_kwargs = dict(
        std_scales=scales,
        role_normalization=model.role_normalization,
        finger_exploration_max=model.finger_exploration_max,
    )
    model.rollout_buffer = ContextDictRolloutBuffer(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        **model.rollout_buffer_kwargs,
    )
    model.rollout_buffer.scale_provider = lambda: model.policy._last_action_std_scales
    model.algorithm_metadata = {
        **model.algorithm_metadata,
        "name": "context-decoupled-ppo-v1",
        "std_scales": list(scales),
        "deployment_std_scale": 1.0,
        "role_normalization": model.role_normalization,
        "role_gradient_balance": model.role_gradient_balance,
        "gradient_balance_max_weight": model.gradient_balance_max_weight,
        "finger_exploration_max": list(model.finger_exploration_max),
        "finger_exploration_quiet_m": FINGER_EXPLORATION_QUIET_M,
        "finger_exploration_full_m": FINGER_EXPLORATION_FULL_M,
        "runtime_arm_scales": list(model.runtime_arm_scales),
    }


def load_context_exploration_ppo(
    path,
    env,
    device="auto",
    *,
    std_scales=(10.0, 25.0, 1.0, 1.0),
    critic_learning_rate=0.0003,
    role_normalization=None,
    allow_context_reconfigure=False,
    role_gradient_balance=None,
    gradient_balance_max_weight=None,
    finger_exploration_max=None,
    actor_update_scope=None,
    wide_finger_residual=None,
    arm_residual_mode=None,
):
    """Resume exact contexts, or explicitly start an experiment with fresh buffer.

    role_normalization=None preserves a saved setting (False for conversions).
    Changed scales, normalization, gradient balancing, finger exploration, residual
    architecture, or actor update scope require allow_context_reconfigure=True on contextual checkpoints.
    actor_update_scope=None, wide_finger_residual=None and arm_residual_mode=None
    inherit saved settings. Arm modes 'wide'/'always' enable one four-output head;
    explicit reconfiguration can switch its gate without discarding learned weights.
    Enabling the residual adds a zero-initialized head and a separate actor Adam
    parameter group without resetting old state; disabling it is rejected.
    Reconfiguration preserves
    weights and both optimizer states; subsequent finger-head steps advance the
    shared head Adam clock while preserving inactive arm rows and row moments.
    """
    arm_residual_mode = validate_arm_residual_mode(arm_residual_mode)
    scales = validate_std_scales(std_scales)
    if not np.isfinite(critic_learning_rate) or critic_learning_rate <= 0:
        raise ValueError("critic_learning_rate must be positive and finite")
    from .policy_loading import checkpoint_algorithm

    if checkpoint_algorithm(path) == "context-decoupled-ppo-v1":
        model = ContextExplorationPPO.load(path, env=env, device=device)
        old_arm = getattr(model, "arm_residual_mode", None)
        desired_arm = old_arm if arm_residual_mode is None else arm_residual_mode
        old_wide = model.wide_finger_residual
        desired_wide = (
            old_wide if wide_finger_residual is None else bool(wide_finger_residual)
        )
        if old_wide and (not desired_wide):
            raise ValueError(
                "Cannot disable an existing wide finger residual and discard learned weights"
            )
        old_scope = model.actor_update_scope
        desired_scope = (
            old_scope
            if actor_update_scope is None
            else model.validate_actor_update_scope(actor_update_scope)
        )
        old_scales, old_role_normalization = (
            tuple(model.std_scales),
            model.role_normalization,
        )
        desired_normalization = (
            old_role_normalization
            if role_normalization is None
            else bool(role_normalization)
        )
        old_balance, old_bound = (
            model.role_gradient_balance,
            model.gradient_balance_max_weight,
        )
        desired_balance = (
            old_balance
            if role_gradient_balance is None
            else bool(role_gradient_balance)
        )
        desired_bound = (
            old_bound
            if gradient_balance_max_weight is None
            else float(gradient_balance_max_weight)
        )
        role_gradient_weights([1.0], [1.0], desired_bound)
        old_finger_max = tuple(model.finger_exploration_max)
        desired_finger_max = (
            old_finger_max
            if finger_exploration_max is None
            else validate_finger_exploration(finger_exploration_max, len(scales))
        )
        changed = (
            desired_arm != old_arm
            or desired_wide != old_wide
            or desired_scope != old_scope
            or (desired_finger_max != old_finger_max)
            or (old_scales != scales)
            or (desired_normalization != old_role_normalization)
            or (desired_balance != old_balance)
            or (desired_bound != old_bound)
        )
        if changed and (not allow_context_reconfigure):
            raise ValueError(
                "Cannot change fixed saved context scales, role normalization, gradient balancing, finger exploration, residual architecture, or actor update scope without allow_context_reconfigure=True"
            )
        if changed:
            if desired_wide and (not old_wide):
                _enable_wide_finger(model)
            if desired_arm != old_arm:
                _enable_arm_residual(model, desired_arm)
            model.set_actor_update_scope(desired_scope)
            model.finger_exploration_max = desired_finger_max
            model.role_gradient_balance = desired_balance
            model.gradient_balance_max_weight = desired_bound
            _configure_context(model, scales, desired_normalization)
            model.context_reconfiguration = dict(
                old_std_scales=list(old_scales),
                new_std_scales=list(scales),
                old_role_normalization=old_role_normalization,
                new_role_normalization=desired_normalization,
                old_role_gradient_balance=old_balance,
                new_role_gradient_balance=desired_balance,
                old_gradient_balance_max_weight=old_bound,
                new_gradient_balance_max_weight=desired_bound,
                old_finger_exploration_max=list(old_finger_max),
                new_finger_exploration_max=list(desired_finger_max),
                old_actor_update_scope=old_scope,
                new_actor_update_scope=desired_scope,
                old_wide_finger_residual=old_wide,
                new_wide_finger_residual=desired_wide,
                old_arm_residual_mode=old_arm,
                new_arm_residual_mode=desired_arm,
                fresh_rollout_buffer=True,
            )
        model.critic_learning_rate = float(critic_learning_rate)
        model.algorithm_metadata["critic_learning_rate"] = model.critic_learning_rate
        for group in model.critic_optimizer.param_groups:
            group["lr"] = model.critic_learning_rate
        return model
    original = load_decoupled_ppo(
        path, env, device, critic_learning_rate=critic_learning_rate
    )
    if (
        type(original.policy) is not MultiInputActorCriticPolicy
        or original.use_sde
        or original.policy.squash_output
    ):
        raise ValueError(
            "Conversion requires ordinary unsquashed MultiInputActorCriticPolicy"
        )
    model = ContextExplorationPPO("MultiInputPolicy", env, _init_setup_model=False)
    model.__dict__.update(original.__dict__)
    model.finger_exploration_max = validate_finger_exploration(
        finger_exploration_max, len(scales)
    )
    model.role_gradient_balance = (
        bool(role_gradient_balance) if role_gradient_balance is not None else False
    )
    model.gradient_balance_max_weight = (
        10.0
        if gradient_balance_max_weight is None
        else float(gradient_balance_max_weight)
    )
    role_gradient_weights([1.0], [1.0], model.gradient_balance_max_weight)
    model.base_normalize_advantage = original.normalize_advantage
    model.policy.__class__ = ContextActorCriticPolicy
    model.policy_class = ContextActorCriticPolicy
    model.policy.wide_finger_residual = False
    model.policy.wide_finger_head = None
    model.policy.arm_residual_head = None
    model.policy.arm_residual_mode = None
    model.arm_residual_mode = None
    if wide_finger_residual:
        _enable_wide_finger(model)
    if arm_residual_mode is not None:
        _enable_arm_residual(model, arm_residual_mode)
    model.set_actor_update_scope(
        original.actor_update_scope
        if actor_update_scope is None
        else actor_update_scope
    )
    _configure_context(
        model, scales, False if role_normalization is None else bool(role_normalization)
    )
    return model


def set_runtime_arm_scales(model, scales):
    """Set exploration-only arm multipliers between completed learn chunks.

    The four worker values multiply each worker's existing first four action
    scales. No model tensor, action mean, finger scale, or optimizer changes.
    Stored per-transition scales continue to define current PPO likelihoods.
    """
    if not isinstance(model, ContextExplorationPPO):
        raise TypeError("Runtime arm scales require ContextExplorationPPO")
    values = validate_runtime_arm_scales(scales, len(model.std_scales))
    if any((value != 1 for value in values)) and model.action_space.shape != (5,):
        raise ValueError("Arm exploration requires five force actions")
    if getattr(model, "_collecting_context_rollout", False) or getattr(
        model, "_context_update_active", False
    ):
        raise RuntimeError(
            "Change runtime arm scales only between completed model.learn chunks"
        )
    buffer = model.rollout_buffer
    if getattr(model, "_context_rollout_pending", False) or (
        buffer.pos and (not buffer.generator_ready)
    ):
        raise RuntimeError(
            "Cannot change runtime arm scales while an unconsumed rollout is pending"
        )
    previous = list(model.runtime_arm_scales)
    model.runtime_arm_scales = values
    model.policy.runtime_arm_scales = values
    model.policy_kwargs["runtime_arm_scales"] = values
    model.algorithm_metadata["runtime_arm_scales"] = list(values)
    model.algorithm_metadata["runtime_arm_scale_update"] = dict(
        previous=previous, current=list(values), at_step=int(model.num_timesteps)
    )
    return values
