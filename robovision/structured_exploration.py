"""Temporally persistent gSDE with exact inherited deterministic force means.

Uses Stable-Baselines3's generalized state-dependent exploration algorithm:
Gaussian noise weights are resampled periodically and reused across observations.
PPO stores/reconstructs its per-state marginal Gaussian density. As in standard
SB3 gSDE, this is NOT a claim of exact joint-trajectory likelihood under temporal
correlation. No noise is added after computing the recorded action likelihood.

Actor latents are unit normalized ONLY for the noise basis. The action-mean
network and its residual heads are unchanged. The normalization gives a direct
physical calibration when migrating a five-vector Gaussian standard deviation.
"""

from __future__ import annotations
import copy
import torch
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.distributions import StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, MultiInputActorCriticPolicy
from .context_exploration import ContextActorCriticPolicy, wide_finger_gate
from .decoupled_ppo import DecoupledPPO
from .grasp_benchmark import preserved_rng

ALGORITHM = "structured-decoupled-ppo-v1"
FORCE_LIMITS = (40.0, 55.0, 45.0, 20.0, 10.0)


def unit_noise_features(latent):
    """Unit basis, including a deterministic unit-vector fallback at zero latent."""
    norm = latent.norm(dim=-1, keepdim=True)
    normalized = latent / norm.clamp_min(1e-12)
    fallback = torch.zeros_like(latent)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-12, normalized, fallback)


class UnitLatentNoiseDistribution(StateDependentNoiseDistribution):
    def __init__(self, action_dim):
        super().__init__(
            action_dim,
            full_std=True,
            use_expln=False,
            squash_output=False,
            learn_features=False,
            epsilon=0.0,
        )

    def proba_distribution(self, mean_actions, log_std, latent_sde):
        return super().proba_distribution(
            mean_actions, log_std, unit_noise_features(latent_sde)
        )


class StructuredActorCriticPolicy(MultiInputActorCriticPolicy):
    """Same force means/residuals; gSDE for training, fresh Gaussian for predict."""

    def __init__(
        self, *args, wide_finger_residual=False, arm_residual_mode=None, **kwargs
    ):
        if not kwargs.pop("use_sde", True):
            raise ValueError("Structured exploration requires use_sde=True")
        if (
            kwargs.get("squash_output", False)
            or not kwargs.get("full_std", True)
            or kwargs.get("use_expln", False)
        ):
            raise ValueError("Require unsquashed full-std exponential Gaussian gSDE")
        super().__init__(*args, use_sde=True, **kwargs)
        self.action_dist = UnitLatentNoiseDistribution(self.action_space.shape[0])
        self.action_dist.latent_sde_dim = self.action_net.in_features
        self.wide_finger_residual = False
        self.wide_finger_head = None
        self.arm_residual_mode = None
        self.arm_residual_head = None
        if wide_finger_residual:
            ContextActorCriticPolicy.enable_wide_finger_residual(self)
        if arm_residual_mode is not None:
            ContextActorCriticPolicy.enable_arm_residual(self, arm_residual_mode)
        self.reset_noise()

    def get_distribution(self, obs):
        features = BasePolicy.extract_features(self, obs, self.pi_features_extractor)
        latent = self.mlp_extractor.forward_actor(features)
        means = self.action_net(latent)
        if self.wide_finger_residual:
            finger = wide_finger_gate(obs)[:, None] * self.wide_finger_head(latent)
            means = torch.cat((means[:, :4], means[:, 4:5] + finger), dim=1)
        if self.arm_residual_mode is not None:
            arm = self.arm_residual_head(latent)
            if self.arm_residual_mode == "wide":
                arm = wide_finger_gate(obs)[:, None] * arm
            means = torch.cat((means[:, :4] + arm, means[:, 4:5]), dim=1)
        return self.action_dist.proba_distribution(means, self.log_std, latent)

    def forward(self, obs, deterministic=False):
        distribution = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        return (
            actions.reshape((-1, *self.action_space.shape)),
            self.predict_values(obs),
            distribution.log_prob(actions),
        )

    def evaluate_actions(self, obs, actions):
        distribution = self.get_distribution(obs)
        return (
            self.predict_values(obs),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def _predict(self, observation, deterministic=False):
        distribution = self.get_distribution(observation)
        return (
            distribution.mode() if deterministic else distribution.distribution.sample()
        )

    def _get_constructor_parameters(self):
        return dict(
            super()._get_constructor_parameters(),
            wide_finger_residual=self.wide_finger_residual,
            arm_residual_mode=self.arm_residual_mode,
        )


class StructuredPPO(DecoupledPPO):
    def __init__(self, policy, env, *, sde_sample_freq=25, **kwargs):
        if type(sde_sample_freq) is not int or not 1 <= sde_sample_freq <= 128:
            raise ValueError("Explicit gSDE resampling interval in[1,128] required")
        if kwargs.pop("use_sde", True) is not True:
            raise ValueError("StructuredPPO requires use_sde=True")
        if policy not in ("MultiInputPolicy", StructuredActorCriticPolicy):
            raise ValueError("StructuredPPO requires its residual-compatible policy")
        super().__init__(
            StructuredActorCriticPolicy,
            env,
            use_sde=True,
            sde_sample_freq=sde_sample_freq,
            **kwargs,
        )

    def _setup_model(self):
        super()._setup_model()
        self.checkpoint_algorithm = ALGORITHM
        self.algorithm_metadata.update(
            name=ALGORITHM,
            exploration="gSDE",
            noise_basis="unit-normalized actor latent; detached noise features",
            sde_sample_freq=self.sde_sample_freq,
            deployment="deterministic actor means or fresh per-action Gaussian",
            likelihood="standard SB3 gSDE per-state marginal Gaussian",
            correlated_joint_trajectory_likelihood=False,
        )


def convert_to_structured(model, *, resample_steps=25):
    """Migrate a prepared, fresh-optimizer actor before any new rollout/update.

    Call AFTER train_generalization.prepare_model applies physical noise scaling,
    and BEFORE configuring a new rollout length/discount in the experiment.
    All mean/value tensors are copied exactly. Only the five log standard
    deviations become one repeated vector per noise feature. This changes the
    exploration family; it does not replay actions or supply a teacher.
    """
    if isinstance(model, StructuredPPO) or getattr(model, "use_sde", False):
        raise ValueError("Convert a non-SDE parent exactly once")
    if model.actor_update_scope != "all" or model.policy.share_features_extractor:
        raise ValueError("Prepared all-actor, separate-critic policy required")
    if model.policy.optimizer.state or model.critic_optimizer.state:
        raise ValueError(
            "Call prepare_model first; conversion requires explicit fresh moments"
        )
    if model._last_obs is not None or model.rollout_buffer.pos:
        raise ValueError("Convert before collecting any new rollout")
    if model.policy.log_std.shape != (5,):
        raise ValueError("Expected five inherited Gaussian standard deviations")
    for name in ("std_scales", "runtime_arm_scales", "finger_exploration_max"):
        if any((float(v) != 1.0 for v in getattr(model, name, ()))):
            raise ValueError("Prepare unit exploration contexts before gSDE migration")
    if getattr(model, "role_normalization", False) or getattr(
        model, "role_gradient_balance", False
    ):
        raise ValueError("Role-normalization/gradient-balance migration is unsupported")
    kwargs = copy.deepcopy(model.policy_kwargs)
    for key in ("std_scales", "runtime_arm_scales", "finger_exploration_max"):
        kwargs.pop(key, None)
    kwargs.update(
        wide_finger_residual=bool(getattr(model.policy, "wide_finger_residual", False)),
        arm_residual_mode=getattr(model.policy, "arm_residual_mode", None),
    )
    original = model.policy.state_dict()
    with preserved_rng():
        converted = StructuredPPO(
            "MultiInputPolicy",
            model.get_env(),
            learning_rate=model.learning_rate,
            n_steps=model.n_steps,
            batch_size=model.batch_size,
            n_epochs=model.n_epochs,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            clip_range=model.clip_range,
            clip_range_vf=model.clip_range_vf,
            normalize_advantage=model.normalize_advantage,
            ent_coef=model.ent_coef,
            vf_coef=model.vf_coef,
            max_grad_norm=model.max_grad_norm,
            target_kl=model.target_kl,
            sde_sample_freq=resample_steps,
            policy_kwargs=kwargs,
            seed=model.seed,
            device=model.device,
            verbose=model.verbose,
            tensorboard_log=model.tensorboard_log,
            rollout_buffer_class=DictRolloutBuffer,
            critic_learning_rate=model.critic_learning_rate,
            actor_update_scope="all",
        )
        target = converted.policy.state_dict()
        if set(original) != set(target):
            raise ValueError(
                "Migration must preserve every existing mean/value/residual tensor name"
            )
        with torch.no_grad():
            for name, value in original.items():
                if name == "log_std":
                    target[name].copy_(value[None, :].expand_as(target[name]))
                else:
                    if value.shape != target[name].shape:
                        raise ValueError(f"Mean/value tensor shape changed: {name}")
                    target[name].copy_(value)
        converted.policy.load_state_dict(target)
        converted.policy.reset_noise(converted.n_envs)
    for name in ("num_timesteps", "_n_updates", "_current_progress_remaining"):
        setattr(converted, name, getattr(model, name))
    converted.policy.set_training_mode(model.policy.training)
    converted.exploration_migration = dict(
        source_algorithm=getattr(model, "checkpoint_algorithm", None),
        all_mean_and_value_tensors_exact=True,
        optimizer_moments="fresh",
        initial_marginal_std="exact inherited five-vector at every nondegenerate/zero-fallback latent",
        global_rng_preserved=True,
        inherited_steps=int(model.num_timesteps),
        resample_steps=resample_steps,
        source_log_std_shape=list(model.policy.log_std.shape),
        new_log_std_shape=list(converted.policy.log_std.shape),
    )
    return converted


def exploration_report(model):
    """Physical marginal bounds; matrix weights are not one global action STD."""
    if not isinstance(model.policy, StructuredActorCriticPolicy):
        raise ValueError("Structured policy required")
    std = model.policy.log_std.detach().exp().cpu()
    physical = torch.tensor(FORCE_LIMITS, dtype=std.dtype)
    return dict(
        algorithm=ALGORITHM,
        sde_sample_freq=int(model.sde_sample_freq),
        resample_seconds=float(model.sde_sample_freq * 0.02),
        noise_weight_shape=list(std.shape),
        noise_features="unit norm with zero-vector fallback",
        physical_marginal_std_lower_bound=(std.min(0).values * physical).tolist(),
        physical_marginal_std_upper_bound=(std.max(0).values * physical).tolist(),
        log_std_trainable=bool(model.policy.log_std.requires_grad),
        deployment_stochastic="fresh per-action marginal Gaussian",
        rollout_stochastic="persistent state-dependent Gaussian function between matrix resamples",
        likelihood="standard SB3 gSDE per-state marginal, not full correlated-trajectory density",
        mean_force_override=False,
        action_demonstrations=False,
        runtime_controller=False,
    )
