"""Opt-in PPO with separate actor/critic optimizers and KL-gated actor updates.

Adapted from Stable-Baselines3 2.7 PPO.train (MIT; see
licenses/stable-baselines3-MIT.txt): same clipped actor surrogate, entropy,
GAE targets, and optional value clipping. Requires disjoint feature extractors.
KL stopping freezes only the actor for the rest of the rollout's epochs. Critic
fitting continues on those SAME on-policy targets; no extra samples or teacher.
"""

from contextlib import contextmanager
import copy
import math
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.save_util import load_from_zip_file


class DecoupledPPO(PPO):
    def __init__(
        self, *args, critic_learning_rate=0.0003, actor_update_scope="all", **kwargs
    ):
        self.actor_update_scope = self.validate_actor_update_scope(actor_update_scope)
        if not math.isfinite(critic_learning_rate) or critic_learning_rate <= 0:
            raise ValueError("critic_learning_rate must be positive and finite")
        self.critic_learning_rate = float(critic_learning_rate)
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        self._build_optimizers()

    def _build_optimizers(self, previous=None):
        if self.policy.share_features_extractor:
            raise ValueError("DecoupledPPO requires share_features_extractor=False")
        actor_modules = [
            self.policy.pi_features_extractor,
            self.policy.mlp_extractor.policy_net,
            self.policy.action_net,
        ]
        critic_modules = [
            self.policy.vf_features_extractor,
            self.policy.mlp_extractor.value_net,
            self.policy.value_net,
        ]
        self.actor_parameters = list(
            dict.fromkeys((p for m in actor_modules for p in m.parameters()))
        )
        if hasattr(self.policy, "log_std"):
            self.actor_parameters.append(self.policy.log_std)
        base_actor_parameters = self.actor_parameters.copy()
        actor_groups = [{"params": base_actor_parameters}]
        for name in ("wide_finger_head", "arm_residual_head"):
            residual = getattr(self.policy, name, None)
            if residual is not None:
                parameters = list(residual.parameters())
                actor_groups.append({"params": parameters})
                self.actor_parameters.extend(parameters)
        self.critic_parameters = list(
            dict.fromkeys((p for m in critic_modules for p in m.parameters()))
        )
        actor_ids, critic_ids = (
            set(map(id, self.actor_parameters)),
            set(map(id, self.critic_parameters)),
        )
        if actor_ids & critic_ids:
            raise ValueError("Actor and critic parameters overlap")
        if actor_ids | critic_ids != set(map(id, self.policy.parameters())):
            raise ValueError("Unclassified policy parameters; refusing to omit updates")
        old = previous if previous is not None else self.policy.optimizer
        options = dict(self.policy.optimizer_kwargs)
        self.policy.optimizer = self.policy.optimizer_class(
            actor_groups,
            lr=self.lr_schedule(self._current_progress_remaining),
            **options,
        )
        self.critic_optimizer = self.policy.optimizer_class(
            self.critic_parameters, lr=self.critic_learning_rate, **options
        )
        migrated = 0
        if previous is not None:
            for optimizer in (self.policy.optimizer, self.critic_optimizer):
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter in old.state:
                            optimizer.state[parameter] = copy.deepcopy(
                                old.state[parameter]
                            )
                            migrated += 1
        self.algorithm_metadata = dict(
            name="decoupled-ppo-v1",
            actor_kl_stop_only=True,
            separate_gradient_clipping=True,
            critic_learning_rate=self.critic_learning_rate,
            original_gae_returns=True,
            actor_update_scope=self.actor_update_scope,
            finger_head_optimizer_semantics="Frozen arm weights/row moments; shared head Adam step advances",
            arm_head_optimizer_semantics="Frozen finger weights/row moments; shared head Adam step advances",
        )
        self.optimizer_migration = dict(
            mode="per-parameter-state-copy" if previous is not None else "fresh",
            migrated_parameters=migrated,
        )
        self.set_actor_update_scope(self.actor_update_scope)

    @staticmethod
    def validate_actor_update_scope(scope):
        if scope not in (
            "all",
            "finger_head",
            "arm_head",
            "wide_finger",
            "arm_residual",
            "grasp_residual",
        ):
            raise ValueError(
                "actor_update_scope must be 'all', 'finger_head', 'arm_head', 'wide_finger', 'arm_residual', or 'grasp_residual'"
            )
        return scope

    def set_actor_update_scope(self, scope):
        scope = self.validate_actor_update_scope(scope)
        if (
            scope == "wide_finger"
            and getattr(self.policy, "wide_finger_head", None) is None
        ):
            raise ValueError(
                "wide_finger scope requires an enabled wide_finger_residual"
            )
        if (
            scope == "arm_residual"
            and getattr(self.policy, "arm_residual_head", None) is None
        ):
            raise ValueError("arm_residual scope requires an enabled arm_residual_mode")
        if scope == "grasp_residual" and any(
            (
                getattr(self.policy, name, None) is None
                for name in ("arm_residual_head", "wide_finger_head")
            )
        ):
            raise ValueError(
                "grasp_residual scope requires both arm_residual_head and wide_finger_head"
            )
        if scope in ("finger_head", "arm_head"):
            head = self.policy.action_net
            if (
                not isinstance(head, th.nn.Linear)
                or head.out_features != 5
                or head.bias is None
            ):
                raise ValueError(
                    f"{scope} requires a five-output Linear force head with bias"
                )
            if not isinstance(self.policy.optimizer, (th.optim.Adam, th.optim.AdamW)):
                raise ValueError(f"{scope} currently supports Adam/AdamW optimizers")
        self.actor_update_scope = scope
        self.algorithm_metadata["actor_update_scope"] = scope

    def _active_actor_parameters(self):
        if self.actor_update_scope == "grasp_residual":
            return [
                p
                for head in (
                    self.policy.arm_residual_head,
                    self.policy.wide_finger_head,
                )
                for p in head.parameters()
                if p.requires_grad
            ]
        if self.actor_update_scope == "arm_residual":
            return [
                p for p in self.policy.arm_residual_head.parameters() if p.requires_grad
            ]
        if self.actor_update_scope == "wide_finger":
            return [
                p for p in self.policy.wide_finger_head.parameters() if p.requires_grad
            ]
        candidates = (
            list(self.policy.action_net.parameters())
            if self.actor_update_scope in ("finger_head", "arm_head")
            else self.actor_parameters
        )
        return [p for p in candidates if p.requires_grad]

    @contextmanager
    def _actor_scope(self):
        """Temporarily freeze unselected actor parameters and force-head rows."""
        if self.actor_update_scope == "all":
            yield
            return
        self.set_actor_update_scope(self.actor_update_scope)
        if self.actor_update_scope == "grasp_residual":
            selected_heads = (
                self.policy.arm_residual_head,
                self.policy.wide_finger_head,
            )
        else:
            selected_heads = (
                self.policy.arm_residual_head
                if self.actor_update_scope == "arm_residual"
                else self.policy.wide_finger_head
                if self.actor_update_scope == "wide_finger"
                else self.policy.action_net,
            )
        head_ids = {id(p) for head in selected_heads for p in head.parameters()}
        flags = [(p, p.requires_grad) for p in self.actor_parameters]
        hooks = []

        def masked_gradient(gradient):
            result = gradient.clone()
            frozen = (
                slice(-1, None)
                if self.actor_update_scope == "arm_head"
                else slice(None, -1)
            )
            result[frozen] = 0
            return result

        try:
            for p, enabled in flags:
                p.grad = None
                p.requires_grad_(enabled and id(p) in head_ids)
                if p.requires_grad and self.actor_update_scope in (
                    "finger_head",
                    "arm_head",
                ):
                    hooks.append(p.register_hook(masked_gradient))
            yield
        finally:
            for hook in hooks:
                hook.remove()
            for p, enabled in flags:
                p.requires_grad_(enabled)

    def _step_actor(self):
        """Clip active gradients; undo inactive-row Adam momentum/weight decay.

        Entire frozen parameters receive grad=None, so their complete optimizer
        state remains unchanged. Shared head tensors retain exact inactive row
        moments, but their scalar Adam step advances for the active rows.
        Returning to joint mode uses that shared clock; no moment reset occurs.
        """
        snapshots = []
        if self.actor_update_scope in ("finger_head", "arm_head"):
            frozen = (
                slice(-1, None)
                if self.actor_update_scope == "arm_head"
                else slice(None, -1)
            )
            for p in self.policy.action_net.parameters():
                if p.grad is None:
                    continue
                state = self.policy.optimizer.state.get(p, {})
                rows = {
                    k: v[frozen].clone()
                    for k, v in state.items()
                    if isinstance(v, th.Tensor) and v.shape == p.shape
                }
                snapshots.append((p, frozen, p.detach()[frozen].clone(), rows))
        th.nn.utils.clip_grad_norm_(self._active_actor_parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        with th.no_grad():
            for p, frozen, weights, rows in snapshots:
                p[frozen].copy_(weights)
                for key, value in self.policy.optimizer.state[p].items():
                    if isinstance(value, th.Tensor) and value.shape == p.shape:
                        value[frozen].copy_(rows[key]) if key in rows else value[
                            frozen
                        ].zero_()

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "actor_parameters",
            "critic_parameters",
            "critic_optimizer",
        ]

    def _get_torch_save_params(self):
        return (["policy", "policy.optimizer", "critic_optimizer"], [])

    def _actor_distribution(self, batch):
        return self.policy.get_distribution(batch.observations)

    def _actor_distribution_diagnostics(self, batch, ratio, log_ratio, clip_range):
        pass

    def _actor_losses(self, batch, advantages, ratio, entropy, log_prob, clip_range):
        actor_loss = -th.min(
            advantages * ratio,
            advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
        ).mean()
        entropy_loss = -entropy.mean() if entropy is not None else log_prob.mean()
        return (actor_loss, entropy_loss)

    def _normalize_actor_advantages(self, batch):
        """Default PPO minibatch normalization; subclasses may specialize only this."""
        advantages = batch.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)
        return advantages

    def train(self):
        with self._actor_scope():
            return self._train_impl()

    def _train_impl(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (
            None
            if self.clip_range_vf is None
            else self.clip_range_vf(self._current_progress_remaining)
        )
        actor_active = True
        # A KL stop freezes only the actor; the independent critic can still fit returns.
        actor_updates = critic_updates = 0
        policy_losses, value_losses, entropies, kls, fractions = ([], [], [], [], [])
        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                actions = (
                    batch.actions.long().flatten()
                    if isinstance(self.action_space, spaces.Discrete)
                    else batch.actions
                )
                if actor_active:
                    distribution = self._actor_distribution(batch)
                    log_prob, entropy = (
                        distribution.log_prob(actions),
                        distribution.entropy(),
                    )
                    advantages = self._normalize_actor_advantages(batch)
                    ratio = th.exp(log_prob - batch.old_log_prob)
                    with th.no_grad():
                        log_ratio = log_prob - batch.old_log_prob
                        kl = float((th.exp(log_ratio) - 1 - log_ratio).mean().cpu())
                    self._actor_distribution_diagnostics(
                        batch, ratio, log_ratio, clip_range
                    )
                    kls.append(kl)
                    if self.target_kl is not None and kl > 1.5 * self.target_kl:
                        actor_active = False
                    else:
                        actor_loss, entropy_loss = self._actor_losses(
                            batch, advantages, ratio, entropy, log_prob, clip_range
                        )
                        self.policy.optimizer.zero_grad()
                        (actor_loss + self.ent_coef * entropy_loss).backward()
                        self._step_actor()
                        actor_updates += 1
                        policy_losses.append(float(actor_loss.detach().cpu()))
                        entropies.append(float(entropy_loss.detach().cpu()))
                        fractions.append(
                            float(
                                (th.abs(ratio - 1) > clip_range)
                                .float()
                                .mean()
                                .detach()
                                .cpu()
                            )
                        )
                values = self.policy.predict_values(batch.observations).flatten()
                values_pred = (
                    values
                    if clip_range_vf is None
                    else batch.old_values
                    + th.clamp(values - batch.old_values, -clip_range_vf, clip_range_vf)
                )
                value_loss = F.mse_loss(batch.returns, values_pred)
                if self.vf_coef != 0:
                    self.critic_optimizer.zero_grad()
                    (self.vf_coef * value_loss).backward()
                    th.nn.utils.clip_grad_norm_(
                        self.critic_parameters, self.max_grad_norm
                    )
                    self.critic_optimizer.step()
                    critic_updates += 1
                value_losses.append(float(value_loss.detach().cpu()))
            self._n_updates += 1
        self.last_update_stats = dict(
            actor_updates=actor_updates,
            critic_updates=critic_updates,
            actor_kl_stopped=not actor_active,
            mean_value_loss=float(np.mean(value_losses)),
            max_approx_kl=max(kls, default=0.0),
        )
        for name, values in [
            ("policy_gradient_loss", policy_losses),
            ("value_loss", value_losses),
            ("entropy_loss", entropies),
            ("approx_kl", kls),
            ("clip_fraction", fractions),
        ]:
            self.logger.record(
                "train/" + name, float(np.mean(values)) if values else 0.0
            )
        for key, value in self.last_update_stats.items():
            self.logger.record("train/" + key, value)
        self.logger.record(
            "train/explained_variance",
            explained_variance(
                self.rollout_buffer.values.flatten(),
                self.rollout_buffer.returns.flatten(),
            ),
        )
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_learning_rate", self.critic_learning_rate)
        self.logger.record("train/clip_range", clip_range)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", float(th.exp(self.policy.log_std).mean().detach().cpu())
            )


def load_decoupled_ppo(
    path,
    env,
    device="auto",
    *,
    critic_learning_rate=0.0003,
    actor_update_scope=None,
    allow_actor_reconfigure=False,
):
    """Convert ordinary PPO, copying actor and critic Adam moments individually.

    Actor/value parameters and saved rollout hyperparameters remain exact.
    Optimizer groups are split, so original joint gradient-clipping semantics
    deliberately change; both per-parameter histories are retained.
    """
    if not math.isfinite(critic_learning_rate) or critic_learning_rate <= 0:
        raise ValueError("critic_learning_rate must be positive and finite")
    from .policy_loading import checkpoint_algorithm

    if checkpoint_algorithm(path) == "context-decoupled-ppo-v1":
        raise ValueError(
            "Context checkpoint requires load_grasp_policy or load_context_exploration_ppo; base PPO likelihoods would be incorrect"
        )
    _, parameters, _ = load_from_zip_file(path, device=device)
    if parameters is not None and "critic_optimizer" in parameters:
        model = DecoupledPPO.load(path, env=env, device=device)
        desired_scope = (
            model.actor_update_scope
            if actor_update_scope is None
            else model.validate_actor_update_scope(actor_update_scope)
        )
        if desired_scope != model.actor_update_scope and (not allow_actor_reconfigure):
            raise ValueError(
                "Changing actor update scope requires allow_actor_reconfigure=True"
            )
        model.set_actor_update_scope(desired_scope)
        model.critic_learning_rate = float(critic_learning_rate)
        for group in model.critic_optimizer.param_groups:
            group["lr"] = model.critic_learning_rate
        model.algorithm_metadata["critic_learning_rate"] = model.critic_learning_rate
        model.optimizer_migration = dict(
            mode="resumed-both-optimizer-states",
            migrated_parameters=sum(
                (
                    len(opt.state)
                    for opt in (model.policy.optimizer, model.critic_optimizer)
                )
            ),
        )
        return model
    original = PPO.load(path, env=env, device=device)
    converted = DecoupledPPO(original.policy_class, env, _init_setup_model=False)
    converted.__dict__.update(original.__dict__)
    converted.critic_learning_rate = float(critic_learning_rate)
    converted._build_optimizers(previous=original.policy.optimizer)
    converted.set_actor_update_scope(actor_update_scope or "all")
    return converted
