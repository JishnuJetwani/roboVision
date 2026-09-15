"""Pure PPO over approach continuation and irreversible, terminal pickup.

Every expert still infers all five force actions at 50 Hz. The learned binary
manager sees the original pixels/proprioception only. A pickup commitment ends
only when the physical task ends; no geometric rule chooses that commitment.
"""

from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from .centered_arrival_env import CenteredArrivalGraspEnv, load_centered_arrival_pool
from .centered_grasp_env import CenteredGraspEnv
from .cnn import NormalizedGraspCNN
from .generalization_env import GeneralizationGraspEnv
from .grasp_benchmark import preserved_rng
from .hierarchical_policy import assert_parameter_independence
from .io import atomic_json
from .policy_state import policy_state_hash
from .ordered_pickup_policy import (
    OrderedPickupEnv,
    OrderedPickupPolicy,
    freeze_pickup_experts,
)
from .skill_hard_evaluation import basename, resolve_skill_checkpoint, sha256
from .manager_support import (
    PHYSICAL_GAMMA,
    PosthocCenteredAudit,
    _source,
    initialize_manager_features,
    manager_duration_settings,
    manager_evaluation_cases,
    validate_manager_duration,
)
from .training_checkpoint import (
    begin_chunk,
    load_resume,
    save_checkpoint,
    _parameter_digest,
)

VERSION = "ordered-terminal-pickup-manager-v1"
WORKER_ROLES = (
    "full_start_100_140",
    "full_start_25_100",
    "approach_arrival_a",
    "approach_arrival_b",
)
DEFAULT_RESET_RECIPE = "arrivals-v1"
RESET_WORKER_ROLES = {
    DEFAULT_RESET_RECIPE: WORKER_ROLES,
    "full-start-v2": (
        "full_start_100_140",
        "full_start_25_100",
        "full_start_100_140",
        "full_start_25_100",
    ),
}
N_ENVS = 4
ROLLOUT_STEPS = 32
ROLLOUT_TRANSITIONS = 128
MAX_PHYSICAL_PER_DECISION = 500
CHUNK_PHYSICAL_UPPER_BOUND = ROLLOUT_TRANSITIONS * MAX_PHYSICAL_PER_DECISION
GATE_PHYSICAL_INTERVAL = 49152
INITIAL_COMMIT_PROBABILITY = 0.05


def ordered_worker_roles(reset_recipe=DEFAULT_RESET_RECIPE):
    if not isinstance(reset_recipe, str) or reset_recipe not in RESET_WORKER_ROLES:
        raise ValueError(
            "Explicit registered reset_recipe arrivals-v1 or full-start-v2 required"
        )
    return RESET_WORKER_ROLES[reset_recipe]


def recorded_reset_recipe(metadata, *, context):
    """Legacy records imply arrivals-v1 only when all four roles match exactly."""
    recipe = metadata.get("reset_recipe", DEFAULT_RESET_RECIPE)
    try:
        roles = ordered_worker_roles(recipe)
    except ValueError as error:
        raise ValueError(f"{context} reset recipe is not registered") from error
    if metadata.get("worker_roles") != list(roles):
        raise ValueError(
            f"{context} curriculum worker roles differ from its reset recipe"
        )
    return recipe


def validate_training_reward_scale(scale):
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or scale not in (1.0, 0.01)
    ):
        raise ValueError("training_reward_scale must be the registered value1.0 or0.01")
    return float(scale)


class OrderedTrainingRewardScale(gym.RewardWrapper):
    """Uniform training units after option discounting; all info stays raw."""

    def __init__(self, env, scale=1.0):
        self.scale = validate_training_reward_scale(scale)
        super().__init__(env)
        self.manager_gamma = env.manager_gamma

    def reward(self, reward):
        return reward * self.scale

    def specification(self):
        return dict(
            self.env.specification(),
            training_reward_scale=self.scale,
            training_reward_units="Uniform multiplier on already-discounted option reward; all physical reward info remains raw",
        )


def ordered_duration_settings(option_steps=5):
    duration = manager_duration_settings(option_steps)
    return dict(
        option_steps=option_steps,
        manager_gamma=duration["manager_gamma"],
        approach_option_seconds=duration["option_seconds"],
        physical_control_hz=50,
        pickup_commitment="Irreversible until true physical task termination",
        maximum_physical_actions_per_decision=MAX_PHYSICAL_PER_DECISION,
        checkpoint_chunk_manager_steps=ROLLOUT_TRANSITIONS,
        checkpoint_chunk_physical_upper_bound=CHUNK_PHYSICAL_UPPER_BOUND,
        gate_interval_physical_steps=GATE_PHYSICAL_INTERVAL,
    )


def planned_ordered_steps(remaining_physical_steps):
    """Reserve an entire worst-case rollout; never assume a short commitment."""
    if type(remaining_physical_steps) is not int or remaining_physical_steps < 0:
        raise ValueError("Remaining physical budget must be a nonnegative integer")
    return (
        ROLLOUT_TRANSITIONS
        if remaining_physical_steps >= CHUNK_PHYSICAL_UPPER_BOUND
        else 0
    )


def ordered_recovery_bounds(recovery, *, saved_physical_steps):
    lower = recovery["lost_interactions_lower_bound"]
    upper = recovery["lost_interactions_upper_bound"]
    if (
        type(lower) is not int
        or lower < 0
        or (upper is not None and (type(upper) is not int or upper < lower))
    ):
        raise RuntimeError("Invalid lost manager interaction bounds")
    if upper is None:
        if saved_physical_steps or lower:
            raise RuntimeError(
                "Cannot bound missing manager work without a chunk intent"
            )
        upper = 0
    return dict(
        **recovery,
        lost_physical_steps_lower_bound=lower,
        lost_physical_steps_upper_bound=upper * MAX_PHYSICAL_PER_DECISION,
        physical_loss_bound="Each lost decision cost between1 and500 physical actions",
    )


def validate_ordered_manager(manager, option_steps=5):
    validate_manager_duration(manager, option_steps)
    if (
        not isinstance(manager.action_space, gym.spaces.Discrete)
        or manager.action_space.n != 2
    ):
        raise ValueError("Ordered pickup requires a binary categorical manager")


def initialize_commit_prior(manager):
    """Architectural prior only; no labels, demonstrations or action targets."""
    if manager.policy.optimizer.state:
        raise ValueError("Commit prior requires a fresh optimizer")
    head = manager.policy.action_net
    if (
        not isinstance(head, torch.nn.Linear)
        or head.out_features != 2
        or head.bias is None
    ):
        raise ValueError("Commit prior requires a fresh two-logit linear action head")
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(head.bias.new_tensor([np.log(0.95), np.log(0.05)]))
    return dict(
        initial_commit_probability=INITIAL_COMMIT_PROBABILITY,
        action_head_weights="Zero",
        action_head_biases=[float(np.log(0.95)), float(np.log(0.05))],
        architectural_prior=True,
        labels_or_demonstrations=False,
        semantics="Action0 continues approach; action1 commits pickup until task termination",
    )


def _ordered_ppo_config(option_steps):
    return dict(
        learning_rate=0.0001,
        n_steps=ROLLOUT_STEPS,
        n_envs=N_ENVS,
        batch_size=64,
        n_epochs=5,
        gamma=ordered_duration_settings(option_steps)["manager_gamma"],
        gae_lambda=0.98,
        ent_coef=0.01,
    )


def resolve_ordered_continuation(
    volume_root,
    manager_source,
    expert_records,
    arrival_record,
    *,
    option_steps,
    training_reward_scale,
    reset_recipe=DEFAULT_RESET_RECIPE,
):
    """Require the same task, fixed experts and reward units before warm loading."""
    if not isinstance(manager_source, dict) or set(manager_source) != {
        "run",
        "checkpoint",
    }:
        raise ValueError("Warm ordered manager requires an exact run/checkpoint source")
    path, record, metadata, metadata_path = resolve_skill_checkpoint(
        volume_root, manager_source["run"], manager_source["checkpoint"]
    )
    if metadata_path.name != "summary.json" or metadata.get("status") != "complete":
        raise ValueError(
            "Warm manager requires a completed source run with stable summary provenance"
        )
    if (
        metadata.get("version") != VERSION
        or metadata.get("identity", {}).get("recipe") != VERSION
        or metadata.get("manager_algorithm") != "PPO binary categorical"
    ):
        raise ValueError(
            "Warm manager source must be the registered ordered PPO recipe"
        )

    def expert_identity(rows):
        if not isinstance(rows, (list, tuple)) or len(rows) != 2:
            raise ValueError("Warm source must identify its exact two experts")
        return [
            {key: row.get(key) for key in ("run", "checkpoint", "policy_sha256")}
            for row in rows
        ]

    requested_experts = expert_identity(expert_records)
    if (
        expert_identity(metadata.get("experts")) != requested_experts
        or expert_identity(metadata["identity"].get("source")) != requested_experts
    ):
        raise ValueError(
            "Warm manager requires the exact same expert sources and checkpoint hashes"
        )
    recorded_arrival = metadata.get("arrival_source", {})
    if any(
        (
            recorded_arrival.get(key) != arrival_record.get(key)
            for key in ("run", "sha256")
        )
    ):
        raise ValueError("Warm manager requires the exact same approach arrival pool")
    if (
        metadata.get("duration", {}).get("option_steps") != option_steps
        or metadata.get("duration", {}).get("manager_gamma")
        != PHYSICAL_GAMMA**option_steps
        or metadata.get("physical_gamma") != PHYSICAL_GAMMA
    ):
        raise ValueError("Warm manager duration or physical discount differs")
    if metadata.get("training_reward_scale") != training_reward_scale:
        raise ValueError(
            "Warm manager must explicitly record the same training reward scale"
        )
    if metadata.get("ppo") != _ordered_ppo_config(option_steps):
        raise ValueError(
            "Warm source PPO configuration differs from the registered recipe"
        )
    target_roles = ordered_worker_roles(reset_recipe)
    source_reset_recipe = recorded_reset_recipe(metadata, context="Warm manager")
    if (
        "reset_recipe" in metadata.get("identity", {})
        and metadata["identity"]["reset_recipe"] != source_reset_recipe
    ):
        raise ValueError("Warm manager reset recipe differs from its source identity")
    source_steps = (
        record.get("final_steps")
        if manager_source["checkpoint"] == "policy.zip"
        else record.get("steps")
    )
    if type(source_steps) is not int or source_steps < 0:
        raise ValueError(
            "Warm source must record its exact inherited manager step counter"
        )
    return (
        path,
        dict(
            **manager_source,
            policy_sha256=sha256(path),
            metadata_file=metadata_path.name,
            metadata_sha256=sha256(metadata_path),
            source_identity=metadata["identity"],
            expected_manager_steps=source_steps,
            source_new_physical_steps=record.get("new_physical_steps"),
            source_training_reward_scale=training_reward_scale,
            source_parent=metadata.get("manager_source"),
            source_reset_recipe=source_reset_recipe,
            target_reset_recipe=reset_recipe,
            source_worker_roles=list(ordered_worker_roles(source_reset_recipe)),
            target_worker_roles=list(target_roles),
            reset_distribution_changed=source_reset_recipe != reset_recipe,
            exact_recipe_match=source_reset_recipe == reset_recipe,
            exact_non_reset_recipe_match=True,
        ),
    )


def load_warm_ordered_manager(
    path, env, *, option_steps, seed, expected_manager_steps, device="cuda"
):
    """Restore PPO/Adam exactly; only simulator episodes and RNG start fresh."""
    from stable_baselines3.common.save_util import load_from_zip_file

    saved_data, saved_parameters, _ = load_from_zip_file(path, device="cpu")
    if set(saved_parameters) != {"policy", "policy.optimizer"}:
        raise ValueError(
            "Warm PPO ZIP must contain both complete policy and optimizer state"
        )
    expected = _parameter_digest(saved_parameters)
    manager = PPO.load(path, env=env, device=device, force_reset=True)
    validate_ordered_manager(manager, option_steps)
    expected_settings = dict(
        _ordered_ppo_config(option_steps),
        normalize_advantage=True,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        use_sde=False,
        sde_sample_freq=-1,
    )
    for key, expected_value in expected_settings.items():
        actual = getattr(manager, key, None)
        if (
            actual != expected_value
            or saved_data.get(key) != expected_value
            or (isinstance(expected_value, bool) and actual is not expected_value)
        ):
            raise ValueError(f"Warm loaded PPO configuration differs: {key}")
    for name, target in (("lr_schedule", 0.0001), ("clip_range", 0.2)):
        schedule = getattr(manager, name, None)
        if not callable(schedule) or any(
            (
                not math.isclose(float(schedule(p)), target, rel_tol=0.0, abs_tol=1e-12)
                for p in (0.0, 0.5, 1.0)
            )
        ):
            raise ValueError(
                f"Warm PPO {name} must retain its registered constant schedule"
            )
    if (
        manager.clip_range_vf is not None
        or manager.policy.share_features_extractor
        or (not manager.policy.normalize_images)
        or (manager.policy.net_arch != [128, 128])
        or (not isinstance(manager.policy.optimizer, torch.optim.Adam))
        or (type(manager.policy.pi_features_extractor) is not NormalizedGraspCNN)
        or (type(manager.policy.vf_features_extractor) is not NormalizedGraspCNN)
        or (manager.action_space.start != 0)
    ):
        raise ValueError(
            "Warm PPO value/feature/optimizer setup differs from the registered recipe"
        )
    if int(manager.num_timesteps) != expected_manager_steps:
        raise ValueError("Warm manager step counter differs from its source record")
    loaded = _parameter_digest(manager.get_parameters())
    if loaded != expected:
        raise RuntimeError("Warm PPO load changed saved actor, critic or Adam tensors")
    counters = dict(
        num_timesteps=int(manager.num_timesteps),
        n_updates=int(manager._n_updates),
        episode_num=int(manager._episode_num),
    )
    manager.seed = seed
    manager.set_random_seed(seed)
    actual_seeds = list(env.seed(seed))
    manager._last_obs = None
    manager._last_original_obs = None
    manager._last_episode_starts = np.ones(manager.n_envs, dtype=bool)
    manager.rollout_buffer.reset()
    prepared = _parameter_digest(manager.get_parameters())
    if prepared != expected or counters != dict(
        num_timesteps=int(manager.num_timesteps),
        n_updates=int(manager._n_updates),
        episode_num=int(manager._episode_num),
    ):
        raise RuntimeError(
            "Fresh warm-run setup changed learned parameters, Adam or inherited counters"
        )
    return (
        manager,
        dict(
            parameter_sha256_in_source=expected,
            parameter_sha256_after_load=loaded,
            parameter_sha256_before_training=prepared,
            policy_sha256_before_training=_parameter_digest(saved_parameters["policy"]),
            optimizer_sha256_before_training=_parameter_digest(
                saved_parameters["policy.optimizer"]
            ),
            inherited_counters=counters,
            optimizer_state_entries=len(manager.policy.optimizer.state),
            new_seed=seed,
            actual_queued_worker_seeds=actual_seeds,
            fresh_simulator_episodes=True,
            fresh_global_rng=True,
            rollout_buffer_empty=True,
            source_simulator_state_restored=False,
            feature_initialization_repeated=False,
            commit_prior_initialization_repeated=False,
            exact_policy_and_adam_restore=True,
        ),
    )


def _freeze_pair(experts):
    return freeze_pickup_experts(experts)


def _validate_sources(sources):
    if not isinstance(sources, (list, tuple)) or len(sources) != 2:
        raise ValueError("Provide two explicit ordered approach/pickup source pairs")
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"run", "checkpoint"}:
            raise ValueError("Each source requires exactly run and checkpoint")
        basename(source["run"])
        basename(source["checkpoint"])
        if not source["checkpoint"].endswith(".zip"):
            raise ValueError("Each source checkpoint must be an explicit ZIP")


def load_ordered_arrival(volume_root, arrival_source):
    if not isinstance(arrival_source, dict) or set(arrival_source) != {"run", "sha256"}:
        raise ValueError("One exact approach arrival source requires run and sha256")
    basename(arrival_source["run"])
    expected = arrival_source["sha256"]
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any((c not in "0123456789abcdef" for c in expected))
    ):
        raise ValueError("Approach arrival source requires an exact lowercase SHA256")
    path = Path(volume_root) / arrival_source["run"] / "approach-arrivals.pkl.gz"
    pool = load_centered_arrival_pool(
        path, expected_sha256=expected, expected_phase="approach"
    )
    return (
        pool,
        dict(**arrival_source, filename=path.name, pool_provenance=pool["provenance"]),
        path,
    )


def make_ordered_env(
    worker,
    seed,
    experts,
    arrival_pool,
    *,
    option_steps=5,
    render_images=True,
    training_reward_scale=1.0,
    reset_recipe=DEFAULT_RESET_RECIPE,
):
    ordered_duration_settings(option_steps)
    scale = validate_training_reward_scale(training_reward_scale)
    if type(worker) is not int or worker not in range(N_ENVS):
        raise ValueError("Ordered manager requires four explicit reset workers")
    roles = ordered_worker_roles(reset_recipe)
    kwargs = dict(
        seed=seed + worker,
        observation="pixels",
        render_images=render_images,
        gamma=PHYSICAL_GAMMA,
    )
    if roles[worker].startswith("full_start_"):
        low, high = (
            (0.1, 0.14) if roles[worker] == "full_start_100_140" else (0.025, 0.1)
        )
        base = CenteredGraspEnv(height_bands=((low, high, 1.0),), **kwargs)
    else:
        base = CenteredArrivalGraspEnv(arrival_pool=arrival_pool, **kwargs)
    ordered = OrderedPickupEnv(
        base,
        experts,
        option_steps=option_steps,
        physical_gamma=PHYSICAL_GAMMA,
        expert_deterministic=True,
    )
    return Monitor(OrderedTrainingRewardScale(ordered, scale))


class OrderedAudit(BaseCallback):
    def __init__(
        self, saved=None, *, option_steps=5, reset_recipe=DEFAULT_RESET_RECIPE
    ):
        super().__init__()
        ordered_duration_settings(option_steps)
        self.worker_roles = ordered_worker_roles(reset_recipe)
        self.reset_recipe = reset_recipe
        if (
            saved is not None
            and recorded_reset_recipe(saved, context="Saved audit") != reset_recipe
        ):
            raise ValueError("Saved audit reset recipe differs from this recipe")
        saved = saved or {}
        if saved.get("option_steps", option_steps) != option_steps:
            raise ValueError("Saved audit duration differs from this recipe")
        self.option_steps = option_steps
        self.physical_steps = int(saved.get("physical_steps", 0))
        self.episodes = list(saved.get("episodes", []))
        if any(
            (
                type(row.get("worker")) is not int
                or row["worker"] not in range(N_ENVS)
                or row.get("worker_role") != self.worker_roles[row["worker"]]
                for row in self.episodes
            )
        ):
            raise ValueError("Saved audit episode role differs from this recipe")
        self.worker_physical_steps = list(
            saved.get("worker_physical_steps", [0] * N_ENVS)
        )
        self.worker_decisions = list(saved.get("worker_decisions", [0] * N_ENVS))
        self.worker_commits = list(saved.get("worker_commits", [0] * N_ENVS))
        self.worker_expert_actions = [
            list(row)
            for row in saved.get(
                "worker_expert_actions", [[0, 0] for _ in range(N_ENVS)]
            )
        ]

    def _on_step(self):
        infos, dones = (self.locals["infos"], self.locals["dones"])
        if len(infos) != N_ENVS or len(dones) != N_ENVS:
            raise RuntimeError("Expected four ordered workers")
        for worker, (info, done) in enumerate(zip(infos, dones)):
            count, option = (info["option_steps_executed"], info["option_index"])
            if (
                type(count) is not int
                or not 1 <= count <= MAX_PHYSICAL_PER_DECISION
                or option not in (0, 1)
                or (option == 0 and count > self.option_steps)
                or (option == 1 and (not done))
            ):
                raise RuntimeError("Ordered option duration/terminal contract violated")
            self.physical_steps += count
            self.worker_physical_steps[worker] += count
            self.worker_decisions[worker] += 1
            self.worker_commits[worker] += int(option == 1)
            self.worker_expert_actions[worker][option] += count
            if done:
                self.episodes.append(
                    dict(
                        worker=worker,
                        worker_role=self.worker_roles[worker],
                        manager_steps=int(self.model.num_timesteps),
                        physical_steps=self.physical_steps,
                        task_success=bool(info["is_success"]),
                        centered_success=bool(info["centered_success"]),
                        original_success=bool(info["original_success_ever"]),
                        reason=info["reason"],
                        committed=bool(option == 1),
                        episode_physical_steps=info["episode_physics_steps"],
                        expert_action_counts=info["expert_action_counts"],
                        physical_return=info["episode_physical_return"],
                        height=info.get("reset_height"),
                        offset=info.get("cup_offset"),
                        arrival_source_phase=info.get("arrival_source_phase"),
                        arrival_source_step=info.get("arrival_source_step"),
                        **info["episode"],
                    )
                )
        return True

    def state(self):
        return dict(
            option_steps=self.option_steps,
            reset_recipe=self.reset_recipe,
            worker_roles=list(self.worker_roles),
            physical_steps=self.physical_steps,
            episodes=self.episodes,
            worker_physical_steps=self.worker_physical_steps,
            worker_decisions=self.worker_decisions,
            worker_commits=self.worker_commits,
            worker_expert_actions=self.worker_expert_actions,
        )

    def report(self):
        workers = []
        for worker, role in enumerate(self.worker_roles):
            rows = [row for row in self.episodes if row["worker"] == worker]
            workers.append(
                dict(
                    worker=worker,
                    role=role,
                    completed_episodes=len(rows),
                    task_successes=sum((r["task_success"] for r in rows)),
                    centered_successes=sum((r["centered_success"] for r in rows)),
                    original_successes=sum((r["original_success"] for r in rows)),
                    physical_actions=self.worker_physical_steps[worker],
                    manager_decisions=self.worker_decisions[worker],
                    commit_decisions=self.worker_commits[worker],
                    physical_actions_by_expert=self.worker_expert_actions[worker],
                )
            )
        return dict(
            scope="All retained training interactions; lost-work bounds reported separately",
            reset_recipe=self.reset_recipe,
            worker_roles=list(self.worker_roles),
            completed_episodes=len(self.episodes),
            task_successes=sum((r["task_success"] for r in self.episodes)),
            centered_successes=sum((r["centered_success"] for r in self.episodes)),
            original_successes=sum((r["original_success"] for r in self.episodes)),
            commit_decisions=sum(self.worker_commits),
            physical_actions=self.physical_steps,
            manager_decisions=sum(self.worker_decisions),
            by_worker=workers,
            failures=dict(
                Counter((r["reason"] for r in self.episodes if not r["task_success"]))
            ),
        )


def optimizer_report(manager):
    values = getattr(getattr(manager, "logger", None), "name_to_value", {})
    logged = {
        key: float(value)
        for key, value in values.items()
        if key.startswith("train/")
        and isinstance(value, (int, float, np.number))
        and np.isfinite(value)
    }
    steps = [
        float(state["step"])
        for state in manager.policy.optimizer.state.values()
        if "step" in state
    ]
    return dict(
        logged=logged,
        ppo_reported_n_updates=int(manager._n_updates),
        optimizer_parameter_steps_min=min(steps) if steps else 0,
        optimizer_parameter_steps_max=max(steps) if steps else 0,
        reported_epochs_are_not_claimed_as_accepted_minibatch_updates=True,
    )


def manager_decision_probabilities(manager, observation):
    """Read binary probabilities without sampling, changing RNG, or training mode."""
    was_training = manager.policy.training
    with preserved_rng():
        try:
            manager.policy.eval()
            with torch.no_grad():
                tensors, _ = manager.policy.obs_to_tensor(observation)
                distribution = manager.policy.get_distribution(tensors).distribution
                values = (
                    distribution.probs.detach()
                    .to(device="cpu", dtype=torch.float64)
                    .numpy()
                )
            if (
                values.shape != (1, 2)
                or not np.isfinite(values).all()
                or (values < 0.0).any()
                or (values > 1.0).any()
                or (not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-06))
            ):
                raise ValueError(
                    "Manager decision probabilities must be two finite normalized values"
                )
            return values[0].tolist()
        finally:
            manager.policy.train(was_training)


def evaluate_ordered(
    manager,
    experts,
    *,
    final=False,
    seed=460100,
    option_steps=5,
    profile="fixed",
    centered_task=True,
    progress=lambda rows: None,
    registered_cases=None,
    deterministic_modes=None,
):
    """Frozen full starts; physical actions counted independently of decisions."""
    validate_ordered_manager(manager, option_steps)
    if type(centered_task) is not bool:
        raise ValueError("centered_task must be explicit boolean")
    if registered_cases is None:
        cases = manager_evaluation_cases(profile=profile, final=final, seed=seed)
    else:
        if (
            not isinstance(registered_cases, list)
            or not 1 <= len(registered_cases) <= 40
        ):
            raise ValueError("Registered evaluation requires1–40 explicit cases")
        cases = []
        for case in registered_cases:
            if (
                not isinstance(case, dict)
                or set(case) != {"height", "offset"}
                or (not isinstance(case["height"], (int, float)))
                or (not 0.025 <= case["height"] <= 0.14)
                or (np.asarray(case["offset"]).shape != (2,))
                or (not np.isfinite(case["offset"]).all())
                or (np.any(np.asarray(case["offset"]) != 0.0))
            ):
                raise ValueError(
                    "Registered evaluation requires a centered start at 25–140 mm"
                )
            cases.append(
                dict(
                    height=float(case["height"]),
                    offset=list(map(float, case["offset"])),
                )
            )
    modes = (
        ((True, False) if final or profile == "fresh40" else (True,))
        if deterministic_modes is None
        else deterministic_modes
    )
    if (
        not isinstance(modes, (tuple, list))
        or not modes
        or len(set(modes)) != len(modes)
        or any((type(mode) is not bool for mode in modes))
        or (type(seed) is not int)
        or (seed < 0)
        or (seed + len(cases) * len(modes) - 1 >= 700000000)
    ):
        raise ValueError(
            "Explicit unique boolean evaluation modes and development seeds required"
        )
    models = (manager, *experts)
    before = [_parameter_digest(model.get_parameters()) for model in models]
    counters = [int(model.num_timesteps) for model in models]
    training = [model.policy.training for model in models]
    rows = []
    with preserved_rng():
        try:
            for deterministic in modes:
                for index, case in enumerate(cases):
                    episode_seed = seed + len(rows)
                    env_class = (
                        CenteredGraspEnv if centered_task else GeneralizationGraspEnv
                    )
                    env = env_class(
                        fixed_height=case["height"],
                        seed=episode_seed,
                        observation="pixels",
                        gamma=PHYSICAL_GAMMA,
                    )
                    try:
                        stack = OrderedPickupPolicy(
                            manager,
                            experts,
                            option_steps=option_steps,
                            expert_deterministic=True,
                        )
                        torch.manual_seed(episode_seed)
                        obs, info = env.reset(seed=episode_seed)
                        stack.reset()
                        center = PosthocCenteredAudit()
                        decisions, trace, total, physical_actions = ([], [], 0.0, 0)
                        while True:
                            previous_decisions = stack.manager_decisions
                            before_state = dict(
                                physical_action=physical_actions,
                                time=physical_actions * env.control_dt,
                                hand=env.grasp_position.tolist(),
                                cup=env.cup_position.tolist(),
                                proprio=np.asarray(obs["proprio"]).tolist(),
                                image_sha256=hashlib.sha256(
                                    np.ascontiguousarray(obs["image"]).tobytes()
                                ).hexdigest(),
                            )
                            action, _ = stack.predict(obs, deterministic=deterministic)
                            decision = stack.manager_decisions != previous_decisions
                            if decision:
                                decisions.append(
                                    dict(
                                        **before_state,
                                        selected_option=stack.current_option,
                                        manager_probabilities=manager_decision_probabilities(
                                            manager, obs
                                        ),
                                        first_force_action=np.asarray(action).tolist(),
                                    )
                                )
                            obs, reward, done, truncated, info = env.step(action)
                            physical_actions += 1
                            total += reward
                            metrics = center.observe(
                                env, info, physical_actions * env.control_dt
                            )
                            if (
                                decision
                                or physical_actions % 10 == 0
                                or done
                                or truncated
                            ):
                                trace.append(
                                    dict(
                                        physical_action=physical_actions,
                                        option=stack.current_option,
                                        action=np.asarray(action).tolist(),
                                        centered_metrics=metrics,
                                        contacts=info["contacts"],
                                        clearance=info["clearance"],
                                    )
                                )
                            if done or truncated:
                                break
                            if physical_actions >= MAX_PHYSICAL_PER_DECISION:
                                raise RuntimeError(
                                    "Evaluation exceeded the registered physical task horizon"
                                )
                        observed = center.report()
                        if centered_task:
                            observed["observation_window"] = (
                                "Centered task termination; shallow original success does not terminate"
                            )
                        endpoint = env.grasp_position - env.cup_position
                        commitment = next(
                            (row for row in decisions if row["selected_option"] == 1),
                            None,
                        )
                        rows.append(
                            dict(
                                **case,
                                case_index=index,
                                seed=episode_seed,
                                deterministic=deterministic,
                                task_success=bool(info["is_success"]),
                                original_success=bool(
                                    info["original_success_ever"]
                                    if centered_task
                                    else info["is_success"]
                                ),
                                centered_task_success=bool(info["centered_success"])
                                if centered_task
                                else None,
                                physical_steps=physical_actions,
                                manager_decisions=stack.manager_decisions,
                                committed=any(
                                    (row["selected_option"] == 1 for row in decisions)
                                ),
                                reason=info["reason"],
                                commit_step=commitment["physical_action"]
                                if commitment
                                else None,
                                commit_z_relative_m=float(
                                    commitment["hand"][2] - commitment["cup"][2]
                                )
                                if commitment
                                else None,
                                endpoint_relative_xyz_m=endpoint.tolist(),
                                endpoint_xy_error_m=float(np.linalg.norm(endpoint[:2])),
                                endpoint_z_relative_m=float(endpoint[2]),
                                expert_action_counts=list(stack.expert_action_counts),
                                episode_return=total,
                                option_history=list(stack.option_history),
                                decision_trace=decisions,
                                trace=trace,
                                **observed,
                            )
                        )
                        progress(rows)
                    finally:
                        env.close()
        finally:
            for model, was_training in zip(models, training):
                model.policy.train(was_training)
    after = [_parameter_digest(model.get_parameters()) for model in models]
    if before != after or counters != [int(model.num_timesteps) for model in models]:
        raise RuntimeError(
            "Frozen ordered evaluation changed policy/optimizer tensors or counters"
        )
    scores = {}
    for deterministic in (True, False):
        selected = [row for row in rows if row["deterministic"] == deterministic]
        if selected:
            high = [row for row in selected if row["height"] >= 0.1]
            scores["deterministic" if deterministic else "stochastic_manager"] = dict(
                episodes=len(selected),
                centered_successes=sum((r["centered_success"] for r in selected)),
                task_successes=sum((r["task_success"] for r in selected)),
                original_successes=sum((r["original_success"] for r in selected)),
                commit_episodes=sum((r["committed"] for r in selected)),
                high100plus_episodes=len(high),
                high100plus_centered_successes=sum(
                    (r["centered_success"] for r in high)
                ),
                physical_actions=sum((r["physical_steps"] for r in selected)),
                manager_decisions=sum((r["manager_decisions"] for r in selected)),
            )
    return dict(
        rows=rows,
        scores=scores,
        final=final,
        profile=profile,
        scene_cases=cases,
        scene_cases_sha256=hashlib.sha256(
            json.dumps(cases, sort_keys=True).encode()
        ).hexdigest(),
        task="centered_full_pickup" if centered_task else "original_full_pickup",
        centered_task=centered_task,
        evaluation_physical_steps=sum((r["physical_steps"] for r in rows)),
        evaluation_manager_decisions=sum((r["manager_decisions"] for r in rows)),
        duration=ordered_duration_settings(option_steps),
        policy_and_optimizer_sha256_before=before,
        policy_and_optimizer_sha256_after=after,
        training_counters_before=counters,
        training_counters_after=counters,
        expert_deterministic=True,
        stochastic_semantics="Binary categorical manager; deterministic frozen force experts",
        manager_probability_order=["continue_approach", "commit_pickup"],
        manager_probability_semantics="Analytic probabilities at the exact decision observation; diagnostic only, no sampling or threshold override",
        seed_provenance="Explicit development scenes; reserved acceptance seeds untouched",
        acceptance=False,
        new_physical_training_steps=0,
    )


def evaluate_ordered_checkpoint(
    volume_root,
    manager_source,
    expert_sources,
    *,
    seed=460100,
    option_steps=5,
    profile="fixed",
    original_task_control=False,
    device="cuda",
):
    _validate_sources(expert_sources)
    if type(original_task_control) is not bool:
        raise ValueError("original_task_control must be explicit boolean")
    manager_evaluation_cases(profile=profile, final=True, seed=seed)
    with preserved_rng():
        manager, record, path = _source(
            volume_root, manager_source, loader=PPO.load, device=device
        )
        loaded = [
            _source(volume_root, source, device=device) for source in expert_sources
        ]
        experts = _freeze_pair([row[0] for row in loaded])
        records, paths = (
            [record, *[row[1] for row in loaded]],
            [path, *[row[2] for row in loaded]],
        )
        primary = evaluate_ordered(
            manager,
            experts,
            final=True,
            seed=seed,
            option_steps=option_steps,
            profile=profile,
        )
        control = (
            evaluate_ordered(
                manager,
                experts,
                final=True,
                seed=seed,
                option_steps=option_steps,
                profile=profile,
                centered_task=False,
            )
            if original_task_control
            else None
        )
        if [sha256(path) for path in paths] != [
            record["policy_sha256"] for record in records
        ]:
            raise RuntimeError(
                "Frozen ordered evaluation changed source checkpoint bytes"
            )
    return dict(
        status="complete",
        version=VERSION,
        manager=record,
        experts=records[1:],
        final=primary,
        scores=primary["scores"],
        original_task_control=control,
        evaluation_physical_steps=primary["evaluation_physical_steps"],
        control_evaluation_physical_steps=control["evaluation_physical_steps"]
        if control
        else 0,
        source_checkpoint_bytes_unchanged=True,
        optimizer_unchanged=True,
        new_physical_training_steps=0,
    )


def train_ordered_pickup(
    volume_root,
    out,
    expert_sources,
    *,
    arrival_source,
    target_physics_steps=196608,
    seed=460000,
    option_steps=5,
    training_reward_scale=1.0,
    manager_source=None,
    reset_recipe=DEFAULT_RESET_RECIPE,
    commit=lambda: None,
    device="cuda",
):
    duration = ordered_duration_settings(option_steps)
    training_reward_scale = validate_training_reward_scale(training_reward_scale)
    worker_roles = ordered_worker_roles(reset_recipe)
    if reset_recipe == "full-start-v2" and manager_source is None:
        raise ValueError(
            "full-start-v2 requires an explicit completed warm manager source"
        )
    _validate_sources(expert_sources)
    if (
        type(target_physics_steps) is not int
        or not CHUNK_PHYSICAL_UPPER_BOUND <= target_physics_steps <= 1048576
        or type(seed) is not int
        or (not 0 <= seed < 699980000)
    ):
        raise ValueError(
            "Provide a bounded physical budget and explicit development seed"
        )
    pool, pool_record, pool_path = load_ordered_arrival(volume_root, arrival_source)
    loaded = [_source(volume_root, source, device=device) for source in expert_sources]
    experts = _freeze_pair([row[0] for row in loaded])
    records, paths = ([row[1] for row in loaded], [row[2] for row in loaded])
    expert_hashes = [policy_state_hash(expert) for expert in experts]
    warm_path = warm_record = None
    if manager_source is not None:
        warm_path, warm_record = resolve_ordered_continuation(
            volume_root,
            manager_source,
            records,
            pool_record,
            option_steps=option_steps,
            training_reward_scale=training_reward_scale,
            reset_recipe=reset_recipe,
        )
        if warm_path.parent.resolve() == Path(out).resolve():
            raise ValueError(
                "Warm continuation requires a new output directory, never the source run"
            )
    project = Path(__file__).resolve().parents[1]
    code = {
        str(path.relative_to(project)): sha256(path)
        for path in (project / "robovision").glob("*.py")
    }
    code["assets/cup_arm.xml"] = sha256(project / "assets/cup_arm.xml")
    config = dict(
        version=VERSION,
        experts=records,
        arrival_source=pool_record,
        seed=seed,
        target_physics_steps=target_physics_steps,
        duration=duration,
        code_sha256=code,
        training_reward_scale=training_reward_scale,
        evaluation_reward_units="Raw physical task rewards; training multiplier is not applied",
        reset_recipe=reset_recipe,
        worker_roles=list(worker_roles),
        physical_gamma=PHYSICAL_GAMMA,
        arrival_resets_used=reset_recipe == DEFAULT_RESET_RECIPE,
        arrival_pool_provenance_required=True,
        arrival_pool_usage="Two training reset workers"
        if reset_recipe == DEFAULT_RESET_RECIPE
        else "Verified provenance only; no arrival resets in this recipe",
        initial_commit_probability=INITIAL_COMMIT_PROBABILITY,
        ppo=_ordered_ppo_config(option_steps),
    )
    if manager_source is not None:
        config.update(
            manager_source=warm_record,
            requested_manager_source=manager_source,
            warm_manager_provenance=warm_record,
            initial_commit_probability=None,
            reset_curriculum_transition=dict(
                source_reset_recipe=warm_record["source_reset_recipe"],
                target_reset_recipe=reset_recipe,
                source_worker_roles=warm_record["source_worker_roles"],
                target_worker_roles=list(worker_roles),
                intentional_distribution_change=warm_record[
                    "reset_distribution_changed"
                ],
                reward_ppo_experts_unchanged=True,
            ),
            commit_probability_initialization="Inherited learned state-dependent logits; no reinitialization",
        )
    identity = dict(
        recipe=VERSION,
        reset_recipe=reset_recipe,
        source=records,
        config_sha256=hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "experiment.json").exists():
        existing = json.loads((out / "experiment.json").read_text())
        if existing.get("identity") != identity:
            raise ValueError(
                "Existing experiment has a different recipe, source or configuration"
            )
        if (
            recorded_reset_recipe(existing, context="Existing experiment")
            != reset_recipe
        ):
            raise ValueError("Existing experiment reset recipe differs")
    if (out / "summary.json").exists():
        complete = json.loads((out / "summary.json").read_text())
        if complete.get("identity") != identity or complete.get("status") != "complete":
            raise ValueError("Existing completion has a different identity or status")
        if (
            recorded_reset_recipe(complete, context="Existing completion")
            != reset_recipe
        ):
            raise ValueError("Existing completion reset recipe differs")
        if sha256(out / "policy.zip") != complete["policy_sha256"]:
            raise ValueError("Completed policy checkpoint bytes changed")
        if warm_path is not None and sha256(warm_path) != warm_record["policy_sha256"]:
            raise ValueError("Warm source manager checkpoint bytes changed")
        return complete
    env = DummyVecEnv(
        [
            lambda worker=worker: make_ordered_env(
                worker,
                seed,
                experts,
                pool,
                option_steps=option_steps,
                training_reward_scale=training_reward_scale,
                reset_recipe=reset_recipe,
            )
            for worker in range(N_ENVS)
        ]
    )
    try:
        recovered = load_resume(
            out,
            expected_identity=identity,
            env=env,
            device=device,
            loader=PPO.load,
            resumed_env_seed=seed + 10000,
        )
        if recovered is None:
            warm_initialization = None
            if warm_path is None:
                manager = PPO(
                    "MultiInputPolicy",
                    env,
                    learning_rate=0.0001,
                    n_steps=ROLLOUT_STEPS,
                    batch_size=64,
                    n_epochs=5,
                    gamma=duration["manager_gamma"],
                    gae_lambda=0.98,
                    ent_coef=0.01,
                    seed=seed,
                    device=device,
                    policy_kwargs=dict(
                        features_extractor_class=NormalizedGraspCNN,
                        net_arch=[128, 128],
                        share_features_extractor=False,
                    ),
                )
                features = initialize_manager_features(
                    manager, experts[0], option_steps=option_steps
                )
                prior = initialize_commit_prior(manager)
            else:
                manager, warm_initialization = load_warm_ordered_manager(
                    warm_path,
                    env,
                    option_steps=option_steps,
                    seed=seed,
                    expected_manager_steps=warm_record["expected_manager_steps"],
                    device=device,
                )
                features = prior = None
            start, training_seconds = (int(manager.num_timesteps), 0.0)
            audit = OrderedAudit(option_steps=option_steps, reset_recipe=reset_recipe)
            gates, recoveries, optimizer_reports = ([], [], [])
            lost_lower = lost_upper = 0
            experiment = dict(
                **config,
                identity=identity,
                source_steps=start,
                starting_steps=start,
                target_new_physical_steps=target_physics_steps,
                feature_initialization=features,
                commit_prior=prior,
                workers=env.env_method("specification"),
                manager_algorithm="PPO binary categorical",
                source_policy_sha256=warm_record["policy_sha256"]
                if warm_record
                else None,
                imported_expert_weights=True,
                expert_weights_frozen=True,
                no_demonstrations=True,
                no_action_teacher=True,
                runtime_geometric_gate=False,
                actor_observation="Two RGB frames and original18 proprioceptive values only",
                variable_duration_discount="Pickup option is terminal; every physical reward is discounted inside the option",
            )
            if warm_initialization is not None:
                experiment["warm_initialization"] = warm_initialization
            atomic_json(out / "experiment.json", experiment)
            commit()
        else:
            manager, saved, recovery = recovered
            start, training_seconds = (saved["start"], saved["training_seconds"])
            experiment = json.loads((out / "experiment.json").read_text())
            if (
                recorded_reset_recipe(experiment, context="Resumed experiment")
                != reset_recipe
            ):
                raise ValueError("Resumed experiment reset recipe differs")
            saved_audit = saved["audit"]
            if "worker_roles" not in saved_audit and "reset_recipe" not in saved_audit:
                if reset_recipe != DEFAULT_RESET_RECIPE:
                    raise ValueError(
                        "Untagged saved audit cannot imply a full-start curriculum"
                    )
                saved_audit = dict(
                    saved_audit,
                    worker_roles=list(worker_roles),
                    reset_recipe=reset_recipe,
                )
            audit = OrderedAudit(
                saved_audit, option_steps=option_steps, reset_recipe=reset_recipe
            )
            recovery = ordered_recovery_bounds(
                recovery, saved_physical_steps=audit.physical_steps
            )
            lost_lower = (
                saved["lost_physical_lower"]
                + recovery["lost_physical_steps_lower_bound"]
            )
            lost_upper = (
                saved["lost_physical_upper"]
                + recovery["lost_physical_steps_upper_bound"]
            )
            recoveries, gates = (saved["recoveries"] + [recovery], saved["gates"])
            optimizer_reports = saved["optimizer_reports"]
        validate_ordered_manager(manager, option_steps)
        assert_parameter_independence((manager, *experts))

        def state():
            return dict(
                identity=identity,
                audit=audit.state(),
                start=start,
                training_seconds=training_seconds,
                gates=gates,
                recoveries=recoveries,
                lost_physical_lower=lost_lower,
                lost_physical_upper=lost_upper,
                optimizer_reports=optimizer_reports,
            )

        def evaluate(phase, *, final=False):
            def progress(rows):
                atomic_json(
                    out / "evaluation-progress.json",
                    dict(
                        phase=phase,
                        episodes=len(rows),
                        new_physical_steps=audit.physical_steps,
                        new_manager_steps=int(manager.num_timesteps) - start,
                    ),
                )

            return evaluate_ordered(
                manager,
                experts,
                final=final,
                seed=seed + 100,
                option_steps=option_steps,
                progress=progress,
            )

        save_checkpoint(manager, out, state(), commit)
        if not (out / "baseline.json").exists():
            atomic_json(out / "baseline.json", evaluate("baseline"))
            commit()
        baseline = json.loads((out / "baseline.json").read_text())

        def gate_if_due(final_boundary=False):
            physical = audit.physical_steps
            if not physical or any(
                (gate["steps"] == int(manager.num_timesteps) for gate in gates)
            ):
                return
            last = max((gate["new_physical_steps"] for gate in gates), default=0)
            if (
                physical // GATE_PHYSICAL_INTERVAL <= last // GATE_PHYSICAL_INTERVAL
                and (not final_boundary)
            ):
                return
            checkpoint = f"step-{manager.num_timesteps}.zip"
            manager.save(out / checkpoint)
            evaluation = evaluate("gate")
            gates.append(
                dict(
                    steps=int(manager.num_timesteps),
                    new_steps=int(manager.num_timesteps) - start,
                    new_physical_steps=physical,
                    checkpoint=checkpoint,
                    policy_sha256=sha256(out / checkpoint),
                    evaluation=evaluation,
                )
            )
            atomic_json(out / "gates.json", gates)
            save_checkpoint(manager, out, state(), commit, tagged_gate=True)

        gate_if_due()
        while True:
            remaining = max(0, target_physics_steps - audit.physical_steps - lost_upper)
            chunk = planned_ordered_steps(remaining)
            if not chunk:
                break
            begin_chunk(out, chunk, commit, expected_identity=identity)
            before, before_decisions = (
                audit.physical_steps,
                int(manager.num_timesteps),
            )
            began = time.perf_counter()
            manager.learn(chunk, reset_num_timesteps=False, callback=audit)
            training_seconds += time.perf_counter() - began
            consumed = audit.physical_steps - before
            if (
                int(manager.num_timesteps) - before_decisions != chunk
                or not chunk <= consumed <= chunk * MAX_PHYSICAL_PER_DECISION
                or audit.physical_steps + lost_upper > target_physics_steps
                or (sum(audit.worker_decisions) != int(manager.num_timesteps) - start)
            ):
                raise RuntimeError(
                    "Ordered manager physical/decision accounting mismatch"
                )
            if [policy_state_hash(expert) for expert in experts] != expert_hashes:
                raise RuntimeError("Training changed frozen expert policy tensors")
            optimizer_reports.append(
                dict(
                    new_manager_steps=int(manager.num_timesteps) - start,
                    new_physical_steps=audit.physical_steps,
                    **optimizer_report(manager),
                )
            )
            atomic_json(
                out / "progress.json",
                dict(
                    phase="training",
                    new_steps=int(manager.num_timesteps) - start,
                    new_manager_steps=int(manager.num_timesteps) - start,
                    new_physical_steps=audit.physical_steps,
                    actual_physical_lower_bound=audit.physical_steps + lost_lower,
                    actual_physical_upper_bound=audit.physical_steps + lost_upper,
                    training_seconds=training_seconds,
                    scores=audit.report(),
                    optimizer=optimizer_reports[-1],
                    recoveries=recoveries,
                ),
            )
            save_checkpoint(manager, out, state(), commit)
            gate_if_due()
        gate_if_due(final_boundary=True)
        manager.save(out / "policy.zip")
        final = evaluate("final", final=True)
        atomic_json(out / "training-episodes.json", audit.episodes)
        if [sha256(path) for path in paths] != [
            record["policy_sha256"] for record in records
        ] or sha256(pool_path) != pool_record["sha256"]:
            raise RuntimeError(
                "Frozen expert or arrival source bytes changed during training"
            )
        if warm_path is not None and sha256(warm_path) != warm_record["policy_sha256"]:
            raise RuntimeError(
                "Warm source manager checkpoint bytes changed during training"
            )
        summary = dict(
            **experiment,
            status="complete",
            final_steps=int(manager.num_timesteps),
            new_steps=int(manager.num_timesteps) - start,
            new_manager_steps=int(manager.num_timesteps) - start,
            new_physical_steps=audit.physical_steps,
            actual_physical_lower_bound=audit.physical_steps + lost_lower,
            actual_physical_upper_bound=audit.physical_steps + lost_upper,
            unused_physical_budget=target_physics_steps
            - audit.physical_steps
            - lost_upper,
            budget_stop="Remaining cap cannot reserve the64000-action worst-case full PPO rollout",
            training_seconds=training_seconds,
            policy_sha256=sha256(out / "policy.zip"),
            baseline=baseline,
            gates=gates,
            final=final,
            scores=final["scores"],
            training_scores=audit.report(),
            optimizer_reports=optimizer_reports,
            recoveries=recoveries,
        )
        if warm_path is not None:
            inherited = experiment["warm_initialization"]["inherited_counters"]
            summary.update(
                source_manager_bytes_unchanged=True,
                inherited_manager_steps=inherited["num_timesteps"],
                inherited_ppo_updates=inherited["n_updates"],
                inherited_model_episode_counter=inherited["episode_num"],
                new_ppo_updates=int(manager._n_updates) - inherited["n_updates"],
                model_episode_counter_delta=int(manager._episode_num)
                - inherited["episode_num"],
                new_completed_training_episodes=len(audit.episodes),
            )
        atomic_json(out / "summary.json", summary)
        commit()
        return summary
    finally:
        env.close()
