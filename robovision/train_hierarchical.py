"""Independent force-control PPO specialists and learned option selection.

Training-time reset curricula and reward geometry never enter actor observations.
All policy files stay in the remote volume; reports contain JSON diagnostics only.
"""

from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from collections import Counter
import copy
import hashlib
import json
import time
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from .grasp_benchmark import preserved_rng
from .io import atomic_json
from .policy_state import policy_state_hash
from .skill_hard_evaluation import resolve_skill_checkpoint, sha256
from .training_checkpoint import save_checkpoint, load_resume, begin_chunk

VERSION = "hierarchical-force-ppo-v1"
CHUNK = 8192
GATE_INTERVAL = 49152


@dataclass(frozen=True)
class SkillRecipe:
    skill: str
    stage: int = 0
    gamma: float = 0.999
    gae_lambda: float = 0.995
    n_steps: int = 512
    noise_scale: float = 10.0
    learning_rate: float = 3e-06
    ent_coef: float = 0.0
    centered_approach: bool = False
    grasp_bootstrap: bool = False
    moving_grasp: bool = False
    structured: bool = False
    arrival_pool_run: str = ""
    arrival_pool_sha256: str = ""
    arrival_probability: float = 0.75
    approach_reward: str = "original"


def validate_recipe(recipe):
    if recipe.approach_reward not in ("original",):
        raise ValueError("Registered approach_reward required: original")
    if recipe.approach_reward != "original" and (
        recipe.skill != "approach" or not recipe.centered_approach or recipe.stage != 0
    ):
        raise ValueError("Clearance reward requires centered approach stage 0")
    maximum = 1 if recipe.centered_approach else 3 if recipe.skill == "grasp" else 2
    if (
        recipe.skill not in ("approach", "grasp", "lift")
        or type(recipe.stage) is not int
        or (not 0 <= recipe.stage <= maximum)
    ):
        raise ValueError("Explicit skill and registered stage required")
    if not 0 < recipe.noise_scale <= 50 or not 0 < recipe.learning_rate <= 0.0003:
        raise ValueError("Invalid new-experiment noise scale or actor learning rate")
    if recipe.grasp_bootstrap and recipe.skill != "grasp":
        raise ValueError(
            "Centered closure bootstrap applies only to the grasp specialist"
        )
    if recipe.centered_approach and recipe.skill != "approach":
        raise ValueError("Centered approach applies only to the approach specialist")
    if recipe.moving_grasp and (not recipe.grasp_bootstrap):
        raise ValueError("Moving grasp requires the explicit bootstrap reset family")
    if bool(recipe.arrival_pool_run) != bool(recipe.arrival_pool_sha256):
        raise ValueError("Arrival source run and exact file hash are both required")
    if recipe.arrival_pool_run:
        from .skill_hard_evaluation import basename

        basename(recipe.arrival_pool_run)
        if (
            recipe.skill not in ("grasp", "lift")
            or len(recipe.arrival_pool_sha256) != 64
        ):
            raise ValueError(
                "Only successors may use a hash-verified predecessor arrival pool"
            )
    if not 0 < recipe.arrival_probability <= 1:
        raise ValueError("Arrival reset fraction must lie in (0,1]")
    return recipe


def make_skill_env(
    worker,
    seed,
    recipe,
    render_images=True,
    use_arrivals=True,
    full_bootstrap_pickup=False,
    **reset_kwargs,
):
    from .hierarchical_skills import ApproachSkillEnv, GraspSkillEnv, LiftSkillEnv

    kwargs = dict(
        seed=seed + worker,
        observation="pixels",
        render_images=render_images,
        gamma=recipe.gamma,
    )
    if recipe.skill == "lift":
        if reset_kwargs:
            raise ValueError("Lift uses its recorded held-object reset")
        env = LiftSkillEnv(skill_stage=recipe.stage, **kwargs)
    else:
        cls = ApproachSkillEnv if recipe.skill == "approach" else GraspSkillEnv
        if recipe.centered_approach:
            from .centered_approach_skill import CenteredApproachSkillEnv

            cls = CenteredApproachSkillEnv
        if recipe.grasp_bootstrap:
            from .centered_grasp_bootstrap import (
                CenteredGraspBootstrapEnv,
                MovingGraspBootstrapEnv,
                FullPickupBootstrapEnv,
            )

            cls = (
                FullPickupBootstrapEnv
                if full_bootstrap_pickup
                else MovingGraspBootstrapEnv
                if recipe.moving_grasp
                else CenteredGraspBootstrapEnv
            )
        env = cls(skill_stage=recipe.stage, **kwargs, **reset_kwargs)
    if use_arrivals and recipe.arrival_pool_run:
        from .hierarchical_replay import ArrivalResetEnv

        predecessor = "approach" if recipe.skill == "grasp" else "grasp"
        path = (
            Path("/runs") / recipe.arrival_pool_run / f"{predecessor}-arrivals.pkl.gz"
        )
        env = ArrivalResetEnv(
            env,
            pool_path=path,
            pool_sha256=recipe.arrival_pool_sha256,
            probability=recipe.arrival_probability,
            seed=seed + worker,
        )
    return Monitor(env)


def configure_skill(model, recipe, seed):
    """Fresh experiment configuration; never called when resuming this recipe."""
    model.seed = seed
    model.set_random_seed(seed)
    if recipe.structured and (not getattr(model, "use_sde", False)):
        from .structured_exploration import convert_to_structured

        model = convert_to_structured(model, resample_steps=25)
    model.gamma = recipe.gamma
    model.gae_lambda = recipe.gae_lambda
    model.n_steps = recipe.n_steps
    model.ent_coef = recipe.ent_coef
    model.rollout_buffer = model.rollout_buffer_class(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        **model.rollout_buffer_kwargs,
    )
    if not recipe.structured:
        model.rollout_buffer.scale_provider = (
            lambda: model.policy._last_action_std_scales
        )
    model._last_obs = None
    return model


def initialize_skill(path, env, recipe, seed, device="cuda"):
    """Initialize from either an older PPO donor or a trained specialist.

    New curriculum experiments explicitly start fresh Adam moments, whereas
    load_resume above restores moments exactly and never enters this function.
    """
    from .context_exploration import (
        load_context_exploration_ppo,
        set_runtime_arm_scales,
    )
    from .policy_loading import checkpoint_algorithm, load_grasp_policy
    from .policy_metrics import exploration_report
    from stable_baselines3.common.utils import get_schedule_fn

    inherited_structured = checkpoint_algorithm(path) == "structured-decoupled-ppo-v1"
    if inherited_structured:
        if not recipe.structured:
            raise ValueError(
                "A structured donor requires an explicitly structured successor"
            )
        model = load_grasp_policy(path, env=env, device=device)
        model.set_actor_update_scope("all")
        model.critic_learning_rate = 0.0003
        model.algorithm_metadata["critic_learning_rate"] = 0.0003
        for parameter in model.policy.parameters():
            parameter.requires_grad_(True)
    else:
        model = load_context_exploration_ppo(
            path,
            env,
            device,
            std_scales=(1.0,) * 4,
            critic_learning_rate=0.0003,
            role_normalization=False,
            role_gradient_balance=False,
            finger_exploration_max=(1.0,) * 4,
            actor_update_scope="all",
            allow_context_reconfigure=True,
        )
        set_runtime_arm_scales(model, (1.0,) * 4)
    before = exploration_report(model)
    with torch.no_grad():
        model.policy.log_std.add_(np.log(recipe.noise_scale))
    model.learning_rate = recipe.learning_rate
    model.lr_schedule = get_schedule_fn(recipe.learning_rate)
    model.policy.optimizer.state.clear()
    model.critic_optimizer.state.clear()
    for group in model.policy.optimizer.param_groups:
        group["lr"] = recipe.learning_rate
    for group in model.critic_optimizer.param_groups:
        group["lr"] = model.critic_learning_rate
    model = configure_skill(model, recipe, seed)
    return (
        model,
        dict(
            exploration_before=before,
            exploration_after=exploration_report(model),
            actor_learning_rate=recipe.learning_rate,
            critic_learning_rate=0.0003,
            actor_update_scope="all",
            optimizer_moments="fresh for this new experiment",
            inherited_actor_mean_weights=True,
            inherited_structured=inherited_structured,
            seed=seed,
        ),
    )


class SkillAudit(BaseCallback):
    def __init__(self, source_steps, saved=None):
        super().__init__()
        self.source_steps = source_steps
        saved = saved or {}
        self.episodes = saved.get("episodes", [])
        self.updates = saved.get("updates", [])
        self.last_recorded = self.updates[-1]["steps"] if self.updates else source_steps
        self.running = [{} for _ in range(4)]

    def capture_update(self):
        if self.model.num_timesteps > self.last_recorded and hasattr(
            self.model, "last_update_stats"
        ):
            self.updates.append(
                dict(
                    steps=int(self.model.num_timesteps), **self.model.last_update_stats
                )
            )
            self.last_recorded = int(self.model.num_timesteps)

    def _on_rollout_start(self):
        self.capture_update()

    def _on_step(self):
        for worker, (done, info) in enumerate(
            zip(self.locals["dones"], self.locals["infos"])
        ):
            running = self.running[worker]
            running["bilateral_ever"] = running.get("bilateral_ever", False) or all(
                info.get("contacts", [False, False])
            )
            running["peak_clearance"] = max(
                running.get("peak_clearance", 0.0), info.get("clearance", 0.0)
            )
            if done:
                self.episodes.append(
                    dict(
                        worker=worker,
                        steps=int(self.model.num_timesteps),
                        phase_success=bool(
                            info.get("phase_success", info.get("is_success", False))
                        ),
                        full_pickup_success=bool(
                            info.get(
                                "full_pickup_success_ever",
                                info.get("is_success", False),
                            )
                        ),
                        reason=info.get("reason"),
                        height=info.get("reset_height"),
                        offset=info.get("cup_offset"),
                        actual_arrival_reset=bool(
                            info.get("actual_arrival_reset", False)
                        ),
                        arrival_entry=info.get("arrival_entry"),
                        phase_metrics=info.get("phase_metrics"),
                        skill_stage=info.get("skill_stage"),
                        **running,
                        **info["episode"],
                    )
                )
                self.running[worker] = {}
        return True

    def state(self):
        return dict(episodes=self.episodes, updates=self.updates)

    def report(self):
        rows = self.episodes[-100:]
        by_reset = {}
        for actual in (False, True):
            selected = [
                r for r in rows if bool(r.get("actual_arrival_reset", False)) == actual
            ]
            if selected:
                by_reset["actual_arrival" if actual else "synthetic_start"] = dict(
                    episodes=len(selected),
                    physical_actions=sum((r["l"] for r in selected)),
                    phase_successes=sum((r["phase_success"] for r in selected)),
                )
        return dict(
            total_episodes=len(self.episodes),
            recent_episodes=len(rows),
            phase_successes=sum((r["phase_success"] for r in rows)),
            full_pickup_successes=sum((r["full_pickup_success"] for r in rows)),
            bilateral=sum((r["bilateral_ever"] for r in rows)),
            failures=dict(
                Counter((r["reason"] for r in rows if not r["phase_success"]))
            ),
            recent_by_reset=by_reset,
        )


def skill_cases(skill, final=False, *, bootstrap=False, stage=0):
    if skill == "lift":
        return [dict(name=f"hold-{i}", reset={}) for i in range(8 if final else 3)]
    if bootstrap:
        span = (0.0005, 0.003, 0.007, 0.007)[stage]
        offsets = ((0.0, 0.0),)
        return [
            dict(
                name=f"bootstrap-z{z:.4f}-{x:+.4f}-{y:+.4f}",
                reset=dict(fixed_relative_z=z),
            )
            for z in (0.005, 0.0075, 0.01)
            for x, y in offsets
        ]
    heights = (0.025, 0.065, 0.1, 0.14) if skill == "approach" else (0.0, 0.01, 0.02)
    offsets = ((0.0, 0.0),)
    return [
        dict(name=f"{skill}-{h:.3f}-{x:+.3f}-{y:+.3f}", reset=dict(fixed_height=h))
        for h in heights
        for x, y in offsets
    ]


class ApproachXYTrace:
    """Per-action measurements, separate from rewards and sampled trajectories.

    Precontact measurements stop before the first tick with positive measured
    finger-cup normal force (or an original task contact flag). This does not
    infer visual attention or distinguish an arm-cup contact absent those forces.
    """

    def __init__(self, env, info):
        self.initial_hand = env.grasp_position.copy()
        self.initial_cup = env.cup_position.copy()
        self.initial_xy_error = float(
            np.linalg.norm((self.initial_hand - self.initial_cup)[:2])
        )
        self.first_contact = None
        self.first_bilateral_step = None
        self.precontact = None
        self.minimum_precontact_xy_error = None
        self.peak_cup_xy_displacement = 0.0
        self.all_robot_contacts = info.get("phase") == "alignment"
        self.observe(env, info)

    def observe(self, env, info):
        hand, cup = (env.grasp_position.copy(), env.cup_position.copy())
        point = dict(
            step=int(env.step_count),
            time_s=float(env.step_count * env.control_dt),
            hand_m=hand.tolist(),
            cup_m=cup.tolist(),
            xy_error_m=float(np.linalg.norm((hand - cup)[:2])),
            hand_xy_displacement_m=float(
                np.linalg.norm((hand - self.initial_hand)[:2])
            ),
            cup_xy_displacement_m=float(np.linalg.norm((cup - self.initial_cup)[:2])),
        )
        contacts = info.get("contacts", [False, False])
        forces = (info.get("phase_metrics") or {}).get("contact_forces_n", [0.0, 0.0])
        robot_contact = (info.get("phase_metrics") or {}).get(
            "robot_cup_contact", False
        )
        if self.first_contact is None and (
            any(contacts) or any((force > 0.0 for force in forces)) or robot_contact
        ):
            self.first_contact = point.copy()
        if self.first_bilateral_step is None and all(contacts):
            self.first_bilateral_step = int(env.step_count)
        if self.first_contact is None:
            self.precontact = point.copy()
            self.minimum_precontact_xy_error = (
                point["xy_error_m"]
                if self.minimum_precontact_xy_error is None
                else min(self.minimum_precontact_xy_error, point["xy_error_m"])
            )
        self.peak_cup_xy_displacement = max(
            self.peak_cup_xy_displacement, point["cup_xy_displacement_m"]
        )
        self.final = point

    def report(self):
        return dict(
            initial_hand_m=self.initial_hand.tolist(),
            initial_cup_m=self.initial_cup.tolist(),
            initial_xy_error_m=self.initial_xy_error,
            final=self.final,
            first_contact=self.first_contact,
            first_bilateral_step=self.first_bilateral_step,
            last_precontact=self.precontact,
            minimum_precontact_xy_error_m=self.minimum_precontact_xy_error,
            precontact_xy_error_reduction_m=None
            if self.precontact is None
            else self.initial_xy_error - self.precontact["xy_error_m"],
            best_precontact_xy_error_reduction_m=None
            if self.minimum_precontact_xy_error is None
            else self.initial_xy_error - self.minimum_precontact_xy_error,
            peak_cup_xy_displacement_m=self.peak_cup_xy_displacement,
            contact_definition="Any robot-cup pair in the post-action50Hz MuJoCo contact array; transient substep contacts may be missed"
            if self.all_robot_contacts
            else "Positive measured finger-cup normal force or original contact flag",
            precontact_excludes_first_contact_tick=True,
            sampled_every_physical_action=True,
        )


def evaluate_skill(
    model,
    recipe,
    *,
    final=False,
    seed=310100,
    progress=lambda rows: None,
    full_bootstrap_pickup=False,
    force_open_start=False,
    registered_cases=None,
    registered_env_factory=None,
):
    validate_recipe(recipe)
    rows = []
    before = policy_state_hash(model)
    was_training = model.policy.training
    if (registered_cases is None) != (registered_env_factory is None):
        raise ValueError(
            "Custom registered evaluation requires both explicit scenes and environment factory"
        )
    cases = (
        skill_cases(
            recipe.skill, final, bootstrap=recipe.grasp_bootstrap, stage=recipe.stage
        )
        if registered_cases is None
        else copy.deepcopy(registered_cases)
    )
    environment_factory = (
        make_skill_env if registered_env_factory is None else registered_env_factory
    )
    with preserved_rng():
        try:
            for deterministic in [True, False] if final else [True]:
                for index, original_case in enumerate(cases):
                    case = copy.deepcopy(original_case)
                    if force_open_start:
                        if not recipe.grasp_bootstrap:
                            raise ValueError(
                                "Explicit opening overrides use the bootstrap reset family"
                            )
                        case["reset"]["fixed_opening"] = 0.045
                    case_seed = seed + index
                    env = environment_factory(
                        0,
                        case_seed,
                        recipe,
                        use_arrivals=False,
                        full_bootstrap_pickup=full_bootstrap_pickup,
                        **case["reset"],
                    )
                    try:
                        torch.manual_seed(case_seed)
                        obs, info = env.reset(seed=case_seed)
                        initial_proprio = None
                        xy_trace = None
                        trajectory = []
                        total = 0.0
                        bilateral = False
                        while True:
                            if set(obs) != {"image", "proprio"}:
                                raise RuntimeError("Unexpected actor input")
                            with torch.no_grad():
                                action, _ = model.predict(
                                    obs, deterministic=deterministic
                                )
                            obs, reward, done, truncated, info = env.step(action)
                            total += reward
                            bilateral |= all(info.get("contacts", [False, False]))
                            base = env.unwrapped
                            if xy_trace is not None:
                                xy_trace.observe(base, info)
                            if (
                                base.step_count <= 10
                                or base.step_count % 10 == 0
                                or done
                                or truncated
                                or (False and base.step_count == 25)
                            ):
                                trajectory.append(
                                    dict(
                                        time=base.step_count * base.control_dt,
                                        hand=base.grasp_position.tolist(),
                                        cup=base.cup_position.tolist(),
                                        joints=base.data.qpos[:6].tolist(),
                                        velocity=base.data.qvel[:6].tolist(),
                                        action=np.asarray(action).tolist(),
                                        contacts=info["contacts"],
                                        phase_metrics=info.get("phase_metrics"),
                                        clearance=info["clearance"],
                                    )
                                )
                            if done or truncated:
                                break
                        rows.append(
                            dict(
                                case=case["name"],
                                reset=case["reset"],
                                seed=case_seed,
                                deterministic=deterministic,
                                phase_success=bool(
                                    info.get("phase_success", info["is_success"])
                                ),
                                full_pickup_success=bool(
                                    info.get(
                                        "full_pickup_success_ever", info["is_success"]
                                    )
                                ),
                                centered_full_pickup_success=bool(
                                    info.get("centered_full_pickup_success_ever", False)
                                ),
                                reason=info["reason"],
                                steps=base.step_count,
                                episode_return=total,
                                bilateral=bilateral,
                                peak_clearance=info["peak_clearance"],
                                final_metrics=info.get("phase_metrics"),
                                trajectory=trajectory,
                            )
                        )
                        if xy_trace is not None:
                            rows[-1].update(
                                case_group=case["group"], xy_movement=xy_trace.report()
                            )
                        progress(rows)
                    finally:
                        env.close()
        finally:
            model.policy.train(was_training)
    if before != policy_state_hash(model):
        raise RuntimeError("Frozen evaluation changed expert tensors")
    scores = {}
    grouped_scores = {}
    for deterministic in (True, False):
        selected = [r for r in rows if r["deterministic"] == deterministic]
        if selected:
            scores["deterministic" if deterministic else "stochastic"] = dict(
                episodes=len(selected),
                phase_successes=sum((r["phase_success"] for r in selected)),
                full_pickup_successes=sum((r["full_pickup_success"] for r in selected)),
                centered_full_pickup_successes=sum(
                    (r["centered_full_pickup_success"] for r in selected)
                ),
                bilateral=sum((r["bilateral"] for r in selected)),
            )
            for group in (
                "nominal_retention",
                "displaced_xy",
                "fresh_xy",
                "height_extension",
                "fresh_height_retention",
                "fresh_height_extension",
            ):
                grouped = [row for row in selected if row.get("case_group") == group]
                if grouped:
                    grouped_scores.setdefault(
                        "deterministic" if deterministic else "stochastic", {}
                    )[group] = dict(
                        episodes=len(grouped),
                        phase_successes=sum((row["phase_success"] for row in grouped)),
                    )
    from .structured_exploration import StructuredActorCriticPolicy

    structured_predict = isinstance(model.policy, StructuredActorCriticPolicy)
    frozen = {}
    return dict(
        rows=rows,
        scores=scores,
        skill=recipe.skill,
        stage=recipe.stage,
        **frozen,
        policy_state_sha256=before,
        final=final,
        acceptance=False,
        actual_arrival_reset_evaluation=False,
        grasp_bootstrap=recipe.grasp_bootstrap,
        full_bootstrap_pickup_evaluation=full_bootstrap_pickup,
        force_open_start=force_open_start,
        scores_by_case_group=grouped_scores,
        approach_reward=recipe.approach_reward,
        evaluation_physical_actions=sum((row["steps"] for row in rows)),
        stochastic_evaluation=dict(
            call="model.predict(deterministic=False)",
            structured_policy_fresh_marginal_per_action=structured_predict,
            training_sde_resample_steps=getattr(model, "sde_sample_freq", None),
            matches_persistent_training_sde=False if structured_predict else None,
        ),
    )


def registered_evaluation_once(
    model,
    recipe,
    out,
    label,
    registration,
    *,
    final=False,
    commit=lambda: None,
    evaluator=None,
):
    """Bound and cache one registered phase evaluation; never silently replay."""
    identity = dict(
        policy_sha256=policy_state_hash(model),
        recipe=asdict(recipe),
        final=final,
        registration=registration,
    )
    out = Path(out)
    result_path, intent_path = (out / (label + ".json"), out / (label + "-intent.json"))
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if (
            cached.get("evaluation_identity") != identity
            or cached.get("frozen_integrity_verified") is not True
        ):
            raise RuntimeError(
                "Registered phase evaluation cache identity or frozen integrity differs"
            )
        return cached
    if intent_path.exists():
        phase = "nominal"
        raise RuntimeError(
            f"Interrupted {phase} evaluation requires explicit audit before replay"
        )
    maximum_episodes = (
        registration["evaluation"]["final_episode_count"]
        if final
        else registration["evaluation"]["gate_case_count"]
    )
    atomic_json(
        intent_path,
        dict(
            evaluation_identity=identity,
            maximum_physical_actions=maximum_episodes * 500,
        ),
    )
    commit()

    def progress(rows):
        atomic_json(
            out / (label + "-progress.json"),
            dict(
                evaluation_identity=identity,
                rows=rows,
                completed_physical_actions=sum((row["steps"] for row in rows)),
                remaining_evaluation_physical_upper_bound=(maximum_episodes - len(rows))
                * 500,
                frozen_integrity_verified=False,
            ),
        )
        commit()

    evaluate = evaluate_skill if evaluator is None else evaluator
    result = evaluate(model, recipe, final=final, progress=progress)
    result["evaluation_identity"] = identity
    atomic_json(result_path, result)
    commit()
    return result


def train_skill(
    volume_root,
    out,
    source,
    checkpoint,
    *,
    recipe,
    target_steps=196608,
    seed=310000,
    commit=lambda: None,
):
    validate_recipe(recipe)
    if (
        type(target_steps) is not int
        or not CHUNK <= target_steps <= 1048576
        or target_steps % CHUNK
    ):
        raise ValueError("Bounded exact skill budget must be a multiple of8192")
    if type(seed) is not int or not 0 <= seed < 700000000:
        raise ValueError("Development seed required")
    out = Path(out)
    path, record, metadata, metadata_path = resolve_skill_checkpoint(
        volume_root, source, checkpoint
    )
    source_hash = sha256(path)
    root = Path(__file__).resolve().parents[1]
    code_hashes = {
        str(p.relative_to(root)): sha256(p) for p in (root / "robovision").glob("*.py")
    }
    code_hashes["assets/cup_arm.xml"] = sha256(root / "assets/cup_arm.xml")
    config = dict(
        version=VERSION,
        recipe=asdict(recipe),
        seed=seed,
        target_steps=target_steps,
        source_sha256=source_hash,
        code_sha256=code_hashes,
    )
    identity = dict(
        recipe=asdict(recipe),
        source=f"{source}/{checkpoint}",
        config_sha256=hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
    )
    env = SubprocVecEnv(
        [partial(make_skill_env, w, seed, recipe) for w in range(4)],
        start_method="spawn",
    )
    try:
        recovered = load_resume(
            out,
            expected_identity=identity,
            env=env,
            device="cuda",
            resumed_env_seed=seed + 10000,
        )
        if recovered is None:
            model, inherited = initialize_skill(path, env, recipe, seed)
            start = int(model.num_timesteps)
            audit = SkillAudit(start)
            training_seconds = 0.0
            recoveries = []
            gates = []
            experiment = dict(
                **config,
                source_run=source,
                source_checkpoint=checkpoint,
                source_policy_sha256=source_hash,
                source_metadata_sha256=sha256(metadata_path),
                source_steps=start,
                starting_steps=start,
                target_new_steps=target_steps,
                inherited_setup=inherited,
                workers=env.env_method("specification"),
                original_physics=True,
                full_five_force_control=True,
                no_demonstrations=True,
                no_action_teacher=True,
                imported_learned_weights=True,
                objective=f"Independent {recipe.skill} specialist; phase success is not full-task success",
            )
            atomic_json(out / "experiment.json", experiment)
            commit()
        else:
            model, saved, recovery = recovered
            start = saved["start"]
            audit = SkillAudit(start, saved["audit"])
            training_seconds = saved["training_seconds"]
            recoveries = saved["recoveries"] + [recovery]
            gates = saved["gates"]
            experiment = json.loads((out / "experiment.json").read_text())
        lost_lower = sum((r["lost_interactions_lower_bound"] for r in recoveries))
        lost_upper = sum((r["lost_interactions_upper_bound"] or 0 for r in recoveries))
        for recovery in recoveries:
            if (
                recovery["lost_interactions_upper_bound"] is None
                and recovery["restored_model_steps"] != start
            ):
                raise RuntimeError(
                    "Missing work intent after training began; cannot bound recovery budget"
                )

        def checkpoint_state():
            return dict(
                identity=identity,
                audit=audit.state(),
                start=start,
                training_seconds=training_seconds,
                recoveries=recoveries,
                gates=gates,
            )

        save_checkpoint(model, out, checkpoint_state(), commit)
        if not (out / "baseline.json").exists():
            baseline = evaluate_skill(model, recipe)
            atomic_json(out / "baseline.json", baseline)
            commit()
        else:
            baseline = json.loads((out / "baseline.json").read_text())

        def finish_pending_gate(force=False):
            new_steps = int(model.num_timesteps) - start
            due = new_steps > 0 and (new_steps % GATE_INTERVAL == 0 or force)
            if due and (not any((g["new_steps"] == new_steps for g in gates))):
                name = f"step-{model.num_timesteps}.zip"
                model.save(out / name)
                evaluation = evaluate_skill(model, recipe)
                gates.append(
                    dict(
                        steps=int(model.num_timesteps),
                        new_steps=new_steps,
                        checkpoint=name,
                        policy_sha256=sha256(out / name),
                        training_seconds=training_seconds,
                        evaluation=evaluation,
                    )
                )
                atomic_json(out / "gates.json", gates)
                atomic_json(out / "training-episodes.json", audit.episodes)
                save_checkpoint(model, out, checkpoint_state(), commit)

        finish_pending_gate(
            force=int(model.num_timesteps) - start + lost_upper + CHUNK > target_steps
        )
        while model.num_timesteps - start + lost_upper + CHUNK <= target_steps:
            begin_chunk(out, CHUNK, commit, expected_identity=identity)
            began = time.perf_counter()
            model.learn(CHUNK, reset_num_timesteps=False, callback=audit)
            training_seconds += time.perf_counter() - began
            audit.capture_update()
            new_steps = int(model.num_timesteps) - start
            if new_steps % CHUNK or new_steps > target_steps:
                raise RuntimeError("Skill interaction accounting mismatch")
            from .policy_metrics import exploration_report

            progress = dict(
                phase="training",
                new_steps=new_steps,
                final_steps=int(model.num_timesteps),
                training_seconds=training_seconds,
                scores=audit.report(),
                updates=audit.updates[-4:],
                exploration=exploration_report(model),
                recoveries=recoveries,
                actual_training_interactions_lower_bound=new_steps + lost_lower,
                actual_training_interactions_upper_bound=new_steps + lost_upper,
            )
            atomic_json(out / "progress.json", progress)
            save_checkpoint(model, out, checkpoint_state(), commit)
            finish_pending_gate(force=new_steps + lost_upper + CHUNK > target_steps)
        model.save(out / "policy.zip")
        final = evaluate_skill(model, recipe, final=True)
        open_start_evaluation = (
            evaluate_skill(model, recipe, final=True, force_open_start=True)
            if recipe.grasp_bootstrap
            else None
        )
        bootstrap_full_pickup_evaluation = (
            evaluate_skill(model, recipe, final=True, full_bootstrap_pickup=True)
            if recipe.grasp_bootstrap
            else None
        )
        atomic_json(out / "training-episodes.json", audit.episodes)
        if sha256(path) != source_hash:
            raise RuntimeError("Parent checkpoint mutated")
        return dict(
            **experiment,
            final_steps=int(model.num_timesteps),
            new_steps=int(model.num_timesteps) - start,
            training_seconds=training_seconds,
            policy_sha256=sha256(out / "policy.zip"),
            baseline=baseline,
            gates=gates,
            final=final,
            open_start_evaluation=open_start_evaluation,
            bootstrap_full_pickup_evaluation=bootstrap_full_pickup_evaluation,
            scores=final["scores"],
            training_scores=audit.report(),
            recoveries=recoveries,
            actual_training_interactions_lower_bound=int(model.num_timesteps)
            - start
            + lost_lower,
            actual_training_interactions_upper_bound=int(model.num_timesteps)
            - start
            + lost_upper,
            retained_training_budget_complete=int(model.num_timesteps) - start
            == target_steps,
            status="complete",
        )
    finally:
        env.close()
