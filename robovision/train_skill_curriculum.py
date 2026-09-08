"""PPO skill experiments with transition-balanced rehearsal and explicit gates."""

import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from collections import Counter, deque
from stable_baselines3.common.vec_env import SubprocVecEnv
from .approach_subtask import ApproachSubtaskEnv
from .dense_approach import DenseApproachEnv
from .open_bootstrap import OpenBootstrapEnv
from .hover_bootstrap import (
    HoverBootstrapEnv,
    FineHoverBootstrapEnv,
    FineDescentBootstrapEnv,
)
from .near_pickup import NearPickupEnv, FineNearPickupEnv, MillimeterNearPickupEnv
from .reverse_curriculum import ReverseGraspEnv
from .reward_scale import RoleRewardScale
from .pickup_evaluation import evaluate as pickup_evaluate
from .io import atomic_json, file_hash

VARIANT_ENVS = {
    "subtask": ApproachSubtaskEnv,
    "dense": DenseApproachEnv,
    "open_bootstrap": OpenBootstrapEnv,
    "hover_bootstrap": HoverBootstrapEnv,
    "fine_hover_bootstrap": FineHoverBootstrapEnv,
    "fine_descent_bootstrap": FineDescentBootstrapEnv,
    "near_pickup": NearPickupEnv,
    "fine_near_pickup": FineNearPickupEnv,
    "millimeter_near_pickup": MillimeterNearPickupEnv,
}


def curriculum_definition(variant):
    cls = VARIANT_ENVS[variant]
    if variant == "dense":
        return (cls, [], None, None)
    lessons = cls.lessons
    approach_last = (
        next((i for i, l in enumerate(lessons) if l.approach_fraction < 1.0)) - 1
    )
    pickup_first = next(
        (i for i, l in enumerate(lessons) if l.approach_fraction == 0.0)
    )
    return (cls, lessons, approach_last, pickup_first)


def gate_configurations(variant, stage):
    """Explicit reset heights and task labels used by frozen stage gates."""
    cls, _, approach_last, pickup_first = curriculum_definition(variant)
    if variant in ("near_pickup", "fine_near_pickup", "millimeter_near_pickup"):
        return [(cls.lessons[stage].reset_height, stage, "pickup")]
    eval_stages = (
        [stage]
        if stage <= approach_last
        else [approach_last, pickup_first]
        if stage < pickup_first
        else [stage]
    )
    return [
        (height, es, "approach" if es <= approach_last else "pickup")
        for height in [0.07, 0.075, 0.08]
        for es in eval_stages
    ]


def gate_transition(
    stage, final_stage, passed, *, stop_after_passed_gate=False, baseline=False
):
    """Decide after a saved gate; stopping preserves the evaluated lesson."""
    if passed and stop_after_passed_gate and (not baseline):
        return (stage, True)
    if passed and final_stage is not None and (stage < final_stage):
        return (stage + 1, False)
    return (stage, False)


def validate_start_stage(variant, stage):
    _, lessons, _, _ = curriculum_definition(variant)
    count = len(lessons) if lessons else 1
    if (
        isinstance(stage, bool)
        or not isinstance(stage, int)
        or (not 0 <= stage < count)
    ):
        raise ValueError(
            f"Invalid start stage for {variant}: expected integer 0..{count - 1}"
        )
    return stage


def resolve_source_lesson(
    variant, source_metadata, source_record, requested_stage=None
):
    """Map hover or near-pickup curricula by exact lesson semantics, never by numeric index."""
    if requested_stage is not None:
        validate_start_stage(variant, requested_stage)
    source_variant = source_metadata.get("variant")
    source_stage = source_record.get("lesson", source_metadata.get("final_lesson", 0))
    hover_variants = {
        "hover_bootstrap",
        "fine_hover_bootstrap",
        "fine_descent_bootstrap",
    }
    near_variants = {"near_pickup", "fine_near_pickup", "millimeter_near_pickup"}
    mapping = None
    if source_variant != variant and (
        source_variant in hover_variants
        and variant in hover_variants
        or (source_variant in near_variants and variant in near_variants)
    ):
        source_lessons = VARIANT_ENVS[source_variant].lessons
        if (
            isinstance(source_stage, bool)
            or not isinstance(source_stage, int)
            or (not 0 <= source_stage < len(source_lessons))
        ):
            raise ValueError("Invalid source hover lesson")
        lesson = source_lessons[source_stage]
        candidates = [
            i for i, l in enumerate(VARIANT_ENVS[variant].lessons) if l == lesson
        ]
        if len(candidates) != 1:
            raise ValueError("No exact source hover lesson mapping exists")
        destination = candidates[0]
        if requested_stage is not None and requested_stage != destination:
            raise ValueError("Explicit stage conflicts with exact hover lesson mapping")
        from dataclasses import asdict

        mapping = dict(
            source_variant=source_variant,
            source_lesson=source_stage,
            destination_variant=variant,
            destination_lesson=destination,
            matched_lesson=asdict(lesson),
            method="exact_dataclass_equality",
        )
        return (validate_start_stage(variant, destination), mapping)
    stage = (
        (source_stage if source_variant == variant else 0)
        if requested_stage is None
        else requested_stage
    )
    return (validate_start_stage(variant, stage), mapping)


def resolve_frontier_fixed_height(value, source_metadata, variant):
    """None/'inherit' restores metadata; 'off' explicitly removes the experiment."""
    if value is None or value == "inherit":
        value = source_metadata.get("frontier_fixed_height")
    if value is None or value == "off":
        return None
    if isinstance(value, bool):
        raise ValueError("Frontier fixed height must be finite meters in [0, .14]")
    try:
        height = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Frontier fixed height must be inherit, off, or meters"
        ) from exc
    if not np.isfinite(height) or not 0 <= height <= 0.14:
        raise ValueError("Frontier fixed height must be finite meters in [0, .14]")
    if variant not in (
        "open_bootstrap",
        "hover_bootstrap",
        "fine_hover_bootstrap",
        "fine_descent_bootstrap",
    ):
        raise ValueError(
            "Frontier fixed height requires an open or hover bootstrap variant"
        )
    return height


def resolve_frontier_attempt_limit(value, source_metadata, variant):
    if value is None:
        value = source_metadata.get("frontier_attempt_limit_steps", 0)
    if type(value) is not int or not 0 <= value <= 500:
        raise ValueError("Frontier attempt limit must be integer 0..500")
    if value and variant not in (
        "near_pickup",
        "fine_near_pickup",
        "millimeter_near_pickup",
    ):
        raise ValueError("Frontier attempt limit requires near pickup")
    return value


def validate_interaction_schedule(target_new_steps, gate_step_interval):
    for value in (target_new_steps, gate_step_interval):
        if type(value) is not int or value < 0 or value % 2048:
            raise ValueError(
                "Interaction budgets and gates must be nonnegative multiples of 2048"
            )
    if gate_step_interval and (not target_new_steps):
        raise ValueError("Step gates require an interaction budget")


def make_env(
    variant,
    stage,
    worker,
    early_close_failure=False,
    frontier_reward_scale=1.0,
    strict_descent=False,
    bootstrap_height_jitter=0.0,
    pickup_potential_scale=0.0,
    pickup_curriculum=False,
    pickup_curriculum_state=None,
    pickup_initial_level=12,
    pickup_curriculum_kind="coarse",
    pickup_initial_opening_index=0,
    frontier_fixed_height=None,
    frontier_attempt_limit_steps=0,
):
    frontier_attempt_limit_steps = resolve_frontier_attempt_limit(
        frontier_attempt_limit_steps, {}, variant
    )
    frontier_fixed_height = resolve_frontier_fixed_height(
        frontier_fixed_height, {}, variant
    )
    if worker == 2:
        return Monitor(
            ReverseGraspEnv(
                curriculum_level=5,
                replay_fraction=0,
                observation="pixels",
                seed=501 + worker,
            )
        )
    if worker == 3:
        from .closure_potential import PickupPotential

        if pickup_curriculum and pickup_curriculum_kind == "fine":
            from .fine_pickup import FinePickupEnv

            env = FinePickupEnv(
                state=pickup_curriculum_state,
                initial_opening_index=pickup_initial_opening_index,
                observation="pixels",
                seed=501 + worker,
            )
        elif pickup_curriculum:
            from .adaptive_pickup import AdaptivePickupEnv

            env = AdaptivePickupEnv(
                state=pickup_curriculum_state,
                initial_level=pickup_initial_level,
                observation="pixels",
                seed=501 + worker,
            )
        else:
            env = ReverseGraspEnv(
                curriculum_level=17,
                replay_fraction=0,
                observation="pixels",
                seed=501 + worker,
            )
        return Monitor(PickupPotential(env, scale=pickup_potential_scale))
    cls = VARIANT_ENVS[variant]
    if variant in ("near_pickup", "fine_near_pickup", "millimeter_near_pickup"):
        if bootstrap_height_jitter or frontier_fixed_height is not None:
            raise ValueError(
                "Near pickup uses the exact lesson height, no jitter or fixed override"
            )
        frontier = cls(
            subtask_stage=stage,
            replay_fraction=0,
            observation="pixels",
            seed=501 + worker,
        )
        frontier.set_frontier_attempt_limit(frontier_attempt_limit_steps or None)
        frontier = frontier
        frontier = frontier
        return Monitor(RoleRewardScale(frontier, frontier_reward_scale))
    kw = (
        dict(subtask_stage=stage, height_range=(0.07, 0.08))
        if variant != "dense"
        else dict(bridge_stage=3)
    )
    if issubclass(cls, OpenBootstrapEnv):
        kw.update(
            early_close_failure=early_close_failure,
            strict_descent=strict_descent,
            bootstrap_height_jitter=bootstrap_height_jitter,
        )
    if worker == 0 and frontier_fixed_height is not None:
        kw["fixed_height"] = frontier_fixed_height
    return Monitor(
        RoleRewardScale(
            cls(**kw, replay_fraction=0, observation="pixels", seed=501 + worker),
            frontier_reward_scale,
        )
    )


def rollouts(model, factory, seeds, deterministic):
    rows = []
    env = factory()
    try:
        for seed in seeds:
            torch.manual_seed(seed)
            obs, initial = env.reset(seed=seed)
            snapshots = []
            initial_relative_height = float(
                env.grasp_position[2] - env.cup_position[2] - 0.014
            )
            minimum_relative_height = initial_relative_height
            minimum_aligned_open_height = None
            best_quality = 0.0
            bilateral_steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, _, done, truncated, info = env.step(action)
                bilateral_steps += int(all(info["contacts"]))
                best_quality = max(
                    best_quality, float(info.get("approach_quality", 0.0))
                )
                minimum_relative_height = min(
                    minimum_relative_height,
                    float(env.grasp_position[2] - env.cup_position[2] - 0.014),
                )
                if (
                    np.min(env.data.qpos[4:6]) >= 0.038
                    and np.linalg.norm((env.grasp_position - env.cup_position)[:2])
                    < 0.012
                ):
                    h = float(env.grasp_position[2] - env.cup_position[2] - 0.014)
                    minimum_aligned_open_height = (
                        h
                        if minimum_aligned_open_height is None
                        else min(h, minimum_aligned_open_height)
                    )
                if deterministic and (
                    env.step_count <= 25 or env.step_count % 25 == 0 or done
                ):
                    snapshots.append(
                        dict(
                            time=env.step_count * 0.02,
                            hand=env.grasp_position.tolist(),
                            cup=env.cup_position.tolist(),
                            openings=env.data.qpos[4:6].tolist(),
                            action=action.tolist(),
                            quality=info.get("approach_quality"),
                            contacts=info["contacts"],
                            approach_stable_steps=info.get("approach_stable_steps"),
                        )
                    )
                if done or truncated:
                    break
            rows.append(
                dict(
                    seed=seed,
                    deterministic=deterministic,
                    success=bool(info["is_success"]),
                    reason=info["reason"],
                    approach_episode=info.get("approach_episode", False),
                    approach_success=info.get("approach_success", False),
                    pickup_success=info.get("pickup_success", info["is_success"]),
                    quality=info.get("approach_quality"),
                    best_quality=best_quality,
                    snapshots=snapshots,
                    duration=env.step_count * 0.02,
                    bilateral_steps=info.get(
                        "bilateral_contact_steps", bilateral_steps
                    ),
                    peak_clearance=info["peak_clearance"],
                    initial_relative_height=initial_relative_height,
                    minimum_relative_height=minimum_relative_height,
                    terminal_relative_height=float(
                        env.grasp_position[2] - env.cup_position[2] - 0.014
                    ),
                    maximum_descent=initial_relative_height - minimum_relative_height,
                    minimum_xy_aligned_open_relative_height=minimum_aligned_open_height,
                )
            )
    finally:
        env.close()
    return rows


def gate(model, variant, stage, *, final=False, strict_descent=False):
    result = {}
    n = 10 if final else 3
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        if variant != "dense":
            cls, _, approach_last, pickup_first = curriculum_definition(variant)
            for h, eval_stage, label in gate_configurations(variant, stage):
                kw = (
                    {}
                    if variant
                    in ("near_pickup", "fine_near_pickup", "millimeter_near_pickup")
                    else dict(fixed_height=h)
                )
                if issubclass(cls, OpenBootstrapEnv):
                    kw["strict_descent"] = strict_descent
                factory = lambda es=eval_stage, kw=kw: cls(
                    subtask_stage=es, replay_fraction=0, observation="pixels", **kw
                )
                key = f"{h}-{label}"
                result[key] = rollouts(model, factory, [190000], True) + rollouts(
                    model, factory, range(190000, 190000 + n), False
                )
        else:
            result["pickup"] = pickup_evaluate(
                model, [0.065, 0.07, 0.075], noisy_episodes=n, seed_start=190000
            )
        for level in [5, 17]:
            factory = lambda level=level: ReverseGraspEnv(
                curriculum_level=level, replay_fraction=0, observation="pixels"
            )
            result[f"retention{level}"] = rollouts(
                model, factory, [191000], True
            ) + rollouts(model, factory, range(191000, 191000 + n), False)
    passed = True
    for key, rows in result.items():
        groups = (
            [rows]
            if key != "pickup"
            else [[r for r in rows if r["height"] == h] for h in [0.065, 0.07, 0.075]]
        )
        for group in groups:
            for deterministic in [True, False]:
                r = [x for x in group if x["deterministic"] == deterministic]
                passed &= bool(r) and np.mean([x["success"] for x in r]) >= 0.8
    return dict(passed=bool(passed), rows=result)


FIXED_COMPARISON_STAGES = (45, 53, 57, 58, 60, 65, 68, 70)


def fixed_height_evaluation(model):
    from .grasp_benchmark import preserved_rng

    results = {}
    with preserved_rng():
        for stage in FIXED_COMPARISON_STAGES:
            factory = lambda stage=stage: FineNearPickupEnv(
                subtask_stage=stage, replay_fraction=0, observation="pixels"
            )
            rows = rollouts(model, factory, [198000], True) + rollouts(
                model, factory, range(198000, 198010), False
            )
            results[f"{FineNearPickupEnv.heights[stage]:.3f}-pickup"] = rows
        for level in (5, 17):
            factory = lambda level=level: ReverseGraspEnv(
                curriculum_level=level, replay_fraction=0, observation="pixels"
            )
            results[f"retention{level}"] = rollouts(
                model, factory, [198000], True
            ) + rollouts(model, factory, range(198000, 198010), False)
    return dict(
        seed_start=198000, episodes_per_height=11, reward_wrappers=False, rows=results
    )


def arm_ramp_scales(spent, starting=(1.0, 1.0, 1.0, 1.0)):
    if not np.isfinite(spent) or spent < 0:
        raise ValueError("Invalid elapsed training time")
    if len(starting) != 4 or not 1 <= starting[1] <= 6:
        raise ValueError("Invalid starting arm ramp")
    return (
        1.0,
        float(starting[1] + (6.0 - starting[1]) * min(spent / 600.0, 1.0)),
        1.0,
        1.0,
    )


class EpisodeCounter(BaseCallback):
    def __init__(self):
        super().__init__()
        self.counts = Counter()
        self.worker_counts = Counter()
        self.height_counts = Counter()
        self.recent = {i: deque(maxlen=100) for i in range(4)}

    def recent_summary(self):
        return {
            str(i): dict(
                episodes=len(rows),
                success_rate=float(np.mean([r[0] for r in rows])),
                mean_steps=float(np.mean([r[1] for r in rows])),
                mean_return=float(np.mean([r[2] for r in rows])),
            )
            for i, rows in self.recent.items()
            if rows
        }

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            if "episode" in info:
                role = "frontier" if i < 2 else "hold" if i == 2 else "pickup"
                self.counts[f"{role}:{info['reason']}"] += 1
                self.worker_counts[f"{i}:{info['reason']}"] += 1
                if i < 2 and "near_pickup_height" in info:
                    self.height_counts[
                        f"{i}:{info['near_pickup_height']:.3f}:{info['reason']}"
                    ] += 1
                self.recent[i].append(
                    (
                        bool(info["is_success"]),
                        info["episode"]["l"],
                        info["episode"]["r"],
                    )
                )
        return True


def resolve_training_source(source, checkpoint=None):
    """Final policy aliases require final metadata; snapshots require a gate record."""
    source = Path(source)
    name = checkpoint or "policy.zip"
    if Path(name).name != name:
        raise ValueError("Checkpoint must be a basename")
    final = name == "policy.zip"
    summary_path = source / "summary.json"
    if not summary_path.exists():
        if final:
            raise ValueError("Final policy requires completed summary.json")
        summary_path = source / "experiment.json"
    summary = json.loads(summary_path.read_text())
    record = summary
    if not final:
        gates = summary.get("gates")
        if gates is None:
            gates = json.loads((source / "gates.json").read_text())
        record = next((g for g in gates if g["checkpoint"] == name), None)
        if record is None:
            raise ValueError("Checkpoint is not in source gates")
    path = source / name
    if file_hash(path) != record.get("policy_sha256"):
        raise ValueError("Source checkpoint hash mismatch")
    return (path, summary, record)


def train(
    source,
    out,
    variant,
    budget=300.0,
    commit=lambda: None,
    stage=None,
    noise_scale=1.0,
    early_close_failure=False,
    gate_interval=60.0,
    frontier_reward_scale=1.0,
    strict_descent=False,
    role_normalization=False,
    source_checkpoint=None,
    arm_noise_scale=1.0,
    learning_rate_scale=1.0,
    bootstrap_height_jitter=0.0,
    decoupled=False,
    critic_learning_rate=0.0003,
    context_exploration=False,
    context_std_scales=None,
    context_role_normalization=None,
    allow_context_reconfigure=False,
    role_gradient_balance=None,
    gradient_balance_max_weight=None,
    pickup_potential_scale=None,
    pickup_curriculum=None,
    pickup_initial_level=12,
    pickup_curriculum_kind=None,
    pickup_initial_opening_index=0,
    finger_exploration_max=None,
    actor_update_scope=None,
    wide_finger_residual=None,
    pickup_freeze_opening=None,
    pickup_attempt_limit_steps=None,
    frontier_fixed_height=None,
    stop_after_passed_gate=False,
    arm_exploration_ramp=False,
    promote_passed_baseline=True,
    fixed_evaluation_interval=0.0,
    frontier_attempt_limit_steps=None,
    target_new_steps=0,
    gate_step_interval=0,
    arm_residual_mode=None,
):
    if fixed_evaluation_interval and variant not in (
        "fine_near_pickup",
        "millimeter_near_pickup",
    ):
        raise ValueError("Fixed height evaluation requires fine/millimeter near pickup")
    if not np.isfinite(fixed_evaluation_interval) or fixed_evaluation_interval < 0:
        raise ValueError("Invalid fixed evaluation interval")
    if type(stop_after_passed_gate) is not bool:
        raise ValueError("stop_after_passed_gate must be bool")
    if variant not in VARIANT_ENVS:
        raise ValueError(variant)
    if (
        variant in ("near_pickup", "fine_near_pickup", "millimeter_near_pickup")
        and bootstrap_height_jitter
    ):
        raise ValueError("Near pickup does not support height jitter")
    if not all(
        (
            np.isfinite(v) and v > 0
            for v in [
                noise_scale,
                arm_noise_scale,
                frontier_reward_scale,
                learning_rate_scale,
            ]
        )
    ):
        raise ValueError("Reward and exploration scales must be positive and finite")
    source, out = (Path(source), Path(out))
    out.mkdir(parents=True, exist_ok=False)
    validate_interaction_schedule(target_new_steps, gate_step_interval)
    path, source_summary, source_record = resolve_training_source(
        source, source_checkpoint
    )
    frontier_attempt_limit_steps = resolve_frontier_attempt_limit(
        frontier_attempt_limit_steps, source_summary, variant
    )
    frontier_fixed_height = resolve_frontier_fixed_height(
        frontier_fixed_height, source_summary, variant
    )
    pickup_potential_scale = (
        source_summary.get("pickup_potential_scale", 0.0)
        if pickup_potential_scale is None
        else float(pickup_potential_scale)
    )
    if not np.isfinite(pickup_potential_scale) or pickup_potential_scale < 0:
        raise ValueError("Pickup potential scale must be finite and nonnegative")
    pickup_curriculum = bool(
        source_summary.get("pickup_curriculum", False)
        if pickup_curriculum is None
        else pickup_curriculum
    )
    source_pickup_kind = source_summary.get("pickup_curriculum_kind") or "coarse"
    pickup_curriculum_kind = pickup_curriculum_kind or source_pickup_kind
    if pickup_curriculum_kind not in ("coarse", "fine"):
        raise ValueError("Pickup curriculum kind must be coarse or fine")
    if (
        pickup_curriculum
        and source_summary.get("pickup_curriculum")
        and (pickup_curriculum_kind != source_pickup_kind)
    ):
        raise ValueError("Cannot silently change a saved pickup curriculum kind")
    pickup_curriculum_state = (
        source_record.get("pickup_curriculum_state") if pickup_curriculum else None
    )
    if (
        pickup_curriculum
        and source_summary.get("pickup_curriculum")
        and (pickup_curriculum_state is None)
    ):
        raise ValueError("Adaptive pickup checkpoint lacks exact curriculum state")
    requested_start_stage = stage
    source_lesson = source_record.get("lesson", source_summary.get("final_lesson", 0))
    default_stage, _ = resolve_source_lesson(variant, source_summary, source_record)
    stage, source_lesson_mapping = resolve_source_lesson(
        variant, source_summary, source_record, stage
    )
    explicit_curriculum_override = None
    if requested_start_stage is not None and stage != default_stage:
        explicit_curriculum_override = dict(
            from_lesson=default_stage,
            to_lesson=stage,
            direction="forward" if stage > default_stage else "backward",
            omitted_intermediate_lessons=list(range(default_stage + 1, stage)),
            note="Experimental curriculum override; omitted lessons are not claimed as passed.",
        )
    from .torch_precision import configure_policy_precision

    precision = configure_policy_precision()
    torch.set_num_threads(2)
    torch.manual_seed(501)
    np.random.seed(501)
    context_exploration = bool(
        context_exploration or source_summary.get("context_exploration", False)
    )
    decoupled = bool(
        decoupled or context_exploration or source_summary.get("decoupled", False)
    )
    if decoupled and role_normalization:
        raise ValueError(
            "Use separate experiments for role normalization and decoupled optimization"
        )
    if (
        actor_update_scope is not None
        or wide_finger_residual is not None
        or arm_residual_mode is not None
    ) and (not context_exploration):
        raise ValueError(
            "Actor update scope requires context exploration in this trainer"
        )
    if not context_exploration and (
        context_role_normalization is not None
        or allow_context_reconfigure
        or role_gradient_balance is not None
        or (gradient_balance_max_weight is not None)
        or (finger_exploration_max is not None)
    ):
        raise ValueError("Context options require context exploration")
    env = SubprocVecEnv(start_method="spawn")

    def current_pickup_state():
        return (
            env.env_method("get_pickup_curriculum_state", indices=[3])[0]
            if pickup_curriculum
            else None
        )

    if pickup_freeze_opening is not None:
        if not pickup_curriculum or pickup_curriculum_kind != "fine":
            env.close()
            raise ValueError(
                "Freezing pickup opening requires the fine pickup curriculum"
            )
        env.env_method(
            "set_pickup_promotion_enabled", not pickup_freeze_opening, indices=[3]
        )
    if pickup_attempt_limit_steps is not None:
        if not pickup_curriculum or pickup_curriculum_kind != "fine":
            env.close()
            raise ValueError("Pickup attempt limits require the fine pickup curriculum")
        env.env_method(
            "set_pickup_attempt_limit",
            None if pickup_attempt_limit_steps == 0 else pickup_attempt_limit_steps,
            indices=[3],
        )
    if context_exploration:
        from .context_exploration import load_context_exploration_ppo

        scales = (
            context_std_scales
            or (
                source_summary.get("worker_std_scales")
                if source_summary.get("context_exploration")
                else None
            )
            or (10.0, 25.0, 1.0, 1.0)
        )
        model = load_context_exploration_ppo(
            path,
            env=env,
            device="cuda",
            critic_learning_rate=critic_learning_rate,
            std_scales=scales,
            role_normalization=context_role_normalization,
            allow_context_reconfigure=allow_context_reconfigure,
            role_gradient_balance=role_gradient_balance,
            gradient_balance_max_weight=gradient_balance_max_weight,
            finger_exploration_max=finger_exploration_max,
            actor_update_scope=actor_update_scope,
            wide_finger_residual=wide_finger_residual,
            arm_residual_mode=arm_residual_mode,
        )
    elif decoupled:
        from .decoupled_ppo import load_decoupled_ppo

        model = load_decoupled_ppo(
            path, env=env, device="cuda", critic_learning_rate=critic_learning_rate
        )
    elif role_normalization:
        from .role_ppo import load_role_normalized_ppo

        model = load_role_normalized_ppo(path, env=env, device="cuda")
    else:
        model = PPO.load(path, env=env, device="cuda")
    if arm_exploration_ramp and (not context_exploration):
        env.close()
        raise ValueError("Arm ramp requires context exploration")
    arm_ramp_start = tuple(getattr(model, "runtime_arm_scales", (1.0, 1.0, 1.0, 1.0)))
    if arm_exploration_ramp:
        arm_ramp_scales(0.0, arm_ramp_start)
    if pickup_potential_scale and model.gamma != 0.995:
        env.close()
        raise ValueError(
            "Pickup potential requires policy discount matching environment gamma=.995"
        )
    from stable_baselines3.common.utils import get_schedule_fn

    initial_learning_rate = float(model.lr_schedule(1.0)) * learning_rate_scale
    model.learning_rate = initial_learning_rate
    model.lr_schedule = get_schedule_fn(initial_learning_rate)
    model.gae_lambda = 0.98
    model.rollout_buffer.gae_lambda = 0.98
    with torch.no_grad():
        model.policy.log_std[-1].add_(np.log(noise_scale))
        model.policy.log_std[:-1].add_(np.log(arm_noise_scale))

    def frozen_actor_hash():
        scope = getattr(model, "actor_update_scope", "all")
        if scope not in (
            "finger_head",
            "wide_finger",
            "arm_head",
            "arm_residual",
            "grasp_residual",
        ):
            return None
        actor_ids = {id(p) for p in model.actor_parameters}
        digest = hashlib.sha256()
        for name, parameter in sorted(model.policy.named_parameters()):
            if id(parameter) not in actor_ids:
                continue
            value = parameter.detach()
            if scope in ("arm_residual", "grasp_residual") and name.startswith(
                "arm_residual_head."
            ):
                continue
            if scope in ("wide_finger", "grasp_residual") and name.startswith(
                "wide_finger_head."
            ):
                continue
            if scope == "finger_head" and name.startswith("action_net."):
                value = value[:-1]
            elif scope == "arm_head" and name.startswith("action_net."):
                value = value[-1:]
            digest.update(name.encode())
            digest.update(value.cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    initial_frozen_actor_hash = frozen_actor_hash()
    starting = model.num_timesteps
    root = Path(__file__).resolve().parents[1]
    files = [
        "robovision/train_skill_curriculum.py",
        "robovision/approach_subtask.py",
        "robovision/dense_approach.py",
        "robovision/near_pickup.py",
        "robovision/approach_bridge.py",
        "robovision/open_bootstrap.py",
        "robovision/hover_bootstrap.py",
        "robovision/fine_pickup.py",
        "robovision/adaptive_pickup.py",
        "robovision/closure_potential.py",
        "robovision/torch_precision.py",
        "robovision/context_exploration.py",
        "robovision/policy_loading.py",
        "robovision/decoupled_ppo.py",
        "robovision/role_ppo.py",
        "robovision/reward_scale.py",
        "robovision/cnn.py",
        "robovision/env.py",
        "robovision/reverse_curriculum.py",
        "robovision/joint_env.py",
        "assets/cup_arm.xml",
    ]
    metadata = dict(
        frontier_attempt_limit_steps=frontier_attempt_limit_steps,
        target_new_steps=target_new_steps,
        gate_step_interval=gate_step_interval,
        arm_exploration_ramp=arm_exploration_ramp,
        arm_ramp_start=arm_ramp_start,
        arm_ramp_target=[1.0, 6.0, 1.0, 1.0],
        arm_ramp_training_seconds=600.0,
        fixed_evaluation_interval=fixed_evaluation_interval,
        fixed_evaluation_stages=FIXED_COMPARISON_STAGES
        if fixed_evaluation_interval
        else None,
        near_pickup_specification=VARIANT_ENVS[variant].specification()
        if variant in ("near_pickup", "fine_near_pickup", "millimeter_near_pickup")
        else None,
        fine_descent_specification=FineDescentBootstrapEnv.specification()
        if variant == "fine_descent_bootstrap"
        else None,
        requested_start_stage=requested_start_stage,
        source_lesson=source_lesson,
        source_variant=source_summary.get("variant"),
        explicit_curriculum_override=explicit_curriculum_override,
        stop_after_passed_gate=stop_after_passed_gate,
        source_lesson_mapping=source_lesson_mapping,
        frontier_fixed_height=frontier_fixed_height,
        frontier_fixed_height_worker=0 if frontier_fixed_height is not None else None,
        pickup_curriculum_kind=pickup_curriculum_kind,
        pickup_curriculum=pickup_curriculum,
        initial_pickup_curriculum_state=current_pickup_state(),
        pickup_potential_scale=pickup_potential_scale,
        torch_precision=precision,
        variant=variant,
        source_run=source.name,
        source_policy_sha256=file_hash(path),
        source_final_steps=starting,
        source_checkpoint=path.name,
        source_training_seconds=source_record.get("training_seconds"),
        starting_lesson=stage,
        training_budget_seconds=budget,
        gae_lambda=0.98,
        context_exploration=context_exploration,
        worker_std_scales=list(model.std_scales)
        if context_exploration
        else [1.0, 1.0, 1.0, 1.0],
        decoupled=decoupled,
        critic_learning_rate=critic_learning_rate if decoupled else None,
        learning_rate=initial_learning_rate,
        learning_rate_scale=learning_rate_scale,
        finger_noise_scale=noise_scale,
        arm_noise_scale=arm_noise_scale,
        early_close_failure=early_close_failure,
        gate_interval=gate_interval,
        frontier_reward_scale=frontier_reward_scale,
        strict_descent=strict_descent,
        bootstrap_height_jitter=bootstrap_height_jitter,
        role_normalization=role_normalization,
        workers=["frontier", "frontier", "hold5", "pickup17"],
        arm_residual_mode=getattr(model.policy, "arm_residual_mode", None),
        actor_update_scope=getattr(model, "actor_update_scope", "all"),
        frozen_actor_sha256=initial_frozen_actor_hash,
        wide_finger_residual=bool(getattr(model.policy, "wide_finger_residual", False)),
        pickup_freeze_opening=not current_pickup_state().get("promotion_enabled", True)
        if pickup_curriculum
        else False,
        pickup_attempt_limit_steps=current_pickup_state().get("attempt_limit")
        if pickup_curriculum
        else None,
        finger_head_optimizer_semantics=getattr(model, "algorithm_metadata", {}).get(
            "finger_head_optimizer_semantics"
        ),
        context_role_normalization=getattr(model, "role_normalization", False)
        if context_exploration
        else None,
        finger_exploration_max=list(
            getattr(model, "finger_exploration_max", (1.0, 1.0, 1.0, 1.0))
        ),
        finger_exploration_quiet_m=getattr(model, "algorithm_metadata", {}).get(
            "finger_exploration_quiet_m"
        ),
        finger_exploration_full_m=getattr(model, "algorithm_metadata", {}).get(
            "finger_exploration_full_m"
        ),
        context_reconfiguration=getattr(model, "context_reconfiguration", None),
        role_gradient_balance=getattr(model, "role_gradient_balance", False),
        gradient_balance_max_weight=getattr(model, "gradient_balance_max_weight", None),
        source_sha256={f: file_hash(root / f) for f in files},
        promote_passed_baseline=promote_passed_baseline,
        semantics="Explicit PPO fork; weights/optimizer retained, optional initial noise scales, fresh RNG streams, no demonstrations",
    )
    atomic_json(out / "experiment.json", metadata)
    spent = 0.0
    next_eval = gate_interval
    next_fixed = (
        fixed_evaluation_interval if fixed_evaluation_interval else float("inf")
    )
    checkpoints = []
    next_step_eval = gate_step_interval if gate_step_interval else float("inf")
    stop_reason = "budget"
    counter = EpisodeCounter()
    try:
        baseline = gate(model, variant, stage, strict_descent=strict_descent)
        atomic_json(out / "baseline.json", baseline)
        if fixed_evaluation_interval:
            atomic_json(out / "fixed-baseline.json", fixed_height_evaluation(model))
        commit()
        _, lessons, _, _ = curriculum_definition(variant)
        next_stage, _ = gate_transition(
            stage,
            len(lessons) - 1 if variant != "dense" else None,
            baseline["passed"] and promote_passed_baseline,
            stop_after_passed_gate=stop_after_passed_gate,
            baseline=True,
        )
        if next_stage != stage:
            stage = next_stage
            env.env_method("set_subtask_stage", stage, indices=[0, 1])
        while spent < budget and (
            not target_new_steps or model.num_timesteps - starting < target_new_steps
        ):
            if arm_exploration_ramp:
                from .context_exploration import set_runtime_arm_scales

                set_runtime_arm_scales(model, arm_ramp_scales(spent, arm_ramp_start))
            start = time.monotonic()
            model.learn(
                total_timesteps=2048,
                reset_num_timesteps=False,
                progress_bar=False,
                callback=counter,
            )
            spent += time.monotonic() - start
            atomic_json(
                out / "progress.json",
                dict(
                    runtime_arm_scales=list(
                        getattr(model, "runtime_arm_scales", (1.0, 1.0, 1.0, 1.0))
                    ),
                    pickup_curriculum_state=current_pickup_state(),
                    training_seconds=spent,
                    new_steps=model.num_timesteps - starting,
                    lesson=stage,
                    episodes=dict(counter.counts),
                    worker_episodes=dict(counter.worker_counts),
                    height_episodes=dict(counter.height_counts),
                    recent_episodes=counter.recent_summary(),
                    action_std=model.policy.log_std.detach().exp().cpu().tolist(),
                    role_advantage_stats=getattr(
                        model.rollout_buffer, "raw_role_advantage_stats", None
                    ),
                    train_metrics={
                        k: float(v)
                        for k, v in model.logger.name_to_value.items()
                        if k.startswith("train/") and np.isscalar(v)
                    },
                ),
            )
            new_steps = model.num_timesteps - starting
            interaction_done = bool(target_new_steps and new_steps >= target_new_steps)
            if interaction_done:
                stop_reason = "interaction_budget"
            if (
                (
                    new_steps >= next_step_eval
                    if gate_step_interval
                    else spent >= next_eval
                )
                or spent >= next_fixed
                or spent >= budget
                or interaction_done
            ):
                assert frozen_actor_hash() == initial_frozen_actor_hash, (
                    "Frozen actor parameters changed"
                )
                result = gate(model, variant, stage, strict_descent=strict_descent)
                file = out / f"step-{model.num_timesteps}.zip"
                model.save(file)
                record = dict(
                    frozen_actor_sha256=frozen_actor_hash(),
                    pickup_curriculum_state=current_pickup_state(),
                    training_seconds=spent,
                    steps=model.num_timesteps,
                    lesson=stage,
                    checkpoint=file.name,
                    policy_sha256=file_hash(file),
                    action_std=model.policy.log_std.detach().exp().cpu().tolist(),
                    **result,
                )
                if fixed_evaluation_interval and (
                    spent >= next_fixed or spent >= budget or interaction_done
                ):
                    record["fixed_height_evaluation"] = fixed_height_evaluation(model)
                    next_fixed = spent + fixed_evaluation_interval
                record["runtime_arm_scales"] = list(
                    getattr(model, "runtime_arm_scales", (1.0, 1.0, 1.0, 1.0))
                )
                checkpoints.append(record)
                atomic_json(out / "gates.json", checkpoints)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in record.items()
                            if k not in ("rows", "fixed_height_evaluation")
                        }
                    ),
                    flush=True,
                )
                _, lessons, _, _ = curriculum_definition(variant)
                next_stage, stop = gate_transition(
                    stage,
                    len(lessons) - 1 if variant != "dense" else None,
                    result["passed"],
                    stop_after_passed_gate=stop_after_passed_gate,
                )
                if stop:
                    stop_reason = "passed_gate"
                    commit()
                    break
                if next_stage != stage:
                    stage = next_stage
                    env.env_method("set_subtask_stage", stage, indices=[0, 1])
                next_eval = spent + gate_interval
                next_step_eval = (
                    new_steps + gate_step_interval
                    if gate_step_interval
                    else float("inf")
                )
                commit()
        model.save(out / "policy.zip")
        final = gate(model, variant, stage, final=True, strict_descent=strict_descent)
        final_fixed_height_evaluation = None
        standard = pickup_evaluate(
            model, [0.05, 0.065, 0.075, 0.08], noisy_episodes=10, seed_start=195000
        )
        summary = dict(
            **metadata,
            final_runtime_arm_scales=list(
                getattr(model, "runtime_arm_scales", (1.0, 1.0, 1.0, 1.0))
            ),
            stop_reason=stop_reason,
            pickup_curriculum_state=current_pickup_state(),
            training_seconds=spent,
            new_steps=model.num_timesteps - starting,
            final_steps=model.num_timesteps,
            final_lesson=stage,
            policy_sha256=file_hash(out / "policy.zip"),
            baseline=baseline,
            final=final,
            final_fixed_height_evaluation=final_fixed_height_evaluation,
            standard_pickup=standard,
            gates=checkpoints,
            episodes=dict(counter.counts),
            worker_episodes=dict(counter.worker_counts),
            height_episodes=dict(counter.height_counts),
            recent_episodes=counter.recent_summary(),
        )
        atomic_json(out / "summary.json", summary)
        commit()
        return summary
    finally:
        env.close()
