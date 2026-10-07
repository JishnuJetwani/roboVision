"""Frozen 200-episode confirmation of the complete centered-height pickup task.

Sources are locked before scene generation. Camera conditions affect every
network input; no simulator geometry selects an expert or modifies its forces.
Partial runs are deliberately not resumed, so interrupted work is not silently
replayed or omitted from accounting. Models remain at their supplied paths.
"""

from __future__ import annotations
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from .centered_grasp_env import CenteredGraspEnv
from .grasp_benchmark import preserved_rng, wilson_interval
from .policy_state import _optimizer_hash
from .hierarchical_policy import _observation
from .policy_state import _counters
from .io import atomic_json
from .policy_state import policy_state_hash
from .ordered_pickup_policy import OrderedPickupPolicy
from .policy_loading import load_grasp_policy
from .skill_hard_evaluation import basename, resolve_skill_checkpoint, sha256
from .train_ordered_pickup import VERSION as MANAGER_VERSION, validate_ordered_manager

VERSION = "ordered-centered-height-confirmation-200-v2"
EPISODES = 200
SUCCESS_THRESHOLD = 180
SEED_MIN = 700000000
SEED_MAX = 799999800
PHYSICAL_GAMMA = 0.999
MAX_ACTIONS = 500
CONDITIONS = ("normal", "frozen", "black")


def validate_protocol(seed, conditions=("normal",), option_steps=5):
    if type(seed) is not int or not SEED_MIN <= seed <= SEED_MAX:
        raise ValueError(
            "Use the dedicated development seed range700000000–799999800; reserved800000000..199 is excluded"
        )
    if (
        not isinstance(conditions, (tuple, list))
        or not conditions
        or any((c not in CONDITIONS for c in conditions))
        or (len(set(conditions)) != len(conditions))
        or (conditions[0] != "normal")
    ):
        raise ValueError(
            "Normal must be first, followed by optional unique frozen/black conditions"
        )
    if type(option_steps) is not int or option_steps not in (5, 25):
        raise ValueError("Use the selected manager duration5 or25")
    return tuple(conditions)


def validate_source(source):
    if not isinstance(source, dict) or set(source) != {"run", "checkpoint", "sha256"}:
        raise ValueError(
            "Each frozen source requires exactly run, checkpoint and sha256"
        )
    basename(source["run"])
    basename(source["checkpoint"])
    value = source["sha256"]
    if (
        not source["checkpoint"].endswith(".zip")
        or not isinstance(value, str)
        or len(value) != 64
        or any((c not in "0123456789abcdef" for c in value))
    ):
        raise ValueError("An explicit ZIP and lowercase SHA256 are required")


def confirmation_cases(seed):
    """Two hundred held-out heights with a fixed cup directly below the hand."""
    validate_protocol(seed)
    rng = np.random.default_rng(seed)
    heights = rng.uniform(0.025, 0.14, EPISODES)
    return [
        dict(case_index=i, seed=seed + i, height=float(height), cup_offset=[0.0, 0.0])
        for i, height in enumerate(heights)
    ]


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def image_condition(observation, reset_image, condition):
    """Transform the complete stack input; leave proprioception numerically exact."""
    _observation(observation)
    if condition not in CONDITIONS:
        raise ValueError("Unknown image condition")
    if np.shape(reset_image) != (6, 96, 96):
        raise ValueError("Frozen condition requires the complete reset image pair")
    pixels = (
        np.zeros_like(observation["image"])
        if condition == "black"
        else reset_image
        if condition == "frozen"
        else observation["image"]
    )
    return dict(
        image=np.array(pixels, copy=True),
        proprio=np.array(observation["proprio"], copy=True),
    )


def frozen_state(models):
    return [
        dict(
            policy_sha256=policy_state_hash(model),
            optimizer_sha256=_optimizer_hash(model),
            counters=_counters(model),
        )
        for model in models
    ]


def summarize(rows, conditions):
    scores = {}
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        successes = sum((r["centered_success"] and r["error"] is None for r in subset))
        high = [r for r in subset if r["height"] >= 0.1]
        scores[condition] = dict(
            episodes=len(subset),
            errors=sum((r["error"] is not None for r in subset)),
            centered_successes=successes,
            original_successes=sum((r["original_success_ever"] for r in subset)),
            centered_success_rate=successes / len(subset) if subset else None,
            centered_wilson95=wilson_interval(successes, len(subset)),
            high100plus_episodes=len(high),
            high100plus_centered_successes=sum((r["centered_success"] for r in high)),
            physical_actions=sum((r["physical_actions"] for r in subset)),
            physical_action_attempts=sum(
                (r["physical_action_attempts"] for r in subset)
            ),
            manager_decisions=sum((r["manager_decisions"] for r in subset)),
            failures=dict(
                Counter((r["reason"] for r in subset if not r["centered_success"]))
            ),
        )
    normal = scores["normal"]
    nominal_pass = (
        normal["centered_successes"] >= SUCCESS_THRESHOLD
        if normal["episodes"] == EPISODES and (not normal["errors"])
        else None
    )
    drops = {}
    normal_rows = {r["case_index"]: r for r in rows if r["condition"] == "normal"}
    for condition in conditions[1:]:
        paired = [
            (normal_rows[b["case_index"]], b)
            for b in rows
            if b["condition"] == condition and b["case_index"] in normal_rows
        ]
        drops[condition] = dict(
            paired_episodes=len(paired),
            centered_success_count_drop=sum(
                (
                    int(a["centered_success"]) - int(b["centered_success"])
                    for a, b in paired
                )
            ),
            normal_success_ablation_failure=sum(
                (
                    a["centered_success"] and (not b["centered_success"])
                    for a, b in paired
                )
            ),
            normal_failure_ablation_success=sum(
                (not a["centered_success"] and b["centered_success"] for a, b in paired)
            ),
        )
    return dict(
        by_condition=scores,
        nominal_stage=dict(
            required_normal_episodes=EPISODES,
            minimum_centered_successes=SUCCESS_THRESHOLD,
            passed=nominal_pass,
            scope="Complete centered pickup across 25–140 mm starting heights",
        ),
        paired_ablation_drops=drops,
        physical_actions=sum((r["physical_actions"] for r in rows)),
        physical_action_attempts=sum((r["physical_action_attempts"] for r in rows)),
        manager_decisions=sum((r["manager_decisions"] for r in rows)),
    )


def evaluate_confirmation(
    manager,
    experts,
    *,
    seed,
    cases=None,
    conditions=("normal",),
    option_steps=5,
    progress=lambda report: None,
    case_indices=None,
):
    conditions = validate_protocol(seed, conditions, option_steps)
    expected = confirmation_cases(seed)
    if cases is not None and cases != expected:
        raise ValueError(
            "Cases differ from the registered deterministic scene generator"
        )
    cases = expected
    if case_indices is not None:
        indices = list(case_indices)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < EPISODES for i in indices)
        ):
            raise ValueError(
                "Case indices must be distinct integers within the registered 200 scenes"
            )
        cases = [expected[i] for i in indices]
    validate_ordered_manager(manager, option_steps)
    if len(experts) != 2:
        raise ValueError("Exactly approach and pickup experts required")
    models = (manager, *experts)
    before = frozen_state(models)
    modes = [
        (module, module.training)
        for model in models
        for module in model.policy.modules()
    ]
    flags = [
        (p, p.requires_grad) for model in models for p in model.policy.parameters()
    ]
    rows = []
    started = time.monotonic()
    with preserved_rng():
        try:
            stack = OrderedPickupPolicy(
                manager,
                experts,
                option_steps=option_steps,
                expert_deterministic=True,
                manager_deterministic=True,
            )
            for condition in conditions:
                for case in cases:
                    row = dict(
                        **copy.deepcopy(case),
                        condition=condition,
                        physical_actions=0,
                        physical_action_attempts=0,
                        manager_decisions=0,
                        expert_action_counts=[0, 0],
                        centered_success=False,
                        original_success_ever=False,
                        first_original_success_step=None,
                        first_centered_success_step=None,
                        first_contact_step=None,
                        first_bilateral_contact_step=None,
                        reason="error",
                        error=None,
                        episode_return=0.0,
                        committed=False,
                        commit_step=None,
                    )
                    progress(
                        dict(
                            status="episode_running",
                            rows=rows,
                            active_case=copy.deepcopy(row),
                            scores=summarize(rows, conditions),
                            incomplete_episode_physical_upper_bound=MAX_ACTIONS,
                        )
                    )
                    stack.reset()
                    random.seed(case["seed"])
                    np.random.seed(case["seed"])
                    torch.manual_seed(case["seed"])
                    env = None
                    try:
                        env = CenteredGraspEnv(
                            fixed_height=case["height"],
                            seed=case["seed"],
                            observation="pixels",
                            render_images=True,
                            gamma=PHYSICAL_GAMMA,
                        )
                        if env.max_steps != MAX_ACTIONS or not np.isclose(
                            env.control_dt, 0.02, rtol=0.0, atol=1e-12
                        ):
                            raise RuntimeError(
                                "Confirmation requires the original500-action deadline and50Hz forces"
                            )
                        observation, info = env.reset(seed=case["seed"])
                        if env.step_count != 0:
                            raise RuntimeError(
                                "Confirmation must start a new complete physical episode"
                            )
                        # Freeze both camera frames together so the ablation cannot leak motion.
                        reset_image = observation["image"].copy()
                        for _ in range(MAX_ACTIONS):
                            observed = image_condition(
                                observation, reset_image, condition
                            )
                            action, state = stack.predict(observed, deterministic=True)
                            if state is not None:
                                raise RuntimeError(
                                    "Ordered stack unexpectedly returned recurrent state"
                                )
                            row["physical_action_attempts"] += 1
                            observation, reward, terminated, truncated, info = env.step(
                                action
                            )
                            row["physical_actions"] += 1
                            if env.step_count != row["physical_actions"]:
                                raise RuntimeError(
                                    "Each force prediction must advance one physical action"
                                )
                            if not np.isfinite(reward):
                                raise RuntimeError("Nonfinite physical task reward")
                            row["episode_return"] += float(reward)
                            row["original_success_ever"] |= bool(
                                info["original_success_ever"]
                            )
                            row["centered_success"] |= bool(info["centered_success"])
                            for key, yes in (
                                (
                                    "first_original_success_step",
                                    info["original_is_success"],
                                ),
                                (
                                    "first_centered_success_step",
                                    info["centered_success"],
                                ),
                                ("first_contact_step", any(info["contacts"])),
                                ("first_bilateral_contact_step", all(info["contacts"])),
                            ):
                                if yes and row[key] is None:
                                    row[key] = env.step_count
                            if terminated or truncated:
                                row.update(
                                    reason=info["reason"],
                                    terminated=bool(terminated),
                                    truncated=bool(truncated),
                                )
                                break
                        else:
                            raise RuntimeError(
                                "Environment failed to terminate by its original physical deadline"
                            )
                        row["centered_success_by_first_original_success"] = bool(
                            row["first_centered_success_step"] is not None
                            and row["first_original_success_step"] is not None
                            and (
                                row["first_centered_success_step"]
                                <= row["first_original_success_step"]
                            )
                        )
                        row["post_original_success_actions"] = (
                            0
                            if row["first_original_success_step"] is None
                            else row["physical_actions"]
                            - row["first_original_success_step"]
                        )
                    except Exception as error:
                        row["error"] = dict(
                            type=type(error).__name__, message=str(error)
                        )
                        raise
                    finally:
                        row.update(
                            manager_decisions=stack.manager_decisions,
                            expert_action_counts=list(stack.expert_action_counts),
                            committed=stack.committed,
                            commit_step=stack.commit_episode_step,
                            option_history=copy.deepcopy(stack.option_history),
                        )
                        if env is not None:
                            env.close()
                        rows.append(row)
                        progress(
                            dict(
                                status="failed" if row["error"] else "evaluating",
                                rows=rows,
                                active_case=None,
                                scores=summarize(rows, conditions),
                            )
                        )
        finally:
            for parameter, flag in flags:
                parameter.requires_grad_(flag)
            for module, mode in modes:
                module.training = mode
            after = frozen_state(models)
            if before != after:
                raise RuntimeError(
                    "Frozen confirmation changed policy/Adam tensors or training counters"
                )
    return dict(
        status="complete",
        version=VERSION,
        rows=rows,
        scores=summarize(rows, conditions),
        frozen_before=before,
        frozen_after=after,
        deterministic=True,
        control_hz=50,
        original_deadline_actions=500,
        training_modes_restored=True,
        gradient_flags_restored=True,
        image_transform_scope="Complete observation passed to both manager and every physical expert",
        original_success_scope="Passive record within each centered-task rollout; not a separate original termination rollout",
        new_physical_training_steps=0,
        learn_calls=0,
        train_calls=0,
        rollout_collection_calls=0,
        evaluation_seconds=time.monotonic() - started,
    )


def _resolve_sources(volume_root, manager_source, expert_sources, option_steps):
    if not isinstance(expert_sources, (tuple, list)) or len(expert_sources) != 2:
        raise ValueError("Provide exactly two ordered physical expert sources")
    sources = [manager_source, *expert_sources]
    for source in sources:
        validate_source(source)
    paths, records, metadata = ([], [], [])
    for role, source in zip(("manager", "approach", "pickup"), sources):
        path, gate, meta, meta_path = resolve_skill_checkpoint(
            volume_root, source["run"], source["checkpoint"]
        )
        actual = sha256(path)
        if actual != source["sha256"]:
            raise ValueError(
                f"Frozen {role} source differs from the explicitly selected SHA256"
            )
        paths.append(path)
        metadata.append(meta)
        records.append(
            dict(
                role=role,
                **source,
                metadata_file=meta_path.name,
                metadata_sha256=sha256(meta_path),
                recorded_steps=gate.get("steps", meta.get("final_steps")),
            )
        )
    meta = metadata[0]
    if (
        meta.get("version") != MANAGER_VERSION
        or meta.get("duration", {}).get("option_steps") != option_steps
    ):
        raise ValueError(
            "Selected manager provenance does not match the ordered architecture/duration"
        )
    inherited = meta.get("experts", [])
    if len(inherited) != 2 or any(
        (
            any(
                (
                    old.get(k) != new[v]
                    for k, v in [
                        ("run", "run"),
                        ("checkpoint", "checkpoint"),
                        ("policy_sha256", "sha256"),
                    ]
                )
            )
            for old, new in zip(inherited, expert_sources)
        )
    ):
        raise ValueError(
            "Physical experts differ from those frozen during manager training"
        )
    return (paths, records)


def _verify_complete(report, identity, cases, conditions, records):
    rows = report.get("rows", [])
    expected = [(condition, case) for condition in conditions for case in cases]
    before = report.get("frozen_before")
    valid_frozen = isinstance(before, list) and len(before) == 3
    if valid_frozen:
        valid_frozen = all(
            (
                isinstance(item, dict)
                and set(item) == {"policy_sha256", "optimizer_sha256", "counters"}
                and all(
                    (
                        isinstance(item[key], str) and len(item[key]) == 64
                        for key in ("policy_sha256", "optimizer_sha256")
                    )
                )
                and isinstance(item["counters"], dict)
                and (item["counters"].get("num_timesteps") == record["recorded_steps"])
                for item, record in zip(before, records)
            )
        )
    if (
        report.get("status") != "complete"
        or report.get("identity") != identity
        or len(rows) != len(expected)
        or (report.get("frozen_before") != report.get("frozen_after"))
        or (not valid_frozen)
        or (report.get("source_sha256_after") != [r["sha256"] for r in records])
        or (report.get("scores") != summarize(rows, conditions))
        or any(
            (
                report.get(k) != 0
                for k in (
                    "learn_calls",
                    "train_calls",
                    "rollout_collection_calls",
                    "new_physical_training_steps",
                )
            )
        )
    ):
        raise ValueError(
            "Existing completed confirmation failed frozen integrity verification"
        )
    for row, (condition, case) in zip(rows, expected):
        if (
            row.get("condition") != condition
            or any((row.get(k) != v for k, v in case.items()))
            or row.get("error") is not None
            or (not 1 <= row.get("physical_actions", 0) <= MAX_ACTIONS)
            or (row.get("physical_action_attempts") != row["physical_actions"])
            or (sum(row.get("expert_action_counts", [])) != row["physical_actions"])
        ):
            raise ValueError(
                "Existing completed confirmation has invalid episode provenance or accounting"
            )


def run_ordered_confirmation(
    volume_root,
    out,
    manager_source,
    expert_sources,
    *,
    seed=740000000,
    conditions=("normal",),
    option_steps=5,
    device="cuda",
    commit=lambda: None,
):
    """Remote-path loader; writes JSON only. Partial outputs require inspection."""
    out = Path(out)
    if out.exists() and any(out.iterdir()) and (not (out / "summary.json").exists()):
        raise ValueError(
            "Partial confirmation output exists; inspect its work accounting and use a new name, not silent replay"
        )
    out.mkdir(parents=True, exist_ok=True)
    existing = (out / "summary.json").exists()
    try:
        conditions = validate_protocol(seed, conditions, option_steps)
        paths, records = _resolve_sources(
            volume_root, manager_source, expert_sources, option_steps
        )
        if any((out.resolve() == path.parent.resolve() for path in paths)):
            raise ValueError(
                "Confirmation output must be separate from every policy source"
            )
        project = Path(__file__).resolve().parents[1]
        files = sorted((project / "robovision").glob("*.py")) + [
            project / "assets/cup_arm.xml",
            project / "scripts/ordered_confirmation_modal.py",
        ]
        hashes = {str(path.relative_to(project)): sha256(path) for path in files}
        lock = dict(
            version=VERSION,
            sources=records,
            seed=seed,
            conditions=list(conditions),
            option_steps=option_steps,
            code_sha256=hashes,
            source_selection="Explicit policy hashes fixed before generating or evaluating cases",
        )
        lock_path = out / "source-lock.json"
        if existing:
            if not lock_path.exists() or json.loads(lock_path.read_text()) != lock:
                raise ValueError("Existing confirmation source/protocol lock differs")
        else:
            atomic_json(lock_path, lock)
            commit()
        cases = confirmation_cases(seed)
        specification = dict(
            **lock,
            cases=cases,
            scene_sha256=_digest(cases),
            episode_count=EPISODES,
            nominal_height_m=[0.025, 0.14],
            fixed_cup_xy_m=[0.32, 0.0],
            fixed_nominal_physics_and_appearance=True,
            physical_gamma=PHYSICAL_GAMMA,
            success_threshold=SUCCESS_THRESHOLD,
            seed_scope="Dedicated development block; no claim that external users have never viewed these cases",
        )
        identity = _digest(specification)
        if existing:
            result = json.loads((out / "summary.json").read_text())
            _verify_complete(result, identity, cases, conditions, records)
            return result
        atomic_json(out / "experiment.json", dict(identity=identity, **specification))
        commit()

        def progress(report):
            atomic_json(
                out / "progress.json",
                dict(identity=identity, frozen_integrity_verified=False, **report),
            )
            commit()

        with preserved_rng():
            try:
                manager = PPO.load(paths[0], device=device)
                experts = [load_grasp_policy(path, device=device) for path in paths[1:]]
                for model, record in zip((manager, *experts), records):
                    if (
                        type(record["recorded_steps"]) is not int
                        or record["recorded_steps"] < 0
                        or int(model.num_timesteps) != record["recorded_steps"]
                    ):
                        raise ValueError(
                            "Loaded training counter differs from the selected checkpoint record"
                        )
                result = evaluate_confirmation(
                    manager,
                    experts,
                    seed=seed,
                    cases=cases,
                    conditions=conditions,
                    option_steps=option_steps,
                    progress=progress,
                )
            finally:
                after = [sha256(path) for path in paths]
                if after != [r["sha256"] for r in records]:
                    raise RuntimeError(
                        "Frozen confirmation source checkpoint bytes changed"
                    )
        result.update(
            identity=identity, specification=specification, source_sha256_after=after
        )
        _verify_complete(result, identity, cases, conditions, records)
        atomic_json(out / "summary.json", result)
        commit()
        return result
    except Exception as error:
        if not existing:
            progress_path = out / "progress.json"
            progress = (
                json.loads(progress_path.read_text())
                if progress_path.exists()
                else None
            )
            atomic_json(
                out / "failure.json",
                dict(
                    type=type(error).__name__,
                    message=str(error),
                    traceback=traceback.format_exc(),
                    last_persisted_progress=progress,
                    incomplete_episode_work="If an episode was active, up to500 uncommitted actions may have occurred; do not infer zero from missing progress",
                    new_physical_training_steps=0,
                ),
            )
            commit()
        raise
