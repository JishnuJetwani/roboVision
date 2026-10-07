"""Final paired 200-scene evaluation on the centered final-height task.

Call once seeds have been reserved outside training/model selection. The module
records supplied provenance but cannot prove that seeds were untouched. Frozen
images preserve the reset view; black images remove all camera information.
Neither ablation changes proprioception, physical state, or success criteria.
"""

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from .joint_env import JointGraspEnv

VERSION = "final-height-200-v1"
ENV_CONFIG = dict(
    stage=3, observation="pixels", render_images=True, gamma=0.995, max_steps=500
)


def wilson_interval(successes, episodes, z=1.959963984540054):
    if episodes < 0 or successes < 0 or successes > episodes:
        raise ValueError("Invalid success counts")
    if episodes == 0:
        return [None, None]
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes * episodes))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _counts(rows):
    n = len(rows)
    successes = sum((bool(row["success"]) for row in rows))
    return dict(
        episodes=n,
        successes=successes,
        success_rate=successes / n if n else None,
        wilson95=wilson_interval(successes, n),
    )


def summarize(rows):
    result = _counts(rows)
    result.update(
        failures=dict(Counter((row["reason"] for row in rows if not row["success"]))),
        any_contact=sum((row["any_contact"] for row in rows)),
        bilateral_contact=sum((row["bilateral_contact"] for row in rows)),
        lifted_6cm=sum((row["peak_clearance"] >= 0.06 for row in rows)),
        lifted_6cm_with_bilateral_contact=sum((row["gripped_lift"] for row in rows)),
    )
    boundaries = dict(spawn_x=0.32, spawn_y=0.0, cup_mass=0.085, grip_friction=0.95)
    result["strata"] = {
        key: {
            "split": split,
            "below": _counts([row for row in rows if row["scene"][key] < split]),
            "at_or_above": _counts([row for row in rows if row["scene"][key] >= split]),
        }
        for key, split in boundaries.items()
    }
    return result


@contextmanager
def preserved_rng():
    numpy_state, python_state = (np.random.get_state(), random.getstate())
    try:
        with torch.random.fork_rng():
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _observation(obs, frozen, condition):
    if set(obs) != {"image", "proprio"}:
        raise ValueError("Benchmark actor must receive only image and proprio")
    image = (
        obs["image"]
        if condition == "normal"
        else frozen
        if condition == "frozen"
        else np.zeros_like(obs["image"])
    )
    return {"image": image.copy(), "proprio": obs["proprio"].copy()}


def evaluate_hard_grasper(
    policy,
    seeds,
    *,
    stochastic=False,
    conditions=("normal", "frozen", "black"),
    policy_sha256=None,
    training_config_sha256=None,
    seed_provenance=None,
):
    """Evaluate PPO-like .predict() objects without training or changing external RNG.

    Exactly 200 distinct nonnegative integer seeds are required. Caller supplies
    policy/config hashes and evidence of seed reservation. No curriculum env,
    easier reset, alternative reward, deadline override, or state observation is
    accepted. Stateful policies exposing reset() are reset at every physical
    episode boundary. Report confidence intervals rather than task completion.
    """
    seeds = list(seeds)
    if (
        len(seeds) != 200
        or any(
            (
                isinstance(s, (bool, np.bool_))
                or not isinstance(s, (int, np.integer))
                or s < 0
                or (s >= 2**32)
                for s in seeds
            )
        )
        or len(set(seeds)) != 200
    ):
        raise ValueError("Provide exactly 200 distinct integer seeds in [0, 2**32)")
    seeds = [int(s) for s in seeds]
    conditions = tuple(conditions)
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or any((c not in ("normal", "frozen", "black") for c in conditions))
    ):
        raise ValueError("Invalid or repeated image conditions")
    model_policy = getattr(policy, "policy", policy)
    training_mode = getattr(model_policy, "training", None)
    source_path = Path(__file__).with_name("joint_env.py")
    result = dict(
        version=VERSION,
        environment=dict(ENV_CONFIG),
        seeds=seeds,
        seed_provenance=seed_provenance,
        seed_reservation_verified_by_evaluator=False,
        policy_sha256=policy_sha256,
        training_config_sha256=training_config_sha256,
        hard_environment_source_sha256=hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        environment_config_sha256=hashlib.sha256(
            json.dumps(ENV_CONFIG, sort_keys=True).encode()
        ).hexdigest(),
        evaluations={},
    )
    env = None
    try:
        with preserved_rng():
            env = JointGraspEnv(**ENV_CONFIG)
            for deterministic in [True, False] if stochastic else [True]:
                for condition in conditions:
                    rows = []
                    for seed in seeds:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        random.seed(seed)
                        obs, info = env.reset(seed=seed)
                        policy_reset = getattr(policy, "reset", None)
                        if callable(policy_reset):
                            policy_reset()
                        frozen = obs["image"].copy()
                        scene = {k: float(v) for k, v in env.params.items()}
                        any_contact = bilateral = gripped_lift = False
                        peak = float(info.get("clearance", 0.0))
                        total_reward = 0.0
                        for step in range(ENV_CONFIG["max_steps"]):
                            action, _ = policy.predict(
                                _observation(obs, frozen, condition),
                                deterministic=deterministic,
                            )
                            obs, reward, done, truncated, info = env.step(action)
                            contacts = info["contacts"]
                            any_contact |= any(contacts)
                            bilateral |= all(contacts)
                            clearance = float(info["clearance"])
                            peak = max(peak, clearance)
                            gripped_lift |= all(contacts) and clearance >= 0.06
                            total_reward += float(reward)
                            if done or truncated:
                                break
                        else:
                            raise RuntimeError(
                                "Final-height environment did not terminate at its deadline"
                            )
                        rows.append(
                            dict(
                                seed=seed,
                                success=bool(info["is_success"]),
                                reason=info["reason"],
                                steps=step + 1,
                                reward=total_reward,
                                scene=scene,
                                any_contact=bool(any_contact),
                                bilateral_contact=bool(bilateral),
                                gripped_lift=bool(gripped_lift),
                                peak_clearance=peak,
                            )
                        )
                    name = f"{condition}-" + (
                        "deterministic" if deterministic else "stochastic"
                    )
                    result["evaluations"][name] = {**summarize(rows), "rows": rows}
    finally:
        if env is not None:
            env.close()
        if training_mode is not None:
            model_policy.train(training_mode)
    return result
