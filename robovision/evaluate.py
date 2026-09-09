"""Evaluate target and grasp policies on fixed simulation seeds."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from .control import action_to_target, execute_approach, run_neural_grasp
from .env import VisionCupEnv
from .io import atomic_json, file_hash
from .models import load_target, load_grasp

ROOT = Path(__file__).resolve().parents[1]


def wilson_interval(successes, episodes):
    z = 1.959963984540054
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    radius = z * np.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes**2)) / denominator
    return [float(center - radius), float(center + radius)]


def evaluate(target_path, grasp_path, seeds, *, ablation="normal"):
    if ablation not in ("normal", "black", "zero-grasp"):
        raise ValueError("Unknown ablation")
    torch.set_num_threads(2)
    target_policy = load_target(target_path)
    grasp_policy = load_grasp(grasp_path)
    if ablation == "zero-grasp":
        with torch.no_grad():
            for parameter in grasp_policy.parameters():
                parameter.zero_()
    env = VisionCupEnv(max_steps=160)
    rows = []
    started = time.monotonic()
    try:
        for seed in seeds:
            env.render_images = True
            obs, _ = env.reset(seed=int(seed))
            if ablation == "black":
                obs["image"][:] = 0
            action, _ = target_policy.predict(obs)
            target = action_to_target(action)
            execute_approach(env, target)
            env.render_images = False
            alignment = float(np.linalg.norm(env.grasp_position[:2] - env.cup_position[:2]))
            info = run_neural_grasp(env, target, grasp_policy)
            info.pop("trace")
            rows.append({
                "seed": int(seed), "target_xy": target.tolist(),
                "initial_alignment_error": alignment,
                "duration_seconds": .8 + info["neural_steps"] * env.control_dt,
                **info,
            })
    finally:
        env.close()
    if not rows:
        raise ValueError("Evaluation needs at least one episode")
    successes = sum(row["is_success"] for row in rows)
    return {
        "target_sha256": file_hash(target_path), "grasp_sha256": file_hash(grasp_path),
        "ablation": ablation, "episodes": len(rows), "successes": successes,
        "success_rate": successes / len(rows),
        "wilson_95": wilson_interval(successes, len(rows)),
        "mean_duration_seconds": float(np.mean([row["duration_seconds"] for row in rows])),
        "failures": dict(Counter(row["reason"] for row in rows if not row["is_success"])),
        "wall_seconds": time.monotonic() - started, "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=ROOT / "models/target_seed101_curriculum.pt")
    parser.add_argument("--grasp", type=Path, default=ROOT / "models/grasp.pt")
    parser.add_argument("--seed-start", type=int, default=81000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--ablation", choices=["normal", "black", "zero-grasp"], default="normal")
    parser.add_argument("--out", type=Path, default=Path("runs/evaluation.json"))
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    result = evaluate(args.target, args.grasp, range(args.seed_start, args.seed_start + args.episodes), ablation=args.ablation)
    atomic_json(args.out, result)
    print(f"{result['successes']}/{result['episodes']} successful grasps; saved {args.out}")


if __name__ == "__main__":
    main()
