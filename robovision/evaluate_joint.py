"""Evaluate or record a full-episode PPO policy, including camera ablations."""

import argparse
from collections import Counter
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
import torch
from .grasp_benchmark import wilson_interval
from .io import atomic_json, file_hash
from .joint_env import JointGraspEnv, VERSION
from .grasp_reward import GraspReward


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        if (path / "latest.json").exists():
            path = path / json.loads((path / "latest.json").read_text())["checkpoint"]
        path = path / "policy.zip"
    return path


def evaluate(args):
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    path = resolve_checkpoint(args.model)
    metadata = json.loads(path.with_name("metadata.json").read_text())
    if metadata.get("version") != VERSION:
        raise ValueError(
            "Incompatible environment/reward version: this checkpoint requires its original environment"
        )
    if file_hash(path) != metadata["policy_sha256"]:
        raise ValueError("Policy checkpoint hash mismatch")
    mode = metadata["config"]["observation"]
    if mode == "state" and args.ablation != "normal":
        raise ValueError("Camera ablations require a pixel policy")
    torch.set_num_threads(2)
    model = PPO.load(path, device=args.device)
    env = JointGraspEnv(
        stage=args.stage,
        observation=mode,
        gamma=metadata["config"]["gamma"],
        render_images=mode == "pixels",
        max_steps=metadata["config"]["max_episode_steps"],
    )
    writer = None
    rows = []
    try:
        if args.video:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video, fps=env.metadata["render_fps"])
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            obs, _ = env.reset(seed=seed)
            frozen = obs.get("image", np.empty(0)).copy()
            total_reward = 0.0
            for _ in range(env.max_steps):
                if args.ablation == "black":
                    obs["image"] = np.zeros_like(obs["image"])
                elif args.ablation == "frozen":
                    obs["image"] = frozen.copy()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if writer:
                    frame = Image.fromarray(env.render())
                    if mode == "pixels":
                        camera = obs["image"][-3:].transpose(1, 2, 0)
                        if args.ablation == "black":
                            camera = np.zeros_like(camera)
                        elif args.ablation == "frozen":
                            camera = frozen[-3:].transpose(1, 2, 0)
                        frame.paste(
                            Image.fromarray(camera).resize((192, 192)), (432, 16)
                        )
                    ImageDraw.Draw(frame).text(
                        (12, 12),
                        f"Torque PPO | {mode} | seed {seed} | step {info['step']}\n{args.ablation} | {info['reason']}",
                        fill="white",
                        stroke_width=1,
                        stroke_fill="black",
                    )
                    writer.append_data(np.asarray(frame))
                if terminated or truncated:
                    break
            rows.append(
                dict(
                    seed=seed,
                    success=bool(info["is_success"]),
                    reason=info["reason"],
                    steps=info["step"],
                    duration_seconds=info["step"] * env.control_dt,
                    peak_clearance=info["peak_clearance"],
                    reward=total_reward,
                    reward_components=info["episode_reward_components"],
                )
            )
    finally:
        env.close()
        if writer:
            writer.close()
    successes = sum((row["success"] for row in rows))
    result = dict(
        env_version=VERSION,
        reward=GraspReward().specification(),
        control=JointGraspEnv.control_spec,
        policy_sha256=file_hash(path),
        observation=mode,
        stage=args.stage,
        ablation=args.ablation,
        episodes=len(rows),
        successes=successes,
        success_rate=successes / len(rows),
        wilson_95=wilson_interval(successes, len(rows)),
        failures=dict(Counter((row["reason"] for row in rows if not row["success"]))),
        rows=rows,
    )
    atomic_json(args.out, result)
    print(f"{successes}/{len(rows)} successful grasps; saved {args.out}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Run directory, checkpoint directory, or policy.zip",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=91000)
    parser.add_argument("--stage", type=int, choices=range(4), default=2)
    parser.add_argument(
        "--ablation", choices=["normal", "black", "frozen"], default="normal"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("runs/joint-evaluation.json"))
    parser.add_argument("--video", type=Path)
    evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
