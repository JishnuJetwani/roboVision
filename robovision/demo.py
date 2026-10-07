"""Render a deterministic episode from the registered evaluation scenes."""

import hashlib
import json
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from stable_baselines3 import PPO

from .centered_grasp_env import CenteredGraspEnv
from .ordered_confirmation import confirmation_cases, frozen_state, _resolve_sources
from .ordered_pickup_policy import OrderedPickupPolicy
from .policy_loading import load_grasp_policy


def render_demo(checkpoint_root, output, manifest, *, seed=741000000, device="cpu"):
    """Replay the highest registered start with unchanged policy parameters."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        {key: source[key] for key in ("run", "checkpoint", "sha256")}
        for source in manifest["sources"]
    ]
    paths, records = _resolve_sources(checkpoint_root, sources[0], sources[1:], 5)
    models = [
        PPO.load(paths[0], device=device),
        *[load_grasp_policy(p, device=device) for p in paths[1:]],
    ]
    before = frozen_state(models)
    case = max(confirmation_cases(seed), key=lambda case: case["height"])
    random.seed(case["seed"])
    np.random.seed(case["seed"])
    torch.manual_seed(case["seed"])
    stack = OrderedPickupPolicy(models[0], models[1:], option_steps=5)
    env = CenteredGraspEnv(
        fixed_height=case["height"],
        seed=case["seed"],
        observation="pixels",
        render_images=True,
        gamma=0.999,
    )
    frames = []
    try:
        observation, _ = env.reset(seed=case["seed"])
        initial_cup = env.cup_position.tolist()
        initial_hand = env.grasp_position.tolist()
        frames.append(env.render())
        for _ in range(500):
            action, _ = stack.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
        steps = env.step_count
        success, reason = bool(info["centered_success"]), info["reason"]
    finally:
        env.close()
    assert before == frozen_state(models), "Rendering changed policy state"
    video = output / "demo.mp4"
    with imageio.get_writer(video, fps=50) as writer:
        for frame in frames:
            writer.append_data(frame)
    indices = np.linspace(0, len(frames) - 1, 6, dtype=int).tolist()
    sheet = Image.new("RGB", (960, 536), "white")
    draw = ImageDraw.Draw(sheet)
    for tile, index in enumerate(indices):
        x, y = (tile % 3) * 320, (tile // 3) * 268
        sheet.paste(Image.fromarray(frames[index]).resize((320, 240)), (x, y))
        draw.text(
            (x + 6, y + 244), f"{index / 50:.2f} s | action {index}", fill="black"
        )
    sheet.save(output / "demo.png")
    report = dict(
        case,
        seed_provenance="Registered centered-height evaluation scene",
        selection="Highest start in the 200 registered evaluation scenes",
        initial_cup_position=initial_cup,
        initial_hand_position=initial_hand,
        sources=records,
        centered_success=success,
        reason=reason,
        total_actions=steps,
        physical_duration_seconds=steps / 50,
        new_training_steps=0,
        source_and_policy_hashes_unchanged=True,
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        frame_indices=indices,
    )
    (output / "demo.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
