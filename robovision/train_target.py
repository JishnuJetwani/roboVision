"""PPO training for the camera-conditioned XY target policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from torch import nn

from robovision.curriculum import ReachCurriculum
from robovision.io import atomic_json, file_hash, stop_after_update
from robovision.models import export_target
from robovision.reach_env import HARD_TOLERANCE, VERSION, PrecisionReachEnv
from robovision.vision import CupMomentExtractor


def parameter_hash(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode() + value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def checkpoint(model, run_dir, metadata):
    root = run_dir / "checkpoints"
    root.mkdir(exist_ok=True)
    name = f"step_{model.num_timesteps:09d}_{time.time_ns()}"
    temporary = root / (name + ".tmp")
    temporary.mkdir()
    model.save(temporary / "policy.zip")
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (numpy_state[0], torch.from_numpy(numpy_state[1].copy()), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "environments": [],
    }
    if model.device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    if model.device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    for wrapped in model.get_env().envs:
        env = wrapped.unwrapped
        state["environments"].append({"scenario_seed": env.scenario_seed,
                                      "task_rng": env.task_rng.bit_generator.state,
                                      "tolerance": env.tolerance})
    torch.save(state, temporary / "state.pt")
    record = {**metadata, "ppo_steps": model.num_timesteps,
              "ppo_epochs": model._n_updates,
              "policy_sha256": file_hash(temporary / "policy.zip"),
              "state_sha256": file_hash(temporary / "state.pt")}
    atomic_json(temporary / "metadata.json", record)
    os.replace(temporary, root / name)
    atomic_json(run_dir / "latest.json", {"checkpoint": f"checkpoints/{name}"})
    atomic_json(run_dir / "metadata.json", record)
    return record


def restore(model, checkpoint_path):
    state = torch.load(checkpoint_path / "state.pt", map_location="cpu", weights_only=True)
    observations = []
    for wrapped, saved in zip(model.get_env().envs, state["environments"], strict=True):
        env = wrapped.unwrapped
        env.set_tolerance(saved["tolerance"])
        observation, _ = wrapped.reset(seed=saved["scenario_seed"])
        env.task_rng.bit_generator.state = saved["task_rng"]
        observations.append(observation)
    model._last_obs = {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}
    model._last_episode_starts = np.ones(len(observations), dtype=bool)
    model.get_env()._reset_seeds()
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], numpy_state[1].numpy(), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


class Recorder(BaseCallback):
    def __init__(self, args, curriculum, started, previous_elapsed):
        super().__init__()
        self.args = args
        self.curriculum = curriculum
        self.started = started
        self.previous_elapsed = previous_elapsed
        self.rows = []

    def _on_rollout_start(self):
        fraction = min(1., self.num_timesteps / max(1., .75 * self.args.steps))
        with torch.no_grad():
            self.model.policy.log_std.fill_(-.5 - 2.5 * fraction)
        if self.args.method == "curriculum":
            promotion = self.curriculum.advance(self.num_timesteps)
            if promotion is not None:
                with (self.args.run_dir / "promotions.jsonl").open("a") as handle:
                    handle.write(json.dumps(promotion) + "\n")
                print("Promotion: " + json.dumps(promotion), flush=True)
            tolerance = self.curriculum.tolerance
        else:
            tolerance = HARD_TOLERANCE
        self.training_env.env_method("set_tolerance", tolerance)

    def _on_step(self):
        infos = self.locals["infos"]
        self.rows.extend(infos)
        self.curriculum.observe(info["is_success"] for info in infos)
        return True

    def _on_rollout_end(self):
        row = {"steps": self.num_timesteps,
               "elapsed_seconds": self.previous_elapsed + time.monotonic() - self.started,
               "log_std": self.model.policy.log_std.detach().cpu().tolist(),
               "tolerance": self.rows[-1]["tolerance"],
               "success_rate": float(np.mean([row["is_success"] for row in self.rows])),
               "mean_alignment_error": float(np.mean([row["alignment_error"] for row in self.rows]))}
        self.rows.clear()
        with (self.args.run_dir / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


def train(args):
    rollout_size = args.envs * args.rollout_steps
    if args.steps < 1 or args.steps % rollout_size:
        raise ValueError("Steps must be a positive multiple of envs * rollout_steps")
    if args.batch_size > rollout_size or rollout_size % args.batch_size:
        raise ValueError("Batch size must divide the rollout size")
    if args.max_seconds <= 0 or args.checkpoint_seconds <= 0:
        raise ValueError("Time limits must be positive")
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume:
        raise FileExistsError("Use a new run directory or --resume")
    if args.resume and not (args.run_dir / "latest.json").exists():
        raise FileNotFoundError("No resumable checkpoint")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    tolerance = .06 if args.method == "curriculum" else HARD_TOLERANCE
    env = DummyVecEnv([lambda i=i: Monitor(PrecisionReachEnv(
        seed=args.seed * 100000 + i, tolerance=tolerance)) for i in range(args.envs)])
    try:
        model = PPO("MultiInputPolicy", env, device=args.device, seed=args.seed,
                    learning_rate=args.learning_rate, n_steps=args.rollout_steps,
                    batch_size=args.batch_size, n_epochs=args.epochs, gamma=0., gae_lambda=1.,
                    ent_coef=0., clip_range=.2, target_kl=.04,
                    policy_kwargs={"features_extractor_class": CupMomentExtractor,
                                   "net_arch": {"pi": [128, 64], "vf": [128, 64]},
                                   "activation_fn": nn.ReLU, "log_std_init": -.5}, verbose=0)
        config = {key: value for key, value in vars(args).items()
                  if key not in ("resume", "max_seconds", "checkpoint_seconds", "run_dir")}
        root = Path(__file__).resolve().parent.parent
        files = ["train_target.py", "reach_env.py", "curriculum.py", "env.py", "vision.py", "models.py", "io.py"]
        sources = [root / "robovision" / name for name in files] + [root / "assets/cup_arm.xml"]
        source = {str(path.relative_to(root)): file_hash(path) for path in sources}
        metadata = {"task_version": VERSION, "config": config, "source_sha256": source,
                    "initial_weights_sha256": parameter_hash(model.policy), "ppo_steps": 0,
                    "elapsed_seconds": 0., "complete": False}
        if args.resume:
            pointer = json.loads((args.run_dir / "latest.json").read_text())
            saved = args.run_dir / pointer["checkpoint"]
            metadata = json.loads((saved / "metadata.json").read_text())
            if metadata["config"] != config or metadata["source_sha256"] != source:
                raise ValueError("Resume requires the same configuration and source files")
            for name in ("policy", "state"):
                extension = ".zip" if name == "policy" else ".pt"
                if file_hash(saved / (name + extension)) != metadata[name + "_sha256"]:
                    raise ValueError("Checkpoint hash mismatch")
            model = PPO.load(saved / "policy.zip", env=env, device=args.device)
            restore(model, saved)
        else:
            model._last_obs = env.reset()
            model._last_episode_starts = np.ones(args.envs, dtype=bool)
        model.policy.log_std.requires_grad_(False)
        curriculum = ReachCurriculum(metadata.get("curriculum_state"))
        previous_elapsed = metadata["elapsed_seconds"]
        callback = Recorder(args, curriculum, started, previous_elapsed)

        def save():
            metadata.update(curriculum_state=curriculum.state_dict(),
                            elapsed_seconds=previous_elapsed + time.monotonic() - started,
                            complete=model.num_timesteps >= args.steps)
            checkpoint(model, args.run_dir, metadata)

        if not args.resume:
            save()
        last_save = time.monotonic()
        with stop_after_update() as stopped:
            while model.num_timesteps < args.steps:
                if stopped[0] or time.monotonic() - started >= args.max_seconds:
                    break
                # Each call completes one rollout and its optimizer updates.
                model.learn(total_timesteps=rollout_size, callback=callback, reset_num_timesteps=False)
                if time.monotonic() - last_save >= args.checkpoint_seconds:
                    save()
                    last_save = time.monotonic()
            save()
        export_target(model, args.run_dir / "target.pt")
        if model.num_timesteps < args.steps:
            print("Stopped after a complete PPO update. Continue with --resume.", flush=True)
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["hard", "curriculum"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=1200.)
    parser.add_argument("--checkpoint-seconds", type=float, default=120.)
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args(argv))


if __name__ == "__main__":
    main()
