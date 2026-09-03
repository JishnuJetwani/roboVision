"""Teacher demonstrations and behavior cloning for vertical motion and grip."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from robovision.control import actuator_action, robot_features, run_neural_grasp
from robovision.env import VisionCupEnv
from robovision.io import atomic_json, file_hash, stop_after_update
from robovision.models import FEATURES, GRASP_VERSION, GraspNetwork, load_grasp


class DemonstrationTeacher:
    def __init__(self):
        self.phase = "descend"

    def act(self, env):
        height = env.grasp_position[2]
        if self.phase == "descend" and height <= .302:
            self.phase = "close"
        if self.phase == "close" and all(env._contacts()):
            self.phase = "lift"
        target = .415 if self.phase == "lift" else .30
        vertical = np.clip((target - height) / env.action_delta, -1., 1.)
        jaw = 1. if self.phase == "descend" else -1.
        return np.array([vertical, jaw], dtype=np.float32)


def set_training_start(env, seed, noise_rng):
    env.reset(seed=seed)
    # Object coordinates are used only to create aligned training starts.
    target = env.cup_position[:2] + noise_rng.uniform(-.004, .004, 2)
    env.data.qpos[:4] = env.inverse_kinematics(np.r_[target, .45])
    env.data.qpos[4:6] = noise_rng.uniform(.038, .045)
    env.data.qvel[:6] = 0.
    env.data.ctrl[:] = env.data.qpos[:6]
    mujoco.mj_forward(env.model, env.data)
    mujoco.mj_step(env.model, env.data, nstep=50)
    env.last_action[:] = [0., 0., 0., 1.]
    env._hold_steps = 0
    env._last_potential = env._potential(env._info())
    return target


def collect(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.episodes < 1 or args.max_seconds <= 0:
        raise ValueError("Episode count and time limit must be positive")
    started = time.monotonic()
    env = VisionCupEnv(render_images=False, max_steps=160)
    features, actions, episode_ids, validation = [], [], [], []
    attempts = []
    completed = 0
    try:
        for attempt in range(args.episodes * 3):
            if completed >= args.episodes or time.monotonic() - started >= args.max_seconds:
                break
            seed = args.seed_start + attempt
            rng = np.random.default_rng(np.random.SeedSequence([seed, 559]))
            target = set_training_start(env, seed, rng)
            teacher = DemonstrationTeacher()
            episode_features, episode_actions = [], []
            for step in range(160):
                observation = robot_features(env, target)
                label = teacher.act(env)
                executed = label.copy()
                if rng.random() < .2:
                    executed += rng.normal(0., .15, 2).astype(np.float32)
                episode_features.append(observation)
                episode_actions.append(label)
                _, _, terminated, truncated, info = env.step(actuator_action(env, target, executed))
                if terminated or truncated:
                    break
            attempts.append({"seed": seed, "success": bool(info["is_success"]), "steps": step + 1})
            if info["is_success"]:
                features.extend(episode_features)
                actions.extend(episode_actions)
                episode_ids.extend([seed] * len(episode_features))
                validation.extend([completed % 5 == 4] * len(episode_features))
                completed += 1
            if (attempt + 1) % 50 == 0:
                print(json.dumps({"attempts": attempt + 1, "successful": completed,
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    except KeyboardInterrupt:
        print("Collection interrupted; saving completed episodes.", flush=True)
    finally:
        env.close()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("wb") as handle:
            np.savez_compressed(handle, x=np.asarray(features, dtype=np.float32).reshape(-1, 12),
                                y=np.asarray(actions, dtype=np.float32).reshape(-1, 2),
                                episode_id=np.asarray(episode_ids, dtype=np.int64),
                                validation=np.asarray(validation, dtype=bool))
        atomic_json(args.out.with_suffix(".json"), {
            "version": GRASP_VERSION, "features": FEATURES, "successful_episodes": completed,
            "requested_episodes": args.episodes, "transitions": len(features),
            "elapsed_seconds": time.monotonic() - started, "attempts": attempts,
            "data_sha256": file_hash(args.out), "start_height": .45,
        })
    print(f"Collected {completed} successful trajectories", flush=True)


def load_demonstrations(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in ("x", "y", "episode_id", "validation")}
    count = len(data["x"])
    if (data["x"].shape != (count, 12) or data["y"].shape != (count, 2)
            or data["episode_id"].shape != (count,) or data["validation"].shape != (count,)
            or data["validation"].dtype != np.bool_):
        raise ValueError("Invalid demonstration shapes or split mask")
    if not np.isfinite(data["x"]).all() or not np.isfinite(data["y"]).all():
        raise ValueError("Demonstrations contain non-finite values")
    train_ids = set(data["episode_id"][~data["validation"]])
    validation_ids = set(data["episode_id"][data["validation"]])
    if not train_ids or not validation_ids or train_ids & validation_ids:
        raise ValueError("Training and validation must contain disjoint, nonempty episode sets")
    return data


def train(args):
    if args.updates < 1 or args.batch_size < 1 or args.max_seconds <= 0:
        raise ValueError("Updates, batch size and time limit must be positive")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise FileExistsError("Use a new output directory or --resume")
    if args.resume and not (args.out / "checkpoint.pt").exists():
        raise FileNotFoundError("No resumable grasp checkpoint")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    data = load_demonstrations(args.data)
    torch.manual_seed(args.seed)
    network = GraspNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    train_indices = np.flatnonzero(~data["validation"])
    validation_indices = np.flatnonzero(data["validation"])
    rng = np.random.default_rng(args.seed + 531)
    root = Path(__file__).resolve().parent.parent
    files = [root / "robovision" / name
             for name in ("train_grasp.py", "models.py", "control.py", "env.py", "io.py")]
    files.append(root / "assets/cup_arm.xml")
    config = {"version": GRASP_VERSION, "data_sha256": file_hash(args.data),
              "seed": args.seed, "updates": args.updates, "batch_size": args.batch_size,
              "learning_rate": args.learning_rate,
              "source_sha256": {str(path.relative_to(root)): file_hash(path) for path in files}}
    initial_hash = hashlib.sha256(b"".join(value.detach().numpy().tobytes()
                                         for value in network.state_dict().values())).hexdigest()
    completed = 0
    previous_elapsed = 0.
    if args.resume:
        saved = torch.load(args.out / "checkpoint.pt", map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["features"] != FEATURES:
            raise ValueError("Resume requires the same data, configuration and source files")
        network.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        completed = saved["updates"]
        previous_elapsed = saved["elapsed_seconds"]
        initial_hash = saved["initial_weights_sha256"]
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
    x = torch.as_tensor(data["x"], dtype=torch.float32)
    y = torch.as_tensor(data["y"], dtype=torch.float32)
    started = time.monotonic()
    last_save = started

    def save():
        state = {"version": GRASP_VERSION, "features": FEATURES, "config": config,
                 "weights": network.state_dict(), "optimizer": optimizer.state_dict(),
                 "updates": completed, "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                 "initial_weights_sha256": initial_hash, "numpy_rng": rng.bit_generator.state,
                 "torch_rng": torch.get_rng_state()}
        temporary = args.out / "checkpoint.tmp.pt"
        torch.save(state, temporary)
        temporary.replace(args.out / "checkpoint.pt")
        metadata = {key: value for key, value in state.items()
                    if key not in ("weights", "optimizer", "numpy_rng", "torch_rng")}
        metadata.update(complete=completed == args.updates,
                        model_sha256=file_hash(args.out / "checkpoint.pt"))
        atomic_json(args.out / "metadata.json", metadata)

    with stop_after_update() as stopped:
        for _ in range(completed, args.updates):
            if stopped[0] or time.monotonic() - started >= args.max_seconds:
                break
            indices = rng.choice(train_indices, args.batch_size)
            predicted = network(x[indices])
            loss = torch.nn.functional.mse_loss(predicted, y[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.)
            optimizer.step()
            completed += 1
            if completed % 500 == 0:
                print(json.dumps({"updates": completed, "loss": float(loss.detach()),
                                  "elapsed_seconds": previous_elapsed + time.monotonic() - started}), flush=True)
            if time.monotonic() - last_save >= args.checkpoint_seconds:
                save()
                last_save = time.monotonic()
        save()
    with torch.no_grad():
        validation_mse = float(torch.nn.functional.mse_loss(network(x[validation_indices]), y[validation_indices]))
    metadata = json.loads((args.out / "metadata.json").read_text())
    metadata["validation_mse"] = validation_mse
    atomic_json(args.out / "metadata.json", metadata)


def validate(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    network = load_grasp(args.model)
    env = VisionCupEnv(render_images=False, max_steps=160)
    rows = []
    try:
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 559]))
            target = set_training_start(env, seed, rng)
            info = run_neural_grasp(env, target, network, record=args.record)
            rows.append({"seed": seed, **info})
    finally:
        env.close()
    result = {"version": GRASP_VERSION, "model_sha256": file_hash(args.model),
              "successes": sum(row["is_success"] for row in rows), "episodes": len(rows), "rows": rows}
    atomic_json(args.out, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collection = commands.add_parser("collect")
    collection.add_argument("--out", type=Path, required=True)
    collection.add_argument("--episodes", type=int, default=600)
    collection.add_argument("--seed-start", type=int, default=300000)
    collection.add_argument("--max-seconds", type=float, default=1200.)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--seed", type=int, default=201)
    training.add_argument("--updates", type=int, default=5000)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--max-seconds", type=float, default=1200.)
    training.add_argument("--checkpoint-seconds", type=float, default=120.)
    training.add_argument("--resume", action="store_true")
    validation = commands.add_parser("validate")
    validation.add_argument("--model", type=Path, required=True)
    validation.add_argument("--out", type=Path, required=True)
    validation.add_argument("--seed-start", type=int, default=8300)
    validation.add_argument("--episodes", type=int, default=100)
    validation.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)
    {"collect": collect, "train": train, "validate": validate}[args.command](args)


if __name__ == "__main__":
    main()
