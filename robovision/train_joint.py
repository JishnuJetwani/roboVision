"""Train direct-torque PPO with a CNN camera policy or privileged-state diagnostic."""

from __future__ import annotations
import argparse
from collections import deque
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from .cnn import GraspCNN, NormalizedGraspCNN
from .io import atomic_json, file_hash, stop_after_update
from .joint_env import JointGraspEnv, VERSION
from .grasp_reward import GraspReward
from .reverse_curriculum import ReverseGraspEnv, ReverseGraspCurriculum, FINAL_LEVEL


class GraspCurriculum:
    def __init__(self, stage=0, state=None):
        state = state or {}
        self.stage = state.get("stage", stage)
        self.recent = deque(state.get("recent", []), maxlen=100)
        self.episodes = state.get("episodes", 0)
        self.promotions = state.get("promotions", [])

    def observe(self, info, steps):
        if info["stage"] != self.stage:
            return False
        self.episodes += 1
        self.recent.append(bool(info["is_success"]))
        if (
            self.stage < 3
            and self.episodes >= 256
            and (len(self.recent) == 100)
            and (np.mean(self.recent) >= 0.7)
        ):
            self.stage += 1
            self.promotions.append({"steps": steps, "stage": self.stage})
            self.episodes = 0
            self.recent.clear()
            return True
        return False

    def state_dict(self):
        return dict(
            stage=self.stage,
            recent=list(self.recent),
            episodes=self.episodes,
            promotions=self.promotions,
        )


class Recorder(BaseCallback):
    def __init__(self, args, curriculum):
        super().__init__()
        self.args, self.curriculum = (args, curriculum)
        self.rows = []
        self.started = time.monotonic()

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if not done:
                continue
            self.rows.append(
                dict(
                    steps=self.num_timesteps,
                    stage=info["stage"],
                    success=bool(info["is_success"]),
                    reason=info["reason"],
                    reward=info["episode"]["r"],
                    length=info["episode"]["l"],
                    reward_components=info["episode_reward_components"],
                )
            )
            if isinstance(self.curriculum, ReverseGraspCurriculum):
                self.rows[-1].update(
                    {
                        key: info[key]
                        for key in (
                            "curriculum_kind",
                            "curriculum_level",
                            "curriculum_frontier",
                            "curriculum_replay",
                            "curriculum_name",
                            "hold_target_steps",
                            "max_stable_hold_steps",
                            "bilateral_contact_steps",
                            "first_contact_loss_step",
                        )
                    }
                )
            if self.args.curriculum and self.curriculum.observe(
                info, self.num_timesteps
            ):
                setter = (
                    "set_curriculum_level"
                    if isinstance(self.curriculum, ReverseGraspCurriculum)
                    else "set_stage"
                )
                self.training_env.env_method(setter, self.curriculum.stage)
                label = (
                    "level"
                    if isinstance(self.curriculum, ReverseGraspCurriculum)
                    else "stage"
                )
                print(
                    f"Curriculum advanced to {label} {self.curriculum.stage}",
                    flush=True,
                )
        return True

    def _on_rollout_end(self):
        if self.rows:
            with (self.args.run_dir / "episodes.jsonl").open("a") as handle:
                for row in self.rows:
                    handle.write(json.dumps(row) + "\n")
        row = dict(
            steps=self.num_timesteps,
            stage=self.curriculum.stage,
            completed_episodes=len(self.rows),
            success_rate=float(np.mean([r["success"] for r in self.rows]))
            if self.rows
            else None,
            elapsed_seconds=time.monotonic() - self.started,
            action_std_normalized=self.model.policy.log_std.detach()
            .exp()
            .cpu()
            .tolist(),
        )
        if isinstance(self.curriculum, ReverseGraspCurriculum):
            row.update(
                stage=None,
                curriculum_kind=self.curriculum.state_dict()["kind"],
                curriculum_level=self.curriculum.level,
                frontier_episodes=self.curriculum.episodes,
                frontier_success_rate=float(np.mean(self.curriculum.recent))
                if self.curriculum.recent
                else None,
                replay_episodes=sum((r["curriculum_replay"] for r in self.rows)),
            )
        with (self.args.run_dir / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        self.rows.clear()


def save_checkpoint(model, args, curriculum, metadata):
    root = args.run_dir / "checkpoints"
    root.mkdir(exist_ok=True)
    name = f"step_{model.num_timesteps:09d}_{time.time_ns()}"
    temporary = root / (name + ".tmp")
    temporary.mkdir()
    model.save(temporary / "policy.zip")
    state = dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        env_rng=model.get_env().env_method("get_rng_state"),
    )
    if model.device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    if model.device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    torch.save(state, temporary / "rng.pt")
    record = {
        **metadata,
        "steps": model.num_timesteps,
        "curriculum": curriculum.state_dict(),
        "policy_sha256": file_hash(temporary / "policy.zip"),
        "rng_sha256": file_hash(temporary / "rng.pt"),
        "resume_semantics": "Restore model, optimizer, curriculum and RNG; start fresh episodes.",
    }
    atomic_json(temporary / "metadata.json", record)
    os.replace(temporary, root / name)
    atomic_json(args.run_dir / "latest.json", {"checkpoint": f"checkpoints/{name}"})
    atomic_json(args.run_dir / "metadata.json", record)
    return root / name


def anneal_hold_exploration(model, curriculum, metadata, scale):
    """One-time noise reduction after the first milestone, between PPO updates.

    Leaves actor means and torque limits untouched. Persisting the event prevents
    a resumed run from applying the scale again. PPO still learns log_std later.
    """
    if (
        scale == 1.0
        or not isinstance(curriculum, ReverseGraspCurriculum)
        or curriculum.level < 1
        or (metadata.get("hold_exploration_event") is not None)
    ):
        return False
    before = model.policy.log_std.detach().exp().cpu().numpy()
    with torch.no_grad():
        model.policy.log_std.add_(float(np.log(scale)))
    after = model.policy.log_std.detach().exp().cpu().numpy()
    metadata["hold_exploration_event"] = dict(
        steps=model.num_timesteps,
        curriculum_level=curriculum.level,
        scale=scale,
        std_normalized_before=before.tolist(),
        std_normalized_after=after.tolist(),
    )
    print(
        "Reduced hold exploration: " + json.dumps(metadata["hold_exploration_event"]),
        flush=True,
    )
    return True


def train(args, on_checkpoint=None):
    args.vec_env = getattr(args, "vec_env", "dummy")
    args.curriculum_kind = getattr(args, "curriculum_kind", "reverse")
    args.curriculum_level = getattr(args, "curriculum_level", 0)
    args.curriculum_replay = getattr(args, "curriculum_replay", 0.2)
    args.curriculum_min_episodes = getattr(args, "curriculum_min_episodes", 256)
    args.curriculum_window = getattr(args, "curriculum_window", 100)
    args.curriculum_threshold = getattr(args, "curriculum_threshold", 0.8)
    if args.curriculum_kind not in ("reverse", "legacy"):
        raise ValueError("Curriculum kind must be reverse or legacy")
    if any(
        getattr(args, field, None)
        for field in ("initial_model", "demo_anchor", "reference_model")
    ):
        raise ValueError(
            "This trainer learns PPO from scratch; teacher and reference initialization are unsupported"
        )
    reverse = args.curriculum and args.curriculum_kind == "reverse"
    if reverse and args.stage != 0:
        raise ValueError(
            "Reverse curriculum uses --curriculum-level, not --stage; use --no-curriculum for a fixed hard run"
        )
    args.exploration_std = list(
        getattr(args, "exploration_std", [1.0, 2.0, 1.5, 0.6, 1.0])
    )
    args.hold_exploration_scale = getattr(args, "hold_exploration_scale", 1.0)
    if (
        not np.isfinite(args.hold_exploration_scale)
        or not 0 < args.hold_exploration_scale <= 1
    ):
        raise ValueError("Hold exploration scale must be in (0, 1]")
    if args.hold_exploration_scale != 1.0 and (not reverse):
        raise ValueError("Hold exploration schedule requires the reverse curriculum")
    args.ent_coef = getattr(args, "ent_coef", 0.0)
    args.normalized_features = getattr(args, "normalized_features", True)
    if (
        len(args.exploration_std) != 5
        or not np.all(np.isfinite(args.exploration_std))
        or min(args.exploration_std) <= 0
    ):
        raise ValueError(
            "Five finite positive physical exploration standard deviations required"
        )
    if not np.isfinite(args.ent_coef) or args.ent_coef < 0:
        raise ValueError("Entropy coefficient must be finite and nonnegative")
    if args.vec_env not in ("dummy", "subproc"):
        raise ValueError("vec_env must be dummy or subproc")
    rollout = args.envs * args.rollout_steps
    if min(args.envs, args.rollout_steps, args.epochs, args.threads) < 1 or rollout < 2:
        raise ValueError(
            "Positive environment/rollout/epoch/thread counts and at least two rollout samples required"
        )
    if (
        args.steps < 1
        or args.steps % rollout
        or args.batch_size < 2
        or rollout % args.batch_size
    ):
        raise ValueError(
            "Steps must be divisible by rollout size; batch size >= 2 must divide rollout size"
        )
    if (
        args.max_seconds <= 0
        or args.checkpoint_seconds <= 0
        or args.max_episode_steps < 1
    ):
        raise ValueError("Time limits and episode length must be positive")
    if not 0 < args.gamma < 1:
        raise ValueError("Full-episode training requires 0 < gamma < 1")
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and (not args.resume):
        raise FileExistsError("Choose an empty run directory or use --resume")
    config = {
        k: v
        for k, v in vars(args).items()
        if k not in ("run_dir", "resume", "steps", "max_seconds", "checkpoint_seconds")
    }
    root = Path(__file__).resolve().parents[1]
    sources = {
        name: file_hash(root / name)
        for name in (
            "robovision/joint_env.py",
            "robovision/env.py",
            "robovision/cnn.py",
            "robovision/train_joint.py",
            "robovision/reverse_curriculum.py",
            "robovision/grasp_reward.py",
            "assets/cup_arm.xml",
        )
    }
    metadata = dict(
        version=VERSION,
        control=JointGraspEnv.control_spec,
        reward=GraspReward().specification(),
        config=config,
        source_sha256=sources,
    )
    curriculum = (
        ReverseGraspCurriculum(
            args.curriculum_level,
            minimum_episodes=args.curriculum_min_episodes,
            window=args.curriculum_window,
            threshold=args.curriculum_threshold,
        )
        if reverse
        else GraspCurriculum(args.stage)
    )
    if reverse:
        metadata["reset_curriculum"] = ReverseGraspCurriculum.specification()
    saved = None
    if args.resume:
        pointer = json.loads((args.run_dir / "latest.json").read_text())
        saved = args.run_dir / pointer["checkpoint"]
        previous = json.loads((saved / "metadata.json").read_text())
        if previous.get("version") != VERSION:
            raise ValueError(
                "Incompatible environment/reward version: start a new training run"
            )
        if previous["config"] != config or previous["source_sha256"] != sources:
            raise ValueError(
                "Resume requires matching configuration and source; only budget/time limits may change"
            )
        for name in ("policy", "rng"):
            path = saved / ("policy.zip" if name == "policy" else "rng.pt")
            if file_hash(path) != previous[f"{name}_sha256"]:
                raise ValueError(f"{name} checkpoint hash mismatch")
        metadata["hold_exploration_event"] = previous.get("hold_exploration_event")
        curriculum = (
            ReverseGraspCurriculum(state=previous["curriculum"])
            if reverse
            else GraspCurriculum(state=previous["curriculum"])
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    vector = DummyVecEnv if args.vec_env == "dummy" else SubprocVecEnv
    vector_kwargs = {} if args.vec_env == "dummy" else {"start_method": "spawn"}
    env_class = ReverseGraspEnv if reverse else JointGraspEnv
    reset_options = (
        dict(curriculum_level=curriculum.level, replay_fraction=args.curriculum_replay)
        if reverse
        else dict(stage=curriculum.stage)
    )
    env = vector(
        [
            lambda i=i: Monitor(
                env_class(
                    seed=args.seed + i,
                    observation=args.observation,
                    gamma=args.gamma,
                    max_steps=args.max_episode_steps,
                    render_images=args.observation == "pixels",
                    **reset_options,
                )
            )
            for i in range(args.envs)
        ],
        **vector_kwargs,
    )
    started = time.monotonic()
    try:
        if saved:
            model = PPO.load(saved / "policy.zip", env=env, device=args.device)
            state = torch.load(saved / "rng.pt", map_location="cpu", weights_only=False)
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
            if "cuda" in state:
                torch.cuda.set_rng_state_all(state["cuda"])
            if "mps" in state:
                torch.mps.set_rng_state(state["mps"])
            if len(state["env_rng"]) != env.num_envs:
                raise ValueError("Checkpoint environment count mismatch")
            for index, rng in enumerate(state["env_rng"]):
                env.env_method("set_rng_state", rng, indices=index)
        else:
            policy_kwargs = dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]), log_std_init=-1.0
            )
            if args.observation == "pixels":
                policy_kwargs["features_extractor_class"] = (
                    NormalizedGraspCNN if args.normalized_features else GraspCNN
                )
                policy_kwargs["share_features_extractor"] = not args.normalized_features
            model = PPO(
                "MultiInputPolicy",
                env,
                seed=args.seed,
                device=args.device,
                n_steps=args.rollout_steps,
                batch_size=args.batch_size,
                n_epochs=args.epochs,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                gae_lambda=0.95,
                ent_coef=args.ent_coef,
                target_kl=0.03,
                policy_kwargs=policy_kwargs,
                verbose=0,
            )
            limits = np.r_[
                JointGraspEnv.torque_limits, JointGraspEnv.finger_force_limit
            ]
            with torch.no_grad():
                model.policy.log_std.copy_(
                    torch.as_tensor(
                        np.log(np.asarray(args.exploration_std) / limits),
                        dtype=model.policy.log_std.dtype,
                        device=model.device,
                    )
                )
        callback = Recorder(args, curriculum)
        last_save = time.monotonic()

        def save():
            metadata["elapsed_seconds_this_run"] = time.monotonic() - started
            path = save_checkpoint(model, args, curriculum, metadata)
            if on_checkpoint is not None:
                on_checkpoint(path)
            return path

        save()
        with stop_after_update() as stopped:
            while model.num_timesteps < args.steps:
                if stopped[0] or time.monotonic() - started >= args.max_seconds:
                    break
                model.learn(
                    total_timesteps=rollout,
                    reset_num_timesteps=False,
                    callback=callback,
                )
                anneal_hold_exploration(
                    model, curriculum, metadata, args.hold_exploration_scale
                )
                metrics = {
                    key: float(value)
                    for key, value in model.logger.name_to_value.items()
                    if key.startswith("train/") and np.isscalar(value)
                }
                with (args.run_dir / "optimizer.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(dict(steps=model.num_timesteps, **metrics)) + "\n"
                    )
                if time.monotonic() - last_save >= args.checkpoint_seconds:
                    save()
                    last_save = time.monotonic()
        path = save()
        print(
            f"Saved {path / 'policy.zip'} ({model.num_timesteps}/{args.steps} interactions)",
            flush=True,
        )
        return path
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--observation", choices=["pixels", "state"], default="pixels")
    parser.add_argument("--steps", type=int, default=1024000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument(
        "--exploration-std",
        type=float,
        nargs=5,
        default=[1.0, 2.0, 1.5, 0.6, 1.0],
        help="Initial noise SD: four joint torques in Nm, then finger force in N",
    )
    parser.add_argument(
        "--hold-exploration-scale",
        type=float,
        default=1.0,
        help="Scale learned noise once after first reverse-curriculum promotion; 1 disables",
    )
    parser.add_argument(
        "--normalized-features", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--stage", type=int, choices=range(4), default=0)
    parser.add_argument(
        "--curriculum", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--curriculum-kind", choices=["reverse", "legacy"], default="reverse"
    )
    parser.add_argument(
        "--curriculum-level", type=int, choices=range(FINAL_LEVEL + 1), default=0
    )
    parser.add_argument(
        "--curriculum-replay",
        type=float,
        default=0.2,
        help="Fraction of resets rehearsing earlier levels",
    )
    parser.add_argument("--curriculum-min-episodes", type=int, default=256)
    parser.add_argument("--curriculum-window", type=int, default=100)
    parser.add_argument("--curriculum-threshold", type=float, default=0.8)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=1200.0)
    parser.add_argument("--checkpoint-seconds", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args(argv))


if __name__ == "__main__":
    main()
