"""Run bounded Modal torque-PPO experiments with persistent reports and checkpoints."""

import json
from pathlib import Path
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("robovision-torque-training")
volume = modal.Volume.from_name("robovision-training", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libegl1", "libgl1", "libgles2", "libopengl0", "libglib2.0-0")
    .pip_install(
        "mujoco==3.3.7",
        "gymnasium==1.2.1",
        "stable-baselines3==2.7.0",
        "torch==2.8.0",
        "numpy==2.2.6",
        "matplotlib==3.10.7",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "pillow==11.3.0",
    )
    .env(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "PYTHONPATH": "/root/project",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    .add_local_dir(
        ROOT / "robovision", "/root/project/robovision", ignore=["__pycache__/**"]
    )
    .add_local_file(ROOT / "assets/cup_arm.xml", "/root/project/assets/cup_arm.xml")
)


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=8192,
    timeout=600,
    scaledown_window=2,
    volumes={"/runs": volume},
)
def evaluate_snapshot(run_name, checkpoint_name, episodes=30):
    """Independent process: evaluation cannot consume training RNG or its budget."""
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from robovision.reverse_curriculum import ReverseGraspEnv
    from robovision.io import atomic_json, file_hash

    volume.reload()
    root = Path("/runs") / run_name
    checkpoint = root / "checkpoints" / checkpoint_name
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert file_hash(checkpoint / "policy.zip") == metadata["policy_sha256"]
    torch.set_num_threads(2)
    policy = PPO.load(checkpoint / "policy.zip", device="cuda")
    result = dict(
        checkpoint=checkpoint_name,
        steps=metadata["steps"],
        policy_sha256=metadata["policy_sha256"],
        evaluations={},
    )
    for level in sorted({0, 5, metadata["curriculum"]["level"]}):
        for deterministic in (True, False):
            env = ReverseGraspEnv(
                curriculum_level=level,
                replay_fraction=0,
                observation="pixels",
                gamma=metadata["config"]["gamma"],
            )
            rows = []
            try:
                for seed in range(130000, 130000 + episodes):
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    obs, _ = env.reset(seed=seed)
                    while True:
                        action, _ = policy.predict(obs, deterministic=deterministic)
                        obs, _, done, truncated, info = env.step(action)
                        if done or truncated:
                            break
                    rows.append(
                        dict(
                            seed=seed,
                            success=info["is_success"],
                            reason=info["reason"],
                            steps=env.step_count,
                            max_hold=info["max_stable_hold_steps"],
                            contact_steps=info["bilateral_contact_steps"],
                            first_contact_loss=info["first_contact_loss_step"],
                        )
                    )
            finally:
                env.close()
            result["evaluations"][
                f"level{level}-{('deterministic' if deterministic else 'stochastic')}"
            ] = dict(
                successes=sum((r["success"] for r in rows)),
                episodes=len(rows),
                max_hold=max((r["max_hold"] for r in rows)),
                mean_contact_seconds=float(
                    np.mean([r["contact_steps"] for r in rows]) * 0.02
                ),
                mean_initial_contact_seconds=float(
                    np.mean(
                        [
                            r["first_contact_loss"] - 1
                            if r["first_contact_loss"] is not None
                            else r["steps"]
                            for r in rows
                        ]
                    )
                    * 0.02
                ),
                rows=rows,
            )
    out = root / "checkpoint-evaluations"
    out.mkdir(exist_ok=True)
    atomic_json(out / f"{checkpoint_name}.json", result)
    volume.commit()
    print(
        json.dumps(
            {
                **result,
                "evaluations": {
                    k: {a: b for a, b in v.items() if a != "rows"}
                    for k, v in result["evaluations"].items()
                },
            }
        ),
        flush=True,
    )
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=1500,
    scaledown_window=2,
    volumes={"/runs": volume},
)
def run_training(
    run_name, budget=600.0, eval_episodes=50, overrides=None, resume_from=None
):
    from argparse import Namespace
    from collections import Counter
    import zipfile
    from robovision.train_joint import train
    from robovision.evaluate_joint import evaluate
    from robovision.io import atomic_json

    directory = Path("/runs") / run_name
    args = Namespace(
        run_dir=directory,
        observation="pixels",
        steps=1024000,
        envs=4,
        vec_env="subproc",
        rollout_steps=128,
        batch_size=128,
        epochs=5,
        learning_rate=0.0003,
        gamma=0.995,
        seed=301,
        stage=0,
        curriculum=True,
        max_episode_steps=500,
        device="cuda",
        threads=2,
        max_seconds=budget,
        checkpoint_seconds=120.0,
        resume=False,
    )
    for key, value in (overrides or {}).items():
        setattr(args, key, value)
    lineage = None
    if resume_from:
        import shutil

        source = Path("/runs") / resume_from
        pointer = json.loads((source / "latest.json").read_text())
        snapshot = source / pointer["checkpoint"]
        inherited = json.loads((snapshot / "metadata.json").read_text())
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copytree(snapshot, directory / pointer["checkpoint"])
        atomic_json(directory / "latest.json", pointer)
        atomic_json(directory / "metadata.json", inherited)
        args = Namespace(
            **inherited["config"],
            run_dir=directory,
            resume=True,
            steps=inherited["steps"] + 1024000,
            max_seconds=budget,
            checkpoint_seconds=(overrides or {}).get("checkpoint_seconds", 60.0),
        )
        lineage = dict(
            source_run=resume_from,
            source_checkpoint=pointer["checkpoint"],
            source_policy_sha256=inherited["policy_sha256"],
            starting_steps=inherited["steps"],
            starting_curriculum=inherited["curriculum"],
            inherited_configuration=True,
        )
        atomic_json(directory / "lineage.json", lineage)
        print("Resuming: " + json.dumps(lineage), flush=True)
    snapshot_calls = []

    def checkpoint_saved(path):
        volume.commit()
        if args.checkpoint_seconds <= 60.0:
            snapshot_calls.append(
                evaluate_snapshot.spawn(run_name, path.name, eval_episodes)
            )

    started = time.monotonic()
    checkpoint = train(args, on_checkpoint=checkpoint_saved)
    training_seconds = time.monotonic() - started
    metadata = json.loads((directory / "metadata.json").read_text())
    episodes = [
        json.loads(line)
        for line in (directory / "episodes.jsonl").read_text().splitlines()
    ]
    summary = dict(
        training_wall_seconds=training_seconds,
        steps=metadata["steps"],
        final_stage=None
        if metadata["curriculum"].get("kind", "").startswith("reverse-grasp-")
        else metadata["curriculum"]["stage"],
        curriculum_state=metadata["curriculum"],
        final_curriculum_level=metadata["curriculum"].get("level"),
        completed_training_episodes=len(episodes),
        training_successes=sum((row["success"] for row in episodes)),
        training_failures=dict(
            Counter((row["reason"] for row in episodes if not row["success"]))
        ),
        evaluations={},
    )
    if lineage:
        summary["lineage"] = lineage
        summary["new_training_steps"] = metadata["steps"] - lineage["starting_steps"]
    for stage in (0, 3):
        options = Namespace(
            model=directory,
            episodes=eval_episodes,
            seed_start=120000,
            stage=stage,
            ablation="normal",
            device="cuda",
            out=directory / f"evaluation-stage{stage}.json",
            video=None,
        )
        result = evaluate(options)
        summary["evaluations"][str(stage)] = {
            key: value for key, value in result.items() if key != "rows"
        }
        atomic_json(directory / "summary.json", summary)
        volume.commit()
        print(json.dumps(summary["evaluations"][str(stage)]), flush=True)
    if metadata["curriculum"].get("kind", "").startswith("reverse-grasp-"):
        import torch
        from stable_baselines3 import PPO
        from robovision.reverse_curriculum import ReverseGraspEnv

        policy = PPO.load(checkpoint / "policy.zip", device="cuda")
        frontier = metadata["curriculum"]["level"]
        summary["hold_evaluations"] = {}
        for level in sorted({0, 5, frontier}):
            for deterministic in (True, False):
                rows = []
                env = ReverseGraspEnv(
                    curriculum_level=level,
                    replay_fraction=0,
                    observation="pixels",
                    max_steps=args.max_episode_steps,
                    gamma=args.gamma,
                )
                try:
                    for seed in range(130000, 130000 + eval_episodes):
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        obs, _ = env.reset(seed=seed)
                        for _ in range(env.max_steps):
                            action, _ = policy.predict(obs, deterministic=deterministic)
                            obs, _, done, truncated, info = env.step(action)
                            if done or truncated:
                                break
                        rows.append(
                            dict(
                                seed=seed,
                                success=info["is_success"],
                                reason=info["reason"],
                                steps=env.step_count,
                                target=info["hold_target_steps"],
                                max_hold=info["max_stable_hold_steps"],
                                contact_steps=info["bilateral_contact_steps"],
                                first_contact_loss=info["first_contact_loss_step"],
                            )
                        )
                finally:
                    env.close()
                key = f"level{level}-{('deterministic' if deterministic else 'stochastic')}"
                result = dict(
                    level=level,
                    deterministic=deterministic,
                    successes=sum((row["success"] for row in rows)),
                    episodes=len(rows),
                    max_hold=max((row["max_hold"] for row in rows)),
                    rows=rows,
                )
                atomic_json(directory / f"evaluation-{key}.json", result)
                summary["hold_evaluations"][key] = {
                    k: v for k, v in result.items() if k != "rows"
                }
        atomic_json(directory / "summary.json", summary)
        volume.commit()
    if snapshot_calls:
        snapshot_results = [call.get() for call in snapshot_calls]
        atomic_json(directory / "checkpoint-evaluations.json", snapshot_results)
        summary["checkpoint_evaluations"] = [
            {
                **r,
                "evaluations": {
                    k: {a: b for a, b in v.items() if a != "rows"}
                    for k, v in r["evaluations"].items()
                },
            }
            for r in snapshot_results
        ]
        atomic_json(directory / "summary.json", summary)
    for stage in (0, 3):
        evaluate(
            Namespace(
                model=directory,
                episodes=1,
                seed_start=120000,
                stage=stage,
                ablation="normal",
                device="cuda",
                out=directory / f"demo-stage{stage}.json",
                video=directory / f"demo-stage{stage}.mp4",
            )
        )
    volume.commit()
    archive_path = Path("/tmp") / f"{run_name}.zip"
    files = [p for p in directory.iterdir() if p.is_file()] + list(checkpoint.iterdir())
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(directory))
    return archive_path.read_bytes()


@app.local_entrypoint()
def main(
    budget: float = 240.0,
    eval_episodes: int = 20,
    tag: str = "normalized",
    wait: bool = False,
    stage: int = 0,
    curriculum: bool = True,
    noise: float = 0.0,
    learning_rate: float = 0.0003,
    curriculum_kind: str = "reverse",
    curriculum_level: int = 0,
    curriculum_replay: float = 0.2,
    curriculum_min_episodes: int = 256,
    curriculum_window: int = 100,
    curriculum_threshold: float = 0.8,
    checkpoint_seconds: float = 120.0,
    hold_exploration_scale: float = 1.0,
    resume_run: str = "",
):
    from datetime import datetime, timezone
    import io
    import zipfile

    run_name = (
        "modal-" + tag + "-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    print(f"Run: {run_name}", flush=True)
    overrides = dict(
        stage=stage,
        curriculum=curriculum,
        hold_exploration_scale=hold_exploration_scale,
        checkpoint_seconds=checkpoint_seconds,
        learning_rate=learning_rate,
    )
    overrides.update(
        curriculum_kind=curriculum_kind,
        curriculum_level=curriculum_level,
        curriculum_replay=curriculum_replay,
        curriculum_min_episodes=curriculum_min_episodes,
        curriculum_window=curriculum_window,
        curriculum_threshold=curriculum_threshold,
    )
    if noise > 0:
        overrides["exploration_std"] = [noise] * 5
    if not wait:
        call = run_training.spawn(
            run_name, budget, eval_episodes, overrides, resume_run or None
        )
        print(
            f"Function call: {call.object_id}; artifacts on robovision-training/{run_name}",
            flush=True,
        )
        return
    payload = run_training.remote(
        run_name, budget, eval_episodes, overrides, resume_run or None
    )
    directory = ROOT / "runs" / run_name
    directory.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(directory)
    print(f"Downloaded final model, reports and videos to {directory}", flush=True)
    print((directory / "summary.json").read_text(), flush=True)
