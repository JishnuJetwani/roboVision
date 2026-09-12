"""Bounded remote PPO specialist/manager experiments. Return JSON, never weights."""

import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("robovision-hierarchical-force-ppo")
volume = modal.Volume.from_name("robovision-training")
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


def smoke_skill(out, source, checkpoint, recipe, seed, commit):
    from functools import partial
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from robovision.train_hierarchical import make_skill_env, initialize_skill
    from robovision.skill_hard_evaluation import resolve_skill_checkpoint, sha256
    from robovision.training_checkpoint import save_checkpoint, load_resume, begin_chunk
    from robovision.near_pickup_probe import policy_state_hash

    path, _, _, _ = resolve_skill_checkpoint("/runs", source, checkpoint)
    env = SubprocVecEnv(
        [partial(make_skill_env, w, seed, recipe) for w in range(4)],
        start_method="spawn",
    )
    try:
        model, settings = initialize_skill(path, env, recipe, seed)
        initial = int(model.num_timesteps)
        identity = dict(
            recipe="hierarchical-smoke-" + recipe.skill,
            source=sha256(path),
            config_sha256="0" * 64,
        )
        save_checkpoint(model, out, dict(identity=identity, audit={}), commit)
        transitions = model.n_steps * 4
        begin_chunk(out, transitions, commit, expected_identity=identity)
        _, callback = model._setup_learn(transitions, reset_num_timesteps=False)
        model.collect_rollouts(
            env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps
        )
        maximum = 0.0
        for batch in model.rollout_buffer.get(128):
            with torch.no_grad():
                logs = model._actor_distribution(batch).log_prob(batch.actions)
                maximum = max(maximum, float((logs - batch.old_log_prob).abs().max()))
        if maximum > 0.001:
            raise RuntimeError(f"Sampler log-probability mismatch: {maximum}")
        model.train()
        save_checkpoint(model, out, dict(identity=identity, audit={}), commit)
        restored, _, recovery = load_resume(
            out,
            expected_identity=identity,
            env=env,
            device="cuda",
            resumed_env_seed=seed + 10000,
        )
        if policy_state_hash(restored) != policy_state_hash(model):
            raise RuntimeError("Save/reload changed policy")
        return dict(
            status="complete",
            diagnostic_training_transitions=int(model.num_timesteps) - initial,
            likelihood_max_abs_error=maximum,
            updates=model.last_update_stats,
            recovery=recovery,
            settings=settings,
            workers=env.env_method("specification"),
            candidate=False,
        )
    finally:
        env.close()


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=14400,
    scaledown_window=2,
    volumes={"/runs": volume},
)
def run(
    name,
    source,
    checkpoint,
    skill="approach",
    stage=0,
    mode="train",
    target_steps=196608,
    seed=310000,
    noise_scale=10.0,
    learning_rate=3e-06,
    grasp_bootstrap=False,
    moving_grasp=False,
    centered_approach=False,
    structured=False,
    arrival_pool_run="",
    arrival_pool_sha256="",
    force_open_start=False,
    approach_reward="original",
):
    import traceback
    import torch
    from robovision.io import atomic_json
    from robovision.skill_hard_evaluation import basename, sha256
    from robovision.torch_precision import configure_policy_precision
    from robovision.train_hierarchical import SkillRecipe, train_skill, validate_recipe

    for value in (name, source, checkpoint):
        basename(value)
    recipe = validate_recipe(
        SkillRecipe(
            skill=skill,
            stage=stage,
            noise_scale=noise_scale,
            learning_rate=learning_rate,
            grasp_bootstrap=grasp_bootstrap,
            moving_grasp=moving_grasp,
            centered_approach=centered_approach,
            structured=structured,
            arrival_pool_run=arrival_pool_run,
            arrival_pool_sha256=arrival_pool_sha256,
            approach_reward=approach_reward,
        )
    )
    torch.set_num_threads(2)
    configure_policy_precision()
    root = Path("/root/project")
    hashes = {
        str(p.relative_to(root)): sha256(p) for p in (root / "robovision").glob("*.py")
    }
    hashes["assets/cup_arm.xml"] = sha256(root / "assets/cup_arm.xml")
    args = dict(
        name=name,
        source=source,
        checkpoint=checkpoint,
        skill=skill,
        stage=stage,
        mode=mode,
        target_steps=target_steps,
        seed=seed,
        noise_scale=noise_scale,
        learning_rate=learning_rate,
        grasp_bootstrap=grasp_bootstrap,
        moving_grasp=moving_grasp,
        centered_approach=centered_approach,
        structured=structured,
        arrival_pool_run=arrival_pool_run,
        arrival_pool_sha256=arrival_pool_sha256,
        force_open_start=force_open_start,
        approach_reward=approach_reward,
        source_sha256=hashes,
    )
    out = Path("/runs") / name
    out.mkdir(exist_ok=True)
    config_path = out / "launch-config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != args:
            raise ValueError(
                "Existing run has a different configuration or source tree"
            )
        if (out / "summary.json").exists():
            return json.dumps(dict(status="already_complete", name=name))
        if mode == "smoke":
            raise ValueError(
                "Interrupted smoke requires inspection; no automatic restart"
            )
    else:
        atomic_json(config_path, args)
        volume.commit()
    try:
        if mode == "smoke":
            result = smoke_skill(out, source, checkpoint, recipe, seed, volume.commit)
        elif mode == "train":
            result = train_skill(
                "/runs",
                out,
                source,
                checkpoint,
                recipe=recipe,
                target_steps=target_steps,
                seed=seed,
                commit=volume.commit,
            )
        elif mode in ("evaluate", "evaluate_full_bootstrap"):
            from robovision.train_hierarchical import evaluate_skill
            from robovision.skill_hard_evaluation import resolve_skill_checkpoint
            from robovision.policy_loading import load_grasp_policy

            path, _, _, _ = resolve_skill_checkpoint("/runs", source, checkpoint)
            model = load_grasp_policy(path, device="cuda")
            final = evaluate_skill(
                model,
                recipe,
                final=True,
                seed=seed,
                full_bootstrap_pickup=mode == "evaluate_full_bootstrap",
                force_open_start=force_open_start,
            )
            result = dict(
                status="complete",
                source_run=source,
                source_checkpoint=checkpoint,
                source_policy_sha256=sha256(path),
                diagnostic_training_transitions=0,
                final=final,
                scores=final["scores"],
                source_exploration_unchanged=True,
                approach_reward=approach_reward,
            )
        else:
            raise ValueError("Unknown mode")
        result["source_sha256"] = hashes
        atomic_json(out / "summary.json", result)
        volume.commit()
        return json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "new_steps",
                    "training_seconds",
                    "scores",
                    "diagnostic_training_transitions",
                )
                if k in result
            }
        )
    except Exception as error:
        atomic_json(
            out / "failure.json",
            dict(
                type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            ),
        )
        volume.commit()
        raise


@app.local_entrypoint()
def main(
    name: str,
    source: str,
    checkpoint: str,
    skill: str = "approach",
    stage: int = 0,
    mode: str = "train",
    target_steps: int = 196608,
    seed: int = 310000,
    noise_scale: float = 10.0,
    learning_rate: float = 3e-06,
    grasp_bootstrap: bool = False,
    moving_grasp: bool = False,
    centered_approach: bool = False,
    structured: bool = False,
    arrival_pool_run: str = "",
    arrival_pool_sha256: str = "",
    force_open_start: bool = False,
    approach_reward: str = "original",
):
    from robovision.skill_hard_evaluation import basename

    for value in (name, source, checkpoint):
        basename(value)
    if skill not in ("approach", "grasp", "lift") or stage not in (
        (0, 1, 2, 3) if skill == "grasp" else (0, 1, 2)
    ):
        raise ValueError("Registered skill and stage required")
    if (
        mode not in ("train", "smoke", "evaluate", "evaluate_full_bootstrap")
        or not 8192 <= target_steps <= 1048576
        or target_steps % 8192
    ):
        raise ValueError("Bounded exact interaction budget required")
    if (
        not 0 <= seed < 700000000
        or not 0 < noise_scale <= 50
        or (not 0 < learning_rate <= 0.0003)
    ):
        raise ValueError("Invalid training configuration")
    if grasp_bootstrap and skill != "grasp":
        raise ValueError("Closure bootstrap applies to grasp only")
    if moving_grasp and (not grasp_bootstrap):
        raise ValueError("Moving grasp requires the bootstrap family")
    if centered_approach and (skill != "approach" or stage not in (0, 1)):
        raise ValueError("Centered approach requires approach skill and stage 0 or 1")
    if approach_reward not in ("original",):
        raise ValueError("Registered approach reward required")
    if approach_reward != "original" and (
        skill != "approach"
        or not centered_approach
        or type(stage) is not int
        or (stage != 0)
    ):
        raise ValueError("Clearance reward requires centered approach stage 0")
    if mode == "evaluate_full_bootstrap" and (not grasp_bootstrap):
        raise ValueError("Full bootstrap evaluation requires bootstrap resets")
    if force_open_start and (
        not grasp_bootstrap or mode not in ("evaluate", "evaluate_full_bootstrap")
    ):
        raise ValueError("Opening override is for frozen bootstrap evaluation only")
    if bool(arrival_pool_run) != bool(arrival_pool_sha256):
        raise ValueError("Provide both arrival source and hash")
    if arrival_pool_run:
        basename(arrival_pool_run)
        if skill not in ("grasp", "lift") or len(arrival_pool_sha256) != 64:
            raise ValueError("Invalid successor arrival pool")
    out = ROOT / "runs" / name
    if out.exists():
        raise ValueError(
            "Existing local run: inspect its call before considering a separate resume"
        )
    out.mkdir()
    args = dict(
        name=name,
        source=source,
        checkpoint=checkpoint,
        skill=skill,
        stage=stage,
        mode=mode,
        target_steps=target_steps,
        seed=seed,
        noise_scale=noise_scale,
        learning_rate=learning_rate,
        grasp_bootstrap=grasp_bootstrap,
        moving_grasp=moving_grasp,
        centered_approach=centered_approach,
        structured=structured,
        arrival_pool_run=arrival_pool_run,
        arrival_pool_sha256=arrival_pool_sha256,
        force_open_start=force_open_start,
        approach_reward=approach_reward,
    )
    call = run.spawn(**args)
    launch = dict(
        **args,
        call=call.object_id,
        app=app.app_id,
        returns="JSON only; models stay on Modal",
    )
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps(launch), flush=True)
