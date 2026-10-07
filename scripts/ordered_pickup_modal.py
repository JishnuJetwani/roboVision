"""Train or evaluate one learned approach-to-pickup commitment on Modal."""

import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("robovision-ordered-pickup")
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
    expert_sources,
    arrival_source=None,
    target_physics_steps=196608,
    seed=319800,
    option_steps=5,
    mode="train",
    manager_source=None,
    profile="fixed",
    training_reward_scale=1.0,
    reset_recipe="arrivals-v1",
):
    import traceback
    import torch
    from robovision.io import atomic_json
    from robovision.skill_hard_evaluation import basename, sha256
    from robovision.torch_precision import configure_policy_precision
    from robovision.train_ordered_pickup import (
        train_ordered_pickup,
        evaluate_ordered_checkpoint,
        ordered_worker_roles,
    )

    basename(name)
    ordered_worker_roles(reset_recipe)
    if reset_recipe != "arrivals-v1" and (mode != "train" or manager_source is None):
        raise ValueError("full-start-v2 is an explicit warm-training reset comparison")
    torch.set_num_threads(2)
    configure_policy_precision()
    project = Path("/root/project")
    hashes = {
        str(p.relative_to(project)): sha256(p)
        for p in (project / "robovision").glob("*.py")
    }
    hashes["assets/cup_arm.xml"] = sha256(project / "assets/cup_arm.xml")
    args = dict(
        name=name,
        expert_sources=expert_sources,
        arrival_source=arrival_source,
        target_physics_steps=target_physics_steps,
        seed=seed,
        option_steps=option_steps,
        mode=mode,
        manager_source=manager_source,
        profile=profile,
        training_reward_scale=training_reward_scale,
        reset_recipe=reset_recipe,
        source_sha256=hashes,
    )
    out = Path("/runs") / name
    out.mkdir(exist_ok=True)
    config_path = out / "launch-config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != args:
            raise ValueError("Existing run differs in configuration or source")
        if (out / "summary.json").exists():
            return json.dumps(dict(status="already_complete", name=name))
    else:
        atomic_json(config_path, args)
        volume.commit()
    try:
        if mode == "train":
            result = train_ordered_pickup(
                "/runs",
                out,
                expert_sources,
                arrival_source=arrival_source,
                target_physics_steps=target_physics_steps,
                seed=seed,
                option_steps=option_steps,
                training_reward_scale=training_reward_scale,
                manager_source=manager_source,
                reset_recipe=reset_recipe,
                commit=volume.commit,
            )
        elif mode == "evaluate":
            result = evaluate_ordered_checkpoint(
                "/runs",
                manager_source,
                expert_sources,
                seed=seed,
                option_steps=option_steps,
                profile=profile,
                original_task_control=True,
            )
        else:
            raise ValueError(
                "Registered ordered training or frozen evaluation mode required"
            )
        result["source_sha256"] = hashes
        atomic_json(out / "summary.json", result)
        volume.commit()
        return json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "new_steps",
                    "new_physical_steps",
                    "training_seconds",
                    "scores",
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
    approach_source: str,
    approach_checkpoint: str,
    pickup_source: str,
    pickup_checkpoint: str,
    arrival_run: str = "",
    arrival_sha256: str = "",
    target_physics_steps: int = 196608,
    seed: int = 319800,
    option_steps: int = 5,
    mode: str = "train",
    manager_run: str = "",
    manager_checkpoint: str = "",
    profile: str = "fixed",
    training_reward_scale: float = 1.0,
    reset_recipe: str = "arrivals-v1",
):
    from robovision.skill_hard_evaluation import basename

    for value in (
        name,
        approach_source,
        approach_checkpoint,
        pickup_source,
        pickup_checkpoint,
    ):
        basename(value)
    if mode not in ("train", "evaluate") or option_steps not in (5, 25):
        raise ValueError("Registered mode and manager duration required")
    if not 64000 <= target_physics_steps <= 1048576 or not 0 <= seed < 700000000:
        raise ValueError("Bounded physical budget and development seed required")
    if profile not in ("fixed", "fresh40") or (mode == "train" and profile != "fixed"):
        raise ValueError("Fresh profile is for frozen evaluation")
    if training_reward_scale not in (1.0, 0.01) or (
        mode != "train" and training_reward_scale != 1.0
    ):
        raise ValueError(
            "Training reward scale must be 1 or .01; evaluation uses raw task rewards"
        )
    if bool(manager_run) != bool(manager_checkpoint) or (
        mode == "evaluate" and (not manager_run)
    ):
        raise ValueError(
            "Provide an exact manager run/checkpoint pair; frozen evaluation requires one"
        )
    if reset_recipe not in ("arrivals-v1", "full-start-v2") or (
        reset_recipe == "full-start-v2" and (mode != "train" or not manager_run)
    ):
        raise ValueError(
            "Registered reset recipe required; full-start-v2 requires warm training"
        )
    arrival_source = None
    if arrival_run or arrival_sha256:
        basename(arrival_run)
        if len(arrival_sha256) != 64 or any(
            (c not in "0123456789abcdef" for c in arrival_sha256)
        ):
            raise ValueError("Exact approach arrival pool SHA256 required")
        arrival_source = dict(run=arrival_run, sha256=arrival_sha256)
    if mode == "train" and arrival_source is None:
        raise ValueError("Ordered training requires a verified approach arrival pool")
    manager_source = None
    if manager_run:
        basename(manager_run)
        basename(manager_checkpoint)
        if not manager_checkpoint.endswith(".zip"):
            raise ValueError("Manager checkpoint must be an explicit ZIP")
        if mode == "train" and manager_run == name:
            raise ValueError("Warm continuation requires a new run name")
        manager_source = dict(run=manager_run, checkpoint=manager_checkpoint)
    out = ROOT / "runs" / name
    if out.exists():
        raise ValueError("Inspect existing call; do not duplicate")
    out.mkdir()
    args = dict(
        name=name,
        expert_sources=[
            dict(run=approach_source, checkpoint=approach_checkpoint),
            dict(run=pickup_source, checkpoint=pickup_checkpoint),
        ],
        arrival_source=arrival_source,
        target_physics_steps=target_physics_steps,
        seed=seed,
        option_steps=option_steps,
        mode=mode,
        manager_source=manager_source,
        profile=profile,
        training_reward_scale=training_reward_scale,
        reset_recipe=reset_recipe,
    )
    call = run.spawn(**args)
    launch = dict(
        **args,
        call=call.object_id,
        app=app.app_id,
        returns="JSON only; all models and arrival pools remain on Modal",
    )
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps(launch), flush=True)
