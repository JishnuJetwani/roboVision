"""Run an explicitly hash-pinned frozen nominal confirmation on Modal."""

import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("robovision-ordered-nominal-confirmation")
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
    .add_local_file(
        ROOT / "scripts/ordered_confirmation_modal.py",
        "/root/project/scripts/ordered_confirmation_modal.py",
    )
)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=1800,
    scaledown_window=2,
    volumes={"/runs": volume},
)
def run(
    name,
    manager_source,
    expert_sources,
    seed=740000000,
    conditions=("normal",),
    option_steps=5,
):
    _basename(name)
    out = Path("/runs") / name
    had_output = out.exists() and any(out.iterdir())
    try:
        import torch
        from robovision.ordered_confirmation import run_ordered_confirmation
        from robovision.torch_precision import configure_policy_precision

        torch.set_num_threads(2)
        configure_policy_precision()
        result = run_ordered_confirmation(
            "/runs",
            out,
            manager_source,
            expert_sources,
            seed=seed,
            conditions=conditions,
            option_steps=option_steps,
            device="cuda",
            commit=volume.commit,
        )
    except Exception as error:
        if not had_output and (not (out / "failure.json").exists()):
            import traceback
            from robovision.io import atomic_json

            out.mkdir(parents=True, exist_ok=True)
            atomic_json(
                out / "failure.json",
                dict(
                    type=type(error).__name__,
                    message=str(error),
                    traceback=traceback.format_exc(),
                    new_physical_training_steps=0,
                    evaluation_work="Unknown if no evaluator progress was persisted; do not infer zero",
                ),
            )
            volume.commit()
        raise
    return json.dumps(
        dict(
            status=result["status"],
            scores=result["scores"],
            identity=result["identity"],
            new_physical_training_steps=0,
        )
    )


def _basename(value):
    if (
        not isinstance(value, str)
        or value in ("", ".", "..")
        or Path(value).name != value
    ):
        raise ValueError("Run and checkpoint names must be explicit basenames")


def _validate_cli(seed, conditions, option_steps, sources):
    """Stdlib-only validation: the local Modal CLI does not have Torch/SB3."""
    if type(seed) is not int or not 700000000 <= seed <= 799999800:
        raise ValueError("Use the dedicated development seed range700000000–799999800")
    if (
        not conditions
        or conditions[0] != "normal"
        or len(set(conditions)) != len(conditions)
        or any((c not in ("normal", "frozen", "black") for c in conditions))
    ):
        raise ValueError(
            "Normal must be first, with optional unique frozen/black conditions"
        )
    if type(option_steps) is not int or option_steps not in (5, 25):
        raise ValueError("Selected manager duration must be5 or25")
    for source in sources:
        _basename(source["run"])
        _basename(source["checkpoint"])
        value = source["sha256"]
        if (
            not source["checkpoint"].endswith(".zip")
            or not isinstance(value, str)
            or len(value) != 64
            or any((c not in "0123456789abcdef" for c in value))
        ):
            raise ValueError("Each source needs an explicit ZIP and lowercase SHA256")


@app.local_entrypoint()
def main(
    name: str,
    manager_run: str,
    manager_checkpoint: str,
    manager_sha256: str,
    approach_run: str,
    approach_checkpoint: str,
    approach_sha256: str,
    pickup_run: str,
    pickup_checkpoint: str,
    pickup_sha256: str,
    seed: int = 740000000,
    conditions: str = "normal",
    option_steps: int = 5,
):
    _basename(name)
    parsed = tuple(conditions.split(","))
    manager = dict(
        run=manager_run, checkpoint=manager_checkpoint, sha256=manager_sha256
    )
    experts = [
        dict(run=approach_run, checkpoint=approach_checkpoint, sha256=approach_sha256),
        dict(run=pickup_run, checkpoint=pickup_checkpoint, sha256=pickup_sha256),
    ]
    _validate_cli(seed, parsed, option_steps, [manager, *experts])
    for source in [manager, *experts]:
        if source["run"] == name:
            raise ValueError(
                "Confirmation requires a new output name separate from source runs"
            )
    out = ROOT / "runs" / name
    if out.exists():
        raise ValueError(
            "Inspect existing confirmation call; do not launch duplicate episodes"
        )
    out.mkdir(parents=True)
    args = dict(
        name=name,
        manager_source=manager,
        expert_sources=experts,
        seed=seed,
        conditions=parsed,
        option_steps=option_steps,
    )
    (out / "selection.json").write_text(json.dumps(args, indent=2) + "\n")
    call = run.spawn(**args)
    launch = dict(
        **args,
        call=call.object_id,
        app=app.app_id,
        returns="JSON only; all model files remain on Modal",
    )
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps(launch), flush=True)
