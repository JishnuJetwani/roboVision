"""Frozen video and trajectory probe; returns JSON only."""

import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("robovision-skill-video")
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
    timeout=900,
    scaledown_window=2,
    volumes={"/runs": volume},
)
def record(name, source, checkpoint, variant, stage, seed, mode):
    from robovision.skill_video import record_skill_video

    result = record_skill_video(
        "/runs",
        name,
        source,
        checkpoint,
        variant=variant,
        stage=stage,
        seed=seed,
        mode=mode,
    )
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    name: str,
    source: str,
    checkpoint: str,
    variant: str,
    seed: int,
    stage: int = -1,
    mode: str = "deterministic",
):
    if variant not in (
        "near_pickup",
        "fine_near_pickup",
        "millimeter_near_pickup",
        "final_height",
    ) or mode not in ("deterministic", "base"):
        raise ValueError("Invalid variant or sampling mode")
    if not 0 <= seed < 2**32:
        raise ValueError("Invalid seed")
    if (
        variant in ("near_pickup", "fine_near_pickup", "millimeter_near_pickup")
        and stage < 0
    ):
        raise ValueError("Near pickup requires --stage")
    if variant == "final_height" and stage != -1:
        raise ValueError("Final-height forbids --stage")
    from robovision.skill_hard_evaluation import basename

    for value in (name, source, checkpoint):
        basename(value)
    output = ROOT / "runs" / name
    if (output / "launch.json").exists():
        raise ValueError("Video run already has a handle")
    output.mkdir(exist_ok=True)
    call = record.spawn(
        name, source, checkpoint, variant, None if stage == -1 else stage, seed, mode
    )
    launch = dict(
        name=name,
        source=source,
        checkpoint=checkpoint,
        variant=variant,
        stage=stage,
        seed=seed,
        mode=mode,
        call=call.object_id,
        app=app.app_id,
        returns="JSON only; MP4 remains remote",
    )
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps(launch), flush=True)
