"""Explicit final-height evaluation of skill policies; returns JSON only."""

import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("robovision-skill-hard-evaluation")
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
    timeout=7200,
    scaledown_window=2,
    volumes={"/runs": volume},
)
def run_evaluation(
    name, source, checkpoint, seeds, conditions, stochastic, seed_provenance
):
    from robovision.skill_hard_evaluation import evaluate_skill_hard, basename
    from robovision.io import atomic_json

    output = Path("/runs") / basename(name)
    output.mkdir(exist_ok=False)
    result = evaluate_skill_hard(
        "/runs",
        source,
        checkpoint,
        seeds,
        conditions=conditions,
        stochastic=stochastic,
        seed_provenance=seed_provenance,
    )
    atomic_json(output / "summary.json", result)
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    name: str,
    source: str,
    checkpoint: str,
    seed_file: str,
    conditions: str = "normal,frozen,black",
    stochastic: bool = False,
):
    from robovision.skill_hard_evaluation import protocol, basename

    for value in (name, source, checkpoint):
        basename(value)
    seed_data = json.loads(Path(seed_file).read_text())
    if not isinstance(seed_data, dict) or "seeds" not in seed_data:
        raise ValueError(
            "seed-file must contain a JSON object with seeds and optional provenance"
        )
    spec = protocol(
        seed_data["seeds"],
        conditions=conditions.split(","),
        stochastic=stochastic,
        seed_provenance=seed_data.get("provenance"),
    )
    output = ROOT / "runs" / name
    if (output / "launch.json").exists():
        raise ValueError("Evaluation already has a launch handle")
    output.mkdir(exist_ok=True)
    call = run_evaluation.spawn(
        name,
        source,
        checkpoint,
        spec["seeds"],
        spec["conditions"],
        spec["stochastic"],
        spec["seed_provenance"],
    )
    launch = dict(
        name=name,
        source_run=source,
        checkpoint=checkpoint,
        protocol=spec,
        call=call.object_id,
        app=app.app_id,
        returns="JSON report only; weights remain on Modal",
    )
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps(launch), flush=True)
