"""Inspect recorded results or evaluate the three pinned final checkpoints."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def report(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize the recorded evaluation; does not run policies"
    )
    parser.parse_args(argv)
    result = json.loads(
        (ROOT / "benchmarks/centered-height-evaluation.json").read_text()
    )
    score = result["scores"]
    successes, episodes = score["centered_successes"], score["episodes"]
    low, high = score["centered_wilson95"]
    print(
        f"Recorded complete pickups: {successes}/{episodes} ({successes / episodes:.1%})"
    )
    print(f"95% Wilson interval: {low:.2%}–{high:.2%}")
    print(f"Failures: {score['failures']}")
    print("Scope: fixed cup and scene; centered starting heights from 2.5 to 14 cm.")


def evaluate_final(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate the hash-pinned approach, pickup, and manager policies"
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "models",
        help="Checkpoint directory (default: bundled models)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        default=741000000,
        help="Default replays the recorded scenes; a new seed is a separate evaluation",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("normal", "frozen", "black"),
        default=["normal"],
        help="Camera conditions; normal must be first",
    )
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / "benchmarks/final-model.json").read_text())
    # Check availability before loading any policies or starting simulation.
    for source in manifest["sources"]:
        directory = args.checkpoint_root / source["run"]
        for filename in (source["checkpoint"], source["metadata_file"]):
            if not (directory / filename).is_file():
                parser.error(
                    f"Missing {directory / filename}; see docs/reproduction.md"
                )
    from .skill_hard_evaluation import sha256

    for source in manifest["sources"]:
        directory = args.checkpoint_root / source["run"]
        for name, expected in (
            (source["checkpoint"], source["sha256"]),
            (source["metadata_file"], source["metadata_sha256"]),
        ):
            if sha256(directory / name) != expected:
                parser.error(f"Checksum mismatch: {directory / name}")
    from .ordered_confirmation import run_ordered_confirmation

    sources = [
        {key: source[key] for key in ("run", "checkpoint", "sha256")}
        for source in manifest["sources"]
    ]
    result = run_ordered_confirmation(
        args.checkpoint_root,
        args.out,
        sources[0],
        sources[1:],
        seed=args.seed,
        conditions=tuple(args.conditions),
        option_steps=manifest["manager_option_steps"],
        device=args.device,
    )
    print(json.dumps(result["scores"], indent=2))


def demo(argv=None):
    parser = argparse.ArgumentParser(
        description="Render the highest registered evaluation start"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "runs/demo")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "models")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    from .demo import render_demo

    manifest = json.loads((ROOT / "benchmarks/final-model.json").read_text())
    result = render_demo(args.checkpoint_root, args.out, manifest, device=args.device)
    print(json.dumps(result, indent=2))
