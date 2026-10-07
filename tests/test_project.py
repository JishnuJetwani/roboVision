"""The public commands must accurately expose the saved evidence."""

import hashlib
import json
from pathlib import Path
import pytest
from robovision.project import evaluate_final, report

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_result_and_demo_match_selected_checkpoints(capsys):
    result = json.loads(
        (ROOT / "benchmarks/centered-height-evaluation.json").read_text()
    )
    rows = result["rows"]
    assert len(rows) == result["scores"]["episodes"] == 200
    assert sum(row["centered_success"] and row["error"] is None for row in rows) == 199
    manifest = json.loads((ROOT / "benchmarks/final-model.json").read_text())
    demo = json.loads((ROOT / "benchmarks/demo.json").read_text())
    assert [row["sha256"] for row in demo["sources"]] == [
        row["sha256"] for row in manifest["sources"]
    ]
    assert (
        hashlib.sha256((ROOT / "assets/demo.mp4").read_bytes()).hexdigest()
        == demo["video_sha256"]
    )
    report([])
    assert "199/200 (99.5%)" in capsys.readouterr().out


def test_final_evaluation_reports_missing_weights_without_creating_output(
    tmp_path, capsys
):
    output = tmp_path / "evaluation"
    with pytest.raises(SystemExit) as error:
        evaluate_final(["--checkpoint-root", str(tmp_path), "--out", str(output)])
    assert error.value.code == 2
    assert "Missing" in capsys.readouterr().err
    assert not output.exists()


def test_registered_demo_is_the_highest_evaluation_scene():
    result = json.loads(
        (ROOT / "benchmarks/centered-height-evaluation.json").read_text()
    )
    highest = max(result["rows"], key=lambda row: row["height"])
    demo = json.loads((ROOT / "benchmarks/demo.json").read_text())
    assert demo["seed"] == highest["seed"] == 741000152
    assert demo["height"] == highest["height"]
    assert demo["cup_offset"] == highest["cup_offset"] == [0.0, 0.0]
    assert demo["initial_cup_position"][:2] == [0.32, 0.0]


def test_bundled_checkpoint_hashes_and_inference():
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from robovision.ordered_confirmation import _resolve_sources
    from robovision.policy_loading import load_grasp_policy
    from robovision.ordered_pickup_policy import OrderedPickupPolicy

    torch.set_num_threads(1)
    manifest = json.loads((ROOT / "benchmarks/final-model.json").read_text())
    sources = [
        {k: row[k] for k in ("run", "checkpoint", "sha256")}
        for row in manifest["sources"]
    ]
    paths, records = _resolve_sources(ROOT / "models", sources[0], sources[1:], 5)
    models = [
        PPO.load(paths[0], device="cpu"),
        *[load_grasp_policy(p, device="cpu") for p in paths[1:]],
    ]
    assert [model.num_timesteps for model in models] == [
        r["recorded_steps"] for r in records
    ]
    stack = OrderedPickupPolicy(models[0], models[1:])
    observation = {
        "image": np.zeros((6, 96, 96), dtype=np.uint8),
        "proprio": np.zeros(18, dtype=np.float32),
    }
    action, state = stack.predict(observation, deterministic=True)
    assert action.shape == (5,) and np.isfinite(action).all() and state is None


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [200], [True]])
def test_shards_reject_invalid_scene_indices(indices):
    from robovision.ordered_confirmation import evaluate_confirmation

    with pytest.raises(ValueError, match="Case indices"):
        evaluate_confirmation(None, [], seed=741000000, case_indices=indices)
