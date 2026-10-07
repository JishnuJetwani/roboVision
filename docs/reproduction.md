# Running the controller

## Setup

Use Python 3.11 or 3.12 and `uv`:

```sh
uv sync --group dev
uv run python -m robovision report
uv run pytest -q
uv run ruff format --check robovision scripts tests
```

The three trained checkpoints and their source metadata are included in `models/`.
[The manifest](../benchmarks/final-model.json) pins their SHA-256 hashes.
Each run directory contains only the selected checkpoint and its summary; the
summary also contains the gate records needed to identify intermediate checkpoints.
No cloud account is needed for local playback or evaluation.

## Render the demo

```sh
uv run python -m robovision demo --out runs/demo
```

This renders the highest start in the registered evaluation: seed `741000152`,
a 139.795968 mm height offset, and zero initial cup XY offset. It writes a video,
contact sheet, and scene/model metadata. The included video was rendered with an
NVIDIA L4; local rendering and floating-point differences can affect exact replay.

## Evaluate

```sh
uv run python -m robovision evaluate-final --out runs/evaluation
```

The default seed block is `741000000`–`741000199`, matching the recorded 200 scenes.
No training occurs during evaluation.

Use a new output directory for a new run. The evaluator records model, optimizer,
and training-counter fingerprints before and after evaluation. Repeating the same
scene set is a reproducibility check, not another independent sample. Use
`--device cuda` on a CUDA machine; CPU is the local default.

## Training

| Stage | Implementation | Cloud launcher |
| --- | --- | --- |
| Hold and reverse curriculum | `train_joint.py` | `scripts/train_modal.py` |
| Finger opening and height progression | `train_skill_curriculum.py` | `scripts/skill_curriculum_modal.py` |
| Approach and pickup specialists | `train_hierarchical.py` | `scripts/hierarchical_modal.py` |
| Learned handoff | `train_ordered_pickup.py` | `scripts/ordered_pickup_modal.py` |

Start the initial curriculum with:

```sh
uv run python -m robovision train-joint --run-dir runs/hold-curriculum
```

Later stages take explicit source checkpoints, and arrival-based practice also
requires simulator snapshots from approach rollouts. The included final checkpoints
support inference and evaluation; they are not every intermediate training state
or arrival pool. Training the full sequence requires generating those artifacts.
The launchers expose their stage and source arguments through `--help`.

## Code map

- `env.py`, `joint_env.py`, `grasp_reward.py`: MuJoCo scene and motor-force task.
- `cnn.py`, `decoupled_ppo.py`, `context_exploration.py`, `structured_exploration.py`:
  visual features, separate optimization, and exploration.
- `reverse_curriculum.py`, `open_bootstrap.py`, `fine_pickup.py`, `near_pickup.py`:
  hold, opening, and height lessons.
- `hierarchical_skills.py`, `centered_arrival_env.py`: specialists and arrival resets.
- `ordered_pickup_policy.py`: irreversible learned handoff.
- `policy_state.py`, `ordered_confirmation.py`: frozen-state checks and paired evaluation.
