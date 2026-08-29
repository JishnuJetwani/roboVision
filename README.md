# roboVision

A simulated arm that finds, grips and lifts a cup in MuJoCo. PPO chooses where to reach. A second network, trained by behavior cloning, controls the lift and gripper.

The experiment compares a fixed reaching tolerance with a curriculum that tightens the tolerance as the policy improves.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --group dev
```
