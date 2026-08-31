# roboVision

A simulated arm that finds, grips and lifts a cup in MuJoCo. PPO chooses where to reach. A second network, trained by behavior cloning, controls the lift and gripper.

The experiment compares a fixed reaching tolerance with a curriculum that tightens the tolerance as the policy improves.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --group dev
```

## Environment

The arm has base, shoulder, elbow and wrist joints, plus a parallel-jaw gripper. It lifts a hollow cup through finger contact, using finite-force actuators.

`VisionCupEnv` takes four actions: X, Y and Z movement, and jaw opening. Observations contain two 96 x 96 RGB frames and the robot's state.

Success requires a 0.5-second hold with both fingers touching and the cup's bottom at least 6 cm above the table. During the hold, tilt must stay below 20 degrees and speed below 0.1 m/s.

```sh
uv run pytest -q
```

## Policies

Color segmentation measures the cyan cup's position and shape in the image. The target network uses these measurements and the robot's state to choose an XY target. A second network takes twelve robot measurements and controls vertical motion and jaw opening.

Checkpoint loaders check the model version and load tensor weights.

Inverse kinematics moves the hand above the chosen target. An XY servo holds it while the grasp network controls vertical motion and the jaws at 20 Hz. The controllers receive no simulator cup coordinates or contact flags.
