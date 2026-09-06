from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO
import torch
from torch import nn

from robovision.models import export_target, load_target
from robovision.reach_env import PrecisionReachEnv
from robovision.vision import CupMomentExtractor, image_moments


def test_color_moments_and_missing_detection():
    image = torch.zeros(2, 6, 96, 96)
    image[0, -2:, 30:40, 60:70] = .7
    moments = image_moments(image)
    assert moments[0, 0] > 0
    assert moments[0, 1] < 0
    assert moments[0, -1] == 1
    assert torch.count_nonzero(moments[1]) == 0


def test_export_preserves_preprocessing_and_actions(tmp_path):
    env = PrecisionReachEnv(render_images=False)
    try:
        model = PPO("MultiInputPolicy", env, device="cpu", seed=7,
                    n_steps=8, batch_size=8,
                    policy_kwargs={"features_extractor_class": CupMomentExtractor,
                                   "net_arch": {"pi": [128, 64], "vf": [128, 64]},
                                   "activation_fn": nn.ReLU})
        path = tmp_path / "target.pt"
        export_target(model, path)
        actor = load_target(path)
        observation, _ = env.reset(seed=3)
        for x in (20, 60):
            observation["image"][:] = 0
            observation["image"][-2:, 30:45, x:x + 10] = 180
            expected, _ = model.predict(observation, deterministic=True)
            actual, _ = actor.predict(observation)
            np.testing.assert_array_equal(actual, expected)
    finally:
        env.close()


def test_incompatible_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "target.pt"
    torch.save({"version": "unrecognized"}, path)
    with pytest.raises(ValueError, match="Incompatible target checkpoint"):
        load_target(path)
